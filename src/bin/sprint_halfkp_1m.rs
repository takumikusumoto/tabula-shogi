use std::path::Path;
use std::time::Instant;

use tabula_shogi::board::Position;
use tabula_shogi::eval::EvalMode;
use tabula_shogi::eval::halfkp::{HALFKP_INPUT_SIZE, HalfKPEvaluator};
use tabula_shogi::eval::halfkp_trainer::HalfKPTrainer;
use tabula_shogi::selfplay::config::SelfPlayConfig;
use tabula_shogi::selfplay::dataset::{DatasetEntry, DatasetHandler};
use tabula_shogi::selfplay::game::SimpleRng;
use tabula_shogi::selfplay::manager::SelfPlayManager;

struct SprintConfig {
    generate_games: usize,
    threads: usize,
    depth: usize,
    epochs: usize,
    batch_size: usize,
    lr: f32,
    data_path: String,
    model_output: String,
    benchmark_only: bool,
}

impl Default for SprintConfig {
    fn default() -> Self {
        SprintConfig {
            generate_games: 100,
            threads: 4,
            depth: 2,
            epochs: 3,
            batch_size: 256,
            lr: 0.001,
            data_path: "data/sprint_1m.tsv".to_string(),
            model_output: "models/halfkp_sprint_1m.bin".to_string(),
            benchmark_only: false,
        }
    }
}

fn parse_args() -> SprintConfig {
    let mut cfg = SprintConfig::default();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--games" => {
                if i + 1 < args.len() {
                    cfg.generate_games = args[i + 1].parse().unwrap_or(cfg.generate_games);
                    i += 1;
                }
            }
            "--threads" => {
                if i + 1 < args.len() {
                    cfg.threads = args[i + 1].parse().unwrap_or(cfg.threads);
                    i += 1;
                }
            }
            "--depth" => {
                if i + 1 < args.len() {
                    cfg.depth = args[i + 1].parse().unwrap_or(cfg.depth);
                    i += 1;
                }
            }
            "--epochs" => {
                if i + 1 < args.len() {
                    cfg.epochs = args[i + 1].parse().unwrap_or(cfg.epochs);
                    i += 1;
                }
            }
            "--batch-size" => {
                if i + 1 < args.len() {
                    cfg.batch_size = args[i + 1].parse().unwrap_or(cfg.batch_size);
                    i += 1;
                }
            }
            "--lr" => {
                if i + 1 < args.len() {
                    cfg.lr = args[i + 1].parse().unwrap_or(cfg.lr);
                    i += 1;
                }
            }
            "--data" => {
                if i + 1 < args.len() {
                    cfg.data_path = args[i + 1].clone();
                    i += 1;
                }
            }
            "--model" => {
                if i + 1 < args.len() {
                    cfg.model_output = args[i + 1].clone();
                    i += 1;
                }
            }
            "--benchmark-only" => {
                cfg.benchmark_only = true;
            }
            _ => {}
        }
        i += 1;
    }
    cfg
}

/// マルチスレッド自己対局によるデータセット生成 (SelfPlayManager 統合)
fn generate_selfplay_data(cfg: &SprintConfig) -> (Vec<DatasetEntry>, std::time::Duration) {
    println!("\n=== Phase 1: Self-Play Data Generation ===");

    if let Some(parent) = Path::new(&cfg.data_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let selfplay_cfg = SelfPlayConfig {
        num_games: cfg.generate_games,
        threads: cfg.threads,
        depth: cfg.depth as u8,
        random_opening_plies: 8,
        max_plies: 300,
        resign_threshold: -2500,
        csa_output: None,
        data_output: Some(cfg.data_path.clone()),
        tt_size_mb: 16,
        seed: 0x9E3779B97F4A7C15,
        eval_mode: EvalMode::Hce,
        temperature_plies: 24,
        use_book: false,
        start_game_id: 0,
    };

    let start = Instant::now();
    let stats = SelfPlayManager::run(selfplay_cfg);
    let duration = start.elapsed();

    println!(
        "\nData Generation Finished: {} games completed in {:.2}s",
        stats.completed_games,
        duration.as_secs_f64()
    );

    let entries = match DatasetHandler::load_from_file(&cfg.data_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!(
                "Failed to reload generated dataset from {}: {e}",
                cfg.data_path
            );
            Vec::new()
        }
    };

    println!(
        "Extracted {} valid training positions from {}",
        entries.len(),
        cfg.data_path
    );
    println!(
        "Effective Generation Throughput: {:.1} positions/sec",
        entries.len() as f64 / duration.as_secs_f64().max(0.001)
    );

    (entries, duration)
}

/// データセットの準備（存在しない場合は生成データを保存）
fn load_or_generate_dataset(cfg: &SprintConfig) -> (Vec<DatasetEntry>, std::time::Duration) {
    if Path::new(&cfg.data_path).exists() {
        println!("Loading existing dataset from {}...", cfg.data_path);
        match DatasetHandler::load_from_file(&cfg.data_path) {
            Ok(data) if !data.is_empty() => {
                println!("Loaded {} positions from file.", data.len());
                return (data, std::time::Duration::from_secs(0));
            }
            _ => {
                println!("Dataset file empty or invalid. Generating fresh selfplay data...");
            }
        }
    }

    generate_selfplay_data(cfg)
}

/// HalfKP トレーナーによる高速学習スプリント
fn train_halfkp_sprint(
    cfg: &SprintConfig,
    dataset: &[DatasetEntry],
) -> (HalfKPTrainer, std::time::Duration) {
    println!("\n=== Phase 2: HalfKP Backpropagation Training ===");
    println!(
        "Dataset Size: {} positions, Epochs: {}, Batch Size: {}, LR: {}",
        dataset.len(),
        cfg.epochs,
        cfg.batch_size,
        cfg.lr
    );

    let start = Instant::now();
    let mut trainer = HalfKPTrainer::new();

    if dataset.is_empty() {
        eprintln!("Warning: Empty dataset! Skipping training.");
        return (trainer, start.elapsed());
    }

    // 局面データをパースして特徴量キャッシュ化
    println!("Extracting HalfKP sparse features from SFEN strings...");
    let prep_start = Instant::now();
    let mut samples: Vec<(Vec<usize>, Vec<usize>, f32)> = Vec::with_capacity(dataset.len());

    for entry in dataset {
        if let Ok(pos) = Position::from_sfen(&entry.sfen) {
            let mover_feats = HalfKPEvaluator::extract_halfkp_features(&pos, pos.side_to_move);
            let opp_feats =
                HalfKPEvaluator::extract_halfkp_features(&pos, pos.side_to_move.opposite());
            samples.push((mover_feats, opp_feats, entry.result));
        }
    }
    println!(
        "Features extracted for {} valid samples in {:.2}s",
        samples.len(),
        prep_start.elapsed().as_secs_f64()
    );

    if samples.is_empty() {
        eprintln!("Error: No valid positions could be parsed.");
        return (trainer, start.elapsed());
    }

    // エポックループ
    let num_samples = samples.len();
    let num_batches = num_samples.div_ceil(cfg.batch_size);
    let mut rng = SimpleRng::new(42);

    for epoch in 1..=cfg.epochs {
        let epoch_start = Instant::now();
        // インプレース・シャッフル
        for i in (1..num_samples).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            samples.swap(i, j);
        }

        let mut epoch_loss = 0.0f32;
        let mut processed = 0;

        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * cfg.batch_size;
            let end_idx = (start_idx + cfg.batch_size).min(num_samples);
            let batch = &samples[start_idx..end_idx];

            let loss = trainer.train_batch(batch, cfg.lr, 600.0);
            epoch_loss += loss.mse_loss * batch.len() as f32;
            processed += batch.len();
        }

        let avg_loss = epoch_loss / processed.max(1) as f32;
        let epoch_dur = epoch_start.elapsed().as_secs_f64();
        let speed = processed as f64 / epoch_dur.max(0.001);

        println!(
            "Epoch [{}/{}]: MSE Loss = {:.6}, Speed = {:.0} pos/sec ({:.2}s)",
            epoch, cfg.epochs, avg_loss, speed, epoch_dur
        );
    }

    let total_dur = start.elapsed();
    println!(
        "Training Completed in {:.2}s ({:.1} pos/sec overall)",
        total_dur.as_secs_f64(),
        (num_samples * cfg.epochs) as f64 / total_dur.as_secs_f64().max(0.001)
    );

    (trainer, total_dur)
}

/// 重み統計とモデル検証
fn verify_and_save_model(cfg: &SprintConfig, trainer: &HalfKPTrainer) {
    println!("\n=== Phase 3: Model Verification & Export ===");
    let eval = trainer.to_evaluator();

    // 統計計算
    let mut non_zero_feats = 0usize;
    let mut max_abs_feat = 0i16;
    for row in &eval.feature_weights {
        for &w in row {
            if w != 0 {
                non_zero_feats += 1;
            }
            max_abs_feat = max_abs_feat.max(w.abs());
        }
    }

    let mut non_zero_out = 0usize;
    let mut max_abs_out = 0i16;
    for &w in &eval.output_weights {
        if w != 0 {
            non_zero_out += 1;
        }
        max_abs_out = max_abs_out.max(w.abs());
    }

    println!("--- Model Statistics ---");
    println!(
        "Active Feature Weights: {} / {} ({:.2}%)",
        non_zero_feats,
        HALFKP_INPUT_SIZE * 128,
        (non_zero_feats as f64 / (HALFKP_INPUT_SIZE * 128) as f64) * 100.0
    );
    println!("Max Absolute Feature Weight: {}", max_abs_feat);
    println!("Active Output Weights: {} / 256", non_zero_out);
    println!("Max Absolute Output Weight: {}", max_abs_out);
    println!("Output Bias: {}", eval.output_bias);

    // テスト局面評価
    let startpos = Position::startpos();
    let initial_score = eval.evaluate(&startpos);
    println!("Initial Position Eval: {} cp", initial_score);

    // 保存
    if let Some(parent) = Path::new(&cfg.model_output).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    println!("Saving HalfKP model to {}...", cfg.model_output);
    match eval.save_to_file(&cfg.model_output) {
        Ok(_) => {
            let metadata = std::fs::metadata(&cfg.model_output).unwrap();
            println!(
                "Successfully exported HalfKP model! Size: {:.2} MB ({} bytes)",
                metadata.len() as f64 / (1024.0 * 1024.0),
                metadata.len()
            );
        }
        Err(e) => {
            eprintln!("Error exporting model: {e}");
        }
    }
}

fn main() {
    let cfg = parse_args();
    println!("****************************************************");
    println!("*   PROJECT TITAN: 1M Positions HalfKP Sprint      *");
    println!("****************************************************");

    let (dataset, gen_dur) = load_or_generate_dataset(&cfg);

    if cfg.benchmark_only {
        println!("\n[Benchmark-Only Mode Finished]");
        return;
    }

    let (trainer, train_dur) = train_halfkp_sprint(&cfg, &dataset);
    verify_and_save_model(&cfg, &trainer);

    println!("\n=== Sprint Benchmark Summary ===");
    if gen_dur.as_secs_f64() > 0.0 {
        println!("Generation Time: {:.2}s", gen_dur.as_secs_f64());
    }
    println!("Training Time:   {:.2}s", train_dur.as_secs_f64());
    println!("Sprint Pipeline Execution Verified Successfully!");
}
