use super::match_runner::{MatchConfig, MatchResult, MatchRunner};
use super::sprt::{SprtConfig, SprtStatus};
use crate::board::Position;
use crate::eval::EvalMode;
use crate::eval::halfkp::HalfKPEvaluator;
use crate::eval::halfkp_trainer::{HalfKPTrainer, TrainStepLoss};
use crate::selfplay::dataset::DatasetHandler;
use crate::selfplay::{SelfPlayConfig, SelfPlayManager};
use std::path::Path;
use std::sync::Arc;

pub struct LoopConfig {
    pub iterations: usize,
    pub start_iteration: Option<usize>,
    pub state_path: String,
    pub games_per_iteration: usize,
    pub eval_pairs: usize,
    pub threads: usize,
    pub depth: u8,
    pub epochs: usize,
    pub lr: f32,
    pub batch_size: usize,
    pub data_path: String,
    pub deep_data_path: String,
    pub best_model_path: String,
    pub candidate_model_path: String,
    pub candidate_ckpt_path: String,
    pub min_promotion_games: usize,
    pub summary_path: String,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            iterations: 3,
            start_iteration: None,
            state_path: "loop_state.txt".to_string(),
            games_per_iteration: 120,
            eval_pairs: 15,
            threads: 2,
            depth: 2,
            epochs: 3,
            lr: 0.001,
            batch_size: 1024,
            data_path: "loop_dataset.tsv".to_string(),
            deep_data_path: "deep_dataset.tsv".to_string(),
            best_model_path: "models/best_halfkp.bin".to_string(),
            candidate_model_path: "models/candidate_halfkp.bin".to_string(),
            candidate_ckpt_path: "models/candidate_halfkp_ckpt.bin".to_string(),
            min_promotion_games: 20,
            summary_path: "loop_summary.csv".to_string(),
        }
    }
}

pub struct SelfImprovementLoop;

impl SelfImprovementLoop {
    /// 自律的自己改善ループを実行 (本格 HalfKP 204,120次元)
    pub fn run(config: &LoopConfig) {
        println!("============================================================");
        println!("=== TabulaShogi Autonomous Self-Improvement Loop (HalfKP) ===");
        println!(
            "Iterations: {}, Games/Iter: {}, EvalPairs: {} ({} games)",
            config.iterations,
            config.games_per_iteration,
            config.eval_pairs,
            config.eval_pairs * 2
        );
        println!(
            "Threads: {}, Depth: {}, Epochs: {}, LR: {}, BatchSize: {}",
            config.threads, config.depth, config.epochs, config.lr, config.batch_size
        );
        println!("Dataset: {}", config.data_path);
        println!("Best Model: {}", config.best_model_path);
        println!("Candidate Model: {}", config.candidate_model_path);
        println!("Candidate Ckpt: {}", config.candidate_ckpt_path);
        println!("Min Promotion Games: {}", config.min_promotion_games);
        println!("Summary Log: {}", config.summary_path);
        println!("============================================================");

        // 初期モデルの確認 (HalfKPモデルが存在するか)
        let mut current_best_eval = if Path::new(&config.best_model_path).exists() {
            match HalfKPEvaluator::load_from_file(&config.best_model_path) {
                Ok(eval) => {
                    println!(
                        "Loaded initial best HalfKP model from '{}'",
                        config.best_model_path
                    );
                    EvalMode::HalfKP(Arc::new(eval))
                }
                Err(e) => {
                    println!(
                        "Failed to load '{}' ({e}), using HCE as base",
                        config.best_model_path
                    );
                    EvalMode::Hce
                }
            }
        } else {
            println!("No existing best HalfKP model found. Starting with initial HCE.");
            EvalMode::Hce
        };

        let start_gen = match config.start_iteration {
            Some(s) => s,
            None => {
                if Path::new(&config.state_path).exists() {
                    std::fs::read_to_string(&config.state_path)
                        .ok()
                        .and_then(|s| s.trim().parse::<usize>().ok())
                        .map(|last| last + 1)
                        .unwrap_or(1)
                } else {
                    1
                }
            }
        };
        println!(
            "Cumulative Generation Offset: Starting at Gen {}",
            start_gen
        );

        for round in 1..=config.iterations {
            let cur_gen = start_gen + round - 1;
            let gen_start_instant = std::time::Instant::now();
            println!(
                "\n>>> Generation {} (Round {} / {}) <<<",
                cur_gen, round, config.iterations
            );

            // Step 1: 自己対局データ生成 (Policy Mismatch / OOD 解消のため Champion 50% / Candidate 50% 混合)
            // 候補モデルが存在する場合は、Candidate による自己対局を 50% 混ぜて未知の局面・疑問手を収集し、
            // Step 2 の深読み教師で再評価・矯正する (DAgger-like 探索データ統合)
            let half_games = config.games_per_iteration / 2;
            let champ_games = config.games_per_iteration - half_games;
            let gen_seed = 0x9E3779B97F4A7C15u64
                .wrapping_add((cur_gen as u64).wrapping_mul(0x517cc1b727220a95));

            println!(
                "\n--- Step 1: Self-Play Data Generation ({} Champion games + {} Candidate exploration games) ---",
                champ_games, half_games
            );

            // 王者による自己対局
            let sp_cfg_champ = SelfPlayConfig {
                num_games: champ_games,
                threads: config.threads,
                depth: config.depth,
                data_output: Some(config.data_path.clone()),
                eval_mode: current_best_eval.clone(),
                seed: gen_seed,
                ..Default::default()
            };
            SelfPlayManager::run(sp_cfg_champ);

            // 候補モデル（Candidate HalfKP）による探査自己対局（存在する場合）
            let candidate_eval_opt = if Path::new(&config.candidate_model_path).exists() {
                HalfKPEvaluator::load_from_file(&config.candidate_model_path).ok()
            } else {
                None
            };

            if let Some(cand_eval) = candidate_eval_opt {
                let sp_cfg_cand = SelfPlayConfig {
                    num_games: half_games,
                    threads: config.threads,
                    depth: config.depth,
                    data_output: Some(config.data_path.clone()),
                    eval_mode: EvalMode::HalfKP(Arc::new(cand_eval)),
                    seed: gen_seed.wrapping_add(0x85ebca6b),
                    start_game_id: champ_games,
                    ..Default::default()
                };
                SelfPlayManager::run(sp_cfg_cand);
            } else if half_games > 0 {
                // 初回等でCandidateが存在しない場合はChampionで全数補完
                let sp_cfg_fallback = SelfPlayConfig {
                    num_games: half_games,
                    threads: config.threads,
                    depth: config.depth,
                    data_output: Some(config.data_path.clone()),
                    eval_mode: current_best_eval.clone(),
                    seed: gen_seed.wrapping_add(0x85ebca6b),
                    start_game_id: champ_games,
                    ..Default::default()
                };
                SelfPlayManager::run(sp_cfg_fallback);
            }

            // Step 2: データセット読込 & IIZ 深読み再評価 & スパース AdamW HalfKP 学習
            println!(
                "\n--- Step 2: Training Candidate Model from Dataset (IIZ Distillation + HalfKP AdamW) ---"
            );
            let mut dataset = match DatasetHandler::load_sampled_with_deep_pool(
                &config.data_path,
                Some(&config.deep_data_path),
                100_000,
                0.5,
                0.5,
                gen_seed,
            ) {
                Ok(d) if !d.is_empty() => d,
                _ => {
                    println!("Warning: No dataset found or empty, skipping iteration.");
                    continue;
                }
            };

            // やねうら王流 IIZ (多重反復雑巾絞り): 最新サンプリングのうち最大 5,000 局面を Depth+2 で深読み再評価
            let relabel_count = 5_000.min(dataset.len());
            let relabel_depth = config.depth.saturating_add(2);
            let t_relabel = std::time::Instant::now();
            let successful_relabelled = DatasetHandler::relabel_deep(
                &mut dataset,
                relabel_count,
                relabel_depth,
                config.threads,
            );
            println!(
                "IIZ Distillation: Successfully re-evaluated {}/{} positions at Depth {} in {:.2}s",
                successful_relabelled.len(),
                relabel_count,
                relabel_depth,
                t_relabel.elapsed().as_secs_f64()
            );

            // 探索が正常完了した真の深読み教師局面のみを永続プールファイルに追記
            if !successful_relabelled.is_empty()
                && let Err(e) =
                    DatasetHandler::append_to_file(&config.deep_data_path, &successful_relabelled)
            {
                eprintln!("Warning: Failed to persist deep relabeled pool: {e}");
            }

            println!(
                "Sampled {} training positions (50% recent / 50% history). Training HalfKP for {} epochs (batch_size: {})...",
                dataset.len(),
                config.epochs,
                config.batch_size
            );

            // 学習スコープを明確に区切り、Trainer(314MB)を対戦前に確実にヒープ解放
            let (candidate_eval, init_mse, final_mse, loss_reduction) = {
                let mut trainer = if Path::new(&config.candidate_ckpt_path).exists() {
                    match HalfKPTrainer::load_checkpoint(&config.candidate_ckpt_path) {
                        Ok(t) => {
                            println!(
                                "[Resume] Loaded candidate AdamW checkpoint from '{}'",
                                config.candidate_ckpt_path
                            );
                            t
                        }
                        Err(e) => {
                            println!(
                                "Failed to load candidate checkpoint '{}' ({e}), initializing from champion",
                                config.candidate_ckpt_path
                            );
                            match &current_best_eval {
                                EvalMode::HalfKP(best) => HalfKPTrainer::from_evaluator(best),
                                _ => HalfKPTrainer::new(),
                            }
                        }
                    }
                } else {
                    match &current_best_eval {
                        EvalMode::HalfKP(best) => {
                            println!(
                                "[WarmStart] Initializing trainer from current best HalfKP model"
                            );
                            HalfKPTrainer::from_evaluator(best)
                        }
                        _ => {
                            println!("[Init] Initializing fresh HalfKPTrainer");
                            HalfKPTrainer::new()
                        }
                    }
                };

                let t_train_start = std::time::Instant::now();
                let mut init_loss = TrainStepLoss::zero();
                let mut final_loss = TrainStepLoss::zero();
                let mut is_first_batch = true;
                let k_scale = 600.0f32;

                if let Some(parent) = Path::new(&config.candidate_model_path).parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                if let Some(parent) = Path::new(&config.candidate_ckpt_path).parent() {
                    let _ = std::fs::create_dir_all(parent);
                }

                let mut rng_seed = gen_seed.wrapping_add(0xdeadbeef);

                for epoch in 1..=config.epochs {
                    println!(
                        "  [Epoch {}/{}] Training {} positions (batch_size: {})...",
                        epoch,
                        config.epochs,
                        dataset.len(),
                        config.batch_size
                    );
                    // 各エポックでインプレースシャッフル
                    if dataset.len() > 1 {
                        let mut rng = crate::selfplay::game::SimpleRng::new(rng_seed);
                        rng_seed = rng_seed.wrapping_add(0x9e3779b97f4a7c15);
                        for i in (1..dataset.len()).rev() {
                            let j = rng.gen_range(i + 1);
                            dataset.swap(i, j);
                        }
                    }

                    for chunk in dataset.chunks(config.batch_size) {
                        let mut batch_samples: Vec<(Vec<usize>, Vec<usize>, f32)> =
                            Vec::with_capacity(chunk.len());

                        for entry in chunk {
                            if let Ok(pos) = Position::from_sfen(&entry.sfen) {
                                let mover_feats = HalfKPEvaluator::extract_halfkp_features(
                                    &pos,
                                    pos.side_to_move,
                                );
                                let opp_feats = HalfKPEvaluator::extract_halfkp_features(
                                    &pos,
                                    pos.side_to_move.opposite(),
                                );
                                // 教師ターゲット: 深読み評価値と勝敗結果のハイブリッド蒸留
                                let pred_eval = HalfKPTrainer::sigmoid(entry.score as f32, k_scale);
                                let target = (0.8 * pred_eval + 0.2 * entry.result).clamp(0.0, 1.0);
                                batch_samples.push((mover_feats, opp_feats, target));
                            }
                        }

                        if !batch_samples.is_empty() {
                            let loss = trainer.train_batch(&batch_samples, config.lr, k_scale);
                            if is_first_batch {
                                init_loss = loss;
                                is_first_batch = false;
                            }
                            final_loss = loss;
                        }
                    }
                }

                let reduction = if init_loss.mse_loss > 0.0 {
                    (init_loss.mse_loss - final_loss.mse_loss) / init_loss.mse_loss * 100.0
                } else {
                    0.0
                };
                println!(
                    "HalfKP Training Complete in {:.2}s! MSE Loss: {:.6} -> {:.6} (Reduction: {:.2}%), Range Loss: {:.6}",
                    t_train_start.elapsed().as_secs_f64(),
                    init_loss.mse_loss,
                    final_loss.mse_loss,
                    reduction,
                    final_loss.range_loss
                );

                let cand_eval = trainer.to_evaluator();
                if let Err(e) = cand_eval.save_to_file(&config.candidate_model_path) {
                    eprintln!("Error saving candidate model: {e}");
                }

                if let Err(e) = trainer.save_checkpoint(&config.candidate_ckpt_path) {
                    eprintln!("Error saving candidate checkpoint: {e}");
                }

                println!(
                    "[Memory] Dropping HalfKPTrainer to reclaim ~314MB heap before arena matches..."
                );
                drop(trainer);

                (
                    cand_eval,
                    init_loss.mse_loss,
                    final_loss.mse_loss,
                    reduction,
                )
            };

            // Step 3: アリーナ対戦 & レーティング検定 (Candidate vs Best)
            println!("\n--- Step 3: Arena Match & SPRT Testing (Peak Memory < 200MB) ---");
            let mut current_pairs = config.eval_pairs;
            let match_cfg = MatchConfig {
                name_a: format!("Candidate_Gen{cur_gen}"),
                name_b: "Best_Model".to_string(),
                eval_a: EvalMode::HalfKP(Arc::new(candidate_eval.clone())),
                eval_b: current_best_eval.clone(),
                pairs: current_pairs,
                depth: config.depth,
                threads: config.threads,
                random_opening: 6,
                max_plies: 320,
                tt_size_mb: 16,
                sprt_config: Some(SprtConfig {
                    elo0: 0.0,
                    elo1: 50.0,
                    alpha: 0.05,
                    beta: 0.05,
                }),
            };

            let mut match_res = MatchRunner::run_match(&match_cfg);

            // SPRT判定が Continue かつ勝ち越し傾向 (勝率52%以上) の場合、動的に延長対局
            let max_pairs = config.eval_pairs * 3;
            while let Some(ref sprt) = match_res.sprt {
                if sprt.status == SprtStatus::Continue
                    && match_res.win_rate_a >= 0.52
                    && current_pairs < max_pairs
                {
                    current_pairs += config.eval_pairs;
                    println!(
                        "\n[SPRT Overtime] Indecisive Continue with positive win rate {:.1}% (LLR: {:.2}). Incrementally extending to {} pairs...",
                        match_res.win_rate_a * 100.0,
                        sprt.llr,
                        current_pairs
                    );
                    match_res = MatchRunner::run_match_extended(
                        &match_cfg,
                        Some(&match_res),
                        current_pairs,
                    );
                } else {
                    break;
                }
            }

            // Step 4: 厳格な昇格判定
            let promoted = Self::handle_promotion(
                cur_gen,
                &match_res,
                &candidate_eval,
                &mut current_best_eval,
                &config.best_model_path,
                config.min_promotion_games,
            );

            // 進捗サマリログ (loop_summary.csv) への追記永続化
            let champ_name = match &current_best_eval {
                EvalMode::HalfKP(_) => "HalfKP",
                EvalMode::Nnue(_) => "NNUE",
                EvalMode::Hce => "HCE",
            };
            let sprt_str = match_res
                .sprt
                .as_ref()
                .map(|s| format!("{:?}", s.status))
                .unwrap_or_else(|| "None".to_string());
            let gen_duration = gen_start_instant.elapsed().as_secs_f64();

            Self::append_summary_csv(
                &config.summary_path,
                cur_gen,
                champ_name,
                gen_duration,
                dataset.len(),
                successful_relabelled.len(),
                init_mse,
                final_mse,
                loss_reduction,
                match_res.total_games,
                match_res.win_rate_a,
                match_res.elo_diff_a,
                &sprt_str,
                promoted,
            );

            // 世代番号を永続化（次回再起動時に自動で直前世代から継続可能）
            if let Err(e) = std::fs::write(&config.state_path, cur_gen.to_string()) {
                eprintln!("Warning: Failed to persist generation state: {e}");
            }

            if promoted {
                println!(
                    "[Progression] Gen {cur_gen} Candidate successfully promoted! Fresh cycle will warm-start from new champion."
                );
                // 昇格時は Candidate チェックポイントを整理し、次代は新王者からウォームスタート
                if Path::new(&config.candidate_ckpt_path).exists() {
                    let _ = std::fs::remove_file(&config.candidate_ckpt_path);
                    let bak_path = format!("{}.bak", config.candidate_ckpt_path);
                    let _ = std::fs::remove_file(&bak_path);
                }
            } else {
                println!(
                    "[Progression] Candidate did not beat champion in Gen {cur_gen}. Retaining trained weights & Adam momentum for Gen {}.",
                    cur_gen + 1
                );
            }
        }

        println!("\n============================================================");
        println!("Autonomous Self-Improvement Session Completed!");
        println!("Best model preserved at: {}", config.best_model_path);
        println!("============================================================");
    }

    pub fn handle_promotion(
        iter: usize,
        match_res: &MatchResult,
        candidate_eval: &HalfKPEvaluator,
        current_best_eval: &mut EvalMode,
        best_model_path: &str,
        min_promotion_games: usize,
    ) -> bool {
        // 昇格条件:
        // 1. SPRT が Pass (統計的有意に強い)
        // 2. 最低 min_promotion_games (通常20対局以上) を消化していること (小標本による偶発的早期誤昇格を完全防止)
        let sprt_passed = match_res
            .sprt
            .as_ref()
            .map(|s| s.status == SprtStatus::Pass)
            .unwrap_or(false);

        let min_games_met = match_res.total_games >= min_promotion_games;

        let promoted = sprt_passed && min_games_met;
        if promoted {
            let reason = format!(
                "SPRT Pass (Statistically Significant Superiority, {} games >= {} required)",
                match_res.total_games, min_promotion_games
            );
            println!(
                "\n>>> [PROMOTION] Gen {} Candidate won ({}) with {:.1}% win rate ({:+.1} Elo). Promoting to Best Model! <<<",
                iter,
                reason,
                match_res.win_rate_a * 100.0,
                match_res.elo_diff_a
            );
            if let Some(parent) = Path::new(best_model_path).parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Err(e) = candidate_eval.save_to_file(best_model_path) {
                eprintln!("Error writing promoted best model: {e}");
                false
            } else {
                *current_best_eval = EvalMode::HalfKP(Arc::new(candidate_eval.clone()));
                println!("Successfully promoted and updated '{}'!", best_model_path);
                true
            }
        } else {
            if sprt_passed && !min_games_met {
                println!(
                    "\n>>> [GATE REJECTED] Gen {} Candidate passed SPRT early but had only {} games (< {} required). Promotion deferred. <<<",
                    iter, match_res.total_games, min_promotion_games
                );
            } else {
                println!(
                    "\n>>> [REJECTED] Gen {} Candidate did not surpass Best Model ({:.1}% win rate, Elo {:+.1}). Keeping existing best. <<<",
                    iter,
                    match_res.win_rate_a * 100.0,
                    match_res.elo_diff_a
                );
            }
            false
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn append_summary_csv(
        summary_path: &str,
        generation: usize,
        champ_mode: &str,
        duration_secs: f64,
        dataset_size: usize,
        relabel_count: usize,
        init_mse: f32,
        final_mse: f32,
        reduction_pct: f32,
        match_games: usize,
        win_rate: f64,
        elo_diff: f64,
        sprt_status: &str,
        promoted: bool,
    ) {
        if let Some(parent) = Path::new(summary_path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let file_exists = Path::new(summary_path).exists();
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(summary_path)
        {
            use std::io::Write;
            if !file_exists {
                let _ = writeln!(
                    file,
                    "generation,timestamp,champion_mode,duration_secs,dataset_size,relabel_count,init_mse,final_mse,reduction_pct,match_games,win_rate,elo_diff,sprt_status,promoted"
                );
            }
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let _ = writeln!(
                file,
                "{generation},{timestamp},{champ_mode},{duration_secs:.2},{dataset_size},{relabel_count},{init_mse:.6},{final_mse:.6},{reduction_pct:.2},{match_games},{win_rate:.4},{elo_diff:+.1},{sprt_status},{promoted}"
            );
        }
    }
}
