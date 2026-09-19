use super::match_runner::{MatchConfig, MatchResult, MatchRunner};
use super::sprt::{SprtConfig, SprtStatus};
use crate::board::Position;
use crate::eval::EvalMode;
use crate::eval::halfkp::HalfKPEvaluator;
use crate::eval::halfkp_trainer::{HalfKPTrainer, TrainStepLoss};
use crate::selfplay::dataset::{DatasetEntry, DatasetHandler};
use crate::selfplay::{SelfPlayConfig, SelfPlayManager};
use std::path::Path;
use std::sync::Arc;

/// やねうら王流 IIZ (多重反復雑巾絞り) で 1 世代あたり深読み再評価する最大局面数
pub const MAX_IIZ_RELABEL_PER_GEN: usize = 5_000;
/// データセットサンプリング時の最大局面数上限
pub const MAX_DATASET_SAMPLE_LIMIT: usize = 100_000;
/// 教師ターゲット作成時の深読み評価値の重み (80%)
pub const DISTILLATION_TEACHER_WEIGHT: f32 = 0.8;
/// 教師ターゲット作成時のゲーム勝敗結果の重み (20%)
pub const DISTILLATION_RESULT_WEIGHT: f32 = 0.2;
/// シグモイド勝率変換のデフォルト感度係数 (600 cp で勝率 ~73%)
pub const DEFAULT_SIGMOID_K: f32 = 600.0;

/// 自己対局・アリーナ対戦パラメータ
#[derive(Clone, Debug)]
pub struct LoopArenaParams {
    pub eval_pairs: usize,
    pub threads: usize,
    pub depth: u8,
    pub min_promotion_games: usize,
}

impl Default for LoopArenaParams {
    fn default() -> Self {
        Self {
            eval_pairs: 15,
            threads: 2,
            depth: 2,
            min_promotion_games: 20,
        }
    }
}

/// Candidate 学習ハイパーパラメータ
#[derive(Clone, Debug)]
pub struct LoopTrainingParams {
    pub epochs: usize,
    pub lr: f32,
    pub batch_size: usize,
}

impl Default for LoopTrainingParams {
    fn default() -> Self {
        Self {
            epochs: 3,
            lr: 0.001,
            batch_size: 1024,
        }
    }
}

/// データセット・モデル永続化パス設定
#[derive(Clone, Debug)]
pub struct LoopStoragePaths {
    pub state_path: String,
    pub data_path: String,
    pub deep_data_path: String,
    pub best_model_path: String,
    pub candidate_model_path: String,
    pub candidate_ckpt_path: String,
    pub summary_path: String,
}

impl Default for LoopStoragePaths {
    fn default() -> Self {
        Self {
            state_path: "data/loop_state.txt".to_string(),
            data_path: "data/loop_dataset.tsv".to_string(),
            deep_data_path: "data/deep_dataset.tsv".to_string(),
            best_model_path: "models/best_halfkp.bin".to_string(),
            candidate_model_path: "models/candidate_halfkp.bin".to_string(),
            candidate_ckpt_path: "models/candidate_halfkp_ckpt.bin".to_string(),
            summary_path: "data/loop_summary.csv".to_string(),
        }
    }
}

/// 自律的自己改善ループ全体設定
#[derive(Clone, Debug)]
pub struct LoopConfig {
    pub iterations: usize,
    pub start_iteration: Option<usize>,
    pub games_per_iteration: usize,
    pub arena: LoopArenaParams,
    pub training: LoopTrainingParams,
    pub paths: LoopStoragePaths,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            iterations: 3,
            start_iteration: None,
            games_per_iteration: 120,
            arena: LoopArenaParams::default(),
            training: LoopTrainingParams::default(),
            paths: LoopStoragePaths::default(),
        }
    }
}

/// 世代ごとの学習・評価メトリクスサマリ
#[derive(Clone, Copy, Debug)]
pub struct GenerationMetrics {
    pub dataset_len: usize,
    pub relabelled_count: usize,
    pub init_mse: f32,
    pub final_mse: f32,
    pub loss_reduction: f32,
}

pub struct SelfImprovementLoop;

impl SelfImprovementLoop {
    /// 自律的自己改善ループを実行 (本格 HalfKP 204,120次元)
    pub fn run(config: &LoopConfig) -> Result<(), String> {
        println!("============================================================");
        println!("=== TabulaShogi Autonomous Self-Improvement Loop (HalfKP) ===");
        println!(
            "Iterations: {}, Games/Iter: {}, EvalPairs: {} ({} games)",
            config.iterations,
            config.games_per_iteration,
            config.arena.eval_pairs,
            config.arena.eval_pairs * 2
        );
        println!(
            "Threads: {}, Depth: {}, Epochs: {}, LR: {}, BatchSize: {}",
            config.arena.threads,
            config.arena.depth,
            config.training.epochs,
            config.training.lr,
            config.training.batch_size
        );
        println!("Dataset: {}", config.paths.data_path);
        println!("Best Model: {}", config.paths.best_model_path);
        println!("Candidate Model: {}", config.paths.candidate_model_path);
        println!("Candidate Ckpt: {}", config.paths.candidate_ckpt_path);
        println!("Min Promotion Games: {}", config.arena.min_promotion_games);
        println!("Summary Log: {}", config.paths.summary_path);
        println!("============================================================");

        // 実行に必要な親ディレクトリ群（data/, models/ 等）を自動作成し、クリーンワークツリーでの起動失敗を防止
        for path_str in [
            &config.paths.data_path,
            &config.paths.deep_data_path,
            &config.paths.best_model_path,
            &config.paths.candidate_model_path,
            &config.paths.candidate_ckpt_path,
            &config.paths.summary_path,
            &config.paths.state_path
        ] {
            if let Some(parent) = Path::new(path_str).parent() {
                let _ = std::fs::create_dir_all(parent);
            }
        }

        // 0. パス重複チェック: best_model_path と candidate_model_path の衝突防止 (昇格ゲート無効化の完全防止)
        let best_canonical = Path::new(&config.paths.best_model_path)
            .canonicalize()
            .unwrap_or_else(|_| std::path::PathBuf::from(&config.paths.best_model_path));
        let cand_canonical = Path::new(&config.paths.candidate_model_path)
            .canonicalize()
            .unwrap_or_else(|_| std::path::PathBuf::from(&config.paths.candidate_model_path));
        if best_canonical == cand_canonical
            || config
                .paths
                .best_model_path
                .eq_ignore_ascii_case(&config.paths.candidate_model_path)
        {
            let err_msg = format!(
                "[Fatal Error] best_model_path ('{}') and candidate_model_path ('{}') resolve to the same file! Aborting to protect champion model.",
                config.paths.best_model_path, config.paths.candidate_model_path
            );
            eprintln!("{err_msg}");
            return Err(err_msg);
        }

        // 初期モデルの確認 (HalfKPモデルが存在するか)
        // 既存モデルの読み込み失敗時はサイレントに HCE へ退行せず fail-closed で中断
        let mut current_best_eval = if Path::new(&config.paths.best_model_path).exists() {
            match HalfKPEvaluator::load_from_file(&config.paths.best_model_path) {
                Ok(eval) => {
                    println!(
                        "Loaded initial best HalfKP model from '{}'",
                        config.paths.best_model_path
                    );
                    EvalMode::HalfKP(Arc::new(eval))
                }
                Err(e) => {
                    let err_msg = format!(
                        "[Fatal Error] Failed to load existing best HalfKP model '{}': {e}. Aborting to prevent silent HCE fallback.",
                        config.paths.best_model_path
                    );
                    eprintln!("{err_msg}");
                    return Err(err_msg);
                }
            }
        } else {
            println!("No existing best HalfKP model found. Starting with initial HCE.");
            EvalMode::Hce
        };

        let start_gen = match config.start_iteration {
            Some(s) => s,
            None => {
                if Path::new(&config.paths.state_path).exists() {
                    std::fs::read_to_string(&config.paths.state_path)
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

            // Step 1: 自己対局データ生成 (I/O障害やworker異常時は fail-closed で即座に中断)
            let gen_seed = 0x9E3779B97F4A7C15u64
                .wrapping_add((cur_gen as u64).wrapping_mul(0x517cc1b727220a95));
            if let Err(e) = Self::generate_selfplay_data(
                config.games_per_iteration,
                &config.arena,
                &config.paths,
                &current_best_eval,
                cur_gen,
                gen_seed,
            ) {
                let err_msg = format!(
                    "[Fatal Error] Step 1 Self-play failed in Gen {cur_gen}: {e}. Aborting pipeline to prevent training on corrupt/incomplete data."
                );
                eprintln!("{err_msg}");
                return Err(err_msg);
            }

            // Step 2: データセット準備 & IIZ 深読み再評価
            let (mut dataset, successful_relabelled_count) = match Self::prepare_training_dataset(
                &config.paths,
                &config.arena,
                gen_seed,
                &current_best_eval,
            ) {
                Some(res) => res,
                None => {
                    println!("Warning: No dataset found or empty, skipping iteration.");
                    continue;
                }
            };

            // Step 3: Candidate 学習 & メモリ即時解放
            let (candidate_eval, init_mse, final_mse, loss_reduction) =
                Self::train_candidate_generation(
                    &config.training,
                    &config.paths,
                    gen_seed,
                    &mut dataset,
                    &current_best_eval,
                );

            // Step 4: アリーナ対戦 & SPRT 検定
            let match_res = Self::run_arena_and_sprt(
                &config.arena,
                cur_gen,
                &candidate_eval,
                &current_best_eval,
            );

            // Step 5: 厳格な昇格判定
            let promoted = Self::handle_promotion(
                cur_gen,
                &match_res,
                &candidate_eval,
                &mut current_best_eval,
                &config.paths.best_model_path,
                config.arena.min_promotion_games,
            );

            // Step 6: サマリ記録 & 世代更新
            let metrics = GenerationMetrics {
                dataset_len: dataset.len(),
                relabelled_count: successful_relabelled_count,
                init_mse,
                final_mse,
                loss_reduction,
            };
            Self::record_generation_summary(
                &config.paths,
                cur_gen,
                &current_best_eval,
                gen_start_instant,
                &metrics,
                &match_res,
                promoted,
            );
        }

        println!("\n============================================================");
        println!("Autonomous Self-Improvement Session Completed!");
        println!("Best model preserved at: {}", config.paths.best_model_path);
        println!("============================================================");
        Ok(())
    }

    /// Step 1: 自己対局データ生成 (Policy Mismatch / OOD 解消のため Champion 50% / Candidate 50% 混合)
    fn generate_selfplay_data(
        games_per_iteration: usize,
        arena: &LoopArenaParams,
        paths: &LoopStoragePaths,
        current_best_eval: &EvalMode,
        cur_gen: usize,
        gen_seed: u64,
    ) -> Result<(), String> {
        let half_games = games_per_iteration / 2;
        let champ_games = games_per_iteration - half_games;

        println!(
            "\n--- Step 1: Self-Play Data Generation (Gen {cur_gen}: {} Champion games + {} Candidate exploration games) ---",
            champ_games, half_games
        );

        // 王者による自己対局
        let sp_cfg_champ = SelfPlayConfig {
            num_games: champ_games,
            threads: arena.threads,
            depth: arena.depth,
            data_output: Some(paths.data_path.clone()),
            eval_mode: current_best_eval.clone(),
            seed: gen_seed,
            ..Default::default()
        };
        let stats_champ = SelfPlayManager::run(sp_cfg_champ);
        if stats_champ.io_errors > 0 || stats_champ.completed_games < champ_games {
            return Err(format!(
                "Champion self-play encountered {} I/O errors or incomplete games ({}/{})",
                stats_champ.io_errors, stats_champ.completed_games, champ_games
            ));
        }

        // 候補モデル（Candidate HalfKP）による探査自己対局（存在する場合）
        let candidate_eval_opt = if Path::new(&paths.candidate_model_path).exists() {
            HalfKPEvaluator::load_from_file(&paths.candidate_model_path).ok()
        } else {
            None
        };

        if let Some(cand_eval) = candidate_eval_opt {
            let sp_cfg_cand = SelfPlayConfig {
                num_games: half_games,
                threads: arena.threads,
                depth: arena.depth,
                data_output: Some(paths.data_path.clone()),
                eval_mode: EvalMode::HalfKP(Arc::new(cand_eval)),
                seed: gen_seed.wrapping_add(0x85ebca6b),
                start_game_id: champ_games,
                ..Default::default()
            };
            let stats_cand = SelfPlayManager::run(sp_cfg_cand);
            if stats_cand.io_errors > 0 || stats_cand.completed_games < half_games {
                return Err(format!(
                    "Candidate exploration self-play encountered {} I/O errors or incomplete games ({}/{})",
                    stats_cand.io_errors, stats_cand.completed_games, half_games
                ));
            }
        } else if half_games > 0 {
            // 初回等でCandidateが存在しない場合はChampionで全数補完
            let sp_cfg_fallback = SelfPlayConfig {
                num_games: half_games,
                threads: arena.threads,
                depth: arena.depth,
                data_output: Some(paths.data_path.clone()),
                eval_mode: current_best_eval.clone(),
                seed: gen_seed.wrapping_add(0x85ebca6b),
                start_game_id: champ_games,
                ..Default::default()
            };
            let stats_fallback = SelfPlayManager::run(sp_cfg_fallback);
            if stats_fallback.io_errors > 0 || stats_fallback.completed_games < half_games {
                return Err(format!(
                    "Fallback champion self-play encountered {} I/O errors or incomplete games ({}/{})",
                    stats_fallback.io_errors, stats_fallback.completed_games, half_games
                ));
            }
        }

        Ok(())
    }

    /// Step 2: データセットサンプリング読込 & IIZ 深読み再評価 (知識蒸留)
    fn prepare_training_dataset(
        paths: &LoopStoragePaths,
        arena: &LoopArenaParams,
        gen_seed: u64,
        current_best_eval: &EvalMode,
    ) -> Option<(Vec<DatasetEntry>, usize)> {
        println!(
            "\n--- Step 2: Training Candidate Model from Dataset (IIZ Distillation + HalfKP AdamW) ---"
        );
        // 1. やねうら王流 IIZ (多重反復雑巾絞り):
        // 直近の自己対局データ (data_path) から最新局面を抽出し、
        // 探索深さ Depth+2 で深読み再評価を行う (深読みプール既存局面との重複再処理を完全防止)
        let mut successful_relabelled_count = 0;
        if Path::new(&paths.data_path).exists()
            && let Ok(mut recent_entries) = DatasetHandler::load_sampled(
                &paths.data_path,
                MAX_IIZ_RELABEL_PER_GEN,
                1.0,
                gen_seed,
            )
            && !recent_entries.is_empty()
        {
            let relabel_count = recent_entries.len();
            let relabel_depth = arena.depth.saturating_add(2);
            let t_relabel = std::time::Instant::now();
            let successful_relabelled = DatasetHandler::relabel_deep(
                &mut recent_entries,
                relabel_count,
                relabel_depth,
                arena.threads,
                current_best_eval,
            );
            println!(
                "IIZ Distillation: Successfully re-evaluated {}/{} recent positions at Depth {} in {:.2}s",
                successful_relabelled.len(),
                relabel_count,
                relabel_depth,
                t_relabel.elapsed().as_secs_f64()
            );

            // 探索が正常完了した真の深読み教師局面のみを永続プールファイルに追記
            if !successful_relabelled.is_empty() {
                if let Err(e) =
                    DatasetHandler::append_to_file(&paths.deep_data_path, &successful_relabelled)
                {
                    eprintln!("Warning: Failed to persist deep relabeled pool: {e}");
                }
                successful_relabelled_count = successful_relabelled.len();
            }
        }

        // 2. 蓄積された深読み高品質プール (50%) と通常自己対局データ (50%) をブレンドして学習用データセットを構築
        let dataset = match DatasetHandler::load_sampled_with_deep_pool(
            &paths.data_path,
            Some(&paths.deep_data_path),
            MAX_DATASET_SAMPLE_LIMIT,
            0.5,
            0.5,
            gen_seed,
        ) {
            Ok(d) if !d.is_empty() => d,
            _ => return None,
        };

        Some((dataset, successful_relabelled_count))
    }

    /// Step 3: Candidate 学習 & メモリ即時解放
    fn train_candidate_generation(
        training: &LoopTrainingParams,
        paths: &LoopStoragePaths,
        gen_seed: u64,
        dataset: &mut [DatasetEntry],
        current_best_eval: &EvalMode,
    ) -> (HalfKPEvaluator, f32, f32, f32) {
        println!(
            "Sampled {} training positions (50% recent / 50% history). Training HalfKP for {} epochs (batch_size: {})...",
            dataset.len(),
            training.epochs,
            training.batch_size
        );

        let mut trainer = if Path::new(&paths.candidate_ckpt_path).exists() {
            match HalfKPTrainer::load_checkpoint(&paths.candidate_ckpt_path) {
                Ok(t) => {
                    println!(
                        "[Resume] Loaded candidate AdamW checkpoint from '{}'",
                        paths.candidate_ckpt_path
                    );
                    t
                }
                Err(e) => {
                    println!(
                        "Failed to load candidate checkpoint '{}' ({e}), initializing from champion",
                        paths.candidate_ckpt_path
                    );
                    match current_best_eval {
                        EvalMode::HalfKP(best) => HalfKPTrainer::from_evaluator(best),
                        _ => HalfKPTrainer::new(),
                    }
                }
            }
        } else {
            match current_best_eval {
                EvalMode::HalfKP(best) => {
                    println!("[WarmStart] Initializing trainer from current best HalfKP model");
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
        let k_scale = DEFAULT_SIGMOID_K;

        if let Some(parent) = Path::new(&paths.candidate_model_path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Some(parent) = Path::new(&paths.candidate_ckpt_path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        let mut rng_seed = gen_seed.wrapping_add(0xdeadbeef);

        for epoch in 1..=training.epochs {
            println!(
                "  [Epoch {}/{}] Training {} positions (batch_size: {})...",
                epoch,
                training.epochs,
                dataset.len(),
                training.batch_size
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

            for chunk in dataset.chunks(training.batch_size) {
                let mut batch_samples: Vec<(Vec<usize>, Vec<usize>, f32)> =
                    Vec::with_capacity(chunk.len());

                for entry in chunk {
                    if let Ok(pos) = Position::from_sfen(&entry.sfen) {
                        let mover_feats =
                            HalfKPEvaluator::extract_halfkp_features(&pos, pos.side_to_move);
                        let opp_feats = HalfKPEvaluator::extract_halfkp_features(
                            &pos,
                            pos.side_to_move.opposite(),
                        );
                        // 教師ターゲット: 深読み評価値と勝敗結果のハイブリッド蒸留
                        let pred_eval = HalfKPTrainer::sigmoid(entry.score as f32, k_scale);
                        let target = (DISTILLATION_TEACHER_WEIGHT * pred_eval
                            + DISTILLATION_RESULT_WEIGHT * entry.result)
                            .clamp(0.0, 1.0);
                        batch_samples.push((mover_feats, opp_feats, target));
                    }
                }

                if !batch_samples.is_empty() {
                    let loss = trainer.train_batch(&batch_samples, training.lr, k_scale);
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
        if let Err(e) = cand_eval.save_to_file(&paths.candidate_model_path) {
            eprintln!("Error saving candidate model: {e}");
        }

        if let Err(e) = trainer.save_checkpoint(&paths.candidate_ckpt_path) {
            eprintln!("Error saving candidate checkpoint: {e}");
        }

        println!("[Memory] Dropping HalfKPTrainer to reclaim ~314MB heap before arena matches...");
        drop(trainer);

        (
            cand_eval,
            init_loss.mse_loss,
            final_loss.mse_loss,
            reduction,
        )
    }

    /// Step 4: アリーナ対戦 & SPRT 検定 (動的延長対局)
    fn run_arena_and_sprt(
        arena: &LoopArenaParams,
        cur_gen: usize,
        candidate_eval: &HalfKPEvaluator,
        current_best_eval: &EvalMode,
    ) -> MatchResult {
        println!("\n--- Step 3: Arena Match & SPRT Testing (Peak Memory < 200MB) ---");
        let mut current_pairs = arena.eval_pairs;
        let match_cfg = MatchConfig {
            name_a: format!("Candidate_Gen{cur_gen}"),
            name_b: "Best_Model".to_string(),
            eval_a: EvalMode::HalfKP(Arc::new(candidate_eval.clone())),
            eval_b: current_best_eval.clone(),
            pairs: current_pairs,
            depth: arena.depth,
            threads: arena.threads,
            random_opening: 6,
            max_plies: 320,
            tt_size_mb: 16,
            sprt_config: Some(SprtConfig {
                elo0: 0.0,
                elo1: 50.0,
                alpha: 0.05,
                beta: 0.05,
                min_games: arena.min_promotion_games,
            }),
        };

        let mut match_res = MatchRunner::run_match(&match_cfg);

        // SPRT判定が Continue かつ勝ち越し傾向 (勝率52%以上) の場合、動的に延長対局
        let max_pairs = arena.eval_pairs * 3;
        while let Some(ref sprt) = match_res.sprt {
            if sprt.status == SprtStatus::Continue
                && match_res.win_rate_a >= 0.52
                && current_pairs < max_pairs
            {
                current_pairs += arena.eval_pairs;
                println!(
                    "\n[SPRT Overtime] Indecisive Continue with positive win rate {:.1}% (LLR: {:.2}). Incrementally extending to {} pairs...",
                    match_res.win_rate_a * 100.0,
                    sprt.llr,
                    current_pairs
                );
                match_res =
                    MatchRunner::run_match_extended(&match_cfg, Some(&match_res), current_pairs);
            } else {
                break;
            }
        }
        match_res
    }

    /// Step 6: サマリ記録 & 世代更新 (CSV追記・状態永続化・クリーンアップ)
    fn record_generation_summary(
        paths: &LoopStoragePaths,
        cur_gen: usize,
        current_best_eval: &EvalMode,
        gen_start_instant: std::time::Instant,
        metrics: &GenerationMetrics,
        match_res: &MatchResult,
        promoted: bool,
    ) {
        let champ_name = match current_best_eval {
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
            &paths.summary_path,
            cur_gen,
            champ_name,
            gen_duration,
            metrics.dataset_len,
            metrics.relabelled_count,
            metrics.init_mse,
            metrics.final_mse,
            metrics.loss_reduction,
            match_res.total_games,
            match_res.win_rate_a,
            match_res.elo_diff_a,
            &sprt_str,
            promoted,
        );

        // 世代番号を永続化（次回再起動時に自動で直前世代から継続可能）
        if let Err(e) = std::fs::write(&paths.state_path, cur_gen.to_string()) {
            eprintln!("Warning: Failed to persist generation state: {e}");
        }

        if promoted {
            println!(
                "[Progression] Gen {cur_gen} Candidate successfully promoted! Fresh cycle will warm-start from new champion."
            );
            // 昇格時は Candidate チェックポイントを整理し、次代は新王者からウォームスタート
            if Path::new(&paths.candidate_ckpt_path).exists() {
                let _ = std::fs::remove_file(&paths.candidate_ckpt_path);
                let bak_path = format!("{}.bak", paths.candidate_ckpt_path);
                let _ = std::fs::remove_file(&bak_path);
            }
        } else {
            println!(
                "[Progression] Candidate did not beat champion in Gen {cur_gen}. Retaining trained weights & Adam momentum for Gen {}.",
                cur_gen + 1
            );
        }
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
