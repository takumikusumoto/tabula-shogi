use super::match_runner::{MatchConfig, MatchResult, MatchRunner};
use super::sprt::{SprtConfig, SprtStatus};
use crate::eval::{EvalMode, NNUEEvaluator, NNUETrainer};
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
            batch_size: 64,
            data_path: "loop_dataset.tsv".to_string(),
            deep_data_path: "deep_dataset.tsv".to_string(),
            best_model_path: "best_nnue.bin".to_string(),
            candidate_model_path: "candidate_nnue.bin".to_string(),
        }
    }
}

pub struct SelfImprovementLoop;

impl SelfImprovementLoop {
    /// 自律的自己改善ループを実行
    pub fn run(config: &LoopConfig) {
        println!("============================================================");
        println!("=== TabulaShogi Autonomous Self-Improvement Loop ===");
        println!(
            "Iterations: {}, Games/Iter: {}, EvalPairs: {} ({} games)",
            config.iterations,
            config.games_per_iteration,
            config.eval_pairs,
            config.eval_pairs * 2
        );
        println!(
            "Threads: {}, Depth: {}, Epochs: {}, LR: {}",
            config.threads, config.depth, config.epochs, config.lr
        );
        println!("Dataset: {}", config.data_path);
        println!("Best Model: {}", config.best_model_path);
        println!("============================================================");

        // 初期モデルの確認
        let mut current_best_eval = if Path::new(&config.best_model_path).exists() {
            match NNUEEvaluator::load_from_file(&config.best_model_path) {
                Ok(nnue) => {
                    println!(
                        "Loaded initial best model from '{}'",
                        config.best_model_path
                    );
                    EvalMode::Nnue(Arc::new(nnue))
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
            println!("No existing best model found. Starting with initial HCE.");
            EvalMode::Hce
        };

        let mut trainer = match &current_best_eval {
            EvalMode::Nnue(best_nnue) => NNUETrainer::from_evaluator(best_nnue),
            EvalMode::HalfKP(_) => {
                // HalfKP Trainer は Step 3 で統合
                NNUETrainer::new()
            }
            EvalMode::Hce => {
                if Path::new(&config.candidate_model_path).exists() {
                    match NNUEEvaluator::load_from_file(&config.candidate_model_path) {
                        Ok(candidate) => {
                            println!(
                                "Loaded existing candidate model from '{}' for training continuation",
                                config.candidate_model_path
                            );
                            NNUETrainer::from_evaluator(&candidate)
                        }
                        Err(e) => {
                            println!(
                                "Failed to load candidate '{}' ({e}), starting from scratch",
                                config.candidate_model_path
                            );
                            NNUETrainer::new()
                        }
                    }
                } else {
                    NNUETrainer::new()
                }
            }
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
            println!(
                "\n>>> Generation {} (Round {} / {}) <<<",
                cur_gen, round, config.iterations
            );

            // Step 1: 自己対局データ生成 (Policy Mismatch / OOD 解消のため Champion 50% / Candidate 50% 混合)
            // 候補モデルが存在する場合は、Candidate による自己対局を 50% 混ぜて未知の局面・疑問手を収集し、
            // Step 2 の深読み教師 (Depth 4) で再評価・矯正する (DAgger-like 探索データ統合)
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

            // 候補モデル（Candidate NNUE）による探査自己対局（存在する場合）
            let candidate_eval_opt = if Path::new(&config.candidate_model_path).exists() {
                NNUEEvaluator::load_from_file(&config.candidate_model_path).ok()
            } else {
                None
            };

            if let Some(cand_eval) = candidate_eval_opt {
                let sp_cfg_cand = SelfPlayConfig {
                    num_games: half_games,
                    threads: config.threads,
                    depth: config.depth,
                    data_output: Some(config.data_path.clone()),
                    eval_mode: EvalMode::Nnue(Arc::new(cand_eval)),
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

            // Step 2: データセット読込 & IIZ 深読み再評価 & 継続 NNUE 学習
            println!("\n--- Step 2: Training Candidate Model from Dataset (IIZ Distillation) ---");
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

            // やねうら王流 IIZ (多重反復雑巾絞り): 最新サンプリングのうち 5,000 局面を Depth 4 で深読み再評価
            let relabel_count = 5_000.min(dataset.len());
            let relabel_depth = config.depth.saturating_add(2); // Depth 2 -> Depth 4
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

            // 探索が正常完了した真の深読み教師局面のみを永続プールファイルに追記 (未完了・中断の混入を完全排除)
            if !successful_relabelled.is_empty()
                && let Err(e) =
                    DatasetHandler::append_to_file(&config.deep_data_path, &successful_relabelled)
            {
                eprintln!("Warning: Failed to persist deep relabeled pool: {e}");
            }

            println!(
                "Sampled {} training positions (50% recent / 50% history). Training for {} epochs...",
                dataset.len(),
                config.epochs
            );

            let (candidate_eval, init_loss, final_loss) =
                trainer.train_dataset(&dataset, config.epochs, config.lr, config.batch_size, 400.0);

            let reduction = if init_loss > 0.0 {
                (init_loss - final_loss) / init_loss * 100.0
            } else {
                0.0
            };
            println!(
                "Training Complete! MSE Loss: {:.4} -> {:.4} (Reduction: {:.2}%)",
                init_loss, final_loss, reduction
            );

            if let Err(e) = candidate_eval.save_to_file(&config.candidate_model_path) {
                eprintln!("Error saving candidate model: {e}");
            }

            // Step 3: アリーナ対戦 & レーティング検定 (Candidate vs Best)
            println!("\n--- Step 3: Arena Match & SPRT Testing ---");
            let mut current_pairs = config.eval_pairs;
            let match_cfg = MatchConfig {
                name_a: format!("Candidate_Gen{cur_gen}"),
                name_b: "Best_Model".to_string(),
                eval_a: EvalMode::Nnue(Arc::new(candidate_eval.clone())),
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

            // SPRT判定が Continue かつ勝ち越し傾向 (勝率52%以上) の場合、
            // 小標本による誤否決を防ぎ統計的有意性を確定させるため最大3倍 (例: 120局) まで動的に延長対局
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

            // Step 4: 昇格判定
            let promoted = Self::handle_promotion(
                cur_gen,
                &match_res,
                &candidate_eval,
                &mut current_best_eval,
                &config.best_model_path,
            );

            // 世代番号を永続化（次回再起動時に自動で直前世代から継続可能）
            if let Err(e) = std::fs::write(&config.state_path, cur_gen.to_string()) {
                eprintln!("Warning: Failed to persist generation state: {e}");
            }

            if !promoted {
                // 昇格しなかった場合:
                // 王者がHCEの間はCandidateの学習進捗とAdam状態を絶対に破棄せず蓄積を継続する！
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

    fn handle_promotion(
        iter: usize,
        match_res: &MatchResult,
        candidate_eval: &NNUEEvaluator,
        current_best_eval: &mut EvalMode,
        best_model_path: &str,
    ) -> bool {
        // 昇格条件: SPRT が Pass（統計的有意に強い）ことのみを要求し、サンプル分散による偶発的昇格を排除
        let sprt_passed = match_res
            .sprt
            .as_ref()
            .map(|s| s.status == SprtStatus::Pass)
            .unwrap_or(false);

        let promoted = sprt_passed;
        if promoted {
            let reason = "SPRT Pass (Statistically Significant Superiority)";
            println!(
                "\n>>> [PROMOTION] Gen {} Candidate won ({}) with {:.1}% win rate ({:+.1} Elo). Promoting to Best Model! <<<",
                iter,
                reason,
                match_res.win_rate_a * 100.0,
                match_res.elo_diff_a
            );
            if let Err(e) = candidate_eval.save_to_file(best_model_path) {
                eprintln!("Error writing promoted best model: {e}");
                false
            } else {
                *current_best_eval = EvalMode::Nnue(Arc::new(candidate_eval.clone()));
                println!("Successfully promoted and updated '{}'!", best_model_path);
                true
            }
        } else {
            println!(
                "\n>>> [REJECTED] Gen {} Candidate did not surpass Best Model ({:.1}% win rate, Elo {:+.1}). Keeping existing best. <<<",
                iter,
                match_res.win_rate_a * 100.0,
                match_res.elo_diff_a
            );
            false
        }
    }
}
