use super::match_runner::{MatchConfig, MatchResult, MatchRunner};
use super::sprt::SprtConfig;
use crate::eval::{EvalMode, NNUEEvaluator, NNUETrainer};
use crate::selfplay::dataset::DatasetHandler;
use crate::selfplay::{SelfPlayConfig, SelfPlayManager};
use std::path::Path;
use std::sync::Arc;

pub struct LoopConfig {
    pub iterations: usize,
    pub games_per_iteration: usize,
    pub eval_pairs: usize,
    pub threads: usize,
    pub depth: u8,
    pub epochs: usize,
    pub lr: f32,
    pub batch_size: usize,
    pub data_path: String,
    pub best_model_path: String,
    pub candidate_model_path: String,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            iterations: 3,
            games_per_iteration: 50,
            eval_pairs: 15,
            threads: 2,
            depth: 2,
            epochs: 10,
            lr: 0.005,
            batch_size: 64,
            data_path: "loop_dataset.tsv".to_string(),
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

        for iter in 1..=config.iterations {
            println!("\n>>> Generation {} / {} <<<", iter, config.iterations);

            // Step 1: 自己対局データ生成
            println!(
                "\n--- Step 1: Self-Play Data Generation ({} games) ---",
                config.games_per_iteration
            );
            let sp_cfg = SelfPlayConfig {
                num_games: config.games_per_iteration,
                threads: config.threads,
                depth: config.depth,
                random_opening_plies: 8,
                data_output: Some(config.data_path.clone()),
                eval_mode: current_best_eval.clone(),
                ..Default::default()
            };
            SelfPlayManager::run(sp_cfg);

            // Step 2: データセット読込 & スクラッチ NNUE 学習
            println!("\n--- Step 2: Training Candidate Model from Dataset ---");
            let dataset = match DatasetHandler::load_from_file(&config.data_path) {
                Ok(d) if !d.is_empty() => d,
                _ => {
                    println!("Warning: No dataset found or empty, skipping iteration.");
                    continue;
                }
            };
            println!(
                "Loaded {} training positions. Training for {} epochs...",
                dataset.len(),
                config.epochs
            );

            let mut trainer = NNUETrainer::new();
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
            let match_cfg = MatchConfig {
                name_a: format!("Candidate_Gen{iter}"),
                name_b: "Best_Model".to_string(),
                eval_a: EvalMode::Nnue(Arc::new(candidate_eval.clone())),
                eval_b: current_best_eval.clone(),
                pairs: config.eval_pairs,
                depth: config.depth,
                threads: config.threads,
                random_opening: 6,
                max_plies: 320,
                tt_size_mb: 16,
                sprt_config: Some(SprtConfig {
                    elo0: 0.0,
                    elo1: 5.0,
                    alpha: 0.05,
                    beta: 0.05,
                }),
            };

            let match_res = MatchRunner::run_match(&match_cfg);

            // Step 4: 昇格判定
            Self::handle_promotion(
                iter,
                &match_res,
                &candidate_eval,
                &mut current_best_eval,
                &config.best_model_path,
            );
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
    ) {
        let promoted = match_res.win_rate_a > 0.50;
        if promoted {
            println!(
                "\n>>> [PROMOTION] Gen {} Candidate won with {:.1}% win rate ({:+.1} Elo). Promoting to Best Model! <<<",
                iter,
                match_res.win_rate_a * 100.0,
                match_res.elo_diff_a
            );
            if let Err(e) = candidate_eval.save_to_file(best_model_path) {
                eprintln!("Error writing promoted best model: {e}");
            } else {
                *current_best_eval = EvalMode::Nnue(Arc::new(candidate_eval.clone()));
                println!("Successfully promoted and updated '{}'!", best_model_path);
            }
        } else {
            println!(
                "\n>>> [REJECTED] Gen {} Candidate did not surpass Best Model ({:.1}% win rate). Keeping existing best. <<<",
                iter,
                match_res.win_rate_a * 100.0
            );
        }
    }
}
