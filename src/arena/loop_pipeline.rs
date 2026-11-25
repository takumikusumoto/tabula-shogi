use super::match_runner::{MatchConfig, MatchResult, MatchRunner, OvertimePolicy};
use super::sprt::SprtConfig;
pub use super::workspace::{GenerationMetrics, LoopStoragePaths};
use super::workspace::{GenerationSummaryRow, LoopWorkspace};
use crate::eval::EvalMode;
use crate::eval::halfkp::HalfKPEvaluator;
use crate::eval::halfkp_trainer::{
    HalfKPInitPolicy, HalfKPTargetPolicy, HalfKPTrainProgress, HalfKPTrainSessionConfig,
    HalfKPTrainSource, train_halfkp_session_with_progress,
};
use crate::selfplay::dataset::{DatasetEntry, DatasetHandler};
use crate::selfplay::{SelfPlayConfig, SelfPlayManager, relabel_deep};
use std::io::ErrorKind;
use std::sync::Arc;

/// やねうら王流 IIZ (多重反復雑巾絞り) で 1 世代あたり深読み再評価する最大局面数
pub const MAX_IIZ_RELABEL_PER_GEN: usize = 5_000;
/// データセットサンプリング時の最大局面数上限
pub const MAX_DATASET_SAMPLE_LIMIT: usize = 100_000;
/// 教師ターゲット作成時の評価スコア重み (80%)
pub const DISTILLATION_TEACHER_WEIGHT: f32 = 0.8;
/// シグモイド勝率変換のデフォルト感度係数 (600 cp で勝率 ~73%)
pub use crate::eval::halfkp_trainer::DEFAULT_SIGMOID_K;

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

        let workspace = LoopWorkspace::open(config.paths.clone()).inspect_err(|e| {
            eprintln!("[Fatal Error] {e}");
        })?;
        let mut current_best_eval = workspace.load_champion().inspect_err(|e| {
            eprintln!("[Fatal Error] {e}");
        })?;
        let start_gen = workspace
            .resolve_start_generation(config.start_iteration)
            .inspect_err(|e| {
                eprintln!("[Fatal Error] {e}");
            })?;
        println!(
            "Cumulative Generation Offset: Starting at Gen {}",
            start_gen
        );

        for round in 1..=config.iterations {
            let cur_gen = start_gen.checked_add(round - 1).ok_or_else(|| {
                let err_msg = format!(
                    "[Fatal Error] Generation counter overflow (start_gen={start_gen}, round={round})"
                );
                eprintln!("{err_msg}");
                err_msg
            })?;
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
                &workspace,
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

            // Step 2: データセット準備 & IIZ 深読み再評価 (永続化障害は fail-closed)
            let (mut dataset, successful_relabelled_count) = match Self::prepare_training_dataset(
                &workspace,
                &config.arena,
                gen_seed,
                &current_best_eval,
            )? {
                Some(res) => res,
                None => {
                    println!("Warning: No dataset found or empty, skipping iteration.");
                    continue;
                }
            };

            // Step 3: Candidate 学習 & 300 MiB トレーナー自動解放
            let dataset_len = dataset.len();
            let (candidate_eval, init_mse, final_mse, loss_reduction) =
                Self::train_candidate_generation(
                    &config.training,
                    workspace.paths(),
                    gen_seed,
                    &mut dataset,
                    &current_best_eval,
                )?;
            drop(dataset);
            let candidate_eval = Arc::new(candidate_eval);

            // Step 4: アリーナ対戦 & SPRT 検定 (動的延長対局)
            let match_res = Self::run_arena_and_sprt(
                &config.arena,
                cur_gen,
                Arc::clone(&candidate_eval),
                &current_best_eval,
            );

            // Step 5: 厳格な昇格判定 & 成果物遷移 (王者保存 + チェックポイント整理 / 維持)
            let verdict = workspace.apply_promotion(
                cur_gen,
                &match_res,
                &candidate_eval,
                &mut current_best_eval,
                config.arena.min_promotion_games,
            )?;

            // Step 6: サマリ CSV 追記 & 世代状態永続化
            let metrics = GenerationMetrics {
                dataset_len,
                relabelled_count: successful_relabelled_count,
                init_mse,
                final_mse,
                loss_reduction,
            };
            workspace.record_generation(&GenerationSummaryRow {
                generation: cur_gen,
                champion_mode: current_best_eval.label(),
                duration_secs: gen_start_instant.elapsed().as_secs_f64(),
                metrics: &metrics,
                match_result: &match_res,
                promoted: verdict.is_promoted(),
            })?;
        }

        println!("\n============================================================");
        println!("Autonomous Self-Improvement Session Completed!");
        println!(
            "Best model preserved at: {}",
            workspace.paths().best_model_path
        );
        println!("============================================================");
        Ok(())
    }

    /// Step 1: 自己対局データ生成 (Policy Mismatch / OOD 解消のため Champion 50% / Candidate 50% 混合)
    fn generate_selfplay_data(
        games_per_iteration: usize,
        arena: &LoopArenaParams,
        workspace: &LoopWorkspace,
        current_best_eval: &EvalMode,
        cur_gen: usize,
        gen_seed: u64,
    ) -> Result<(), String> {
        let half_games = games_per_iteration / 2;
        let champ_games = games_per_iteration - half_games;
        let data_path = &workspace.paths().data_path;

        println!(
            "\n--- Step 1: Self-Play Data Generation (Gen {cur_gen}: {} Champion games + {} Candidate exploration games) ---",
            champ_games, half_games
        );

        Self::run_selfplay_batch(
            "Champion self-play",
            champ_games,
            0,
            gen_seed,
            current_best_eval.clone(),
            arena,
            data_path,
        )?;

        // 候補モデル（Candidate HalfKP）が存在する場合は fail-closed で検証ロードし、探査自己対局に使用
        let candidate_eval_opt = workspace.load_candidate_if_exists()?;
        if half_games > 0 {
            let (label, eval_mode) = match candidate_eval_opt {
                Some(cand_eval) => ("Candidate exploration self-play", cand_eval),
                None => ("Fallback champion self-play", current_best_eval.clone()),
            };
            Self::run_selfplay_batch(
                label,
                half_games,
                champ_games,
                gen_seed.wrapping_add(0x85ebca6b),
                eval_mode,
                arena,
                data_path,
            )?;
        }

        Ok(())
    }

    fn run_selfplay_batch(
        label: &str,
        num_games: usize,
        start_game_id: usize,
        seed: u64,
        eval_mode: EvalMode,
        arena: &LoopArenaParams,
        data_path: &str,
    ) -> Result<(), String> {
        if num_games == 0 {
            return Ok(());
        }
        let sp_cfg = SelfPlayConfig {
            num_games,
            threads: arena.threads,
            depth: arena.depth,
            data_output: Some(data_path.to_string()),
            eval_mode,
            seed,
            start_game_id,
            ..Default::default()
        };
        SelfPlayManager::run(sp_cfg).map_err(|e| format!("{label} failed: {e}"))?;
        Ok(())
    }

    /// Step 2: データセットサンプリング読込 & IIZ 深読み再評価 (知識蒸留)
    fn prepare_training_dataset(
        workspace: &LoopWorkspace,
        arena: &LoopArenaParams,
        gen_seed: u64,
        current_best_eval: &EvalMode,
    ) -> Result<Option<(Vec<DatasetEntry>, usize)>, String> {
        println!(
            "\n--- Step 2: Training Candidate Model from Dataset (IIZ Distillation + HalfKP AdamW) ---"
        );
        let paths = workspace.paths();

        match std::fs::symlink_metadata(&paths.data_path) {
            Ok(_) => {}
            Err(e) if e.kind() == ErrorKind::NotFound => return Ok(None),
            Err(e) => {
                return Err(format!(
                    "Failed to inspect dataset file '{}': {e}",
                    paths.data_path
                ));
            }
        }

        // 1. やねうら王流 IIZ (多重反復雑巾絞り):
        // 直近の自己対局データ (data_path) から最新局面を抽出し、
        // 探索深さ Depth+2 で深読み再評価を行う (深読みプール既存局面との重複再処理を完全防止)
        let mut successful_relabelled_count = 0;
        let mut recent_entries =
            DatasetHandler::load_sampled(&paths.data_path, MAX_IIZ_RELABEL_PER_GEN, 1.0, gen_seed)
                .map_err(|e| {
                    format!(
                        "Failed to load recent dataset '{}' for deep relabeling: {e}",
                        paths.data_path
                    )
                })?;

        if !recent_entries.is_empty() {
            let relabel_count = recent_entries.len();
            let relabel_depth = arena.depth.saturating_add(2);
            let t_relabel = std::time::Instant::now();
            let successful_relabelled = relabel_deep(
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

            // 探索が正常完了した真の深読み教師局面のみを永続プールファイルに fail-closed 追記
            workspace.append_deep_pool(&successful_relabelled)?;
            successful_relabelled_count = successful_relabelled.len();
        }

        // 2. 蓄積された深読み高品質プール (50%) と通常自己対局データ (50%) をブレンドして学習用データセットを構築
        let dataset = DatasetHandler::load_sampled_with_deep_pool(
            &paths.data_path,
            Some(&paths.deep_data_path),
            MAX_DATASET_SAMPLE_LIMIT,
            0.5,
            0.5,
            gen_seed,
        )
        .map_err(|e| format!("Failed to build blended training dataset: {e}"))?;

        if dataset.is_empty() {
            return Ok(None);
        }

        Ok(Some((dataset, successful_relabelled_count)))
    }

    /// Step 3: Candidate 学習 & メモリ即時解放
    fn train_candidate_generation(
        training: &LoopTrainingParams,
        paths: &LoopStoragePaths,
        gen_seed: u64,
        dataset: &mut [DatasetEntry],
        current_best_eval: &EvalMode,
    ) -> Result<(HalfKPEvaluator, f32, f32, f32), String> {
        let dataset_len = dataset.len();
        println!(
            "Sampled {} training positions (50% recent / 50% history). Training HalfKP for {} epochs (batch_size: {})...",
            dataset_len, training.epochs, training.batch_size
        );

        let init = match current_best_eval {
            EvalMode::HalfKP(best) => HalfKPInitPolicy::ResumeOrWarmStart {
                checkpoint_path: paths.candidate_ckpt_path.clone(),
                evaluator: best.as_ref(),
            },
            _ => HalfKPInitPolicy::ResumeOrFresh {
                checkpoint_path: paths.candidate_ckpt_path.clone(),
            },
        };

        let session_cfg = HalfKPTrainSessionConfig {
            init,
            target: HalfKPTargetPolicy::ScoreBlend {
                score_weight: DISTILLATION_TEACHER_WEIGHT,
            },
            epochs: training.epochs,
            batch_size: training.batch_size,
            lr: training.lr,
            k: DEFAULT_SIGMOID_K,
            model_output_path: Some(paths.candidate_model_path.clone()),
            checkpoint_output_path: Some(paths.candidate_ckpt_path.clone()),
            checkpoint_interval_batches: 0,
            persist_each_epoch: false,
        };

        let source = HalfKPTrainSource::InMemory {
            entries: dataset,
            shuffle_seed: gen_seed.wrapping_add(0xdeadbeef),
        };

        let outcome = train_halfkp_session_with_progress(
            session_cfg,
            source,
            |event| match event {
                HalfKPTrainProgress::Initialized { mode, detail } => {
                    println!("[{}] {detail}", mode.label());
                }
                HalfKPTrainProgress::EpochStarted {
                    epoch,
                    total_epochs,
                } => {
                    println!(
                        "  [Epoch {epoch}/{total_epochs}] Training {dataset_len} positions (batch_size: {})...",
                        training.batch_size
                    );
                }
                _ => {}
            },
        )?;

        println!(
            "HalfKP Training Complete in {:.2}s! MSE Loss: {:.6} -> {:.6} (Reduction: {:.2}%), Range Loss: {:.6}",
            outcome.total_duration_secs,
            outcome.init_loss.mse_loss,
            outcome.final_loss.mse_loss,
            outcome.loss_reduction_pct,
            outcome.final_loss.range_loss
        );

        Ok((
            outcome.evaluator,
            outcome.init_loss.mse_loss,
            outcome.final_loss.mse_loss,
            outcome.loss_reduction_pct,
        ))
    }

    /// Step 4: アリーナ対戦 & SPRT 検定 (動的延長対局)
    fn run_arena_and_sprt(
        arena: &LoopArenaParams,
        cur_gen: usize,
        candidate_eval: Arc<HalfKPEvaluator>,
        current_best_eval: &EvalMode,
    ) -> MatchResult {
        println!("\n--- Step 3: Arena Match & SPRT Testing (Peak Memory < 200MB) ---");
        let match_cfg = MatchConfig {
            name_a: format!("Candidate_Gen{cur_gen}"),
            name_b: "Best_Model".to_string(),
            eval_a: EvalMode::HalfKP(candidate_eval),
            eval_b: current_best_eval.clone(),
            pairs: arena.eval_pairs,
            depth: arena.depth,
            threads: arena.threads,
            generation: cur_gen,
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

        MatchRunner::run_match_with_overtime(
            &match_cfg,
            Some(OvertimePolicy::standard(arena.eval_pairs)),
        )
    }
}
