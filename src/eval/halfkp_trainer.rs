use super::atomic_io::{ensure_distinct_paths, write_atomic_with_backup};
use super::halfkp::{
    HALFKP_BIAS_SCALE, HALFKP_FEATURE_SCALE, HALFKP_HIDDEN_SIZE, HALFKP_INPUT_SIZE,
    HALFKP_OUTPUT_SCALE, HalfKPEvaluator, MAX_EVAL_CP,
};
use super::sigmoid_win_rate;
use crate::board::Position;
use crate::movegen::MoveGenerator;
use crate::selfplay::dataset::{DatasetBatchReader, DatasetEntry};
use crate::selfplay::game::SimpleRng;
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::path::Path;
use std::time::Instant;

/// Upper bound on mini-batch size to keep sparse gradient maps safely within the 500 MiB RSS budget.
pub const MAX_BATCH_SIZE: usize = 4096;

/// Default sigmoid sensitivity factor (600 cp -> ~73% win probability).
pub const DEFAULT_SIGMOID_K: f32 = 600.0;

const CHECKPOINT_MAGIC: &[u8; 8] = b"TB_HKPCK";
const EXPORT_VERIFICATION_SEED: u64 = 0x243f6a8885a308d3;
const GOLDEN_RATIO_SEED_STEP: u64 = 0x9e3779b97f4a7c15;

/// Loss metrics recorded for a single mini-batch update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainStepLoss {
    /// Mean squared error between predicted win probability and Training Target.
    pub mse_loss: f32,
    /// Preactivation range penalty outside `[0.0, 64.0]`.
    pub range_loss: f32,
    /// Sum of `mse_loss + range_loss`.
    pub total_loss: f32,
}

impl TrainStepLoss {
    pub fn zero() -> Self {
        Self {
            mse_loss: 0.0,
            range_loss: 0.0,
            total_loss: 0.0,
        }
    }
}

/// Export-time evidence that fractional training updates survived Q6/Q9 fixed-point quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HalfKPQuantizationStats {
    pub feature_values: usize,
    pub float_nonzero_features: usize,
    pub quantized_nonzero_features: usize,
    pub float_changed_from_initial: usize,
    pub legacy_changed_from_initial: usize,
    pub scaled_changed_from_initial: usize,
    pub feature_saturations: usize,
    pub output_nonzero: usize,
    pub output_saturations: usize,
    pub feature_bias_saturations: usize,
    pub output_bias_saturated: bool,
}

/// Read-only summary of a persisted Training Checkpoint (`TB_HKPCK`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HalfKPCheckpointSummary {
    pub active_features: usize,
    pub total_feature_steps: u64,
    pub output_bias: f32,
    pub beta1_pow: f32,
    pub beta2_pow: f32,
}

/// Domain initialization mode actually chosen for a training session (`GLOSSARY.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalfKPInitMode {
    Fresh,
    WarmStart,
    Resume,
}

impl HalfKPInitMode {
    pub fn label(self) -> &'static str {
        match self {
            HalfKPInitMode::Fresh => "Init",
            HalfKPInitMode::WarmStart => "WarmStart",
            HalfKPInitMode::Resume => "Resume",
        }
    }
}

/// Initialization policy for a HalfKP training session.
#[derive(Debug, Clone)]
pub enum HalfKPInitPolicy<'a> {
    /// Start from deterministic initial Evaluation Model weights with fresh optimizer state.
    Fresh,
    /// Resume from a Training Checkpoint if it exists (fail-closed if corrupt); otherwise start Fresh.
    ResumeOrFresh { checkpoint_path: String },
    /// Resume from a Training Checkpoint if it exists (fail-closed if corrupt); otherwise Warm-Start.
    ResumeOrWarmStart {
        checkpoint_path: String,
        evaluator: &'a HalfKPEvaluator,
    },
}

/// How each position's Training Target in `[0.0, 1.0]` is computed from a `DatasetEntry`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HalfKPTargetPolicy {
    /// Fit purely against game outcome (`entry.result`).
    OutcomeOnly,
    /// Fit against `score_weight * sigmoid(entry.score, k) + (1 - score_weight) * entry.result`.
    ScoreBlend { score_weight: f32 },
}

/// Position feed for a HalfKP training session.
pub enum HalfKPTrainSource<'a> {
    /// In-memory dataset slice, shuffled in place before each epoch and encoded in mini-batches.
    InMemory {
        entries: &'a mut [DatasetEntry],
        shuffle_seed: u64,
    },
    /// Completed partition TSV files in a directory, streamed batch by batch under 500 MiB RSS.
    PartitionDir { data_dir: String, shuffle_seed: u64 },
}

/// Configuration for a unified HalfKP training session.
#[derive(Debug, Clone)]
pub struct HalfKPTrainSessionConfig<'a> {
    pub init: HalfKPInitPolicy<'a>,
    pub target: HalfKPTargetPolicy,
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f32,
    pub k: f32,
    pub model_output_path: Option<String>,
    pub checkpoint_output_path: Option<String>,
    pub checkpoint_interval_batches: usize,
    pub persist_each_epoch: bool,
}

/// Lightweight progress events emitted to an optional caller adapter.
#[derive(Debug, Clone, Copy)]
pub enum HalfKPTrainProgress<'a> {
    Initialized {
        mode: HalfKPInitMode,
        detail: &'a str,
    },
    DiscoveredPartitions {
        file_count: usize,
    },
    EpochStarted {
        epoch: usize,
        total_epochs: usize,
    },
    BatchCompleted {
        epoch: usize,
        epoch_batches: usize,
        global_batches: usize,
        epoch_positions: usize,
        loss: TrainStepLoss,
        positions_per_sec: f64,
    },
    CheckpointSaved {
        path: &'a str,
        global_batches: usize,
    },
    EpochCompleted {
        epoch: usize,
        total_epochs: usize,
        positions: usize,
        avg_mse: f64,
        avg_range: f64,
        duration_secs: f64,
    },
}

/// Structured outcome returned after the ~300 MiB trainer heap state has been dropped.
#[derive(Debug, Clone)]
pub struct HalfKPTrainSessionOutcome {
    pub evaluator: HalfKPEvaluator,
    pub epochs_completed: usize,
    pub total_positions_trained: usize,
    pub total_batches: usize,
    pub init_loss: TrainStepLoss,
    pub final_loss: TrainStepLoss,
    pub loss_reduction_pct: f32,
    pub total_duration_secs: f64,
    pub positions_per_sec: f64,
}

/// Verification and reload diagnostics returned by `export_halfkp_checkpoint`.
#[derive(Debug, Clone)]
pub struct HalfKPExportReport {
    pub evaluator: HalfKPEvaluator,
    pub quantization: HalfKPQuantizationStats,
    pub verification_samples: usize,
    pub verification_mae_cp: f64,
    pub verification_max_abs_error_cp: f32,
    pub output_bytes: u64,
    pub startpos_eval_cp: i32,
}

/// Runs a HalfKP training session without progress logging.
pub fn train_halfkp_session(
    config: HalfKPTrainSessionConfig<'_>,
    source: HalfKPTrainSource<'_>,
) -> Result<HalfKPTrainSessionOutcome, String> {
    train_halfkp_session_with_progress(config, source, |_| {})
}

/// Runs a validated, fail-closed HalfKP training session (in-memory or partitioned streaming),
/// persists requested Evaluation Model / Training Checkpoint artifacts atomically, and drops the
/// ~300 MiB trainer heap state before returning the exported `HalfKPEvaluator` and metrics.
pub fn train_halfkp_session_with_progress<F>(
    config: HalfKPTrainSessionConfig<'_>,
    mut source: HalfKPTrainSource<'_>,
    mut on_progress: F,
) -> Result<HalfKPTrainSessionOutcome, String>
where
    F: FnMut(HalfKPTrainProgress<'_>),
{
    let overall_start = Instant::now();
    validate_session_config(&config)?;

    // Initialize reader before allocating the ~300 MiB trainer so missing/empty directories fail fast.
    let mut partition_reader = match &source {
        HalfKPTrainSource::PartitionDir { data_dir, .. } => {
            let reader = DatasetBatchReader::from_partition_dir(data_dir, config.batch_size)
                .map_err(|e| {
                    format!("Failed to initialize DatasetBatchReader from '{data_dir}': {e}")
                })?;
            if reader.file_count() == 0 {
                return Err(format!(
                    "No completed partition files found in '{data_dir}'. Did you run self-play generation first?"
                ));
            }
            on_progress(HalfKPTrainProgress::DiscoveredPartitions {
                file_count: reader.file_count(),
            });
            Some(reader)
        }
        HalfKPTrainSource::InMemory { entries, .. } => {
            if entries.is_empty() {
                return Err(
                    "Training dataset is empty; refusing to run HalfKP session.".to_string()
                );
            }
            None
        }
    };

    let mut trainer = init_trainer(&config.init, &mut on_progress)?;
    let mut session_stats = SessionStats::default();

    for epoch in 1..=config.epochs {
        let mut epoch_stats = EpochStats::start(epoch);
        on_progress(HalfKPTrainProgress::EpochStarted {
            epoch,
            total_epochs: config.epochs,
        });

        match &mut source {
            HalfKPTrainSource::InMemory {
                entries,
                shuffle_seed,
            } => {
                if entries.len() > 1 {
                    let seed = shuffle_seed
                        .wrapping_add((epoch as u64).wrapping_mul(GOLDEN_RATIO_SEED_STEP));
                    let mut rng = SimpleRng::new(seed);
                    for i in (1..entries.len()).rev() {
                        let j = rng.gen_range(i + 1);
                        entries.swap(i, j);
                    }
                }

                for chunk in entries.chunks(config.batch_size) {
                    process_entry_batch(
                        &mut trainer,
                        chunk,
                        &config,
                        &mut epoch_stats,
                        &mut session_stats,
                        &mut on_progress,
                    )?;
                }
            }
            HalfKPTrainSource::PartitionDir { shuffle_seed, .. } => {
                let reader = partition_reader
                    .as_mut()
                    .expect("partition reader initialized for PartitionDir");
                reader.shuffle_files(shuffle_seed.wrapping_add(epoch as u64));

                while let Some(batch_entries) = reader
                    .next_batch()
                    .map_err(|e| format!("I/O error reading batch: {e}"))?
                {
                    if batch_entries.is_empty() {
                        continue;
                    }
                    process_entry_batch(
                        &mut trainer,
                        &batch_entries,
                        &config,
                        &mut epoch_stats,
                        &mut session_stats,
                        &mut on_progress,
                    )?;
                }
            }
        }

        let epoch_elapsed = epoch_stats.started_at.elapsed().as_secs_f64().max(0.001);
        let (avg_mse, avg_range) = epoch_stats.averages();
        on_progress(HalfKPTrainProgress::EpochCompleted {
            epoch,
            total_epochs: config.epochs,
            positions: epoch_stats.positions,
            avg_mse,
            avg_range,
            duration_secs: epoch_elapsed,
        });

        session_stats.epochs_completed += 1;
        session_stats.total_positions_trained += epoch_stats.positions;

        // Persist intermediate epochs only when requested (last epoch is persisted once below).
        if config.persist_each_epoch && epoch < config.epochs && epoch_stats.positions > 0 {
            if let Some(ref model_path) = config.model_output_path {
                let eval = trainer.to_evaluator();
                eval.save_to_file(model_path)
                    .map_err(|e| format!("Failed to save exported model to '{model_path}': {e}"))?;
                drop(eval);
            }
            if let Some(ref ckpt_path) = config.checkpoint_output_path {
                trainer.save_checkpoint(ckpt_path).map_err(|e| {
                    format!("Failed to save epoch checkpoint to '{ckpt_path}': {e}")
                })?;
            }
        }
    }

    if session_stats.total_positions_trained == 0 {
        return Err(
            "Zero valid positions were processed across all epochs. Model and checkpoint were not updated."
                .to_string(),
        );
    }

    // Export Evaluation Model first, persist model before checkpoint, then drop ~300 MiB trainer.
    let evaluator = trainer.to_evaluator();
    if let Some(ref model_path) = config.model_output_path {
        evaluator
            .save_to_file(model_path)
            .map_err(|e| format!("Failed to save exported model to '{model_path}': {e}"))?;
    }
    if let Some(ref ckpt_path) = config.checkpoint_output_path {
        trainer
            .save_checkpoint(ckpt_path)
            .map_err(|e| format!("Failed to save training checkpoint to '{ckpt_path}': {e}"))?;
    }
    drop(trainer);

    let loss_reduction_pct = if session_stats.init_loss.mse_loss > 0.0 {
        (session_stats.init_loss.mse_loss - session_stats.final_loss.mse_loss)
            / session_stats.init_loss.mse_loss
            * 100.0
    } else {
        0.0
    };
    let total_duration_secs = overall_start.elapsed().as_secs_f64().max(0.001);
    let positions_per_sec = session_stats.total_positions_trained as f64 / total_duration_secs;

    Ok(HalfKPTrainSessionOutcome {
        evaluator,
        epochs_completed: session_stats.epochs_completed,
        total_positions_trained: session_stats.total_positions_trained,
        total_batches: session_stats.total_batches,
        init_loss: session_stats.init_loss,
        final_loss: session_stats.final_loss,
        loss_reduction_pct,
        total_duration_secs,
        positions_per_sec,
    })
}

/// Loads a persisted Training Checkpoint, exports a Q6/Q9 `TABU_HK2` Evaluation Model, verifies
/// float-vs-integer quantization error over deterministic legal positions, drops the ~300 MiB
/// trainer before atomic persistence & streaming reload validation, and returns full diagnostics.
pub fn export_halfkp_checkpoint(
    checkpoint_path: &str,
    output_path: &str,
    verification_samples: usize,
) -> Result<HalfKPExportReport, String> {
    ensure_distinct_paths(checkpoint_path, output_path, "checkpoint", "model output")?;

    let trainer = load_checkpoint_fail_closed(checkpoint_path)?;
    let evaluator = trainer.to_evaluator();
    let quantization = trainer.quantization_stats(&evaluator);

    let samples = verification_samples.max(1);
    let mut pos = Position::startpos();
    let mut rng = SimpleRng::new(EXPORT_VERIFICATION_SEED);
    let mut total_abs_error = 0.0f64;
    let mut max_abs_error = 0.0f32;
    for _ in 0..samples {
        let mover = HalfKPEvaluator::extract_halfkp_features(&pos, pos.side_to_move);
        let opponent = HalfKPEvaluator::extract_halfkp_features(&pos, pos.side_to_move.opposite());
        let (float_cp, _, _, _, _, _) = trainer.forward(&mover, &opponent);
        let integer_cp = evaluator.evaluate(&pos);
        let abs_error = (float_cp - integer_cp as f32).abs();
        total_abs_error += abs_error as f64;
        max_abs_error = max_abs_error.max(abs_error);

        let legal_moves = MoveGenerator::generate_legal_moves(&mut pos);
        if legal_moves.is_empty() || pos.repetition_count() >= 4 {
            pos = Position::startpos();
        } else {
            let mv = legal_moves[rng.gen_range(legal_moves.len())];
            pos.do_move(mv);
        }
    }
    let verification_mae_cp = total_abs_error / samples as f64;

    // Reclaim ~300 MiB trainer heap BEFORE saving and reload-validating to stay well under 500 MiB.
    drop(trainer);

    evaluator
        .save_to_file(output_path)
        .map_err(|e| format!("Failed to save HalfKP model '{output_path}': {e}"))?;
    drop(evaluator);

    let loaded = HalfKPEvaluator::load_from_file(output_path)
        .map_err(|e| format!("Exported model failed reload validation: {e}"))?;
    let output_bytes = std::fs::metadata(output_path)
        .map_err(|e| format!("Failed to inspect exported model '{output_path}': {e}"))?
        .len();
    let startpos_eval_cp = loaded.evaluate(&Position::startpos());

    Ok(HalfKPExportReport {
        evaluator: loaded,
        quantization,
        verification_samples: samples,
        verification_mae_cp,
        verification_max_abs_error_cp: max_abs_error,
        output_bytes,
        startpos_eval_cp,
    })
}

/// Validates a persisted Training Checkpoint (`TB_HKPCK`) and returns a lightweight read-only summary
/// without exposing raw weight/optimizer arrays.
pub fn inspect_halfkp_checkpoint(path: &str) -> Result<HalfKPCheckpointSummary, String> {
    let trainer = load_checkpoint_fail_closed(path)?;
    let mut active_features = 0usize;
    let mut total_feature_steps = 0u64;
    for &s in &trainer.step_feat {
        if s > 0 {
            active_features += 1;
            total_feature_steps += s as u64;
        }
    }
    Ok(HalfKPCheckpointSummary {
        active_features,
        total_feature_steps,
        output_bias: trainer.output_bias,
        beta1_pow: trainer.beta1_pow,
        beta2_pow: trainer.beta2_pow,
    })
}

fn validate_session_config(config: &HalfKPTrainSessionConfig<'_>) -> Result<(), String> {
    if config.batch_size == 0 || config.batch_size > MAX_BATCH_SIZE {
        return Err(format!(
            "Invalid batch_size: {}. batch_size must be between 1 and {} to strictly guarantee the 500 MiB memory limit.",
            config.batch_size, MAX_BATCH_SIZE
        ));
    }
    if config.epochs == 0 {
        return Err("Invalid epochs: 0. epochs must be at least 1.".to_string());
    }
    if config.lr <= 0.0 || !config.lr.is_finite() {
        return Err(format!(
            "Invalid learning rate: {}. lr must be positive and finite.",
            config.lr
        ));
    }
    if config.k <= 0.0 || !config.k.is_finite() {
        return Err(format!(
            "Invalid sigmoid sensitivity k: {}. k must be positive and finite.",
            config.k
        ));
    }
    if let HalfKPTargetPolicy::ScoreBlend { score_weight } = config.target
        && (!(0.0..=1.0).contains(&score_weight) || !score_weight.is_finite())
    {
        return Err(format!(
            "Invalid score_weight: {}. score_weight must be finite and within [0.0, 1.0].",
            score_weight
        ));
    }
    if let (Some(model_path), Some(ckpt_path)) = (
        config.model_output_path.as_deref(),
        config.checkpoint_output_path.as_deref(),
    ) {
        ensure_distinct_paths(model_path, ckpt_path, "model output", "checkpoint output")?;
    }
    if let Some(model_path) = config.model_output_path.as_deref() {
        let resume_ckpt = match &config.init {
            HalfKPInitPolicy::ResumeOrFresh { checkpoint_path }
            | HalfKPInitPolicy::ResumeOrWarmStart {
                checkpoint_path, ..
            } => Some(checkpoint_path.as_str()),
            _ => None,
        };
        if let Some(ckpt_path) = resume_ckpt {
            ensure_distinct_paths(model_path, ckpt_path, "model output", "resume checkpoint")?;
        }
    }
    Ok(())
}

fn load_checkpoint_fail_closed(path: &str) -> Result<HalfKPTrainer, String> {
    HalfKPTrainer::load_checkpoint(path).map_err(|err| {
        let bak_path = format!("{path}.bak");
        if Path::new(&bak_path).exists() {
            format!(
                "{err}. A backup checkpoint exists at '{bak_path}'; restore it manually, or remove '{path}' to start cleanly via Warm-Start/Fresh initialization."
            )
        } else {
            format!(
                "{err}. Remove the corrupted file '{path}' if you intend to discard optimizer history and start cleanly via Warm-Start/Fresh initialization."
            )
        }
    })
}

fn init_trainer<F>(
    init: &HalfKPInitPolicy<'_>,
    on_progress: &mut F,
) -> Result<HalfKPTrainer, String>
where
    F: FnMut(HalfKPTrainProgress<'_>),
{
    let (resume_path, fallback_eval): (Option<&str>, Option<&HalfKPEvaluator>) = match init {
        HalfKPInitPolicy::Fresh => (None, None),
        HalfKPInitPolicy::ResumeOrFresh { checkpoint_path } => {
            (Some(checkpoint_path.as_str()), None)
        }
        HalfKPInitPolicy::ResumeOrWarmStart {
            checkpoint_path,
            evaluator,
        } => (Some(checkpoint_path.as_str()), Some(*evaluator)),
    };

    if let Some(ckpt_path) = resume_path {
        let exists = Path::new(ckpt_path)
            .try_exists()
            .map_err(|e| format!("Failed to check checkpoint path '{ckpt_path}': {e}"))?;
        if exists {
            let trainer = load_checkpoint_fail_closed(ckpt_path)?;
            on_progress(HalfKPTrainProgress::Initialized {
                mode: HalfKPInitMode::Resume,
                detail: ckpt_path,
            });
            return Ok(trainer);
        }
    }

    match fallback_eval {
        Some(eval) => {
            on_progress(HalfKPTrainProgress::Initialized {
                mode: HalfKPInitMode::WarmStart,
                detail: "existing Evaluation Model weights (fresh optimizer)",
            });
            Ok(HalfKPTrainer::from_evaluator(eval))
        }
        None => {
            on_progress(HalfKPTrainProgress::Initialized {
                mode: HalfKPInitMode::Fresh,
                detail: "deterministic initial evaluator weights",
            });
            Ok(HalfKPTrainer::new())
        }
    }
}

struct SessionStats {
    init_loss: TrainStepLoss,
    final_loss: TrainStepLoss,
    has_batch: bool,
    epochs_completed: usize,
    total_positions_trained: usize,
    total_batches: usize,
}

impl Default for SessionStats {
    fn default() -> Self {
        Self {
            init_loss: TrainStepLoss::zero(),
            final_loss: TrainStepLoss::zero(),
            has_batch: false,
            epochs_completed: 0,
            total_positions_trained: 0,
            total_batches: 0,
        }
    }
}

struct EpochStats {
    epoch: usize,
    started_at: Instant,
    positions: usize,
    mse_sum: f64,
    range_sum: f64,
    batches: usize,
}

impl EpochStats {
    fn start(epoch: usize) -> Self {
        Self {
            epoch,
            started_at: Instant::now(),
            positions: 0,
            mse_sum: 0.0,
            range_sum: 0.0,
            batches: 0,
        }
    }

    fn averages(&self) -> (f64, f64) {
        if self.positions > 0 {
            (
                self.mse_sum / self.positions as f64,
                self.range_sum / self.positions as f64,
            )
        } else {
            (0.0, 0.0)
        }
    }
}

fn process_entry_batch<F>(
    trainer: &mut HalfKPTrainer,
    entries: &[DatasetEntry],
    config: &HalfKPTrainSessionConfig<'_>,
    epoch_stats: &mut EpochStats,
    session_stats: &mut SessionStats,
    on_progress: &mut F,
) -> Result<(), String>
where
    F: FnMut(HalfKPTrainProgress<'_>),
{
    let mut batch_samples: Vec<(Vec<usize>, Vec<usize>, f32)> = Vec::with_capacity(entries.len());
    for entry in entries {
        if let Ok(pos) = Position::from_sfen(&entry.sfen) {
            let mover_feats = HalfKPEvaluator::extract_halfkp_features(&pos, pos.side_to_move);
            let opp_feats =
                HalfKPEvaluator::extract_halfkp_features(&pos, pos.side_to_move.opposite());
            let target = compute_training_target(entry, config.target, config.k);
            batch_samples.push((mover_feats, opp_feats, target));
        }
    }

    if batch_samples.is_empty() {
        return Ok(());
    }

    let batch_len = batch_samples.len();
    let loss = trainer.train_batch(&batch_samples, config.lr, config.k);
    if !session_stats.has_batch {
        session_stats.init_loss = loss;
        session_stats.has_batch = true;
    }
    session_stats.final_loss = loss;

    epoch_stats.mse_sum += (loss.mse_loss as f64) * (batch_len as f64);
    epoch_stats.range_sum += (loss.range_loss as f64) * (batch_len as f64);
    epoch_stats.positions += batch_len;
    epoch_stats.batches += 1;
    session_stats.total_batches += 1;

    let elapsed = epoch_stats.started_at.elapsed().as_secs_f64().max(0.001);
    on_progress(HalfKPTrainProgress::BatchCompleted {
        epoch: epoch_stats.epoch,
        epoch_batches: epoch_stats.batches,
        global_batches: session_stats.total_batches,
        epoch_positions: epoch_stats.positions,
        loss,
        positions_per_sec: epoch_stats.positions as f64 / elapsed,
    });

    if config.checkpoint_interval_batches > 0
        && session_stats
            .total_batches
            .is_multiple_of(config.checkpoint_interval_batches)
        && let Some(ref ckpt_path) = config.checkpoint_output_path
    {
        trainer
            .save_checkpoint(ckpt_path)
            .map_err(|e| format!("Failed to save periodic checkpoint '{ckpt_path}': {e}"))?;
        on_progress(HalfKPTrainProgress::CheckpointSaved {
            path: ckpt_path,
            global_batches: session_stats.total_batches,
        });
    }

    Ok(())
}

#[inline(always)]
fn compute_training_target(entry: &DatasetEntry, policy: HalfKPTargetPolicy, k: f32) -> f32 {
    match policy {
        HalfKPTargetPolicy::OutcomeOnly => entry.result.clamp(0.0, 1.0),
        HalfKPTargetPolicy::ScoreBlend { score_weight } => {
            let pred_eval = sigmoid_win_rate(entry.score as f32, k);
            let result_weight = 1.0 - score_weight;
            (score_weight * pred_eval + result_weight * entry.result).clamp(0.0, 1.0)
        }
    }
}

/// Internal scratch-built HalfKP backpropagation trainer (Sparse AdamW, ~300 MiB heap state).
struct HalfKPTrainer {
    feature_weights: Vec<[f32; HALFKP_HIDDEN_SIZE]>,
    feature_biases: [f32; HALFKP_HIDDEN_SIZE],
    output_weights: [f32; HALFKP_HIDDEN_SIZE * 2],
    output_bias: f32,

    // Sparse AdamW optimizer moments and per-feature local update steps.
    m_feat: Vec<[f32; HALFKP_HIDDEN_SIZE]>,
    v_feat: Vec<[f32; HALFKP_HIDDEN_SIZE]>,
    step_feat: Vec<u32>,
    m_f_bias: [f32; HALFKP_HIDDEN_SIZE],
    v_f_bias: [f32; HALFKP_HIDDEN_SIZE],
    m_out: [f32; HALFKP_HIDDEN_SIZE * 2],
    v_out: [f32; HALFKP_HIDDEN_SIZE * 2],
    m_bias: f32,
    v_bias: f32,
    beta1_pow: f32,
    beta2_pow: f32,
}

#[derive(Clone, Copy, Debug)]
struct AdamWHyperParams {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
}

impl HalfKPTrainer {
    #[inline(always)]
    fn adamw_step(
        w: &mut f32,
        m: &mut f32,
        v: &mut f32,
        g: f32,
        one_minus_b1: f32,
        one_minus_b2: f32,
        hp: &AdamWHyperParams,
    ) {
        *m = hp.beta1 * *m + (1.0 - hp.beta1) * g;
        *v = hp.beta2 * *v + (1.0 - hp.beta2) * g * g;
        let m_hat = *m / one_minus_b1;
        let v_hat = *v / one_minus_b2;
        *w -= hp.lr * (m_hat / (v_hat.sqrt() + hp.eps) + hp.wd * *w);
    }

    fn new() -> Self {
        let initial_eval = HalfKPEvaluator::new();
        Self::from_evaluator(&initial_eval)
    }

    fn from_evaluator(eval: &HalfKPEvaluator) -> Self {
        let mut feature_weights = Vec::with_capacity(HALFKP_INPUT_SIZE);
        for row in &eval.feature_weights {
            let mut f_row = [0.0f32; HALFKP_HIDDEN_SIZE];
            for (w, &q) in f_row.iter_mut().zip(row.iter()) {
                *w = q as f32 / HALFKP_FEATURE_SCALE as f32;
            }
            feature_weights.push(f_row);
        }

        let mut feature_biases = [0.0f32; HALFKP_HIDDEN_SIZE];
        for (b, &q) in feature_biases.iter_mut().zip(eval.feature_biases.iter()) {
            *b = q as f32 / HALFKP_FEATURE_SCALE as f32;
        }

        let mut output_weights = [0.0f32; HALFKP_HIDDEN_SIZE * 2];
        for (w, &q) in output_weights.iter_mut().zip(eval.output_weights.iter()) {
            *w = q as f32 / HALFKP_OUTPUT_SCALE as f32;
        }

        let output_bias = eval.output_bias as f32 / HALFKP_BIAS_SCALE as f32;

        HalfKPTrainer {
            feature_weights,
            feature_biases,
            output_weights,
            output_bias,
            m_feat: vec![[0.0f32; HALFKP_HIDDEN_SIZE]; HALFKP_INPUT_SIZE],
            v_feat: vec![[0.0f32; HALFKP_HIDDEN_SIZE]; HALFKP_INPUT_SIZE],
            step_feat: vec![0u32; HALFKP_INPUT_SIZE],
            m_f_bias: [0.0f32; HALFKP_HIDDEN_SIZE],
            v_f_bias: [0.0f32; HALFKP_HIDDEN_SIZE],
            m_out: [0.0f32; HALFKP_HIDDEN_SIZE * 2],
            v_out: [0.0f32; HALFKP_HIDDEN_SIZE * 2],
            m_bias: 0.0,
            v_bias: 0.0,
            beta1_pow: 1.0,
            beta2_pow: 1.0,
        }
    }

    fn to_evaluator(&self) -> HalfKPEvaluator {
        HalfKPEvaluator::from_float_weights(
            &self.feature_weights,
            &self.feature_biases,
            &self.output_weights,
            self.output_bias,
        )
    }

    fn quantization_stats(&self, eval: &HalfKPEvaluator) -> HalfKPQuantizationStats {
        let mut stats = HalfKPQuantizationStats {
            feature_values: HALFKP_INPUT_SIZE * HALFKP_HIDDEN_SIZE,
            float_nonzero_features: 0,
            quantized_nonzero_features: 0,
            float_changed_from_initial: 0,
            legacy_changed_from_initial: 0,
            scaled_changed_from_initial: 0,
            feature_saturations: 0,
            output_nonzero: 0,
            output_saturations: 0,
            feature_bias_saturations: 0,
            output_bias_saturated: false,
        };

        for (feat, (float_row, quantized_row)) in self
            .feature_weights
            .iter()
            .zip(eval.feature_weights.iter())
            .enumerate()
        {
            for (hidden, (&float_weight, &quantized_weight)) in
                float_row.iter().zip(quantized_row.iter()).enumerate()
            {
                let initial_quantized = HalfKPEvaluator::initial_feature_weight(feat, hidden);
                let initial_float = initial_quantized as f32 / HALFKP_FEATURE_SCALE as f32;
                stats.float_nonzero_features += usize::from(float_weight != 0.0);
                stats.quantized_nonzero_features += usize::from(quantized_weight != 0);
                stats.float_changed_from_initial +=
                    usize::from((float_weight - initial_float).abs() > 1.0e-6);
                stats.legacy_changed_from_initial += usize::from(
                    float_weight.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
                        != initial_float as i16,
                );
                stats.scaled_changed_from_initial +=
                    usize::from(quantized_weight != initial_quantized);
                let scaled = float_weight * HALFKP_FEATURE_SCALE as f32;
                stats.feature_saturations +=
                    usize::from(scaled < i16::MIN as f32 || scaled > i16::MAX as f32);
            }
        }

        for (&float_weight, &quantized_weight) in
            self.output_weights.iter().zip(eval.output_weights.iter())
        {
            stats.output_nonzero += usize::from(quantized_weight != 0);
            let scaled = float_weight * HALFKP_OUTPUT_SCALE as f32;
            stats.output_saturations +=
                usize::from(scaled < i16::MIN as f32 || scaled > i16::MAX as f32);
        }
        for &bias in &self.feature_biases {
            let scaled = bias * HALFKP_FEATURE_SCALE as f32;
            stats.feature_bias_saturations +=
                usize::from(scaled < i16::MIN as f32 || scaled > i16::MAX as f32);
        }
        let scaled_bias = self.output_bias * HALFKP_BIAS_SCALE as f32;
        stats.output_bias_saturated =
            scaled_bias < i32::MIN as f32 || scaled_bias > i32::MAX as f32;

        stats
    }

    fn forward(
        &self,
        mover_feats: &[usize],
        opp_feats: &[usize],
    ) -> (
        f32,
        f32,
        [f32; HALFKP_HIDDEN_SIZE],
        [f32; HALFKP_HIDDEN_SIZE],
        [f32; HALFKP_HIDDEN_SIZE],
        [f32; HALFKP_HIDDEN_SIZE],
    ) {
        let mut m_acc = self.feature_biases;
        for &f in mover_feats {
            if f < HALFKP_INPUT_SIZE {
                for (a, &w) in m_acc.iter_mut().zip(self.feature_weights[f].iter()) {
                    *a += w;
                }
            }
        }

        let mut o_acc = self.feature_biases;
        for &f in opp_feats {
            if f < HALFKP_INPUT_SIZE {
                for (a, &w) in o_acc.iter_mut().zip(self.feature_weights[f].iter()) {
                    *a += w;
                }
            }
        }

        // ClippedReLU (0.0..=64.0)
        let mut m_hidden = [0.0f32; HALFKP_HIDDEN_SIZE];
        for (h, &a) in m_hidden.iter_mut().zip(m_acc.iter()) {
            *h = a.clamp(0.0, 64.0);
        }

        let mut o_hidden = [0.0f32; HALFKP_HIDDEN_SIZE];
        for (h, &a) in o_hidden.iter_mut().zip(o_acc.iter()) {
            *h = a.clamp(0.0, 64.0);
        }

        let mut output = self.output_bias;
        for (i, &h) in m_hidden.iter().enumerate() {
            output += h * self.output_weights[i];
        }
        for (i, &h) in o_hidden.iter().enumerate() {
            output += h * self.output_weights[HALFKP_HIDDEN_SIZE + i];
        }

        let raw_cp = output / 128.0;
        let score_cp = raw_cp.clamp(-MAX_EVAL_CP as f32, MAX_EVAL_CP as f32);

        (score_cp, output, m_acc, o_acc, m_hidden, o_hidden)
    }

    fn train_batch(
        &mut self,
        batch: &[(Vec<usize>, Vec<usize>, f32)],
        lr: f32,
        k: f32,
    ) -> TrainStepLoss {
        if batch.is_empty() {
            return TrainStepLoss::zero();
        }

        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let epsilon = 1e-8f32;
        let weight_decay = 0.0001f32;
        let lambda_range = 0.001f32;
        let ln10_div_k = 10.0f32.ln() / k;
        let max_eval = MAX_EVAL_CP as f32;

        let mut grad_out_w = [0.0f32; HALFKP_HIDDEN_SIZE * 2];
        let mut grad_out_b = 0.0f32;
        let mut grad_feature_biases = [0.0f32; HALFKP_HIDDEN_SIZE];
        let mut total_mse_loss = 0.0f32;
        let mut total_range_loss = 0.0f32;

        let mut active_feature_updates: HashMap<usize, [f32; HALFKP_HIDDEN_SIZE]> = HashMap::new();

        for (mover_feats, opp_feats, target) in batch {
            let (score_cp, raw_output, m_acc, o_acc, m_hidden, o_hidden) =
                self.forward(mover_feats, opp_feats);

            let pred = sigmoid_win_rate(score_cp, k);
            let error = pred - target;
            total_mse_loss += error * error;

            let stable_score_cp = score_cp.clamp(-6000.0, 6000.0);
            let stable_pred = sigmoid_win_rate(stable_score_cp, k);
            let d_sigmoid = (stable_pred * (1.0 - stable_pred) * ln10_div_k).max(1e-7);
            let d_loss_d_score = 2.0 * error * d_sigmoid;

            let raw_cp = raw_output / 128.0;
            let d_output = if raw_cp >= max_eval {
                if d_loss_d_score > 0.0 {
                    (d_loss_d_score / 128.0).clamp(-10.0, 10.0)
                } else {
                    0.0
                }
            } else if raw_cp <= -max_eval {
                if d_loss_d_score < 0.0 {
                    (d_loss_d_score / 128.0).clamp(-10.0, 10.0)
                } else {
                    0.0
                }
            } else {
                (d_loss_d_score / 128.0).clamp(-10.0, 10.0)
            };

            grad_out_b += d_output;

            let mut d_m_acc = [0.0f32; HALFKP_HIDDEN_SIZE];
            for (i, &h) in m_hidden.iter().enumerate() {
                grad_out_w[i] += d_output * h;
                let task_grad = if m_acc[i] > 0.0 && m_acc[i] < 64.0 {
                    d_output * self.output_weights[i]
                } else {
                    0.0
                };
                let (range_grad, range_loss) = if m_acc[i] < 0.0 {
                    (
                        lambda_range * m_acc[i],
                        0.5 * lambda_range * m_acc[i] * m_acc[i],
                    )
                } else if m_acc[i] > 64.0 {
                    let diff = m_acc[i] - 64.0;
                    (lambda_range * diff, 0.5 * lambda_range * diff * diff)
                } else {
                    (0.0, 0.0)
                };
                total_range_loss += range_loss;
                d_m_acc[i] = task_grad + range_grad;
            }

            let mut d_o_acc = [0.0f32; HALFKP_HIDDEN_SIZE];
            for (i, &h) in o_hidden.iter().enumerate() {
                grad_out_w[HALFKP_HIDDEN_SIZE + i] += d_output * h;
                let task_grad = if o_acc[i] > 0.0 && o_acc[i] < 64.0 {
                    d_output * self.output_weights[HALFKP_HIDDEN_SIZE + i]
                } else {
                    0.0
                };
                let (range_grad, range_loss) = if o_acc[i] < 0.0 {
                    (
                        lambda_range * o_acc[i],
                        0.5 * lambda_range * o_acc[i] * o_acc[i],
                    )
                } else if o_acc[i] > 64.0 {
                    let diff = o_acc[i] - 64.0;
                    (lambda_range * diff, 0.5 * lambda_range * diff * diff)
                } else {
                    (0.0, 0.0)
                };
                total_range_loss += range_loss;
                d_o_acc[i] = task_grad + range_grad;
            }

            for (gb, &dm) in grad_feature_biases.iter_mut().zip(d_m_acc.iter()) {
                *gb += dm;
            }
            for (gb, &do_val) in grad_feature_biases.iter_mut().zip(d_o_acc.iter()) {
                *gb += do_val;
            }

            for &f in mover_feats {
                if f < HALFKP_INPUT_SIZE {
                    let entry = active_feature_updates
                        .entry(f)
                        .or_insert([0.0f32; HALFKP_HIDDEN_SIZE]);
                    for (g, &dm) in entry.iter_mut().zip(d_m_acc.iter()) {
                        *g += dm;
                    }
                }
            }
            for &f in opp_feats {
                if f < HALFKP_INPUT_SIZE {
                    let entry = active_feature_updates
                        .entry(f)
                        .or_insert([0.0f32; HALFKP_HIDDEN_SIZE]);
                    for (g, &do_val) in entry.iter_mut().zip(d_o_acc.iter()) {
                        *g += do_val;
                    }
                }
            }
        }

        let batch_size_f = batch.len() as f32;
        grad_out_b /= batch_size_f;
        for g in &mut grad_out_w {
            *g /= batch_size_f;
        }
        for g in &mut grad_feature_biases {
            *g /= batch_size_f;
        }

        self.beta1_pow *= beta1;
        self.beta2_pow *= beta2;
        let one_minus_beta1 = 1.0 - self.beta1_pow;
        let one_minus_beta2 = 1.0 - self.beta2_pow;

        let hp = AdamWHyperParams {
            lr,
            beta1,
            beta2,
            eps: epsilon,
            wd: weight_decay,
        };

        Self::adamw_step(
            &mut self.output_bias,
            &mut self.m_bias,
            &mut self.v_bias,
            grad_out_b,
            one_minus_beta1,
            one_minus_beta2,
            &hp,
        );

        for (i, &g) in grad_out_w.iter().enumerate() {
            Self::adamw_step(
                &mut self.output_weights[i],
                &mut self.m_out[i],
                &mut self.v_out[i],
                g,
                one_minus_beta1,
                one_minus_beta2,
                &hp,
            );
        }

        for (i, &g) in grad_feature_biases.iter().enumerate() {
            Self::adamw_step(
                &mut self.feature_biases[i],
                &mut self.m_f_bias[i],
                &mut self.v_f_bias[i],
                g,
                one_minus_beta1,
                one_minus_beta2,
                &hp,
            );
        }

        for (f, grad_slice) in active_feature_updates {
            let step = self.step_feat[f].saturating_add(1);
            self.step_feat[f] = step;
            let one_minus_b1_f = if step > 1000 {
                1.0
            } else {
                1.0 - beta1.powi(step as i32)
            };
            let one_minus_b2_f = if step > 10000 {
                1.0
            } else {
                1.0 - beta2.powi(step as i32)
            };

            let m_slice = &mut self.m_feat[f];
            let v_slice = &mut self.v_feat[f];
            let w_slice = &mut self.feature_weights[f];
            for (i, &raw_g) in grad_slice.iter().enumerate() {
                let g = raw_g / batch_size_f;
                Self::adamw_step(
                    &mut w_slice[i],
                    &mut m_slice[i],
                    &mut v_slice[i],
                    g,
                    one_minus_b1_f,
                    one_minus_b2_f,
                    &hp,
                );
            }
        }

        let mse = total_mse_loss / batch_size_f;
        let range = total_range_loss / batch_size_f;
        TrainStepLoss {
            mse_loss: mse,
            range_loss: range,
            total_loss: mse + range,
        }
    }

    fn save_checkpoint(&self, path: &str) -> io::Result<()> {
        write_atomic_with_backup(path, |writer| {
            writer.write_all(CHECKPOINT_MAGIC)?;
            writer.write_all(&self.beta1_pow.to_le_bytes())?;
            writer.write_all(&self.beta2_pow.to_le_bytes())?;
            writer.write_all(&self.output_bias.to_le_bytes())?;
            writer.write_all(&self.m_bias.to_le_bytes())?;
            writer.write_all(&self.v_bias.to_le_bytes())?;

            for &b in &self.feature_biases {
                writer.write_all(&b.to_le_bytes())?;
            }
            for &m in &self.m_f_bias {
                writer.write_all(&m.to_le_bytes())?;
            }
            for &v in &self.v_f_bias {
                writer.write_all(&v.to_le_bytes())?;
            }

            for &w in &self.output_weights {
                writer.write_all(&w.to_le_bytes())?;
            }
            for &m in &self.m_out {
                writer.write_all(&m.to_le_bytes())?;
            }
            for &v in &self.v_out {
                writer.write_all(&v.to_le_bytes())?;
            }

            for row in &self.feature_weights {
                for &w in row {
                    writer.write_all(&w.to_le_bytes())?;
                }
            }
            for row in &self.m_feat {
                for &m in row {
                    writer.write_all(&m.to_le_bytes())?;
                }
            }
            for row in &self.v_feat {
                for &v in row {
                    writer.write_all(&v.to_le_bytes())?;
                }
            }
            for &s in &self.step_feat {
                writer.write_all(&s.to_le_bytes())?;
            }
            Ok(())
        })
    }

    fn load_checkpoint(path: &str) -> Result<Self, String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("Failed to open HalfKP checkpoint '{path}': {e}"))?;

        let expected_len: u64 = 8
            + 5 * 4
            + 3 * (HALFKP_HIDDEN_SIZE as u64) * 4
            + 3 * (HALFKP_HIDDEN_SIZE as u64 * 2) * 4
            + 3 * (HALFKP_INPUT_SIZE as u64) * (HALFKP_HIDDEN_SIZE as u64) * 4
            + (HALFKP_INPUT_SIZE as u64) * 4;
        let file_len = file
            .metadata()
            .map_err(|e| format!("Failed to read checkpoint file metadata: {e}"))?
            .len();
        if file_len != expected_len {
            return Err(format!(
                "HalfKP checkpoint file size mismatch: got {} bytes, expected {} bytes",
                file_len, expected_len
            ));
        }

        let mut reader = io::BufReader::new(file);

        let mut magic = [0u8; 8];
        reader
            .read_exact(&mut magic)
            .map_err(|e| format!("Failed to read checkpoint magic: {e}"))?;
        if &magic != CHECKPOINT_MAGIC {
            return Err("Invalid HalfKP checkpoint magic header".to_string());
        }

        let read_f32 = |r: &mut io::BufReader<std::fs::File>| -> Result<f32, String> {
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf)
                .map_err(|e| format!("Checkpoint read f32 error: {e}"))?;
            Ok(f32::from_le_bytes(buf))
        };
        let read_u32 = |r: &mut io::BufReader<std::fs::File>| -> Result<u32, String> {
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf)
                .map_err(|e| format!("Checkpoint read u32 error: {e}"))?;
            Ok(u32::from_le_bytes(buf))
        };

        let beta1_pow = read_f32(&mut reader)?;
        let beta2_pow = read_f32(&mut reader)?;
        let output_bias = read_f32(&mut reader)?;
        let m_bias = read_f32(&mut reader)?;
        let v_bias = read_f32(&mut reader)?;

        let mut feature_biases = [0.0f32; HALFKP_HIDDEN_SIZE];
        for b in &mut feature_biases {
            *b = read_f32(&mut reader)?;
        }
        let mut m_f_bias = [0.0f32; HALFKP_HIDDEN_SIZE];
        for m in &mut m_f_bias {
            *m = read_f32(&mut reader)?;
        }
        let mut v_f_bias = [0.0f32; HALFKP_HIDDEN_SIZE];
        for v in &mut v_f_bias {
            *v = read_f32(&mut reader)?;
        }

        let mut output_weights = [0.0f32; HALFKP_HIDDEN_SIZE * 2];
        for w in &mut output_weights {
            *w = read_f32(&mut reader)?;
        }
        let mut m_out = [0.0f32; HALFKP_HIDDEN_SIZE * 2];
        for m in &mut m_out {
            *m = read_f32(&mut reader)?;
        }
        let mut v_out = [0.0f32; HALFKP_HIDDEN_SIZE * 2];
        for v in &mut v_out {
            *v = read_f32(&mut reader)?;
        }

        let mut feature_weights = Vec::with_capacity(HALFKP_INPUT_SIZE);
        for _ in 0..HALFKP_INPUT_SIZE {
            let mut row = [0.0f32; HALFKP_HIDDEN_SIZE];
            for w in &mut row {
                *w = read_f32(&mut reader)?;
            }
            feature_weights.push(row);
        }

        let mut m_feat = Vec::with_capacity(HALFKP_INPUT_SIZE);
        for _ in 0..HALFKP_INPUT_SIZE {
            let mut row = [0.0f32; HALFKP_HIDDEN_SIZE];
            for m in &mut row {
                *m = read_f32(&mut reader)?;
            }
            m_feat.push(row);
        }

        let mut v_feat = Vec::with_capacity(HALFKP_INPUT_SIZE);
        for _ in 0..HALFKP_INPUT_SIZE {
            let mut row = [0.0f32; HALFKP_HIDDEN_SIZE];
            for v in &mut row {
                *v = read_f32(&mut reader)?;
            }
            v_feat.push(row);
        }

        let mut step_feat = Vec::with_capacity(HALFKP_INPUT_SIZE);
        for _ in 0..HALFKP_INPUT_SIZE {
            step_feat.push(read_u32(&mut reader)?);
        }

        Ok(HalfKPTrainer {
            feature_weights,
            feature_biases,
            output_weights,
            output_bias,
            m_feat,
            v_feat,
            step_feat,
            m_f_bias,
            v_f_bias,
            m_out,
            v_out,
            m_bias,
            v_bias,
            beta1_pow,
            beta2_pow,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Color;
    use std::collections::HashSet;

    #[test]
    fn trainer_initialization_and_roundtrip_exact() {
        let eval = HalfKPEvaluator::new();
        let trainer = HalfKPTrainer::from_evaluator(&eval);

        assert_eq!(trainer.feature_weights.len(), HALFKP_INPUT_SIZE);
        assert_eq!(trainer.feature_biases.len(), HALFKP_HIDDEN_SIZE);
        assert_eq!(trainer.output_weights.len(), HALFKP_HIDDEN_SIZE * 2);

        let exported_eval = trainer.to_evaluator();
        assert_eq!(exported_eval.feature_weights.len(), HALFKP_INPUT_SIZE);
        assert_eq!(exported_eval.feature_biases, eval.feature_biases);
        assert_eq!(exported_eval.output_weights, eval.output_weights);
        assert_eq!(exported_eval.output_bias, eval.output_bias);
    }

    #[test]
    fn trainer_forward_matches_quantized_evaluator_within_2cp() {
        let eval = HalfKPEvaluator::new();
        let trainer = HalfKPTrainer::from_evaluator(&eval);
        let pos = Position::startpos();

        let mover_feats = HalfKPEvaluator::extract_halfkp_features(&pos, Color::Black);
        let opp_feats = HalfKPEvaluator::extract_halfkp_features(&pos, Color::White);

        let (score_cp, _, _, _, _, _) = trainer.forward(&mover_feats, &opp_feats);
        let eval_score = eval.evaluate(&pos);
        let diff = (score_cp - eval_score as f32).abs();
        assert!(
            diff <= 2.0,
            "Trainer forward score {score_cp} deviates too much from Evaluator {eval_score}"
        );
    }

    #[test]
    fn sparse_gradient_leaves_inactive_features_bit_exact() {
        let mut trainer = HalfKPTrainer::new();
        let pos = Position::startpos();

        let mover_feats = HalfKPEvaluator::extract_halfkp_features(&pos, Color::Black);
        let opp_feats = HalfKPEvaluator::extract_halfkp_features(&pos, Color::White);
        let active_set: HashSet<usize> = mover_feats
            .iter()
            .chain(opp_feats.iter())
            .copied()
            .collect();

        let inactive_idx = 100_000usize;
        assert!(!active_set.contains(&inactive_idx));

        let before = trainer.feature_weights[inactive_idx];
        trainer.train_batch(&[(mover_feats, opp_feats, 1.0f32)], 0.01, DEFAULT_SIGMOID_K);
        assert_eq!(before, trainer.feature_weights[inactive_idx]);
    }

    #[test]
    fn nonzero_weights_quantization_consistency_within_bounds() {
        let mut trainer = HalfKPTrainer::new();
        let mut pos = Position::startpos();
        let mover_feats = HalfKPEvaluator::extract_halfkp_features(&pos, Color::Black);

        for (i, w) in trainer.output_weights.iter_mut().enumerate() {
            *w = ((i as f32 % 17.0) - 8.0) * 0.07;
        }
        trainer.output_bias = 257.25;
        for &f in &mover_feats {
            for (i, w) in trainer.feature_weights[f].iter_mut().enumerate() {
                *w = ((i as f32 % 7.0) - 3.0) * 0.13;
            }
        }

        let eval = trainer.to_evaluator();
        let mut rng = SimpleRng::new(0xd1b54a32d192ed03);
        let mut total_abs_error = 0.0f32;
        let mut max_abs_error = 0.0f32;
        let samples = 64usize;

        for _ in 0..samples {
            let mf = HalfKPEvaluator::extract_halfkp_features(&pos, pos.side_to_move);
            let of = HalfKPEvaluator::extract_halfkp_features(&pos, pos.side_to_move.opposite());
            let (float_cp, _, _, _, _, _) = trainer.forward(&mf, &of);
            let integer_cp = eval.evaluate(&pos);
            let err = (float_cp - integer_cp as f32).abs();
            total_abs_error += err;
            max_abs_error = max_abs_error.max(err);

            let moves = MoveGenerator::generate_legal_moves(&mut pos);
            if moves.is_empty() || pos.repetition_count() >= 4 {
                pos = Position::startpos();
            } else {
                let mv = moves[rng.gen_range(moves.len())];
                pos.do_move(mv);
            }
        }

        let mae = total_abs_error / samples as f32;
        assert!(mae <= 2.0, "MAE must be <= 2cp, got {mae}cp");
        assert!(
            max_abs_error <= 3.0,
            "Max abs error must be <= 3cp, got {max_abs_error}cp"
        );
    }

    #[test]
    fn fractional_update_survives_q6_quantization() {
        let mut trainer = HalfKPTrainer::new();
        let pos = Position::startpos();
        let feature = HalfKPEvaluator::extract_halfkp_features(&pos, Color::Black)[0];
        let initial = trainer.feature_weights[feature][0];
        trainer.feature_weights[feature][0] = initial + 0.1;

        assert_eq!(
            trainer.feature_weights[feature][0].round(),
            initial.round(),
            "legacy scale-1 export would erase this 0.1 update"
        );
        let eval = trainer.to_evaluator();
        let initial_quantized = (initial * HALFKP_FEATURE_SCALE as f32).round() as i16;
        assert_ne!(
            eval.feature_weights[feature][0], initial_quantized,
            "Q6 export must preserve a 0.1-weight update"
        );
    }

    #[test]
    fn sparse_adamw_local_step_prevents_first_seen_feature_spike() {
        let mut trainer = HalfKPTrainer::new();
        let pos = Position::startpos();
        let mover_feats = HalfKPEvaluator::extract_halfkp_features(&pos, Color::Black);
        let opp_feats = HalfKPEvaluator::extract_halfkp_features(&pos, Color::White);

        let dummy_batch = vec![(vec![0usize], vec![1usize], 0.5f32)];
        for _ in 0..100 {
            trainer.train_batch(&dummy_batch, 0.001, DEFAULT_SIGMOID_K);
        }

        let test_feat = mover_feats[0];
        let before = trainer.feature_weights[test_feat][0];
        let lr = 0.01f32;
        trainer.train_batch(&[(mover_feats, opp_feats, 1.0f32)], lr, DEFAULT_SIGMOID_K);
        let delta = (trainer.feature_weights[test_feat][0] - before).abs();
        assert!(
            delta <= lr * 1.5,
            "Delta {delta} exceeds stable bound (max 1.5*lr = {})",
            lr * 1.5
        );
    }

    #[test]
    fn checkpoint_roundtrip_preserves_optimizer_state() {
        let mut trainer = HalfKPTrainer::new();
        let pos = Position::startpos();
        let mf = HalfKPEvaluator::extract_halfkp_features(&pos, Color::Black);
        let of = HalfKPEvaluator::extract_halfkp_features(&pos, Color::White);
        trainer.train_batch(&[(mf, of, 1.0f32)], 0.01, DEFAULT_SIGMOID_K);

        let path = std::env::temp_dir().join(format!(
            "tabula_trainer_unit_ckpt_{}.bin",
            std::process::id()
        ));
        let path_str = path.to_string_lossy().into_owned();
        trainer.save_checkpoint(&path_str).unwrap();
        let loaded = HalfKPTrainer::load_checkpoint(&path_str).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(trainer.beta1_pow, loaded.beta1_pow);
        assert_eq!(trainer.beta2_pow, loaded.beta2_pow);
        assert_eq!(trainer.output_bias, loaded.output_bias);
        assert_eq!(trainer.output_weights, loaded.output_weights);
        assert_eq!(trainer.feature_biases, loaded.feature_biases);
        assert_eq!(trainer.feature_weights[0], loaded.feature_weights[0]);
        assert_eq!(trainer.m_feat[0], loaded.m_feat[0]);
        assert_eq!(trainer.v_feat[0], loaded.v_feat[0]);
        assert_eq!(trainer.step_feat[0], loaded.step_feat[0]);
    }
}
