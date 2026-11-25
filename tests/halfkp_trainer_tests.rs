mod common;
use common::TestTempDir;
use std::fs;
use std::path::Path;
use tabula_shogi::board::Position;
use tabula_shogi::eval::halfkp::HalfKPEvaluator;
use tabula_shogi::eval::halfkp_trainer::{
    DEFAULT_SIGMOID_K, HalfKPInitPolicy, HalfKPTargetPolicy, HalfKPTrainSessionConfig,
    HalfKPTrainSource, MAX_BATCH_SIZE, export_halfkp_checkpoint, inspect_halfkp_checkpoint,
    train_halfkp_session,
};
use tabula_shogi::eval::sigmoid_win_rate;
use tabula_shogi::selfplay::dataset::DatasetEntry;

fn startpos_entry(score: i32, result: f32) -> DatasetEntry {
    DatasetEntry {
        sfen: "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1".to_string(),
        score,
        result,
        move_usi: "7g7f".to_string(),
    }
}

#[test]
fn test_halfkp_session_in_memory_convergence_and_persistence() {
    let ws = TestTempDir::new("hkp_session_conv");
    let model_path = ws.file_path("nested/models/candidate.bin");
    let ckpt_path = ws.file_path("nested/ckpts/candidate_ckpt.bin");

    let mut entries = vec![startpos_entry(0, 1.0); 4];
    let config = HalfKPTrainSessionConfig {
        init: HalfKPInitPolicy::Fresh,
        target: HalfKPTargetPolicy::OutcomeOnly,
        epochs: 10,
        batch_size: 4,
        lr: 0.01,
        k: DEFAULT_SIGMOID_K,
        model_output_path: Some(model_path.clone()),
        checkpoint_output_path: Some(ckpt_path.clone()),
        checkpoint_interval_batches: 0,
        persist_each_epoch: false,
    };

    let outcome = train_halfkp_session(
        config,
        HalfKPTrainSource::InMemory {
            entries: &mut entries,
            shuffle_seed: 42,
        },
    )
    .expect("In-memory HalfKP session must succeed");

    assert_eq!(outcome.epochs_completed, 10);
    assert_eq!(outcome.total_positions_trained, 40);
    assert!(
        outcome.final_loss.mse_loss < outcome.init_loss.mse_loss,
        "Final MSE {} must be lower than initial MSE {}",
        outcome.final_loss.mse_loss,
        outcome.init_loss.mse_loss
    );

    let pos = Position::startpos();
    let eval_cp = outcome.evaluator.evaluate(&pos);
    let pred = sigmoid_win_rate(eval_cp as f32, DEFAULT_SIGMOID_K);
    assert!(
        pred > 0.5,
        "Trained Evaluation Model should predict win bias (> 0.5), got {pred} ({eval_cp} cp)"
    );

    assert!(Path::new(&model_path).exists());
    assert!(Path::new(&ckpt_path).exists());
    let reloaded = HalfKPEvaluator::load_from_file(&model_path).expect("Saved model must reload");
    assert_eq!(reloaded.evaluate(&pos), eval_cp);

    let summary = inspect_halfkp_checkpoint(&ckpt_path).expect("Saved checkpoint must inspect");
    assert!(summary.active_features > 0);
    assert!(summary.total_feature_steps >= 10);
}

#[test]
fn test_halfkp_session_resume_vs_warm_start_and_corrupt_checkpoint_fail_closed() {
    let ws = TestTempDir::new("hkp_session_resume");
    let model_path = ws.file_path("model.bin");
    let ckpt_path = ws.file_path("ckpt.bin");
    let bak_path = format!("{ckpt_path}.bak");

    let mut entries = vec![startpos_entry(400, 0.5); 2];

    // Session 1: Fresh training creates Training Checkpoint (no .bak yet)
    let out1 = train_halfkp_session(
        HalfKPTrainSessionConfig {
            init: HalfKPInitPolicy::Fresh,
            target: HalfKPTargetPolicy::ScoreBlend { score_weight: 0.8 },
            epochs: 2,
            batch_size: 2,
            lr: 0.01,
            k: DEFAULT_SIGMOID_K,
            model_output_path: Some(model_path.clone()),
            checkpoint_output_path: Some(ckpt_path.clone()),
            checkpoint_interval_batches: 0,
            persist_each_epoch: false,
        },
        HalfKPTrainSource::InMemory {
            entries: &mut entries,
            shuffle_seed: 1,
        },
    )
    .expect("Session 1 must succeed");
    let ckpt1 = inspect_halfkp_checkpoint(&ckpt_path).unwrap();
    assert!(!Path::new(&bak_path).exists());

    // Session 2: Resume preserves optimizer state and advances feature steps + creates .bak
    let _out2 = train_halfkp_session(
        HalfKPTrainSessionConfig {
            init: HalfKPInitPolicy::ResumeOrWarmStart {
                checkpoint_path: ckpt_path.clone(),
                evaluator: &out1.evaluator,
            },
            target: HalfKPTargetPolicy::ScoreBlend { score_weight: 0.8 },
            epochs: 2,
            batch_size: 2,
            lr: 0.01,
            k: DEFAULT_SIGMOID_K,
            model_output_path: Some(model_path.clone()),
            checkpoint_output_path: Some(ckpt_path.clone()),
            checkpoint_interval_batches: 0,
            persist_each_epoch: false,
        },
        HalfKPTrainSource::InMemory {
            entries: &mut entries,
            shuffle_seed: 2,
        },
    )
    .expect("Resumed session must succeed");

    let ckpt2 = inspect_halfkp_checkpoint(&ckpt_path).unwrap();
    assert!(ckpt2.total_feature_steps > ckpt1.total_feature_steps);
    assert!(ckpt2.beta1_pow < ckpt1.beta1_pow);
    assert!(Path::new(&bak_path).exists());
    let bak_summary = inspect_halfkp_checkpoint(&bak_path).unwrap();
    assert_eq!(bak_summary.total_feature_steps, ckpt1.total_feature_steps);

    // Corrupt checkpoint must fail closed and point operator to .bak recovery
    fs::write(&ckpt_path, vec![0u8; 100]).unwrap();
    let err = train_halfkp_session(
        HalfKPTrainSessionConfig {
            init: HalfKPInitPolicy::ResumeOrWarmStart {
                checkpoint_path: ckpt_path.clone(),
                evaluator: &out1.evaluator,
            },
            target: HalfKPTargetPolicy::OutcomeOnly,
            epochs: 1,
            batch_size: 2,
            lr: 0.01,
            k: DEFAULT_SIGMOID_K,
            model_output_path: Some(model_path),
            checkpoint_output_path: Some(ckpt_path),
            checkpoint_interval_batches: 0,
            persist_each_epoch: false,
        },
        HalfKPTrainSource::InMemory {
            entries: &mut entries,
            shuffle_seed: 3,
        },
    )
    .expect_err("Corrupt checkpoint must fail closed");
    assert!(
        err.contains("file size mismatch"),
        "Error should mention size mismatch: {err}"
    );
    assert!(
        err.contains(".bak"),
        "Error should guide operator to existing .bak checkpoint: {err}"
    );
}

#[test]
fn test_halfkp_verified_export_and_path_collision_guard() {
    let ws = TestTempDir::new("hkp_export");
    let ckpt_path = ws.file_path("trained_ckpt.bin");
    let out_path = ws.file_path("exported_model.bin");

    let mut entries = vec![startpos_entry(300, 1.0); 4];
    train_halfkp_session(
        HalfKPTrainSessionConfig {
            init: HalfKPInitPolicy::Fresh,
            target: HalfKPTargetPolicy::OutcomeOnly,
            epochs: 3,
            batch_size: 4,
            lr: 0.01,
            k: DEFAULT_SIGMOID_K,
            model_output_path: None,
            checkpoint_output_path: Some(ckpt_path.clone()),
            checkpoint_interval_batches: 0,
            persist_each_epoch: false,
        },
        HalfKPTrainSource::InMemory {
            entries: &mut entries,
            shuffle_seed: 7,
        },
    )
    .expect("Seed checkpoint session must succeed");

    // Same path for checkpoint and output must be rejected
    let collision_err = export_halfkp_checkpoint(&ckpt_path, &ckpt_path, 16)
        .expect_err("Same checkpoint and output path must be rejected");
    assert!(collision_err.contains("must be different"));

    let report =
        export_halfkp_checkpoint(&ckpt_path, &out_path, 32).expect("Verified export must succeed");
    assert_eq!(report.verification_samples, 32);
    assert!(
        report.verification_mae_cp <= 2.0,
        "Quantization MAE must stay <= 2.0 cp, got {}",
        report.verification_mae_cp
    );
    assert!(
        report.verification_max_abs_error_cp <= 3.0,
        "Quantization max error must stay <= 3.0 cp, got {}",
        report.verification_max_abs_error_cp
    );
    assert!(report.quantization.scaled_changed_from_initial > 0);
    assert_eq!(report.output_bytes, 52_255_508);
}

#[test]
fn test_halfkp_session_validation_guards() {
    let ws = TestTempDir::new("hkp_validation");
    let mut entries = vec![startpos_entry(0, 1.0)];

    // 1. batch_size = 0
    assert!(
        train_halfkp_session(
            HalfKPTrainSessionConfig {
                init: HalfKPInitPolicy::Fresh,
                target: HalfKPTargetPolicy::OutcomeOnly,
                epochs: 1,
                batch_size: 0,
                lr: 0.001,
                k: DEFAULT_SIGMOID_K,
                model_output_path: None,
                checkpoint_output_path: None,
                checkpoint_interval_batches: 0,
                persist_each_epoch: false,
            },
            HalfKPTrainSource::InMemory {
                entries: &mut entries,
                shuffle_seed: 1,
            },
        )
        .is_err()
    );

    // 2. batch_size > MAX_BATCH_SIZE (500 MiB protection)
    assert!(
        train_halfkp_session(
            HalfKPTrainSessionConfig {
                init: HalfKPInitPolicy::Fresh,
                target: HalfKPTargetPolicy::OutcomeOnly,
                epochs: 1,
                batch_size: MAX_BATCH_SIZE + 1,
                lr: 0.001,
                k: DEFAULT_SIGMOID_K,
                model_output_path: None,
                checkpoint_output_path: None,
                checkpoint_interval_batches: 0,
                persist_each_epoch: false,
            },
            HalfKPTrainSource::InMemory {
                entries: &mut entries,
                shuffle_seed: 1,
            },
        )
        .is_err()
    );

    // 3. Invalid score_weight
    assert!(
        train_halfkp_session(
            HalfKPTrainSessionConfig {
                init: HalfKPInitPolicy::Fresh,
                target: HalfKPTargetPolicy::ScoreBlend { score_weight: 1.5 },
                epochs: 1,
                batch_size: 1,
                lr: 0.001,
                k: DEFAULT_SIGMOID_K,
                model_output_path: None,
                checkpoint_output_path: None,
                checkpoint_interval_batches: 0,
                persist_each_epoch: false,
            },
            HalfKPTrainSource::InMemory {
                entries: &mut entries,
                shuffle_seed: 1,
            },
        )
        .is_err()
    );

    // 4. Colliding model and checkpoint output paths (even before either file exists)
    let same = ws.file_path("sub/shared.bin");
    let alias = ws.file_path("sub/./shared.bin");
    assert!(
        train_halfkp_session(
            HalfKPTrainSessionConfig {
                init: HalfKPInitPolicy::Fresh,
                target: HalfKPTargetPolicy::OutcomeOnly,
                epochs: 1,
                batch_size: 1,
                lr: 0.001,
                k: DEFAULT_SIGMOID_K,
                model_output_path: Some(same),
                checkpoint_output_path: Some(alias),
                checkpoint_interval_batches: 0,
                persist_each_epoch: false,
            },
            HalfKPTrainSource::InMemory {
                entries: &mut entries,
                shuffle_seed: 1,
            },
        )
        .is_err()
    );

    // 5. Corrupted checkpoint inspection fails closed with size mismatch
    let bad_ckpt = ws.file_path("bad_ckpt.bin");
    fs::write(&bad_ckpt, vec![0u8; 100]).unwrap();
    let err = inspect_halfkp_checkpoint(&bad_ckpt).expect_err("Corrupt checkpoint must fail");
    assert!(err.contains("file size mismatch"));
}
