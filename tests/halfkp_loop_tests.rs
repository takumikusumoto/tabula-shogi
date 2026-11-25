mod common;
use common::TestTempDir;
use std::fs;
use tabula_shogi::arena::{
    LoopArenaParams, LoopConfig, LoopStoragePaths, LoopTrainingParams, LoopWorkspace, MatchResult,
    PromotionVerdict, SelfImprovementLoop, Sprt, SprtConfig, SprtStatus,
};
use tabula_shogi::eval::EvalMode;
use tabula_shogi::eval::halfkp::HalfKPEvaluator;
use tabula_shogi::eval::halfkp_trainer::{
    DEFAULT_SIGMOID_K, HalfKPInitPolicy, HalfKPTargetPolicy, HalfKPTrainSessionConfig,
    HalfKPTrainSource, inspect_halfkp_checkpoint, train_halfkp_session,
};
use tabula_shogi::selfplay::dataset::DatasetEntry;

#[test]
fn test_halfkp_autonomous_loop_single_iteration() {
    let ws = TestTempDir::new("hkp_loop_test");
    let data_path = ws.file_path("data.tsv");
    let deep_data_path = ws.file_path("deep.tsv");
    let best_path = ws.file_path("best.bin");
    let cand_path = ws.file_path("cand.bin");
    let cand_ckpt_path = ws.file_path("cand_ckpt.bin");
    let state_path = ws.file_path("state.txt");
    let summary_path = ws.file_path("summary.csv");

    let config = LoopConfig {
        iterations: 1,
        start_iteration: Some(1),
        games_per_iteration: 4,
        arena: LoopArenaParams {
            eval_pairs: 2,
            threads: 2,
            depth: 1,
            min_promotion_games: 20,
        },
        training: LoopTrainingParams {
            epochs: 2,
            lr: 0.001,
            batch_size: 16,
        },
        paths: LoopStoragePaths {
            state_path: state_path.clone(),
            data_path: data_path.clone(),
            deep_data_path: deep_data_path.clone(),
            best_model_path: best_path.clone(),
            candidate_model_path: cand_path.clone(),
            candidate_ckpt_path: cand_ckpt_path.clone(),
            summary_path: summary_path.clone(),
        },
    };

    assert!(SelfImprovementLoop::run(&config).is_ok());

    // 1. 自己対局データセットが生成されていることを確認
    assert!(fs::metadata(&data_path).is_ok(), "Dataset must be created");

    // 2. 候補モデル（HalfKPバイナリ）が正常保存され、読み込み可能であることを検証
    assert!(
        fs::metadata(&cand_path).is_ok(),
        "Candidate model must be saved"
    );
    let loaded_eval = HalfKPEvaluator::load_from_file(&cand_path);
    assert!(
        loaded_eval.is_ok(),
        "Candidate model must load successfully"
    );

    // 3. 候補チェックポイント（AdamWモーメンタムバイナリ）が正常保存され、読み込み可能であることを検証
    assert!(
        fs::metadata(&cand_ckpt_path).is_ok(),
        "Candidate AdamW checkpoint must be saved"
    );
    let ckpt_summary = inspect_halfkp_checkpoint(&cand_ckpt_path);
    assert!(
        ckpt_summary.is_ok(),
        "Candidate AdamW checkpoint must load successfully"
    );

    // 4. 世代番号が永続化されていることを確認
    assert!(
        fs::metadata(&state_path).is_ok(),
        "State file must be written"
    );
    let saved_gen = fs::read_to_string(&state_path).unwrap();
    assert_eq!(saved_gen.trim(), "1");

    // 5. 進捗サマリ CSV が出力されていることを確認
    assert!(
        fs::metadata(&summary_path).is_ok(),
        "Summary CSV must be written"
    );
    let summary_content = fs::read_to_string(&summary_path).unwrap();
    assert!(summary_content.contains("generation,timestamp,champion_mode"));
    assert!(summary_content.lines().count() >= 2);
}

#[test]
fn test_halfkp_loop_memory_release_and_checkpoint_continuation() {
    let ws = TestTempDir::new("hkp_cont_test");
    let data_path = ws.file_path("data.tsv");
    let deep_data_path = ws.file_path("deep.tsv");
    let best_path = ws.file_path("best.bin");
    let cand_path = ws.file_path("cand.bin");
    let cand_ckpt_path = ws.file_path("cand_ckpt.bin");
    let state_path = ws.file_path("state.txt");
    let summary_path = ws.file_path("summary.csv");

    // 事前に 1 ステップ学習済みの Training Checkpoint を作成
    let mut seed_entries = vec![DatasetEntry {
        sfen: "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1".to_string(),
        score: 100,
        result: 1.0,
        move_usi: "7g7f".to_string(),
    }];
    train_halfkp_session(
        HalfKPTrainSessionConfig {
            init: HalfKPInitPolicy::Fresh,
            target: HalfKPTargetPolicy::OutcomeOnly,
            epochs: 1,
            batch_size: 1,
            lr: 0.01,
            k: DEFAULT_SIGMOID_K,
            model_output_path: None,
            checkpoint_output_path: Some(cand_ckpt_path.clone()),
            checkpoint_interval_batches: 0,
            persist_each_epoch: false,
        },
        HalfKPTrainSource::InMemory {
            entries: &mut seed_entries,
            shuffle_seed: 1,
        },
    )
    .expect("Failed to create seed checkpoint");
    let before_ckpt =
        inspect_halfkp_checkpoint(&cand_ckpt_path).expect("Seed checkpoint must be inspectable");

    let config = LoopConfig {
        iterations: 1,
        start_iteration: Some(2),
        games_per_iteration: 4,
        arena: LoopArenaParams {
            eval_pairs: 2,
            threads: 2,
            depth: 1,
            min_promotion_games: 20,
        },
        training: LoopTrainingParams {
            epochs: 1,
            lr: 0.001,
            batch_size: 16,
        },
        paths: LoopStoragePaths {
            state_path: state_path.clone(),
            data_path: data_path.clone(),
            deep_data_path: deep_data_path.clone(),
            best_model_path: best_path.clone(),
            candidate_model_path: cand_path.clone(),
            candidate_ckpt_path: cand_ckpt_path.clone(),
            summary_path: summary_path.clone(),
        },
    };

    assert!(SelfImprovementLoop::run(&config).is_ok());

    // チェックポイントから Resume されて学習が実行され、オプティマイザステップが前進したことを検証
    let continued_ckpt = inspect_halfkp_checkpoint(&cand_ckpt_path)
        .expect("Must be able to load continued checkpoint");
    assert!(
        continued_ckpt.total_feature_steps > before_ckpt.total_feature_steps,
        "Total feature steps should advance from {}, got {}",
        before_ckpt.total_feature_steps,
        continued_ckpt.total_feature_steps
    );
    assert!(
        continued_ckpt.beta1_pow < before_ckpt.beta1_pow,
        "Adam beta1_pow should decay further from {}, got {}",
        before_ckpt.beta1_pow,
        continued_ckpt.beta1_pow
    );

    // 世代番号が 2 に更新されていることを確認
    let saved_gen = fs::read_to_string(&state_path).unwrap();
    assert_eq!(saved_gen.trim(), "2");
}

#[test]
fn test_halfkp_loop_fails_closed_on_corrupt_checkpoint() {
    let ws = TestTempDir::new("hkp_corrupt_ckpt_test");
    let cand_ckpt_path = ws.file_path("cand_ckpt.bin");
    fs::write(&cand_ckpt_path, vec![0u8; 64]).unwrap();

    let config = LoopConfig {
        iterations: 1,
        start_iteration: Some(1),
        games_per_iteration: 2,
        arena: LoopArenaParams {
            eval_pairs: 1,
            threads: 1,
            depth: 1,
            min_promotion_games: 20,
        },
        training: LoopTrainingParams {
            epochs: 1,
            lr: 0.001,
            batch_size: 16,
        },
        paths: LoopStoragePaths {
            state_path: ws.file_path("state.txt"),
            data_path: ws.file_path("data.tsv"),
            deep_data_path: ws.file_path("deep.tsv"),
            best_model_path: ws.file_path("best.bin"),
            candidate_model_path: ws.file_path("cand.bin"),
            candidate_ckpt_path: cand_ckpt_path,
            summary_path: ws.file_path("summary.csv"),
        },
    };

    let res = SelfImprovementLoop::run(&config);
    assert!(
        res.is_err(),
        "Loop must fail closed when candidate checkpoint is corrupted"
    );
}

#[test]
fn test_halfkp_loop_rejects_best_and_checkpoint_path_collision() {
    let ws = TestTempDir::new("hkp_ckpt_collision_test");
    let shared = ws.file_path("models/shared.bin");
    let shared_alias = ws.file_path("models/./shared.bin");

    let config = LoopConfig {
        iterations: 1,
        start_iteration: Some(1),
        games_per_iteration: 1,
        arena: LoopArenaParams {
            eval_pairs: 1,
            threads: 1,
            depth: 1,
            min_promotion_games: 20,
        },
        training: LoopTrainingParams {
            epochs: 1,
            lr: 0.001,
            batch_size: 16,
        },
        paths: LoopStoragePaths {
            state_path: ws.file_path("state.txt"),
            data_path: ws.file_path("data.tsv"),
            deep_data_path: ws.file_path("deep.tsv"),
            best_model_path: shared,
            candidate_model_path: ws.file_path("models/cand.bin"),
            candidate_ckpt_path: shared_alias,
            summary_path: ws.file_path("summary.csv"),
        },
    };

    let res = SelfImprovementLoop::run(&config);
    assert!(
        res.is_err(),
        "Loop must reject best_model_path == candidate_ckpt_path even before files exist"
    );
}

#[test]
fn test_halfkp_loop_promotion_gate_blocks_small_samples() {
    let ws = TestTempDir::new("hkp_gate_test");
    let best_path = ws.file_path("best.bin");
    let ckpt_path = ws.file_path("cand_ckpt.bin");
    let workspace = LoopWorkspace::open(LoopStoragePaths {
        state_path: ws.file_path("state.txt"),
        data_path: ws.file_path("data.tsv"),
        deep_data_path: ws.file_path("deep.tsv"),
        best_model_path: best_path.clone(),
        candidate_model_path: ws.file_path("cand.bin"),
        candidate_ckpt_path: ckpt_path.clone(),
        summary_path: ws.file_path("summary.csv"),
    })
    .expect("workspace open should succeed");
    fs::write(&ckpt_path, b"pending-ckpt").unwrap();

    let candidate_eval = std::sync::Arc::new(HalfKPEvaluator::new());
    let mut current_best = EvalMode::Hce;

    let mut sprt = Sprt::new(SprtConfig {
        elo0: 0.0,
        elo1: 50.0,
        alpha: 0.05,
        beta: 0.05,
        min_games: 0,
    });
    sprt.record_batch(6, 0, 0);
    assert_eq!(sprt.status, SprtStatus::Pass);

    // 6連勝で SPRT Pass だが対局数が 6局 (< 20局) の状況をシミュレート
    let match_res = MatchResult {
        total_games: 6,
        wins_a: 6,
        wins_b: 0,
        draws: 0,
        win_rate_a: 1.0,
        elo_diff_a: 1000.0,
        sprt: Some(sprt),
    };

    // min_promotion_games = 20 の場合: 6局では昇格がゲートによってブロックされ、チェックポイントは維持される
    let verdict = workspace
        .apply_promotion(1, &match_res, &candidate_eval, &mut current_best, 20)
        .expect("apply_promotion should succeed");

    assert_eq!(
        verdict,
        PromotionVerdict::DeferredInsufficientGames {
            played: 6,
            required: 20,
        },
        "Promotion must be deferred when total_games < min_promotion_games (6 < 20)"
    );
    assert!(
        matches!(current_best, EvalMode::Hce),
        "Current best must remain HCE"
    );
    assert!(
        !std::path::Path::new(&best_path).exists(),
        "Best model file must not be created"
    );
    assert!(
        std::path::Path::new(&ckpt_path).exists(),
        "Candidate checkpoint must be retained when promotion is deferred"
    );
}

#[test]
fn test_halfkp_loop_promotion_gate_permits_when_threshold_met() {
    let ws = TestTempDir::new("hkp_gate_pass");
    let best_path = ws.file_path("best.bin");
    let ckpt_path = ws.file_path("cand_ckpt.bin");
    let ckpt_bak_path = format!("{ckpt_path}.bak");
    let workspace = LoopWorkspace::open(LoopStoragePaths {
        state_path: ws.file_path("state.txt"),
        data_path: ws.file_path("data.tsv"),
        deep_data_path: ws.file_path("deep.tsv"),
        best_model_path: best_path.clone(),
        candidate_model_path: ws.file_path("cand.bin"),
        candidate_ckpt_path: ckpt_path.clone(),
        summary_path: ws.file_path("summary.csv"),
    })
    .expect("workspace open should succeed");
    fs::write(&ckpt_path, b"stale-ckpt").unwrap();
    fs::write(&ckpt_bak_path, b"stale-bak").unwrap();

    let candidate_eval = std::sync::Arc::new(HalfKPEvaluator::new());
    let mut current_best = EvalMode::Hce;

    let mut sprt = Sprt::new(SprtConfig::default());
    sprt.record_batch(30, 0, 0);
    assert_eq!(sprt.status, SprtStatus::Pass);

    // 30局消化して SPRT Pass (20局以上)
    let match_res = MatchResult {
        total_games: 30,
        wins_a: 30,
        wins_b: 0,
        draws: 0,
        win_rate_a: 1.0,
        elo_diff_a: 300.0,
        sprt: Some(sprt),
    };

    // min_promotion_games = 20 の場合: 30局 >= 20局 かつ SPRT Pass なので昇格し、古いチェックポイントと .bak を整理
    let verdict = workspace
        .apply_promotion(1, &match_res, &candidate_eval, &mut current_best, 20)
        .expect("apply_promotion should succeed");

    assert_eq!(
        verdict,
        PromotionVerdict::Promote,
        "Promotion must be accepted when total_games >= min_promotion_games (30 >= 20) and SPRT Pass"
    );
    assert!(
        matches!(current_best, EvalMode::HalfKP(_)),
        "Current best must be updated to HalfKP"
    );
    assert!(
        std::path::Path::new(&best_path).exists(),
        "Best model file must be saved"
    );
    assert!(
        !std::path::Path::new(&ckpt_path).exists(),
        "Stale candidate checkpoint must be removed on promotion"
    );
    assert!(
        !std::path::Path::new(&ckpt_bak_path).exists(),
        "Stale candidate checkpoint backup must be removed on promotion"
    );

    // 保存されたモデルがロード可能であることを検証
    let loaded = HalfKPEvaluator::load_from_file(&best_path);
    assert!(loaded.is_ok(), "Promoted model must be valid HalfKP binary");
}

#[test]
fn test_halfkp_loop_fails_closed_on_unwritable_deep_pool() {
    let ws = TestTempDir::new("hkp_deep_pool_fail");
    let deep_dir = ws.file_path("deep_is_a_dir");
    fs::create_dir_all(&deep_dir).unwrap();

    let config = LoopConfig {
        iterations: 1,
        start_iteration: Some(1),
        games_per_iteration: 2,
        arena: LoopArenaParams {
            eval_pairs: 1,
            threads: 1,
            depth: 1,
            min_promotion_games: 20,
        },
        training: LoopTrainingParams {
            epochs: 1,
            lr: 0.001,
            batch_size: 16,
        },
        paths: LoopStoragePaths {
            state_path: ws.file_path("state.txt"),
            data_path: ws.file_path("data.tsv"),
            deep_data_path: deep_dir,
            best_model_path: ws.file_path("best.bin"),
            candidate_model_path: ws.file_path("cand.bin"),
            candidate_ckpt_path: ws.file_path("cand_ckpt.bin"),
            summary_path: ws.file_path("summary.csv"),
        },
    };

    let res = SelfImprovementLoop::run(&config);
    assert!(
        res.is_err(),
        "Loop must fail closed when persisting deep relabeled pool fails"
    );
}

#[test]
fn test_relabel_deep_with_halfkp_champion() {
    use std::sync::Arc;
    use tabula_shogi::board::Position;
    use tabula_shogi::selfplay::dataset::DatasetEntry;
    use tabula_shogi::selfplay::relabel_deep;

    let start_pos = Position::startpos();
    let mut entries = vec![DatasetEntry {
        sfen: start_pos.to_sfen(),
        score: 9999,
        result: 0.5,
        move_usi: "7g7f".to_string(),
    }];

    let halfkp = Arc::new(HalfKPEvaluator::new());
    let eval_mode = EvalMode::HalfKP(halfkp);

    let successes = relabel_deep(&mut entries, 1, 1, 1, &eval_mode);
    assert_eq!(successes.len(), 1, "HalfKP deep relabeling must succeed");
    assert_ne!(
        entries[0].score, 9999,
        "Score must be re-evaluated using HalfKP model"
    );
}
