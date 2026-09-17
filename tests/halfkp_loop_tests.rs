use std::fs;
use tabula_shogi::arena::{LoopConfig, MatchResult, SelfImprovementLoop, SprtStatus};
use tabula_shogi::eval::EvalMode;
use tabula_shogi::eval::halfkp::HalfKPEvaluator;
use tabula_shogi::eval::halfkp_trainer::HalfKPTrainer;

#[test]
fn test_halfkp_autonomous_loop_single_iteration() {
    let temp_dir = std::env::temp_dir();
    let prefix = format!(
        "hkp_loop_test_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let data_path = temp_dir
        .join(format!("{prefix}_data.tsv"))
        .to_str()
        .unwrap()
        .to_string();
    let deep_data_path = temp_dir
        .join(format!("{prefix}_deep.tsv"))
        .to_str()
        .unwrap()
        .to_string();
    let best_path = temp_dir
        .join(format!("{prefix}_best.bin"))
        .to_str()
        .unwrap()
        .to_string();
    let cand_path = temp_dir
        .join(format!("{prefix}_cand.bin"))
        .to_str()
        .unwrap()
        .to_string();
    let cand_ckpt_path = temp_dir
        .join(format!("{prefix}_cand_ckpt.bin"))
        .to_str()
        .unwrap()
        .to_string();
    let state_path = temp_dir
        .join(format!("{prefix}_state.txt"))
        .to_str()
        .unwrap()
        .to_string();

    let clean_files = || {
        let _ = fs::remove_file(&data_path);
        let _ = fs::remove_file(&deep_data_path);
        let _ = fs::remove_file(&best_path);
        let _ = fs::remove_file(&cand_path);
        let _ = fs::remove_file(&cand_ckpt_path);
        let _ = fs::remove_file(format!("{cand_ckpt_path}.bak"));
        let _ = fs::remove_file(&state_path);
    };
    clean_files();

    let config = LoopConfig {
        iterations: 1,
        start_iteration: Some(1),
        state_path: state_path.clone(),
        games_per_iteration: 4,
        eval_pairs: 2,
        threads: 2,
        depth: 1,
        epochs: 2,
        lr: 0.001,
        batch_size: 16,
        data_path: data_path.clone(),
        deep_data_path: deep_data_path.clone(),
        best_model_path: best_path.clone(),
        candidate_model_path: cand_path.clone(),
        candidate_ckpt_path: cand_ckpt_path.clone(),
        min_promotion_games: 20,
    };

    SelfImprovementLoop::run(&config);

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
    let loaded_trainer = HalfKPTrainer::load_checkpoint(&cand_ckpt_path);
    assert!(
        loaded_trainer.is_ok(),
        "Candidate AdamW checkpoint must load successfully"
    );

    // 4. 世代番号が永続化されていることを確認
    assert!(
        fs::metadata(&state_path).is_ok(),
        "State file must be written"
    );
    let saved_gen = fs::read_to_string(&state_path).unwrap();
    assert_eq!(saved_gen.trim(), "1");

    clean_files();
}

#[test]
fn test_halfkp_loop_memory_release_and_checkpoint_continuation() {
    let temp_dir = std::env::temp_dir();
    let prefix = format!(
        "hkp_cont_test_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let data_path = temp_dir
        .join(format!("{prefix}_data.tsv"))
        .to_str()
        .unwrap()
        .to_string();
    let deep_data_path = temp_dir
        .join(format!("{prefix}_deep.tsv"))
        .to_str()
        .unwrap()
        .to_string();
    let best_path = temp_dir
        .join(format!("{prefix}_best.bin"))
        .to_str()
        .unwrap()
        .to_string();
    let cand_path = temp_dir
        .join(format!("{prefix}_cand.bin"))
        .to_str()
        .unwrap()
        .to_string();
    let cand_ckpt_path = temp_dir
        .join(format!("{prefix}_cand_ckpt.bin"))
        .to_str()
        .unwrap()
        .to_string();
    let state_path = temp_dir
        .join(format!("{prefix}_state.txt"))
        .to_str()
        .unwrap()
        .to_string();

    let clean_files = || {
        let _ = fs::remove_file(&data_path);
        let _ = fs::remove_file(&deep_data_path);
        let _ = fs::remove_file(&best_path);
        let _ = fs::remove_file(&cand_path);
        let _ = fs::remove_file(&cand_ckpt_path);
        let _ = fs::remove_file(format!("{cand_ckpt_path}.bak"));
        let _ = fs::remove_file(&state_path);
    };
    clean_files();

    // 事前にダミーの AdamW チェックポイントを作成 (特定の特徴量の step_feat を 5 に設定)
    {
        let mut trainer = HalfKPTrainer::new();
        trainer.step_feat[42] = 5;
        trainer.step_feat[100] = 12;
        trainer.output_bias = 7.5;
        trainer
            .save_checkpoint(&cand_ckpt_path)
            .expect("Failed to create seed checkpoint");
    }

    let config = LoopConfig {
        iterations: 1,
        start_iteration: Some(2),
        state_path: state_path.clone(),
        games_per_iteration: 4,
        eval_pairs: 2,
        threads: 2,
        depth: 1,
        epochs: 1,
        lr: 0.001,
        batch_size: 16,
        data_path: data_path.clone(),
        deep_data_path: deep_data_path.clone(),
        best_model_path: best_path.clone(),
        candidate_model_path: cand_path.clone(),
        candidate_ckpt_path: cand_ckpt_path.clone(),
        min_promotion_games: 20,
    };

    SelfImprovementLoop::run(&config);

    // チェックポイントから復元されて学習が実行され、チェックポイントがさらに更新されたことを検証
    let continued_trainer = HalfKPTrainer::load_checkpoint(&cand_ckpt_path)
        .expect("Must be able to load continued checkpoint");
    assert!(
        continued_trainer.step_feat[42] >= 5,
        "Step count for feat 42 should be >= 5, got {}",
        continued_trainer.step_feat[42]
    );

    // 世代番号が 2 に更新されていることを確認
    let saved_gen = fs::read_to_string(&state_path).unwrap();
    assert_eq!(saved_gen.trim(), "2");

    clean_files();
}

#[test]
fn test_halfkp_loop_promotion_gate_blocks_small_samples() {
    let temp_dir = std::env::temp_dir();
    let prefix = format!(
        "hkp_gate_test_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let best_path = temp_dir
        .join(format!("{prefix}_best.bin"))
        .to_str()
        .unwrap()
        .to_string();

    let clean_files = || {
        let _ = fs::remove_file(&best_path);
    };
    clean_files();

    let candidate_eval = HalfKPEvaluator::new();
    let mut current_best = EvalMode::Hce;

    let mut sprt = tabula_shogi::arena::Sprt::new(tabula_shogi::arena::SprtConfig::default());
    sprt.record_batch(2, 0, 0);

    // 2連勝で対局数が 2局 (< 20局) の状況をシミュレート
    let match_res = MatchResult {
        total_games: 2,
        wins_a: 2,
        wins_b: 0,
        draws: 0,
        win_rate_a: 1.0,
        elo_diff_a: 400.0,
        sprt: Some(sprt),
    };

    // min_promotion_games = 20 の場合: 2局では昇格がゲートによってブロックされること
    let promoted = tabula_shogi::arena::SelfImprovementLoop::handle_promotion(
        1,
        &match_res,
        &candidate_eval,
        &mut current_best,
        &best_path,
        20,
    );

    assert!(
        !promoted,
        "Promotion must be rejected when total_games < min_promotion_games (2 < 20)"
    );
    assert!(
        matches!(current_best, EvalMode::Hce),
        "Current best must remain HCE"
    );
    assert!(
        !std::path::Path::new(&best_path).exists(),
        "Best model file must not be created"
    );

    clean_files();
}

#[test]
fn test_halfkp_loop_promotion_gate_permits_when_threshold_met() {
    let temp_dir = std::env::temp_dir();
    let prefix = format!(
        "hkp_gate_pass_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let best_path = temp_dir
        .join(format!("{prefix}_best.bin"))
        .to_str()
        .unwrap()
        .to_string();

    let clean_files = || {
        let _ = fs::remove_file(&best_path);
    };
    clean_files();

    let candidate_eval = HalfKPEvaluator::new();
    let mut current_best = EvalMode::Hce;

    let mut sprt = tabula_shogi::arena::Sprt::new(tabula_shogi::arena::SprtConfig::default());
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

    // min_promotion_games = 20 の場合: 30局 >= 20局 かつ SPRT Pass なので昇格が承認されること
    let promoted = tabula_shogi::arena::SelfImprovementLoop::handle_promotion(
        1,
        &match_res,
        &candidate_eval,
        &mut current_best,
        &best_path,
        20,
    );

    assert!(
        promoted,
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

    // 保存されたモデルがロード可能であることを検証
    let loaded = HalfKPEvaluator::load_from_file(&best_path);
    assert!(loaded.is_ok(), "Promoted model must be valid HalfKP binary");

    clean_files();
}
