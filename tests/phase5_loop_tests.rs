use std::fs;
use std::sync::Arc;
use tabula_shogi::arena::{
    LoopConfig, MatchConfig, MatchRunner, SelfImprovementLoop, Sprt, SprtConfig, SprtStatus,
};
use tabula_shogi::eval::{EvalMode, NNUEEvaluator};

#[test]
fn test_sprt_llr_calculation() {
    let config = SprtConfig {
        elo0: 0.0,
        elo1: 10.0,
        alpha: 0.05,
        beta: 0.05,
    };

    // 1. 圧倒的に勝ち越した場合 (Pass 判定)
    let mut sprt_win = Sprt::new(config.clone());
    // 30勝0敗
    sprt_win.record_batch(30, 0, 0);
    assert_eq!(sprt_win.status, SprtStatus::Pass);
    assert!(sprt_win.llr >= sprt_win.upper_bound);
    assert!(sprt_win.win_rate() > 0.9);
    assert!(sprt_win.elo_diff() > 100.0);

    // 2. 圧倒的に負け越した場合 (Fail 判定)
    let mut sprt_loss = Sprt::new(config.clone());
    // 0勝30敗
    sprt_loss.record_batch(0, 30, 0);
    assert_eq!(sprt_loss.status, SprtStatus::Fail);
    assert!(sprt_loss.llr <= sprt_loss.lower_bound);
    assert!(sprt_loss.win_rate() < 0.1);
    assert!(sprt_loss.elo_diff() < -100.0);

    // 3. 互角の対局の場合 (Continue 判定)
    let mut sprt_even = Sprt::new(config);
    sprt_even.record_batch(5, 5, 2);
    assert_eq!(sprt_even.status, SprtStatus::Continue);
    assert!(sprt_even.llr > sprt_even.lower_bound && sprt_even.llr < sprt_even.upper_bound);
    assert!((sprt_even.win_rate() - 0.5).abs() < 0.1);
}

#[test]
fn test_arena_opening_generation() {
    let pos = MatchRunner::generate_opening_position(6, 42);
    assert_eq!(pos.board.len(), 81);
    assert!(!pos.is_in_check(pos.side_to_move));
}

#[test]
fn test_arena_pair_match() {
    let nnue = NNUEEvaluator::new();
    let config = MatchConfig {
        name_a: "HCE".to_string(),
        name_b: "NNUE".to_string(),
        eval_a: EvalMode::Hce,
        eval_b: EvalMode::Nnue(Arc::new(nnue)),
        pairs: 2, // 2ペア = 4局
        depth: 1,
        threads: 2,
        random_opening: 4,
        max_plies: 40,
        tt_size_mb: 8,
        sprt_config: Some(SprtConfig::default()),
    };

    let result = MatchRunner::run_match(&config);

    assert_eq!(result.total_games, 4);
    assert_eq!(result.wins_a + result.wins_b + result.draws, 4);
    assert!(result.win_rate_a >= 0.0 && result.win_rate_a <= 1.0);
    assert!(result.sprt.is_some());
}

#[test]
fn test_autonomous_loop_single_iteration() {
    let temp_dir = std::env::temp_dir();
    let data_path = temp_dir
        .join("test_loop_data.tsv")
        .to_str()
        .unwrap()
        .to_string();
    let best_path = temp_dir
        .join("test_loop_best.bin")
        .to_str()
        .unwrap()
        .to_string();
    let cand_path = temp_dir
        .join("test_loop_cand.bin")
        .to_str()
        .unwrap()
        .to_string();
    let cand_ckpt_path = temp_dir
        .join("test_loop_cand_ckpt.bin")
        .to_str()
        .unwrap()
        .to_string();
    let summary_path = temp_dir
        .join("test_loop_summary.csv")
        .to_str()
        .unwrap()
        .to_string();

    let state_path = temp_dir
        .join("test_loop_state.txt")
        .to_str()
        .unwrap()
        .to_string();
    let deep_data_path = temp_dir
        .join("test_deep_dataset.tsv")
        .to_str()
        .unwrap()
        .to_string();

    let _ = fs::remove_file(&data_path);
    let _ = fs::remove_file(&deep_data_path);
    let _ = fs::remove_file(&best_path);
    let _ = fs::remove_file(&cand_path);
    let _ = fs::remove_file(&cand_ckpt_path);
    let _ = fs::remove_file(&state_path);
    let _ = fs::remove_file(&summary_path);

    let config = LoopConfig {
        iterations: 1,
        start_iteration: Some(1),
        state_path: state_path.clone(),
        games_per_iteration: 4,
        eval_pairs: 2,
        threads: 2,
        depth: 1,
        epochs: 2,
        lr: 0.01,
        batch_size: 16,
        data_path: data_path.clone(),
        deep_data_path: deep_data_path.clone(),
        best_model_path: best_path.clone(),
        candidate_model_path: cand_path.clone(),
        candidate_ckpt_path: cand_ckpt_path.clone(),
        min_promotion_games: 20,
        summary_path: summary_path.clone(),
    };

    SelfImprovementLoop::run(&config);

    // データセットおよび候補モデルが正常生成されたことを確認
    assert!(
        fs::metadata(&data_path).is_ok(),
        "Dataset should be created"
    );
    assert!(
        fs::metadata(&cand_path).is_ok(),
        "Candidate model should be saved"
    );
    assert!(
        fs::metadata(&cand_ckpt_path).is_ok(),
        "Candidate checkpoint should be saved"
    );

    let _ = fs::remove_file(&data_path);
    let _ = fs::remove_file(&deep_data_path);
    let _ = fs::remove_file(&best_path);
    let _ = fs::remove_file(&cand_path);
    let _ = fs::remove_file(&cand_ckpt_path);
    let _ = fs::remove_file(&state_path);
    let _ = fs::remove_file(&summary_path);
}
