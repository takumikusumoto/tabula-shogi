mod common;
use common::TestTempDir;
use std::fs;
use std::sync::Arc;
use tabula_shogi::arena::{
    LoopArenaParams, LoopConfig, LoopStoragePaths, LoopTrainingParams, MatchConfig, MatchRunner,
    SelfImprovementLoop, Sprt, SprtConfig, SprtStatus,
};
use tabula_shogi::eval::{EvalMode, NNUEEvaluator};

#[test]
fn test_sprt_llr_calculation() {
    let config = SprtConfig {
        elo0: 0.0,
        elo1: 10.0,
        alpha: 0.05,
        beta: 0.05,
        min_games: 0,
    };

    // 1. 圧倒的に勝ち越した場合 (Pass 判定)
    let mut sprt_win = Sprt::new(config.clone());
    // 30勝0敗
    sprt_win.record_batch(30, 0, 0);
    assert_eq!(sprt_win.status, SprtStatus::Pass);
    assert!(sprt_win.is_decided());
    assert!(sprt_win.llr >= sprt_win.upper_bound);
    assert!(sprt_win.win_rate() > 0.9);
    assert!(sprt_win.elo_diff() > 100.0);
    // ラッチ検証: Pass確定後に対局を追加してもPass状態が覆らないこと
    sprt_win.record_batch(0, 10, 0);
    assert_eq!(sprt_win.status, SprtStatus::Pass);

    // 2. 圧倒的に負け越した場合 (Fail 判定)
    let mut sprt_loss = Sprt::new(config.clone());
    // 0勝30敗
    sprt_loss.record_batch(0, 30, 0);
    assert_eq!(sprt_loss.status, SprtStatus::Fail);
    assert!(sprt_loss.is_decided());
    assert!(sprt_loss.llr <= sprt_loss.lower_bound);
    assert!(sprt_loss.win_rate() < 0.1);
    assert!(sprt_loss.elo_diff() < -100.0);
    // ラッチ検証: Fail確定後に対局を追加してもFail状態が覆らないこと
    sprt_loss.record_batch(10, 0, 0);
    assert_eq!(sprt_loss.status, SprtStatus::Fail);

    // 3. 互角の対局の場合 (Continue 判定)
    let mut sprt_even = Sprt::new(config);
    sprt_even.record_batch(5, 5, 2);
    assert_eq!(sprt_even.status, SprtStatus::Continue);
    assert!(!sprt_even.is_decided());
    assert!(sprt_even.llr > sprt_even.lower_bound && sprt_even.llr < sprt_even.upper_bound);
    assert!((sprt_even.win_rate() - 0.5).abs() < 0.1);
}

#[test]
fn test_arena_opening_generation() {
    let seed = MatchRunner::opening_seed(42, 3);
    assert_eq!(seed, MatchRunner::opening_seed(42, 3));
    assert_ne!(seed, MatchRunner::opening_seed(43, 3));
    assert_ne!(seed, MatchRunner::opening_seed(42, 4));

    let pos = MatchRunner::generate_opening_position(6, seed);
    let rerun = MatchRunner::generate_opening_position(6, MatchRunner::opening_seed(42, 3));
    assert_eq!(pos.board.len(), 81);
    assert!(!pos.is_in_check(pos.side_to_move));
    assert_eq!(pos.to_sfen(), rerun.to_sfen());

    let next_generation =
        MatchRunner::generate_opening_position(6, MatchRunner::opening_seed(43, 3));
    assert_ne!(
        pos.to_sfen(),
        next_generation.to_sfen(),
        "the same pair index must receive a different opening in a different generation"
    );
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
        generation: 7,
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
    let ws = TestTempDir::new("phase5_loop_test");
    let data_path = ws.file_path("test_loop_data.tsv");
    let best_path = ws.file_path("test_loop_best.bin");
    let cand_path = ws.file_path("test_loop_cand.bin");
    let cand_ckpt_path = ws.file_path("test_loop_cand_ckpt.bin");
    let summary_path = ws.file_path("test_loop_summary.csv");
    let state_path = ws.file_path("test_loop_state.txt");
    let deep_data_path = ws.file_path("test_deep_dataset.tsv");

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
            lr: 0.01,
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
}
