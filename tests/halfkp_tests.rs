use tabula_shogi::board::Position;
use tabula_shogi::eval::halfkp::{
    HALFKP_ACTIVATION_MAX, HALFKP_BIAS_SCALE, HALFKP_FEATURE_SCALE, HALFKP_HIDDEN_SIZE,
    HALFKP_INPUT_SIZE, HALFKP_MAGIC, HALFKP_OUTPUT_DIVISOR, HALFKP_OUTPUT_SCALE, HalfKPEvaluator,
};
use tabula_shogi::movegen::MoveGenerator;
use tabula_shogi::selfplay::game::SimpleRng;
use tabula_shogi::types::Move;

#[test]
fn test_halfkp_initialization_and_dimensions() {
    let eval = HalfKPEvaluator::new();
    assert_eq!(eval.feature_weights.len(), HALFKP_INPUT_SIZE);
    assert_eq!(eval.feature_biases.len(), HALFKP_HIDDEN_SIZE);
    assert_eq!(eval.output_weights.len(), HALFKP_HIDDEN_SIZE * 2);

    for b in &eval.feature_biases {
        assert_eq!(
            *b,
            (32 * HALFKP_FEATURE_SCALE) as i16,
            "Initial hidden biases must encode float 32.0 in Q6"
        );
    }
}

#[test]
fn test_halfkp_fixed_point_units_and_overflow_bounds() {
    assert_eq!(
        HALFKP_BIAS_SCALE,
        HALFKP_FEATURE_SCALE * HALFKP_OUTPUT_SCALE
    );
    assert_eq!(HALFKP_ACTIVATION_MAX, 64 * HALFKP_FEATURE_SCALE);
    assert_eq!(HALFKP_OUTPUT_DIVISOR, 128 * HALFKP_BIAS_SCALE as i64);

    // Even a structurally malformed position that fills all 81 squares and all 14 hand
    // categories to the extraction cap (18) cannot overflow the i32 accumulator.
    let maximum_extracted_features = 81i64 + 14 * 18;
    let maximum_accumulator = (maximum_extracted_features + 1) * i16::MAX as i64;
    assert!(maximum_accumulator < i32::MAX as i64);

    // The worst possible clipped 256-way dot product plus i32 bias fits comfortably in i64.
    let maximum_output =
        HALFKP_ACTIVATION_MAX as i64 * i16::MAX as i64 * (HALFKP_HIDDEN_SIZE * 2) as i64
            + i32::MAX as i64;
    assert!(maximum_output < i64::MAX);
}

#[test]
fn test_halfkp_accumulator_bit_exactness_on_legal_moves() {
    let eval = HalfKPEvaluator::new();
    let mut pos = Position::startpos();
    let mut diff_acc = eval.compute_accumulators_full(&pos);

    // 平手初期局面から 120 手のランダム合法手進行を行い、
    // 全局面において「差分更新結果」と「ゼロからの全再計算結果」が両視点で整数完全一致することを検証
    let mut rng = SimpleRng::new(0x123456789abcdef0);

    for ply in 0..120 {
        let legal_moves = MoveGenerator::generate_legal_moves(&mut pos);
        if legal_moves.is_empty() || pos.repetition_count() >= 4 {
            break;
        }

        let mv = legal_moves[rng.gen_range(legal_moves.len())];
        let pos_before = pos.clone();

        pos.do_move(mv);
        eval.update_accumulator_move(&mut diff_acc, &pos_before, mv, &pos);

        // ゼロからの全再計算アキュムレータ
        let full_acc = eval.compute_accumulators_full(&pos);

        // 先手視点の全 128 要素完全一致検証
        for i in 0..HALFKP_HIDDEN_SIZE {
            assert_eq!(
                diff_acc.accumulation[0][i],
                full_acc.accumulation[0][i],
                "Black accumulator mismatch at ply {ply}, move {}, hidden unit {i}: diff={}, full={}",
                mv.to_usi(),
                diff_acc.accumulation[0][i],
                full_acc.accumulation[0][i]
            );
        }

        // 後手視点の全 128 要素完全一致検証
        for i in 0..HALFKP_HIDDEN_SIZE {
            assert_eq!(
                diff_acc.accumulation[1][i],
                full_acc.accumulation[1][i],
                "White accumulator mismatch at ply {ply}, move {}, hidden unit {i}: diff={}, full={}",
                mv.to_usi(),
                diff_acc.accumulation[1][i],
                full_acc.accumulation[1][i]
            );
        }

        // 評価値の一致検証
        let score_diff = eval.evaluate_with_accumulator(&pos, &diff_acc);
        let score_full = eval.evaluate(&pos);
        assert_eq!(
            score_diff, score_full,
            "Evaluation score mismatch at ply {ply}: diff={score_diff}, full={score_full}"
        );
    }
}

#[test]
fn test_halfkp_accumulator_special_moves_exactness() {
    let eval = HalfKPEvaluator::new();

    // 1. 駒の捕獲・成り局面のテスト (角換わり進行)
    let moves_str = [
        "7g7f", "3c3d", "2g2f", "8c8d", "2f2e", "8d8e", "6i7h", "4a3b",
        "8h2b+", // 角成り & 捕獲
        "3a2b",  // 金で馬を取り返す (成駒捕獲)
        "B*4e",  // 持ち駒の角を打つ (Drop)
    ];

    let mut pos = Position::startpos();
    let mut diff_acc = eval.compute_accumulators_full(&pos);

    for mv_usi in moves_str {
        let mv = Move::from_usi(mv_usi).expect("Valid USI move");
        let pos_before = pos.clone();
        pos.do_move(mv);
        eval.update_accumulator_move(&mut diff_acc, &pos_before, mv, &pos);

        let full_acc = eval.compute_accumulators_full(&pos);
        assert_eq!(
            diff_acc.accumulation, full_acc.accumulation,
            "Accumulator mismatch after special move {mv_usi}"
        );
    }
}

#[test]
fn test_halfkp_roundtrip_serialization() {
    let eval = HalfKPEvaluator::new();
    let test_path = "test_halfkp_temp.bin";

    eval.save_to_file(test_path)
        .expect("Failed to save HalfKP binary");

    let header = std::fs::read(test_path).expect("read saved model header");
    assert_eq!(&header[..8], HALFKP_MAGIC);

    let loaded = HalfKPEvaluator::load_from_file(test_path).expect("Failed to load HalfKP binary");

    assert_eq!(loaded.feature_biases, eval.feature_biases);
    assert_eq!(loaded.output_weights, eval.output_weights);
    assert_eq!(loaded.output_bias, eval.output_bias);

    // 重みの一部サンプリング一致検証
    assert_eq!(loaded.feature_weights[0], eval.feature_weights[0]);
    assert_eq!(
        loaded.feature_weights[100_000],
        eval.feature_weights[100_000]
    );

    let _ = std::fs::remove_file(test_path);
}

#[test]
fn test_halfkp_legacy_format_is_rejected_explicitly() {
    let legacy_path = "target/test_legacy_halfkp.bin";
    std::fs::write(legacy_path, b"TABU_HKP").expect("write legacy header fixture");

    let error = HalfKPEvaluator::load_from_file(legacy_path)
        .expect_err("legacy unscaled weights must never be interpreted as Q6/Q9 weights");
    let _ = std::fs::remove_file(legacy_path);

    assert!(error.contains("Legacy HalfKP format TABU_HKP"));
    assert!(error.contains("export-halfkp"));
}

#[test]
fn test_halfkp_accumulator_after_move_bit_exactness() {
    let eval = HalfKPEvaluator::new();
    let mut pos = Position::startpos();
    let mut diff_acc = eval.compute_accumulators_full(&pos);
    let mut rng = SimpleRng::new(0xDEADBEEFCAFE);

    for ply in 0..120 {
        let legal_moves = MoveGenerator::generate_legal_moves(&mut pos);
        if legal_moves.is_empty() {
            break;
        }

        let mv_idx = (rng.next_u64() as usize) % legal_moves.len();
        let mv = legal_moves[mv_idx];

        pos.do_move(mv);
        eval.update_accumulator_after_move(&mut diff_acc, &pos, mv);

        let full_acc = eval.compute_accumulators_full(&pos);

        // 先手視点・後手視点の完全一致検証
        assert_eq!(
            diff_acc.accumulation[0],
            full_acc.accumulation[0],
            "Black after_move accumulator mismatch at ply {ply}, move {}",
            mv.to_usi()
        );
        assert_eq!(
            diff_acc.accumulation[1],
            full_acc.accumulation[1],
            "White after_move accumulator mismatch at ply {ply}, move {}",
            mv.to_usi()
        );
    }
}

#[test]
fn test_halfkp_search_integration() {
    use std::sync::Arc;
    use tabula_shogi::eval::EvalMode;
    use tabula_shogi::search::SearchEngine;

    let eval = Arc::new(HalfKPEvaluator::new());
    let mut engine = SearchEngine::new(16).with_eval_mode(EvalMode::HalfKP(eval));
    engine.use_book = false;
    let mut pos = Position::startpos();

    // 深さ 3 の探索実行 (アキュムレータ差分更新の探索ツリー全走査)
    let (best_move, score) = engine.search_fixed_depth(&mut pos, 3);

    assert!(best_move.is_some(), "Search must return a best move");
    assert!(engine.nodes() > 10, "Search must visit nodes in tree");
    println!(
        "HalfKP Search returned best move: {:?}, score: {}, nodes: {}",
        best_move,
        score,
        engine.nodes()
    );
}

#[test]
fn test_halfkp_save_load_roundtrip_and_boundary_checks() {
    let mut eval = HalfKPEvaluator::new();
    // 非ゼロ重みの設定
    for row in eval.feature_weights.iter_mut().take(100) {
        row[0] = 42;
        row[127] = -99;
    }
    eval.feature_biases[0] = 15;
    eval.feature_biases[127] = -30;
    eval.output_weights[0] = 120;
    eval.output_weights[255] = -240;
    eval.output_bias = 777;

    let path = "target/test_halfkp_nonzero.bin";
    eval.save_to_file(path)
        .expect("Failed to save HalfKP model");

    let loaded = HalfKPEvaluator::load_from_file(path).expect("Failed to load HalfKP model");
    let _ = std::fs::remove_file(path);

    assert_eq!(eval.output_bias, loaded.output_bias);
    assert_eq!(eval.output_weights, loaded.output_weights);
    assert_eq!(eval.feature_biases, loaded.feature_biases);
    assert_eq!(eval.feature_weights[0], loaded.feature_weights[0]);
    assert_eq!(eval.feature_weights[99], loaded.feature_weights[99]);

    let pos = Position::startpos();
    assert_eq!(eval.evaluate(&pos), loaded.evaluate(&pos));

    // 境界検証テスト: 途中で切れたファイル (100バイト)
    let corrupted_path = "target/corrupted_halfkp.bin";
    std::fs::write(corrupted_path, vec![0u8; 100]).unwrap();
    let res = HalfKPEvaluator::load_from_file(corrupted_path);
    let _ = std::fs::remove_file(corrupted_path);
    assert!(
        res.is_err(),
        "Truncated model file must return Err, not panic"
    );
}

#[test]
fn test_halfkp_search_accumulator_sync_with_full_recomputation() {
    use std::sync::Arc;
    use tabula_shogi::eval::EvalMode;
    use tabula_shogi::search::SearchEngine;

    let mut eval_inst = HalfKPEvaluator::new();
    // 非ゼロ重みを設定して評価値の差異を鋭敏に検出
    for row in eval_inst.feature_weights.iter_mut().take(500) {
        row[0] = 25;
        row[63] = -18;
    }
    eval_inst.output_weights[0] = 40;
    eval_inst.output_weights[128] = -35;
    eval_inst.output_bias = 100;

    let eval = Arc::new(eval_inst);
    let mut engine = SearchEngine::new(16).with_eval_mode(EvalMode::HalfKP(Arc::clone(&eval)));
    engine.use_book = false;

    let mut pos = Position::startpos();
    let mut rng = SimpleRng::new(0xABCDEF1234567890);

    // 60 手対局を進行しながら、各手番でエンジンのルートアキュムレータによる評価値と全再計算評価値の一致を検証
    for ply in 1..=60 {
        engine.init_root_accumulator(&pos);
        let root_acc = engine.halfkp_accumulators[0];
        let diff_eval = eval.evaluate_with_accumulator(&pos, &root_acc);
        let full_eval = eval.evaluate(&pos);

        assert_eq!(
            diff_eval, full_eval,
            "Search accumulator root evaluation mismatch at ply {ply}: diff={diff_eval} vs full={full_eval}"
        );

        let moves = MoveGenerator::generate_legal_moves(&mut pos);
        if moves.is_empty() {
            break;
        }
        let mv = moves[rng.gen_range(moves.len())];
        pos.do_move(mv);
    }
}

#[test]
fn test_halfkp_save_auto_creates_parent_directories() {
    let eval = HalfKPEvaluator::new();
    let deep_path = "target/test_deep_dir_auto_create/sub/models/test_model.bin";
    let _ = std::fs::remove_dir_all("target/test_deep_dir_auto_create");

    // 親ディレクトリが存在しない状態から save_to_file が成功すること
    eval.save_to_file(deep_path)
        .expect("save_to_file must automatically create parent directories");
    assert!(
        std::path::Path::new(deep_path).exists(),
        "Saved model file must exist"
    );

    let loaded = HalfKPEvaluator::load_from_file(deep_path)
        .expect("Must be able to load model from created path");
    assert_eq!(eval.output_bias, loaded.output_bias);

    let _ = std::fs::remove_dir_all("target/test_deep_dir_auto_create");
}
