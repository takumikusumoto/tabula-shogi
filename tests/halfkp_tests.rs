use tabula_shogi::board::Position;
use tabula_shogi::eval::halfkp::{HALFKP_HIDDEN_SIZE, HALFKP_INPUT_SIZE, HalfKPEvaluator};
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
        assert_eq!(*b, 32, "Initial hidden biases must be 32");
    }
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
