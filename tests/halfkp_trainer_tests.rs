use std::sync::Arc;
use tabula_shogi::board::Position;
use tabula_shogi::eval::halfkp::{HALFKP_HIDDEN_SIZE, HALFKP_INPUT_SIZE, HalfKPEvaluator};
use tabula_shogi::eval::halfkp_trainer::HalfKPTrainer;
use tabula_shogi::types::Color;

#[test]
fn test_halfkp_trainer_initialization_and_roundtrip() {
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
fn test_halfkp_trainer_forward_consistency() {
    let eval = Arc::new(HalfKPEvaluator::new());
    let trainer = HalfKPTrainer::from_evaluator(&eval);
    let pos = Position::startpos();

    let mover_feats = HalfKPEvaluator::extract_halfkp_features(&pos, Color::Black);
    let opp_feats = HalfKPEvaluator::extract_halfkp_features(&pos, Color::White);

    let (score_cp, _raw_out, _m_acc, _o_acc, _m_h, _o_h) =
        trainer.forward(&mover_feats, &opp_feats);
    let eval_score = eval.evaluate(&pos);

    // Trainer の浮動小数点 cp と Evaluator の整数 cp の差異が 2cp 以内であることを検証
    let diff = (score_cp - eval_score as f32).abs();
    assert!(
        diff <= 2.0,
        "Trainer forward score {score_cp} deviates too much from Evaluator {eval_score}"
    );
}

#[test]
fn test_halfkp_trainer_loss_convergence() {
    let mut trainer = HalfKPTrainer::new();
    let pos = Position::startpos();

    let mover_feats = HalfKPEvaluator::extract_halfkp_features(&pos, Color::Black);
    let opp_feats = HalfKPEvaluator::extract_halfkp_features(&pos, Color::White);

    // 勝利ターゲット (1.0) に対して複数エポック学習
    let batch = vec![(mover_feats.clone(), opp_feats.clone(), 1.0f32)];

    let initial_loss = trainer.train_batch(&batch, 0.01, 600.0);
    let mut last_loss = initial_loss;

    for _ in 0..10 {
        let loss = trainer.train_batch(&batch, 0.01, 600.0);
        assert!(
            loss <= last_loss + 1e-4,
            "Loss should decrease or stay flat: current {loss} vs last {last_loss}"
        );
        last_loss = loss;
    }

    assert!(
        last_loss < initial_loss,
        "Final loss {last_loss} must be strictly lower than initial loss {initial_loss}"
    );

    // 学習後の予測値が初期値よりも勝利側 (1.0) に近づいていること
    let (trained_cp, _, _, _, _, _) = trainer.forward(&mover_feats, &opp_feats);
    let trained_pred = HalfKPTrainer::sigmoid(trained_cp, 600.0);
    assert!(
        trained_pred > 0.5,
        "Trained prediction {trained_pred} should be biased towards win"
    );
}

#[test]
fn test_halfkp_trainer_sparse_gradient_exactness() {
    let mut trainer = HalfKPTrainer::new();
    let pos = Position::startpos();

    let mover_feats = HalfKPEvaluator::extract_halfkp_features(&pos, Color::Black);
    let opp_feats = HalfKPEvaluator::extract_halfkp_features(&pos, Color::White);

    let active_set: std::collections::HashSet<usize> = mover_feats
        .iter()
        .chain(opp_feats.iter())
        .copied()
        .collect();

    // 未出現の特徴量インデックス (例えば 100,000)
    let inactive_idx = 100_000usize;
    assert!(!active_set.contains(&inactive_idx));

    let inactive_weights_before = trainer.feature_weights[inactive_idx];

    let batch = vec![(mover_feats, opp_feats, 1.0f32)];
    trainer.train_batch(&batch, 0.01, 600.0);

    let inactive_weights_after = trainer.feature_weights[inactive_idx];

    // 未出現特徴量の重みはビット完全一致で不変であること
    assert_eq!(
        inactive_weights_before, inactive_weights_after,
        "Inactive feature weights must remain completely untouched (sparse update guarantee)"
    );
}
