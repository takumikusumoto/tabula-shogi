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

    let initial_loss = trainer.train_batch(&batch, 0.01, 600.0).mse_loss;
    let mut last_loss = initial_loss;

    for _ in 0..10 {
        let loss = trainer.train_batch(&batch, 0.01, 600.0).mse_loss;
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

#[test]
fn test_halfkp_trainer_nonzero_forward_and_quantization_consistency() {
    let mut trainer = HalfKPTrainer::new();
    let pos = Position::startpos();

    let mover_feats = HalfKPEvaluator::extract_halfkp_features(&pos, Color::Black);
    let opp_feats = HalfKPEvaluator::extract_halfkp_features(&pos, Color::White);

    // 非ゼロの重みを意図的に設定 (ゼロ重み盲点の根絶)
    for (i, w) in trainer.output_weights.iter_mut().enumerate() {
        *w = ((i as f32 % 17.0) - 8.0) * 10.0; // -80.0 ~ +80.0
    }
    trainer.output_bias = 50.0;
    for &f in &mover_feats {
        for (i, w) in trainer.feature_weights[f].iter_mut().enumerate() {
            *w = ((i as f32 % 7.0) - 3.0) * 2.0;
        }
    }

    let (score_cp, _raw_out, _m_acc, _o_acc, _m_h, _o_h) =
        trainer.forward(&mover_feats, &opp_feats);

    // 非ゼロ重み状態で Evaluator へエクスポート
    let eval = trainer.to_evaluator();
    let eval_score = eval.evaluate(&pos);

    // 非ゼロの有意な値が出ていることを確認
    assert_ne!(
        eval_score, 0,
        "Nonzero weights must yield non-zero evaluation score"
    );

    // 量子化 (round + integer division) による差異が高々 3cp 以内であることを検証
    let diff = (score_cp - eval_score as f32).abs();
    assert!(
        diff <= 3.0,
        "Trainer forward score {score_cp} deviates too much from Evaluator {eval_score} with nonzero weights (diff={diff})"
    );
}

#[test]
fn test_halfkp_trainer_sparse_adamw_local_step_stability() {
    let mut trainer = HalfKPTrainer::new();
    let pos = Position::startpos();

    let mover_feats = HalfKPEvaluator::extract_halfkp_features(&pos, Color::Black);
    let opp_feats = HalfKPEvaluator::extract_halfkp_features(&pos, Color::White);

    // 100 ステップ別の特徴量をダミー更新してグローバル時刻が進んだ状態を模擬
    let dummy_mover = vec![0usize];
    let dummy_opp = vec![1usize];
    let dummy_batch = vec![(dummy_mover, dummy_opp, 0.5f32)];
    for _ in 0..100 {
        trainer.train_batch(&dummy_batch, 0.001, 600.0);
    }

    // 初めて登場する特徴量（mover_feats）に対する更新量を検証
    let test_feat = mover_feats[0];
    let weight_before = trainer.feature_weights[test_feat][0];

    let lr = 0.01f32;
    let batch = vec![(mover_feats.clone(), opp_feats.clone(), 1.0f32)];
    trainer.train_batch(&batch, lr, 600.0);

    let weight_after = trainer.feature_weights[test_feat][0];
    let delta = (weight_after - weight_before).abs();

    // グローバル時刻が進んでいても、初登場特徴量の更新量は約 1.0 * lr 付近に収まり、
    // 3.16 * lr への跳ね上がりが防止されていることを検証
    assert!(
        delta <= lr * 1.5,
        "Delta {delta} exceeds stable bound (max 1.5*lr = {}), proving local AdamW step stability",
        lr * 1.5
    );
}

#[test]
fn test_halfkp_trainer_checkpoint_roundtrip() {
    let mut trainer = HalfKPTrainer::new();
    let pos = Position::startpos();
    let mover_feats = HalfKPEvaluator::extract_halfkp_features(&pos, Color::Black);
    let opp_feats = HalfKPEvaluator::extract_halfkp_features(&pos, Color::White);
    let batch = vec![(mover_feats, opp_feats, 1.0f32)];

    // 1ステップ学習して状態を変更
    trainer.train_batch(&batch, 0.01, 600.0);

    let ckpt_path = "target/test_trainer_ckpt.bin";
    trainer
        .save_checkpoint(ckpt_path)
        .expect("Failed to save checkpoint");

    let loaded = HalfKPTrainer::load_checkpoint(ckpt_path).expect("Failed to load checkpoint");
    let _ = std::fs::remove_file(ckpt_path);

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

#[test]
fn test_halfkp_trainer_checkpoint_size_validation() {
    let corrupted_path = "target/test_corrupted_ckpt.bin";
    std::fs::write(corrupted_path, vec![0u8; 100]).expect("Failed to write dummy corrupted file");

    let result = HalfKPTrainer::load_checkpoint(corrupted_path);
    let _ = std::fs::remove_file(corrupted_path);

    match result {
        Err(err_msg) => {
            assert!(
                err_msg.contains("file size mismatch"),
                "Error message should mention size mismatch: {err_msg}"
            );
        }
        Ok(_) => panic!("Corrupted or truncated checkpoint file must return Err, not Ok"),
    }
}

#[test]
fn test_halfkp_trainer_checkpoint_auto_creates_parent_directories() {
    let trainer = HalfKPTrainer::new();
    let deep_path = "target/test_deep_dir_ckpt_auto/sub/models/test_ckpt.bin";
    let _ = std::fs::remove_dir_all("target/test_deep_dir_ckpt_auto");

    trainer
        .save_checkpoint(deep_path)
        .expect("save_checkpoint must automatically create parent directories");
    assert!(
        std::path::Path::new(deep_path).exists(),
        "Saved checkpoint file must exist"
    );

    let loaded = HalfKPTrainer::load_checkpoint(deep_path)
        .expect("Must be able to load checkpoint from created path");
    assert_eq!(trainer.beta1_pow, loaded.beta1_pow);

    let _ = std::fs::remove_dir_all("target/test_deep_dir_ckpt_auto");
}
