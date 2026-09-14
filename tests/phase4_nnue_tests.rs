use std::fs;
use std::sync::Arc;
use tabula_shogi::board::Position;
use tabula_shogi::eval::{EvalMode, NNUEEvaluator, NNUETrainer};
use tabula_shogi::search::SearchEngine;
use tabula_shogi::selfplay::dataset::DatasetEntry;
use tabula_shogi::usi::UsiHandler;
use tabula_shogi::usi::parse::UsiCommand;

#[test]
fn test_nnue_roundtrip_and_serialization() {
    let evaluator = NNUEEvaluator::new();
    let bytes = evaluator.to_bytes();

    // 8 (magic) + 4 (input_size) + 4 (hidden_size) + 2520*128*2 + 128*2 + 256*2 + 4 = 645,908 bytes
    assert_eq!(bytes.len(), 645_908);
    assert_eq!(&bytes[0..8], b"TABU_NN2");

    // デシリアライズ検証
    let restored = NNUEEvaluator::from_bytes(&bytes).expect("Deserialization should succeed");
    let pos = Position::startpos();
    let score_orig = evaluator.evaluate(&pos);
    let score_restored = restored.evaluate(&pos);
    assert_eq!(score_orig, score_restored);

    // 一時ファイル保存＆読み込み検証
    let temp_path = std::env::temp_dir().join("test_tabula_nnue.bin");
    let temp_str = temp_path.to_str().unwrap();

    evaluator
        .save_to_file(temp_str)
        .expect("Saving to file should succeed");
    let loaded = NNUEEvaluator::load_from_file(temp_str).expect("Loading from file should succeed");
    assert_eq!(score_orig, loaded.evaluate(&pos));

    let _ = fs::remove_file(temp_path);

    // 不正ヘッダー／サイズ不足のエラーハンドリング検証
    assert!(NNUEEvaluator::from_bytes(&bytes[..100]).is_err());
    let mut bad_magic = bytes.clone();
    bad_magic[0] = b'X';
    assert!(NNUEEvaluator::from_bytes(&bad_magic).is_err());
}

#[test]
fn test_nnue_quantization() {
    let trainer = NNUETrainer::new();
    let nnue = trainer.quantize();
    let pos = Position::startpos();
    let score_int = nnue.evaluate(&pos);

    // Float 推論側のスコア
    let b_feats = NNUEEvaluator::extract_features(&pos, tabula_shogi::types::Color::Black);
    let w_feats = NNUEEvaluator::extract_features(&pos, tabula_shogi::types::Color::White);
    let (score_float, _, _, _, _) = trainer.forward(&b_feats, &w_feats);

    // 初期重みでの初期局面評価値は有限値
    assert!(score_int.abs() < 10_000);
    // 量子化誤差を考慮しても両者の符号とオーダーが一致すること（乖離が許容範囲内）を検証
    let diff = (score_int as f32 - score_float).abs();
    assert!(
        diff < 30.0,
        "Float vs Int inference discrepancy too large: float={score_float}, int={score_int}, diff={diff}"
    );
}

#[test]
fn test_nnue_trainer_loss_convergence() {
    let mut trainer = NNUETrainer::new();

    // 合成データセットの構築:
    // 初期局面 (pred=0.5) に対して Black 勝利 (result=1.0) を学習させ、
    // 予測値が0.5から1.0へと改善し、損失が減少することを検証
    let pos0 = Position::startpos();
    let dataset = vec![DatasetEntry {
        sfen: pos0.to_sfen(),
        score: 0,
        result: 1.0,
        move_usi: "7g7f".to_string(),
    }];

    let (trained_eval, init_loss, final_loss) = trainer.train_dataset(&dataset, 20, 0.05, 1, 400.0);

    // 初期損失 (0.5 - 1.0)^2 = 0.25 から学習により減少
    assert!(
        final_loss < init_loss,
        "Loss must decrease: initial={init_loss}, final={final_loss}"
    );

    // 学習済み重みで評価が計算可能かつ先手スコアが向上
    let score = trained_eval.evaluate(&pos0);
    assert!(
        score > 0,
        "Score should have increased for Black: got {score}"
    );
}

#[test]
fn test_nnue_search_integration() {
    let mut engine = SearchEngine::new(16);
    let nnue = NNUEEvaluator::new();
    engine = engine.with_eval_mode(EvalMode::Nnue(Arc::new(nnue)));

    // 定跡外の局面を作成 (端歩の応酬など)
    let mut pos = Position::startpos();
    let mv1 = tabula_shogi::types::Move::normal(
        tabula_shogi::types::Square::new(0, 6), // 1七
        tabula_shogi::types::Square::new(0, 5), // 1六歩
        false,
    );
    pos.do_move(mv1);
    let mv2 = tabula_shogi::types::Move::normal(
        tabula_shogi::types::Square::new(8, 2), // 9三
        tabula_shogi::types::Square::new(8, 3), // 9四歩
        false,
    );
    pos.do_move(mv2);

    let (best_mv, score) = engine.search_fixed_depth(&mut pos, 2);

    assert!(best_mv.is_some(), "Search should find a move with NNUE");
    assert!(engine.nodes() > 0, "Explored nodes should be positive");
    assert!(score.abs() < 30_000, "Score should be finite");
}

#[test]
fn test_usi_eval_type_and_options() {
    let mut handler = UsiHandler::new();

    // デフォルトは HCE
    match handler.eval_mode() {
        EvalMode::Hce => {}
        _ => panic!("Default evaluation mode must be HCE"),
    }

    // USI コマンド処理: setoption name Eval_Type value NNUE
    let ok = handler.process_command(UsiCommand::SetOption {
        name: "Eval_Type".to_string(),
        value: "NNUE".to_string(),
    });
    assert!(ok);

    match handler.eval_mode() {
        EvalMode::Nnue(_) => {}
        _ => panic!("Evaluation mode must be switched to NNUE"),
    }

    // eval コマンドの正常実行
    let ok_eval = handler.process_command(UsiCommand::Eval);
    assert!(ok_eval);

    // HCE に戻す
    handler.process_command(UsiCommand::SetOption {
        name: "Eval_Type".to_string(),
        value: "HCE".to_string(),
    });
    match handler.eval_mode() {
        EvalMode::Hce => {}
        _ => panic!("Evaluation mode must be switched back to HCE"),
    }

    // NNUE_File オプションでファイルから読み込み
    let temp_path = std::env::temp_dir().join("test_usi_nnue.bin");
    let temp_str = temp_path.to_str().unwrap();
    let nnue = NNUEEvaluator::new();
    nnue.save_to_file(temp_str).unwrap();

    handler.process_command(UsiCommand::SetOption {
        name: "NNUE_File".to_string(),
        value: temp_str.to_string(),
    });

    match handler.eval_mode() {
        EvalMode::Nnue(_) => {}
        _ => panic!("Evaluation mode must be NNUE after loading file"),
    }

    let _ = fs::remove_file(temp_path);
}
