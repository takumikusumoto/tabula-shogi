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
    assert_eq!(&bytes[0..8], b"TABU_NN4");

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

    // Float 推論側のスコア (駒割りベースライン + 残差)
    let b_feats = NNUEEvaluator::extract_features(&pos, tabula_shogi::types::Color::Black);
    let w_feats = NNUEEvaluator::extract_features(&pos, tabula_shogi::types::Color::White);
    let (res_float, _, _, _, _) = trainer.forward(&b_feats, &w_feats);
    let mat_black = NNUEEvaluator::material_black(&pos);
    let score_float = mat_black as f32 + res_float;

    // 初期重みでの初期局面評価値は有限値 (駒割りが等しいので 0)
    assert_eq!(score_int, 0);
    assert_eq!(score_float, 0.0);

    // 初期局面の対称性に頼らない非対称局面での厳密な一致検証
    let asym_pos =
        Position::from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2")
            .expect("Valid asymmetric sfen");
    let asym_score_int = nnue.evaluate(&asym_pos);
    let asym_b_feats =
        NNUEEvaluator::extract_features(&asym_pos, tabula_shogi::types::Color::Black);
    let asym_w_feats =
        NNUEEvaluator::extract_features(&asym_pos, tabula_shogi::types::Color::White);
    let (asym_res_float_raw, _, _, _, _) = trainer.forward(&asym_b_feats, &asym_w_feats);
    let asym_mat_black = NNUEEvaluator::material_black(&asym_pos);
    let asym_score_black_float = asym_mat_black as f32 + asym_res_float_raw;
    let asym_score_float = match asym_pos.side_to_move {
        tabula_shogi::types::Color::Black => asym_score_black_float,
        tabula_shogi::types::Color::White => -asym_score_black_float,
    };
    let asym_diff = (asym_score_int as f32 - asym_score_float).abs();
    assert!(
        asym_diff < 1.0,
        "Asymmetric Float vs Int discrepancy too large: float={asym_score_float}, int={asym_score_int}, diff={asym_diff}"
    );
}

#[test]
fn test_nnue_residual_material_exactness() {
    let nnue = NNUEEvaluator::new();
    let pos_start = Position::startpos();
    let start_eval = nnue.evaluate(&pos_start);
    assert_eq!(start_eval, 0, "Initial position material balance must be 0");

    // 1. 先手の歩を取り除き後手の持ち駒へ移動 (歩損: -100 * 2 = -200 cp)
    let mut pos_pawn_loss = pos_start.clone();
    let pawn_sq = pos_pawn_loss
        .board
        .iter()
        .position(|p| {
            p.is_some_and(|v| {
                v.color == tabula_shogi::types::Color::Black
                    && v.piece_type == tabula_shogi::types::PieceType::Pawn
            })
        })
        .unwrap();
    pos_pawn_loss.board[pawn_sq] = None;
    pos_pawn_loss.hand[tabula_shogi::types::Color::White.index()][0] += 1; // 後手持ち歩+1
    let pawn_eval = nnue.evaluate(&pos_pawn_loss);
    assert_eq!(
        pawn_eval, -200,
        "Losing pawn into opponent hand must be exactly -200 cp"
    );

    // 2. 先手の銀を取り除き後手の持ち駒へ移動 (銀損: -500 * 2 = -1000 cp)
    let mut pos_silver_loss = pos_start.clone();
    let silver_sq = pos_silver_loss
        .board
        .iter()
        .position(|p| {
            p.is_some_and(|v| {
                v.color == tabula_shogi::types::Color::Black
                    && v.piece_type == tabula_shogi::types::PieceType::Silver
            })
        })
        .unwrap();
    pos_silver_loss.board[silver_sq] = None;
    pos_silver_loss.hand[tabula_shogi::types::Color::White.index()][3] += 1;
    let silver_eval = nnue.evaluate(&pos_silver_loss);
    assert_eq!(
        silver_eval, -1000,
        "Losing silver into opponent hand must be exactly -1000 cp"
    );

    // 3. 先手の角を取り除き後手の持ち駒へ移動 (角損: -850 * 2 = -1700 cp)
    let mut pos_bishop_loss = pos_start.clone();
    let bishop_sq = pos_bishop_loss
        .board
        .iter()
        .position(|p| {
            p.is_some_and(|v| {
                v.color == tabula_shogi::types::Color::Black
                    && v.piece_type == tabula_shogi::types::PieceType::Bishop
            })
        })
        .unwrap();
    pos_bishop_loss.board[bishop_sq] = None;
    pos_bishop_loss.hand[tabula_shogi::types::Color::White.index()][5] += 1;
    let bishop_eval = nnue.evaluate(&pos_bishop_loss);
    assert_eq!(
        bishop_eval, -1700,
        "Losing bishop into opponent hand must be exactly -1700 cp"
    );

    // 4. 先手の飛車を取り除き後手の持ち駒へ移動 (飛車損: -1000 * 2 = -2000 cp)
    let mut pos_rook_loss = pos_start.clone();
    let rook_sq = pos_rook_loss
        .board
        .iter()
        .position(|p| {
            p.is_some_and(|v| {
                v.color == tabula_shogi::types::Color::Black
                    && v.piece_type == tabula_shogi::types::PieceType::Rook
            })
        })
        .unwrap();
    pos_rook_loss.board[rook_sq] = None;
    pos_rook_loss.hand[tabula_shogi::types::Color::White.index()][6] += 1;
    let rook_eval = nnue.evaluate(&pos_rook_loss);
    assert_eq!(
        rook_eval, -2000,
        "Losing rook into opponent hand must be exactly -2000 cp"
    );
}

#[test]
fn test_nnue_hidden_bias_gradient_update() {
    let mut trainer = NNUETrainer::new();
    let initial_bias = trainer.feature_biases[0];

    // 1サンプルのバッチで学習を実行
    let pos = Position::startpos();
    let b_feats = NNUEEvaluator::extract_features(&pos, tabula_shogi::types::Color::Black);
    let w_feats = NNUEEvaluator::extract_features(&pos, tabula_shogi::types::Color::White);
    let mat_black = NNUEEvaluator::material_black(&pos);

    // 出力層重みを非ゼロにして勾配を通す
    trainer.output_weights[0] = 0.5;

    // 白番で先手大優勢(1.0)のサンプルを与えることで勾配を発生させる
    let batch = vec![(
        b_feats,
        w_feats,
        mat_black,
        tabula_shogi::types::Color::Black,
        1.0,
    )];
    let _loss = trainer.train_batch(&batch, 0.05, 400.0);

    // feature_biases[0] が更新されたことを検証 (以前は actual_update = 0 でバグっていた)
    assert!(
        (trainer.feature_biases[0] - initial_bias).abs() > 1e-6,
        "feature_biases must be updated by Adam, got initial={initial_bias}, new={}",
        trainer.feature_biases[0]
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
        score: 200,
        result: 1.0,
        move_usi: "7g7f".to_string(),
    }];

    let (trained_eval, init_loss, final_loss) =
        trainer.train_dataset(&dataset, 20, 0.005, 1, 400.0);

    // 初期損失から学習により減少
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

#[test]
fn test_nnue_trainer_from_evaluator_warm_start() {
    let pos0 = Position::startpos();
    let dataset = vec![DatasetEntry {
        sfen: pos0.to_sfen(),
        score: 150,
        result: 1.0,
        move_usi: "7g7f".to_string(),
    }];

    // 世代1の学習
    let mut trainer1 = NNUETrainer::new();
    let (eval_gen1, init_loss1, final_loss1) = trainer1.train_dataset(&dataset, 10, 0.05, 1, 400.0);
    assert!(final_loss1 < init_loss1);

    // 世代2のWarm-start (世代1の学習済み重みを引き継ぐ)
    let mut trainer2 = NNUETrainer::from_evaluator(&eval_gen1);
    let (eval_gen2, init_loss2, final_loss2) = trainer2.train_dataset(&dataset, 10, 0.05, 1, 400.0);

    // Warm-start のため、世代2の初期損失は世代1の初期損失よりも大幅に低いはず
    assert!(
        init_loss2 < init_loss1,
        "Warm-start initial loss ({init_loss2}) should be lower than scratch initial loss ({init_loss1})"
    );
    assert!(final_loss2 <= init_loss2);

    let score1 = eval_gen1.evaluate(&pos0);
    let score2 = eval_gen2.evaluate(&pos0);
    assert!(
        score2 >= score1,
        "Score should further improve: {score1} -> {score2}"
    );
}
