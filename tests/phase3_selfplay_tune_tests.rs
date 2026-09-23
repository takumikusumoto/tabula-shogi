use tabula_shogi::board::Position;
use tabula_shogi::selfplay::csa::CsaSerializer;
use tabula_shogi::selfplay::dataset::{DatasetEntry, DatasetHandler};
use tabula_shogi::selfplay::game::{
    DrawReason, GameEndReason, GameRecord, GameResult, GameRunner, PlyRecord, SimpleRng,
};
use tabula_shogi::selfplay::{SelfPlayConfig, SelfPlayManager};
use tabula_shogi::tune::params::TunableParams;
use tabula_shogi::tune::texel::{PositionFeatures, TexelTuner};
use tabula_shogi::types::{Color, Move, PieceType, Square};

#[test]
fn test_position_to_sfen_startpos_roundtrip() {
    let start_pos = Position::startpos();
    let sfen = start_pos.to_sfen();

    // 平手初期局面のSFEN形式検証
    assert!(sfen.starts_with("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"));

    let restored = Position::from_sfen(&sfen).expect("SFEN parse should succeed");
    assert_eq!(start_pos.side_to_move, restored.side_to_move);
    assert_eq!(start_pos.board, restored.board);
    assert_eq!(start_pos.hand, restored.hand);
    assert_eq!(start_pos.hash, restored.hash);
}

#[test]
fn test_position_to_sfen_with_hand_roundtrip() {
    let mut pos = Position::startpos();
    // 7七歩 -> 7六歩
    let mv1 = Move::normal(Square::new(6, 6), Square::new(6, 5), false);
    pos.do_move(mv1);
    // 3三歩 -> 3四歩
    let mv2 = Move::normal(Square::new(2, 2), Square::new(2, 3), false);
    pos.do_move(mv2);
    // 8八角成 (2二角を取って手駒に角を獲得)
    let mv3 = Move::normal(Square::new(7, 7), Square::new(1, 1), true);
    pos.do_move(mv3);

    assert_eq!(
        pos.hand[Color::Black.index()][PieceType::Bishop.hand_index().unwrap()],
        1
    );

    let sfen = pos.to_sfen();
    assert!(sfen.contains(" B "));

    let restored = Position::from_sfen(&sfen).expect("Roundtrip SFEN parse should succeed");
    assert_eq!(pos.side_to_move, restored.side_to_move);
    assert_eq!(pos.board, restored.board);
    assert_eq!(pos.hand, restored.hand);
    assert_eq!(pos.hash, restored.hash);
}

#[test]
fn test_csa_serializer_output() {
    let mv1 = Move::normal(Square::new(6, 6), Square::new(6, 5), false); // 7七歩(6,6) -> 7六歩(6,5)
    let csa_mv = CsaSerializer::format_move(Color::Black, mv1, PieceType::Pawn);
    assert_eq!(csa_mv, "+7776FU");

    let record = GameRecord {
        game_id: 1,
        plies: vec![PlyRecord {
            ply: 1,
            sfen: "dummy".to_string(),
            side_to_move: Color::Black,
            mv: mv1,
            score: 25,
        }],
        result: GameResult::BlackWin(GameEndReason::Resignation),
        total_plies: 1,
    };

    let csa_text = CsaSerializer::serialize_game(&record);
    assert!(csa_text.contains("V2.2"));
    assert!(csa_text.contains("N+TabulaShogi"));
    assert!(csa_text.contains("+7776FU"));
    assert!(csa_text.contains("%TORYO"));
}

#[test]
fn test_dataset_entry_format_and_parse() {
    let entry = DatasetEntry {
        sfen: "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1".to_string(),
        score: 120,
        result: 1.0,
        move_usi: "7g7f".to_string(),
    };

    let line = DatasetHandler::format_entry(&entry);
    let parsed = DatasetHandler::parse_entry(&line).expect("Should parse TSV line");

    assert_eq!(entry, parsed);
}

fn two_ply_record(result: GameResult) -> GameRecord {
    let pos = Position::startpos();
    GameRecord {
        game_id: 99,
        plies: vec![
            PlyRecord {
                ply: 1,
                sfen: pos.to_sfen(),
                side_to_move: Color::Black,
                mv: Move::normal(Square::new(6, 6), Square::new(6, 5), false),
                score: 12,
            },
            PlyRecord {
                ply: 2,
                sfen: pos.to_sfen(),
                side_to_move: Color::White,
                mv: Move::normal(Square::new(2, 2), Square::new(2, 3), false),
                score: -12,
            }
        ],
        result,
        total_plies: 2,
    }
}

#[test]
fn test_dataset_retains_max_plies_draw_as_half_result() {
    let pos = Position::startpos();
    let sfen = pos.to_sfen();
    let plies = (1..=256)
        .map(|ply| PlyRecord {
            ply,
            sfen: sfen.clone(),
            side_to_move: if ply % 2 == 1 {
                Color::Black
            } else {
                Color::White
            },
            mv: Move::normal(Square::new(6, 6), Square::new(6, 5), false),
            score: 0,
        })
        .collect();
    let game = GameRecord {
        game_id: 256,
        plies,
        result: GameResult::Draw(DrawReason::MaxPliesExceeded),
        total_plies: 256,
    };
    let entries = DatasetHandler::extract_entries_checked(&game, 1)
        .expect("a structurally valid maximum-plies draw must be retained");

    assert_eq!(entries.len(), 256);
    assert!(entries.iter().all(|entry| entry.result == 0.5));
}

#[test]
fn test_dataset_keeps_normal_draw_distinct_and_rejects_invalid_record() {
    let ordinary_draw = two_ply_record(GameResult::Draw(DrawReason::Sennichite));
    let entries = DatasetHandler::extract_entries_checked(&ordinary_draw, 1)
        .expect("an ordinary draw must remain valid");
    assert_eq!(entries.len(), 2);
    assert!(entries.iter().all(|entry| entry.result == 0.5));

    let mut invalid = two_ply_record(GameResult::Draw(DrawReason::MaxPliesExceeded));
    invalid.total_plies = 3;
    let error = DatasetHandler::extract_entries_checked(&invalid, 1)
        .expect_err("a malformed record must not be relabelled as a draw");
    assert!(error.contains("total_plies"));
}

#[test]
fn test_selfplay_single_game_simulation() {
    let config = SelfPlayConfig {
        num_games: 1,
        threads: 1,
        depth: 1, // 高速テストのため深さ1
        random_opening_plies: 2,
        max_plies: 8, // 最大8手で終了
        resign_threshold: -3000,
        csa_output: None,
        data_output: None,
        tt_size_mb: 4,
        seed: 12345,
        eval_mode: tabula_shogi::eval::EvalMode::Hce,
        temperature_plies: 2,
        use_book: false,
        start_game_id: 0,
    };

    let mut engine = tabula_shogi::search::SearchEngine::new(config.tt_size_mb);
    let mut rng = SimpleRng::new(config.seed);

    let game = GameRunner::play_game(1, &config, &mut engine, &mut rng);

    assert_eq!(game.game_id, 1);
    assert!(game.total_plies > 0 && game.total_plies <= 8);
    assert_eq!(game.plies.len(), game.total_plies);

    // 最大手数に達した場合はDraw(MaxPliesExceeded)
    if game.total_plies == 8 {
        assert_eq!(game.result, GameResult::Draw(DrawReason::MaxPliesExceeded));
    }
}

#[test]
fn test_selfplay_manager_small_batch() {
    let config = SelfPlayConfig {
        num_games: 2,
        threads: 2,
        depth: 1,
        random_opening_plies: 2,
        max_plies: 6,
        resign_threshold: -3000,
        csa_output: None,
        data_output: None,
        tt_size_mb: 4,
        seed: 99999,
        eval_mode: tabula_shogi::eval::EvalMode::Hce,
        temperature_plies: 2,
        use_book: false,
        start_game_id: 0,
    };

    let stats = SelfPlayManager::run(config);
    assert_eq!(stats.completed_games, 2);
    assert!(stats.total_plies > 0);
}

#[test]
fn test_search_with_temperature_behavior() {
    let mut pos = Position::startpos();
    let mut engine = tabula_shogi::search::SearchEngine::new(4);

    // 温度0.0での探索: search_fixed_depth と一致すること
    let (fixed_mv, fixed_score) = engine.search_fixed_depth(&mut pos, 1);
    let (temp_zero_mv, temp_zero_score) = engine.search_with_temperature(&mut pos, 1, 0.0, 42);
    assert_eq!(fixed_mv, temp_zero_mv);
    assert_eq!(fixed_score, temp_zero_score);

    // 温度1.0での探索: 合法手が選ばれること
    let legal_moves = tabula_shogi::movegen::MoveGenerator::generate_legal_moves(&mut pos);
    let (temp_mv, score) = engine.search_with_temperature(&mut pos, 1, 1.0, 12345);
    assert!(temp_mv.is_some());
    assert!(legal_moves.contains(&temp_mv.unwrap()));
    // 序盤の評価値は妥当な範囲内であること
    assert!(score.abs() < 500);
}

#[test]
fn test_texel_tuning_convergence() {
    // 合成データセットによる損失収束テスト
    let pos_start = Position::startpos();
    let mut pos_black_adv = Position::startpos();
    // 先手有利局面 (角得)
    pos_black_adv.hand[Color::Black.index()][PieceType::Bishop.hand_index().unwrap()] = 1;

    let feat_start = PositionFeatures::extract(&pos_start, 0.5); // 引分
    let feat_black_adv = PositionFeatures::extract(&pos_black_adv, 1.0); // 先手勝ち

    let dataset = vec![feat_start, feat_black_adv];
    let initial_params = TunableParams::default();

    let (tuned, initial_loss, final_loss) = TexelTuner::train_adam(
        &dataset,
        &initial_params,
        25,  // 25 epochs
        5.0, // learning rate
        TexelTuner::DEFAULT_K,
    );

    assert!(
        final_loss <= initial_loss,
        "Final MSE loss ({final_loss}) should be less than or equal to initial loss ({initial_loss})"
    );
    // 駒価値が最低閾値(10.0以上)を維持していることを検証
    for &v in &tuned.piece_values {
        assert!(v >= 10.0);
    }
}

#[test]
fn test_dataset_load_sampled() {
    let tmp_path = "test_sampled_dataset.tsv";
    let pos_start = Position::startpos();

    // 100件のテスト用エントリーを作成
    let mut entries = Vec::new();
    for i in 1..=100 {
        entries.push(DatasetEntry {
            sfen: pos_start.to_sfen(),
            score: i,
            result: if i % 2 == 0 { 1.0 } else { 0.0 },
            move_usi: "7g7f".to_string(),
        });
    }

    DatasetHandler::append_to_file(tmp_path, &entries).expect("Write test dataset");

    // 20件サンプリング (最新50% = 10件, 過去50% = 10件)
    let sampled =
        DatasetHandler::load_sampled(tmp_path, 20, 0.5, 12345).expect("Load sampled dataset");

    assert_eq!(sampled.len(), 20, "Should sample exactly 20 entries");

    // クリーンアップ
    let _ = std::fs::remove_file(tmp_path);
}

#[test]
fn test_dataset_relabel_deep() {
    let pos_start = Position::startpos();
    let mut entries = vec![
        DatasetEntry {
            sfen: pos_start.to_sfen(),
            score: 9999,
            result: 0.5,
            move_usi: "7g7f".to_string(),
        };
        4
    ];

    let successes =
        DatasetHandler::relabel_deep(&mut entries, 4, 1, 2, &tabula_shogi::eval::EvalMode::Hce);
    assert_eq!(successes.len(), 4, "All valid positions should succeed");

    for entry in &entries {
        assert_ne!(
            entry.score, 9999,
            "Score should be re-evaluated and updated"
        );
    }
}

#[test]
fn test_streaming_batch_reader() {
    use tabula_shogi::selfplay::StreamingBatchReader;

    let test_path = "target/test_streaming_dataset.tsv";
    let pos_start = Position::startpos();
    let entries = vec![
        DatasetEntry {
            sfen: pos_start.to_sfen(),
            score: 10,
            result: 1.0,
            move_usi: "7g7f".to_string(),
        };
        15
    ];
    DatasetHandler::append_to_file(test_path, &entries).expect("Failed to write test dataset");

    // バッチサイズ 4 でストリーミング読込 (4, 4, 4, 3, None)
    let mut reader = StreamingBatchReader::new(test_path, 4).expect("Failed to open reader");
    let mut total_read = 0;
    let mut batch_count = 0;

    while let Some(batch) = reader.next_batch().expect("Failed to read next batch") {
        batch_count += 1;
        total_read += batch.len();
        if batch_count <= 3 {
            assert_eq!(batch.len(), 4);
        } else {
            assert_eq!(batch.len(), 3);
        }
    }

    assert_eq!(total_read, 15);
    assert_eq!(batch_count, 4);

    let _ = std::fs::remove_file(test_path);
}

#[test]
fn test_selfplay_config_use_book_default() {
    let default_cfg = SelfPlayConfig::default();
    assert!(
        !default_cfg.use_book,
        "SelfPlayConfig::default() must have use_book = false for Tabula Rasa"
    );
}
