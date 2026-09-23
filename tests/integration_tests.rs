#[cfg(test)]
mod tests {
    use tabula_shogi::types::{Color, Move, PieceType, Square};
    use tabula_shogi::{MoveGenerator, Position};

    #[test]
    fn test_startpos_legal_moves() {
        let mut pos = Position::startpos();
        let moves = MoveGenerator::generate_legal_moves(&mut pos);
        // 将棋の平手初期局面における合法手は30手
        assert_eq!(moves.len(), 30);
    }

    #[test]
    fn test_do_undo_move() {
        let mut pos = Position::startpos();
        let initial_hash = pos.hash;
        let mv = Move::from_usi("7g7f").expect("Valid USI move 7g7f");

        pos.do_move(mv);
        assert_eq!(pos.side_to_move, Color::White);
        assert_ne!(pos.hash, initial_hash);
        assert_eq!(pos.ply, 2);

        pos.undo_move();
        assert_eq!(pos.side_to_move, Color::Black);
        assert_eq!(pos.hash, initial_hash);
        assert_eq!(pos.ply, 1);
    }

    #[test]
    fn test_nifu_prohibition() {
        // 5筋に既に先手の歩がある状態で、5筋に歩を打つ手は非合法
        let mut pos = Position::startpos();
        // 持ち駒に歩を1枚追加
        let pawn_idx = PieceType::Pawn.hand_index().expect("Pawn has hand index");
        pos.hand[Color::Black.index()][pawn_idx] = 1;
        let moves = MoveGenerator::generate_legal_moves(&mut pos);

        // 5筋への歩打ちは存在しないはず（5gに歩があるため）
        for mv in moves {
            if mv.is_drop() && mv.drop_piece() == Some(PieceType::Pawn) {
                assert_ne!(
                    mv.to().file(),
                    Square::from_usi("5e").expect("Valid square 5e").file()
                );
            }
        }
    }

    #[test]
    fn test_check_evasion() {
        // 先手玉に王手がかかっている局面
        // 相手の飛車が先手玉の直前にいる局面など
        let sfen = "4k4/9/9/9/9/9/9/4r4/4K4 b - 1";
        let mut pos = Position::from_sfen(sfen).expect("Valid test SFEN");
        assert!(pos.is_in_check(Color::Black));

        let moves = MoveGenerator::generate_legal_moves(&mut pos);
        for mv in moves {
            pos.do_move(mv);
            assert!(
                !pos.is_in_check(Color::Black),
                "Escape move must not leave king in check: {:?}",
                mv
            );
            pos.undo_move();
        }
    }

    #[test]
    fn test_uchifudzume_prohibition() {
        // 後手玉が1a(file=0, rank=0)
        // 2a(file=1, rank=0)に金、3a(file=2, rank=0)に金（これで2aの金が紐付きになる）
        // 1c(file=0, rank=2)に香（1bを取れないようにする）
        // 1bに歩を打つと完全な詰みとなる
        let sfen = "6GGk/9/8L/9/9/9/9/9/4K4 b P 1";
        let mut pos = Position::from_sfen(sfen).expect("Valid uchifudzume SFEN");

        let moves = MoveGenerator::generate_legal_moves(&mut pos);
        for mv in moves {
            if mv.is_drop() && mv.drop_piece() == Some(PieceType::Pawn) {
                assert_ne!(
                    mv.to(),
                    Square::from_usi("1b").expect("Valid square 1b"),
                    "P*1b must be prohibited by uchifudzume rule"
                );
            }
        }
    }

    #[test]
    fn test_usi_parser() {
        use tabula_shogi::usi::UsiCommand;

        let cmd = UsiCommand::parse("position startpos moves 7g7f 3c3d");
        match cmd {
            UsiCommand::Position { sfen, moves } => {
                assert!(sfen.is_none());
                assert_eq!(moves.len(), 2);
                assert_eq!(moves[0], Move::from_usi("7g7f").expect("Valid move 7g7f"));
                assert_eq!(moves[1], Move::from_usi("3c3d").expect("Valid move 3c3d"));
            }
            _ => panic!("Expected Position command"),
        }

        let cmd_go = UsiCommand::parse("go btime 10000 wtime 12000 byoyomi 5000");
        match cmd_go {
            UsiCommand::Go(tc) => {
                assert_eq!(tc.btime, Some(10000));
                assert_eq!(tc.wtime, Some(12000));
                assert_eq!(tc.byoyomi, Some(5000));
            }
            _ => panic!("Expected Go command"),
        }
    }

    #[test]
    fn test_search_finds_bestmove() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;
        use tabula_shogi::search::{SearchEngine, TimeControl};

        // 定跡ツリーに存在しない手（例: ▲3六歩 3g3f）を適用して定跡を外す
        let mut pos = Position::startpos();
        let non_book_move = Move::from_usi("3g3f").expect("Valid move 3g3f");
        pos.do_move(non_book_move);

        let mut engine = SearchEngine::new(4);
        let tc = TimeControl {
            byoyomi: Some(100), // 100ms
            ..Default::default()
        };
        let stop_flag = Arc::new(AtomicBool::new(false));
        let best_move = engine.search(&mut pos, &tc, stop_flag);

        assert!(
            best_move.is_some(),
            "Search should find a legal move from non-book position"
        );
        let bm = best_move.expect("Search should return a move");
        let legal_moves = MoveGenerator::generate_legal_moves(&mut pos);
        assert!(
            legal_moves.contains(&bm),
            "Returned move must be legal: {:?}",
            bm
        );
        // 定跡の即時応答ではなく、実際にαβ探索木を走査したことを検証
        assert!(
            engine.nodes() > 0,
            "Engine must have searched nodes via alpha-beta (nodes: {})",
            engine.nodes()
        );
    }

    #[test]
    fn test_repetition() {
        let mut pos = Position::startpos();
        assert_eq!(pos.repetition_count(), 1);

        // 飛車を行ったり来たり動かす
        // 1. 2h3h, 2. 8b7b, 3. 3h2h, 4. 7b8b (これで初期配置に戻る)
        let m1 = Move::from_usi("2h3h").expect("Valid move 2h3h");
        let m2 = Move::from_usi("8b7b").expect("Valid move 8b7b");
        let m3 = Move::from_usi("3h2h").expect("Valid move 3h2h");
        let m4 = Move::from_usi("7b8b").expect("Valid move 7b8b");

        for _ in 0..3 {
            pos.do_move(m1);
            pos.do_move(m2);
            pos.do_move(m3);
            pos.do_move(m4);
        }

        // 3周繰り返したので初期局面のハッシュは 1 + 3 = 4 回出現（千日手）
        assert_eq!(pos.repetition_count(), 4);
    }

    #[test]
    fn test_opening_book() {
        use tabula_shogi::book::OpeningBook;

        let mut pos = Position::startpos();
        // 初手は定跡手 7g7f
        let m1 = OpeningBook::probe(&pos);
        assert_eq!(m1, Some(Move::from_usi("7g7f").expect("Valid move 7g7f")));

        // 7g7f を指す
        pos.do_move(m1.expect("m1 should be Some"));
        // 2手目 (後手) は定跡手 3c3d
        let m2 = OpeningBook::probe(&pos);
        assert_eq!(m2, Some(Move::from_usi("3c3d").expect("Valid move 3c3d")));
    }

    #[test]
    fn test_bug_reported_position() {
        use tabula_shogi::book::OpeningBook;

        // ユーザーから報告されたエラー局面:
        // 1. 7g7f, 2. 8c8d, 3. 2g2f, 4. 8d8e, 5. 6i7h, 6. 4a3b
        let moves_str = ["7g7f", "8c8d", "2g2f", "8d8e", "6i7h", "4a3b"];
        let mut pos = Position::startpos();
        for m_str in moves_str {
            let m = Move::from_usi(m_str).expect("Valid test move sequence");
            pos.do_move(m);
        }

        // 7手目の定跡手
        let next_move = OpeningBook::probe(&pos);
        assert!(next_move.is_some());
        let mv = next_move.expect("Opening book should have move");

        // 2e2d ではなく 2f2e でなければならない！
        assert_eq!(mv, Move::from_usi("2f2e").expect("Valid move 2f2e"));

        // 現在の局面で合法手であることを検証
        let legal_moves = MoveGenerator::generate_legal_moves(&mut pos);
        assert!(
            legal_moves.contains(&mv),
            "Book move must be strictly legal"
        );
    }

    #[test]
    fn test_dfpn_1_ply_mate() {
        use tabula_shogi::search::DfpnSolver;

        let sfen = "4k4/3p1p3/4G4/9/9/9/9/9/4K4 b G 1";
        let mut pos = Position::from_sfen(sfen).expect("Valid 1-ply mate SFEN");

        let mut solver = DfpnSolver::new(10000);
        let (is_mate, mate_move) = solver.solve(&mut pos);

        assert!(is_mate, "df-pn should detect 1-ply mate");
        assert_eq!(
            mate_move,
            Some(Move::from_usi("G*5b").expect("Valid drop move G*5b"))
        );
    }

    #[test]
    fn test_dfpn_3_ply_mate() {
        use tabula_shogi::search::DfpnSolver;

        let sfen = "8k/9/6Gpp/9/9/9/9/9/4K4 b GS 1";
        let mut pos = Position::from_sfen(sfen).expect("Valid 3-ply mate SFEN");

        let mut solver = DfpnSolver::new(20000);
        let (is_mate, mate_move) = solver.solve(&mut pos);

        assert!(is_mate, "df-pn should detect mate");
        assert!(mate_move.is_some(), "df-pn should return a mating move");
    }

    #[test]
    fn test_see_bad_capture_negative() {
        use tabula_shogi::search::SEE;
        // 設計レビュー 13.2 局面:
        // SFEN: k8/9/4g4/4p4/4R4/9/9/9/8K b - 1
        // 5eの飛車で5dの歩を取ると、5cの金で取り返されて大損する
        let sfen = "k8/9/4g4/4p4/4R4/9/9/9/8K b - 1";
        let pos = Position::from_sfen(sfen).expect("Valid SEE test SFEN");
        let mv = Move::from_usi("5e5d").expect("Valid move 5e5d");
        let val = SEE::evaluate(&pos, mv);
        assert!(
            val < 0,
            "SEE for taking a protected pawn with a rook must be negative, got: {}",
            val
        );
    }

    #[test]
    fn test_evasions_filter_uchifudzume() {
        // 設計レビュー 13.3 局面:
        // 王手回避として打つ歩が相手玉への打ち歩詰めになる場合、王手回避手としても非合法
        let sfen = "4k4/K4r3/3GN1B2/9/9/9/9/9/9 b P 1";
        let mut pos = Position::from_sfen(sfen).expect("Valid evasions test SFEN");
        let evasions = MoveGenerator::generate_evasions(&mut pos);
        let illegal_drop = Move::from_usi("P*5b").expect("Valid drop move P*5b");
        assert!(
            !evasions.contains(&illegal_drop),
            "Evasions must not contain an uchifudzume pawn drop!"
        );
    }

    #[test]
    fn test_hce_evaluation_breakdown() {
        use tabula_shogi::eval::Evaluator;
        let pos = Position::startpos();
        let breakdown = Evaluator::evaluate_detailed(&pos);
        // 初期局面は左右対称なので駒得・PST・囲いなどは先後同等で0、手番ボーナス25点のみ
        assert_eq!(breakdown.material_board, 0);
        assert_eq!(breakdown.material_hand, 0);
        assert_eq!(breakdown.piece_square, 0);
        assert_eq!(breakdown.tempo, 25);
        assert_eq!(breakdown.total, Evaluator::evaluate(&pos));
    }

    #[test]
    fn test_hce_hanging_piece_penalty() {
        use tabula_shogi::eval::Evaluator;
        // 先手の飛車が5五で相手の金(5四)に直接睨まれ、味方の利き（紐）が一切ない浮き駒の局面
        // 先手飛車 5e, 後手金 5d (5eに利いている), 先手玉 9i, 後手玉 1a
        let sfen_hanging = "k8/9/9/4g4/4R4/9/9/9/8K b - 1";
        let pos_hanging = Position::from_sfen(sfen_hanging).expect("Valid hanging piece test SFEN");
        let bd_hanging = Evaluator::evaluate_detailed(&pos_hanging);

        // 浮き駒ペナルティが入るため coordination がマイナスになる
        assert!(
            bd_hanging.coordination < 0,
            "Hanging rook under attack should receive a coordination penalty, got: {}",
            bd_hanging.coordination
        );
    }

    #[test]
    fn test_kif_book_parsing_and_branching() {
        use tabula_shogi::book::KifBook;

        let kif_sample = r#"
手合割：平手
先手：Sente
後手：Gote

手数----指手---------消費時間--
   1 ７六歩(77)   ( 0:01/00:00:01)
   2 ３四歩(33)   ( 0:01/00:00:01)
   3 ２六歩(27)   ( 0:01/00:00:01)
   4 ８四歩(83)   ( 0:01/00:00:01)
まで4手で中断

変化：2手
   2 ８四歩(83)   ( 0:01/00:00:01)
   3 ６八銀(79)   ( 0:01/00:00:01)
まで3手で中断
"#;

        let book = KifBook::from_kif_string(kif_sample).expect("Valid KIF parse");
        let mut pos = Position::startpos();

        // 初期局面: 7g7f
        let m1 = book.probe_best(&pos);
        assert_eq!(m1, Some(Move::from_usi("7g7f").unwrap()));

        // 1手進める
        pos.do_move(m1.unwrap());

        // 2手目の候補手: 本線 3c3d と 変化 8c8d の2つが登録されていること
        let moves = book.probe_moves(&pos).expect("Should have moves at ply 1");
        assert_eq!(moves.len(), 2);
        let move_strings: Vec<String> = moves.iter().map(|(m, _)| m.to_usi()).collect();
        assert!(move_strings.contains(&"3c3d".to_string()));
        assert!(move_strings.contains(&"8c8d".to_string()));
    }

    #[test]
    fn test_real_openings_kif_file() {
        use tabula_shogi::book::KifBook;
        let book =
            KifBook::load_from_file("book/openings.kif").expect("Must parse real openings.kif");
        assert!(!book.book_moves.is_empty(), "Book should have moves");
        // 全9戦型（横歩取り、角換わり、四間飛車、三間飛車、中飛車、矢倉、相掛かり、等）から多数の局面が登録されていること
        assert!(
            book.book_moves.len() >= 50,
            "Expected at least 50 positions, got {}",
            book.book_moves.len()
        );

        // 初期局面で 7g7f, 2g2f, 5g5f (先手中飛車) の3手が登録されていること
        let startpos = Position::startpos();
        let moves = book
            .probe_moves(&startpos)
            .expect("Must have moves for startpos");
        assert_eq!(
            moves.len(),
            3,
            "Startpos should have 3 book moves (7g7f, 2g2f, 5g5f)"
        );
        assert_eq!(
            book.probe_best(&startpos),
            Some(Move::from_usi("7g7f").unwrap())
        );
    }

    #[test]
    fn test_usi_halfkp_options_and_switching() {
        use tabula_shogi::UsiHandler;
        use tabula_shogi::usi::UsiCommand;

        let mut handler = UsiHandler::new();
        // 初期状態の確認
        assert!(matches!(
            handler.eval_mode(),
            tabula_shogi::eval::EvalMode::Hce | tabula_shogi::eval::EvalMode::HalfKP(_)
        ));

        // SetOption eval_type HalfKP
        handler.process_command(UsiCommand::SetOption {
            name: "eval_type".to_string(),
            value: "HalfKP".to_string(),
        });
        assert!(matches!(
            handler.eval_mode(),
            tabula_shogi::eval::EvalMode::HalfKP(_)
        ));

        // SetOption eval_type HCE
        handler.process_command(UsiCommand::SetOption {
            name: "eval_type".to_string(),
            value: "HCE".to_string(),
        });
        assert!(matches!(
            handler.eval_mode(),
            tabula_shogi::eval::EvalMode::Hce
        ));

        // SetOption eval_type NNUE
        handler.process_command(UsiCommand::SetOption {
            name: "eval_type".to_string(),
            value: "NNUE".to_string(),
        });
        assert!(matches!(
            handler.eval_mode(),
            tabula_shogi::eval::EvalMode::Nnue(_)
        ));
    }

    #[test]
    fn test_usi_halfkp_search_e2e() {
        use tabula_shogi::UsiHandler;
        use tabula_shogi::search::TimeControl;
        use tabula_shogi::usi::UsiCommand;

        let mut handler = UsiHandler::new();

        // 1. HalfKP モードへ明示的に切り替え
        handler.process_command(UsiCommand::SetOption {
            name: "eval_type".to_string(),
            value: "HalfKP".to_string(),
        });
        assert!(matches!(
            handler.eval_mode(),
            tabula_shogi::eval::EvalMode::HalfKP(_)
        ));

        // 2. 対局準備コマンド群の送信
        handler.process_command(UsiCommand::UsiNewGame);
        handler.process_command(UsiCommand::Position {
            sfen: None,
            moves: vec![],
        });

        // 3. 探索コマンドの送信 (秒読み 150ms で実際の HalfKP 探索を実行)
        handler.process_command(UsiCommand::Go(TimeControl {
            byoyomi: Some(150),
            ..Default::default()
        }));

        // 探索スレッドが完了するのを待機
        std::thread::sleep(std::time::Duration::from_millis(300));

        // 4. 正常終了コマンドの送信
        assert!(!handler.process_command(UsiCommand::Quit));
    }

    #[test]
    fn test_see_capture_promotion_value() {
        use tabula_shogi::search::see::{PROMOTION_SEE_VALUE, SEE};
        // 5四の歩が 5三の金(成る位置) を取る手
        // SFEN: k8/9/4g4/4P4/9/9/9/9/8K b - 1
        let sfen = "k8/9/4g4/4P4/9/9/9/9/8K b - 1";
        let pos = Position::from_sfen(sfen).expect("Valid test SFEN");

        let mv_promote = Move::from_usi("5d5c+").expect("Valid move 5d5c+");
        let mv_unpromote = Move::from_usi("5d5c").expect("Valid move 5d5c");

        let val_promote = SEE::evaluate(&pos, mv_promote);
        let val_unpromote = SEE::evaluate(&pos, mv_unpromote);

        assert_eq!(
            val_promote - val_unpromote,
            PROMOTION_SEE_VALUE,
            "Capture with promotion must include PROMOTION_SEE_VALUE in SEE evaluation"
        );
        assert!(val_promote > 0);
    }

    #[test]
    fn test_usi_eval_type_respects_custom_halfkp_file() {
        use tabula_shogi::eval::HalfKPEvaluator;
        use tabula_shogi::usi::UsiCommand;
        use tabula_shogi::usi::protocol::UsiHandler;

        let temp_dir = std::env::temp_dir();
        let custom_model_path = temp_dir
            .join(format!("test_custom_halfkp_{}.bin", std::process::id()))
            .to_string_lossy()
            .to_string();

        let mut custom_eval = HalfKPEvaluator::new();
        custom_eval.output_bias = 777;
        custom_eval
            .save_to_file(&custom_model_path)
            .expect("Save custom model");

        let mut handler = UsiHandler::new();

        // 1. halfkp_file オプションでカスタムパスを設定
        handler.process_command(UsiCommand::SetOption {
            name: "halfkp_file".to_string(),
            value: custom_model_path.clone(),
        });

        // 2. 一度 HCE に切り替え
        handler.process_command(UsiCommand::SetOption {
            name: "eval_type".to_string(),
            value: "HCE".to_string(),
        });

        // 3. 再度 eval_type: halfkp に切り替え (ここで custom_model_path が使われるべき)
        handler.process_command(UsiCommand::SetOption {
            name: "eval_type".to_string(),
            value: "halfkp".to_string(),
        });

        // 4. 正しくカスタムモデルがロードされているか検証
        if let tabula_shogi::eval::EvalMode::HalfKP(loaded_eval) = handler.eval_mode() {
            assert_eq!(
                loaded_eval.output_bias, 777,
                "Should load custom model weights"
            );
        } else {
            panic!("EvalMode should be HalfKP");
        }

        let _ = std::fs::remove_file(&custom_model_path);
        let _ = std::fs::remove_file(format!("{custom_model_path}.bak"));
    }

    #[test]
    fn test_usi_rejects_legacy_halfkp_without_changing_mode() {
        use tabula_shogi::usi::UsiCommand;
        use tabula_shogi::usi::protocol::UsiHandler;

        let legacy_path = std::env::temp_dir()
            .join(format!("test_legacy_halfkp_{}.bin", std::process::id()))
            .to_string_lossy()
            .to_string();
        std::fs::write(&legacy_path, b"TABU_HKP").expect("write legacy model fixture");

        let mut handler = UsiHandler::new();
        handler.process_command(UsiCommand::SetOption {
            name: "eval_type".to_string(),
            value: "HCE".to_string(),
        });
        handler.process_command(UsiCommand::SetOption {
            name: "halfkp_file".to_string(),
            value: legacy_path.clone(),
        });
        handler.process_command(UsiCommand::SetOption {
            name: "eval_type".to_string(),
            value: "HalfKP".to_string(),
        });

        assert!(
            matches!(handler.eval_mode(), tabula_shogi::eval::EvalMode::Hce),
            "invalid legacy model must not be replaced with a fresh HalfKP evaluator"
        );
        let _ = std::fs::remove_file(legacy_path);
    }
}
