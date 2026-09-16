use tabula_shogi::board::Position;
use tabula_shogi::movegen::MoveGenerator;
use tabula_shogi::search::{NodeType, SearchEngine};

#[test]
fn fixed_depth_detects_attack_with_safe_own_king() {
    let mut pos = Position::from_sfen("4k4/3p1p3/4G4/9/9/9/9/9/4K4 b G 30").unwrap();
    let before = pos.to_sfen();
    let mut engine = SearchEngine::new(1);
    engine.max_nodes = Some(1);
    let (mv, score) = engine.search_fixed_depth(&mut pos, 1);
    assert_eq!(mv.unwrap().to_usi(), "G*5b");
    assert_eq!(score, 27_999);
    assert_eq!(pos.to_sfen(), before);
}

#[test]
fn fixed_depth_abort_preserves_completed_iteration() {
    let mut pos =
        Position::from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 30")
            .unwrap();
    let before = pos.to_sfen();
    let mut reference = SearchEngine::new(1);
    let expected = reference.search_fixed_depth(&mut pos, 1);
    let budget = reference.nodes() + 1;
    let mut engine = SearchEngine::new(1);
    engine.max_nodes = Some(budget);
    assert_eq!(engine.search_fixed_depth(&mut pos, 3), expected);
    let entry = engine.tt.probe(pos.hash).unwrap();
    assert_eq!(entry.depth, 1);
    assert_eq!(entry.node_type, NodeType::Exact);
    assert_eq!(entry.score, expected.1);
    assert_eq!(entry.best_move, expected.0);
    assert_eq!(pos.to_sfen(), before);
}

#[test]
fn fixed_depth_abort_before_depth_one_uses_static_score() {
    let mut pos = Position::from_sfen("4k4/9/9/9/9/9/4P4/9/4K4 b - 30").unwrap();
    let before = pos.to_sfen();
    let mut engine = SearchEngine::new(1);
    engine.max_nodes = Some(1);
    let expected = engine.evaluate(&pos);
    let (mv, score) = engine.search_fixed_depth(&mut pos, 3);
    assert_eq!(score, expected);
    assert!(MoveGenerator::generate_legal_moves(&mut pos).contains(&mv.unwrap()));
    assert!(engine.tt.probe(pos.hash).is_none());
    assert_eq!(pos.to_sfen(), before);
}

#[test]
fn fixed_depth_detail_tracks_completed_depth_accurately() {
    let mut pos = Position::from_sfen("4k4/9/9/9/9/9/4P4/9/4K4 b - 30").unwrap();
    let mut engine = SearchEngine::new(1);

    // 1. 中断なしで深さ2を完了
    let res = engine.search_fixed_depth_detail(&mut pos, 2);
    assert_eq!(res.completed_depth, 2);
    assert!(res.best_move.is_some());

    // 2. 深さ1の途中で即時中断
    let mut abort_engine = SearchEngine::new(1);
    abort_engine.max_nodes = Some(1);
    let abort_res = abort_engine.search_fixed_depth_detail(&mut pos, 2);
    assert_eq!(abort_res.completed_depth, 0);

    // 3. 深さ1を完了し、深さ2の途中で中断
    let mut ref_engine = SearchEngine::new(1);
    let ref_res = ref_engine.search_fixed_depth_detail(&mut pos, 1);
    assert_eq!(ref_res.completed_depth, 1);
    let budget = ref_engine.nodes() + 1;

    let mut partial_engine = SearchEngine::new(1);
    partial_engine.max_nodes = Some(budget);
    let partial_res = partial_engine.search_fixed_depth_detail(&mut pos, 3);
    assert_eq!(partial_res.completed_depth, 1);
    assert_eq!(partial_res.best_move, ref_res.best_move);
    assert_eq!(partial_res.score, ref_res.score);
}

#[test]
fn search_outcome_prevents_book_zero_overwrite_and_tracks_abort() {
    use tabula_shogi::search::SearchOutcome;
    use tabula_shogi::selfplay::{DatasetEntry, DatasetHandler};

    // 平手初期局面 (定跡OpeningBookが確実にヒットする局面)
    let mut start_pos =
        Position::from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
            .unwrap();
    let mut engine = SearchEngine::new(1);

    // 1. use_book: true の場合、定跡ヒット時は SearchOutcome::Book が返り、reliable_score_for_relabel は None を返す
    let outcome = engine.search_fixed_depth_outcome(&mut start_pos, 4);
    match outcome {
        SearchOutcome::Book { best_move } => {
            assert_eq!(best_move.to_usi(), "7g7f");
            assert_eq!(outcome.reliable_score_for_relabel(2), None);
        }
        _ => panic!(
            "Expected SearchOutcome::Book on startpos, got {:?}",
            outcome
        ),
    }

    // 2. 定跡無効の直接探索期待値を算出
    let mut direct_engine = SearchEngine::new(1);
    direct_engine.use_book = false;
    let direct_outcome = direct_engine.search_fixed_depth_outcome(&mut start_pos, 1);
    let expected_direct_score = match direct_outcome {
        SearchOutcome::Completed { score, .. } => score,
        _ => panic!("Expected Completed outcome for direct search"),
    };

    // 3. relabel_deep では engine.use_book = false により定跡手をスキップし、
    //    直接探索期待値と完全に一致する実値で再評価される (0 への誤更新ではないことを厳密検証)
    let mut entries = vec![DatasetEntry {
        sfen: start_pos.to_sfen(),
        score: 9999,
        result: 1.0,
        move_usi: "7g7f".to_string(),
    }];
    DatasetHandler::relabel_deep(&mut entries, 1, 1, 1);
    assert_eq!(
        entries[0].score, expected_direct_score,
        "relabel_deep must exactly match direct search score without book zero contamination"
    );

    // 4. depth: 0 で再評価を実行した場合はガードによりスコアが維持される
    let cur_score = entries[0].score;
    DatasetHandler::relabel_deep(&mut entries, 1, 0, 1);
    assert_eq!(
        entries[0].score, cur_score,
        "depth=0 must NOT overwrite label!"
    );

    // 5. SearchOutcome::Aborted (ノード上限による中断) では reliable_score_for_relabel が None を返し、
    //    深読み再評価でラベルが破壊されないことを直接検証
    let mut abort_engine = SearchEngine::new(1);
    abort_engine.use_book = false;
    abort_engine.max_nodes = Some(1); // 深さ1の最初で即座に打ち切り
    let abort_outcome = abort_engine.search_fixed_depth_outcome(&mut start_pos, 2);
    match abort_outcome {
        SearchOutcome::Aborted {
            completed_depth, ..
        } => {
            assert_eq!(completed_depth, 0);
            assert_eq!(abort_outcome.reliable_score_for_relabel(2), None);
        }
        _ => panic!(
            "Expected SearchOutcome::Aborted on max_nodes=1, got {:?}",
            abort_outcome
        ),
    }
}
