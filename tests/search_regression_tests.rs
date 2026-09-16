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
