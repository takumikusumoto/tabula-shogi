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
