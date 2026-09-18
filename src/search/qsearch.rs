use super::engine::{MATE_SCORE, SearchContext, SearchEngine, TIME_CHECK_INTERVAL};
use super::ordering::MoveOrderer;
use super::see::SEE;
use crate::board::Position;
use crate::movegen::MoveGenerator;
use crate::types::Move;
use std::sync::atomic::Ordering;

/// 静止探索 (Quiescence Search) + SEE Pruning
/// 通常探索の末端ノードで駒の取り合いが落ち着くまでキャプチャ手・成り手のみを再帰探索し、地平線効果を抑制する。
pub fn quiescence(
    engine: &mut SearchEngine,
    pos: &mut Position,
    mut alpha: i32,
    beta: i32,
    ply: usize,
    ctx: &SearchContext,
) -> i32 {
    engine.nodes += 1;
    if let Some(max_n) = engine.max_nodes
        && engine.nodes >= max_n
    {
        ctx.stop_flag.store(true, Ordering::Relaxed);
        return engine.evaluate_at_ply(pos, ply);
    }
    if engine.nodes.is_multiple_of(TIME_CHECK_INTERVAL) && ctx.time_mgr.is_time_up() {
        ctx.stop_flag.store(true, Ordering::Relaxed);
    }
    if ctx.stop_flag.load(Ordering::Relaxed) {
        return 0;
    }

    if ply >= 64 {
        return engine.evaluate_at_ply(pos, ply);
    }

    let in_check = pos.is_in_check(pos.side_to_move);

    if in_check {
        // 王手中の処理: stand-pat 禁止、全王手回避手を探索
        let mut evasions = MoveGenerator::generate_evasions(pos);
        if evasions.is_empty() {
            // 回避手なし = 詰み
            return -MATE_SCORE + (ply as i32);
        }

        MoveOrderer::order_moves(
            &mut evasions,
            pos,
            None,
            &[None, None],
            None,
            Some(&engine.history),
        );

        let mut best_score = -MATE_SCORE + (ply as i32);

        for mv in evasions {
            pos.do_move(mv);
            engine.update_accumulator_after_move_at_ply(pos, mv, ply + 1);
            let score = -quiescence(engine, pos, -beta, -alpha, ply + 1, ctx);
            pos.undo_move();

            if ctx.stop_flag.load(Ordering::Relaxed) {
                return 0;
            }

            if score > best_score {
                best_score = score;
            }
            if score >= beta {
                return beta;
            }
            if score > alpha {
                alpha = score;
            }
        }
        return best_score;
    }

    // 王手されていない通常局面: 静的評価（立合いスコア）
    let stand_pat = engine.evaluate_at_ply(pos, ply);
    if stand_pat >= beta {
        return beta;
    }
    if stand_pat > alpha {
        alpha = stand_pat;
    }

    // 取り合い（捕獲手および成り手）のみを生成
    let all_moves = MoveGenerator::generate_legal_moves(pos);
    let mut tactical_moves: Vec<Move> = all_moves
        .into_iter()
        .filter(|m| pos.board[m.to().index()].is_some() || m.is_promote())
        .collect();

    MoveOrderer::order_moves(
        &mut tactical_moves,
        pos,
        None,
        &[None, None],
        None,
        Some(&engine.history),
    );

    for mv in tactical_moves {
        // SEE Pruning: 駒取り手でSEE < 0（損な取り合い）はスキップ
        if pos.board[mv.to().index()].is_some() && SEE::evaluate(pos, mv) < 0 {
            continue;
        }

        pos.do_move(mv);
        engine.update_accumulator_after_move_at_ply(pos, mv, ply + 1);
        let score = -quiescence(engine, pos, -beta, -alpha, ply + 1, ctx);
        pos.undo_move();

        if ctx.stop_flag.load(Ordering::Relaxed) {
            return 0;
        }

        if score >= beta {
            return beta;
        }
        if score > alpha {
            alpha = score;
        }
    }

    alpha
}
