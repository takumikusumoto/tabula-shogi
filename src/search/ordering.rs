use crate::board::Position;
use crate::types::Move;

const TT_MOVE_BONUS: i32 = 1_000_000;
const GOOD_CAPTURE_BONUS: i32 = 100_000;
const BAD_CAPTURE_PENALTY: i32 = -50_000;
const PROMOTION_BONUS: i32 = 20_000;
const KILLER_1_BONUS: i32 = 10_000;
const KILLER_2_BONUS: i32 = 9_000;
const COUNTER_MOVE_BONUS: i32 = 8_500;
const HISTORY_MAX_BONUS: i32 = 8_000;
const ENEMY_ZONE_DROP_BONUS: i32 = 1_000;

pub struct MoveOrderer;

impl MoveOrderer {
    /// 探索効率を最大化するために指し手をソートする
    pub fn order_moves(
        moves: &mut [Move],
        pos: &Position,
        tt_move: Option<Move>,
        killer_moves: &[Option<Move>; 2],
        counter_move: Option<Move>,
        history: Option<&[[i32; 81]; 81]>,
    ) {
        let mut scores: Vec<i32> = moves
            .iter()
            .map(|mv| Self::score_move(*mv, pos, tt_move, killer_moves, counter_move, history))
            .collect();

        // 簡易選択ソート
        for i in 0..moves.len() {
            let mut best_idx = i;
            for j in (i + 1)..moves.len() {
                if scores[j] > scores[best_idx] {
                    best_idx = j;
                }
            }
            if best_idx != i {
                moves.swap(i, best_idx);
                scores.swap(i, best_idx);
            }
        }
    }

    fn score_move(
        mv: Move,
        pos: &Position,
        tt_move: Option<Move>,
        killer_moves: &[Option<Move>; 2],
        counter_move: Option<Move>,
        history: Option<&[[i32; 81]; 81]>,
    ) -> i32 {
        // 1. 置換表の最善手
        if Some(mv) == tt_move {
            return TT_MOVE_BONUS;
        }

        let mut score = 0;

        // 2. 駒取り (SEE + MVV-LVA)
        let to_sq = mv.to();
        if let Some(target_p) = pos.board[to_sq.index()] {
            let see_val = super::see::SEE::evaluate(pos, mv);
            if see_val >= 0 {
                let victim_val = target_p.piece_type.base_value();
                let attacker_val = if let Some(from_sq) = mv.from() {
                    pos.board[from_sq.index()]
                        .map(|p| p.piece_type.base_value())
                        .unwrap_or(100)
                } else {
                    mv.drop_piece().map(|pt| pt.base_value()).unwrap_or(100)
                };
                score += GOOD_CAPTURE_BONUS + see_val + (victim_val * 10 - attacker_val);
            } else {
                // 損な駒取り (Bad Capture) は優先度を下げる
                score += BAD_CAPTURE_PENALTY + see_val;
            }
        }

        // 3. 成り手
        if mv.is_promote() {
            score += PROMOTION_BONUS;
        }

        // 4. キラー手
        if killer_moves[0] == Some(mv) {
            score += KILLER_1_BONUS;
        } else if killer_moves[1] == Some(mv) {
            score += KILLER_2_BONUS;
        }

        // 5. 応手 (Countermove Heuristic: 直前の相手手に対する好手)
        if counter_move == Some(mv) {
            score += COUNTER_MOVE_BONUS;
        }

        // 6. 歴史ヒューリスティック (過去にカットオフを起こした手の優先)
        if let Some(hist) = history
            && let Some(from_sq) = mv.from()
        {
            let h_val = hist[from_sq.index()][to_sq.index()];
            score += h_val.min(HISTORY_MAX_BONUS);
        }

        // 7. 駒打ち (敵陣への侵入打ち手のみ優遇)
        if mv.is_drop() && to_sq.is_promoted_zone(pos.side_to_move) {
            score += ENEMY_ZONE_DROP_BONUS;
        }

        score
    }
}
