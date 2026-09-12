use super::config::SelfPlayConfig;
use crate::board::Position;
use crate::movegen::MoveGenerator;
use crate::search::SearchEngine;
use crate::types::{Color, Move};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameEndReason {
    Checkmate,
    Resignation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawReason {
    Sennichite,
    MaxPliesExceeded,
    Jishogi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameResult {
    BlackWin(GameEndReason),
    WhiteWin(GameEndReason),
    Draw(DrawReason),
}

impl GameResult {
    pub fn score_black(&self) -> f32 {
        match self {
            GameResult::BlackWin(_) => 1.0,
            GameResult::WhiteWin(_) => 0.0,
            GameResult::Draw(_) => 0.5,
        }
    }

    pub fn to_csa_end_comment(&self) -> &'static str {
        match self {
            GameResult::BlackWin(GameEndReason::Checkmate)
            | GameResult::WhiteWin(GameEndReason::Checkmate) => "%TSUMI",
            GameResult::BlackWin(GameEndReason::Resignation)
            | GameResult::WhiteWin(GameEndReason::Resignation) => "%TORYO",
            GameResult::Draw(DrawReason::Sennichite) => "%SENNICHITE",
            GameResult::Draw(DrawReason::MaxPliesExceeded)
            | GameResult::Draw(DrawReason::Jishogi) => "%JISHOGI",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlyRecord {
    pub ply: usize,
    pub sfen: String,
    pub side_to_move: Color,
    pub mv: Move,
    pub score: i32,
}

#[derive(Debug, Clone)]
pub struct GameRecord {
    pub game_id: usize,
    pub plies: Vec<PlyRecord>,
    pub result: GameResult,
    pub total_plies: usize,
}

/// ゼロ依存の自律的擬似乱数生成器 (XorShift64)
pub struct SimpleRng(pub u64);

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        SimpleRng(if seed == 0 { 0x853c49e6748fea9b } else { seed })
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    #[inline(always)]
    pub fn gen_range(&mut self, upper: usize) -> usize {
        if upper == 0 {
            0
        } else {
            (self.next_u64() as usize) % upper
        }
    }
}

pub struct GameRunner;

impl GameRunner {
    pub fn play_game(
        game_id: usize,
        config: &SelfPlayConfig,
        engine: &mut SearchEngine,
        rng: &mut SimpleRng,
    ) -> GameRecord {
        let mut pos = Position::startpos();
        let mut plies = Vec::with_capacity(128);
        let mut consecutive_low_eval_black = 0;
        let mut consecutive_low_eval_white = 0;

        let result;

        loop {
            // 1. 千日手判定 (4回同形出現)
            if pos.repetition_count() >= 4 {
                result = GameResult::Draw(DrawReason::Sennichite);
                break;
            }

            // 2. 手数上限判定
            if pos.ply > config.max_plies {
                result = GameResult::Draw(DrawReason::MaxPliesExceeded);
                break;
            }

            // 3. 合法手の生成
            let legal_moves = MoveGenerator::generate_legal_moves(&mut pos);
            if legal_moves.is_empty() {
                // 合法手なし = 詰み負け
                result = match pos.side_to_move {
                    Color::Black => GameResult::WhiteWin(GameEndReason::Checkmate),
                    Color::White => GameResult::BlackWin(GameEndReason::Checkmate),
                };
                break;
            }

            let sfen = pos.to_sfen();
            let current_side = pos.side_to_move;

            // 4. 着手の決定
            let (chosen_move, score) = if pos.ply <= config.random_opening_plies {
                // 序盤の多様性確保: ランダム着手
                let idx = rng.gen_range(legal_moves.len());
                let mv = legal_moves[idx];
                (mv, 0)
            } else {
                // 探索による最善手
                let (best_mv, eval_score) = engine.search_fixed_depth(&mut pos, config.depth);
                match best_mv {
                    Some(mv) => (mv, eval_score),
                    None => {
                        // 探索がNoneを返した場合は投了扱い
                        result = match current_side {
                            Color::Black => GameResult::WhiteWin(GameEndReason::Resignation),
                            Color::White => GameResult::BlackWin(GameEndReason::Resignation),
                        };
                        break;
                    }
                }
            };

            // 5. 投了判定 (2手連続で極端な劣勢)
            if score <= config.resign_threshold && pos.ply > config.random_opening_plies {
                match current_side {
                    Color::Black => consecutive_low_eval_black += 1,
                    Color::White => consecutive_low_eval_white += 1,
                }
            } else {
                match current_side {
                    Color::Black => consecutive_low_eval_black = 0,
                    Color::White => consecutive_low_eval_white = 0,
                }
            }

            if consecutive_low_eval_black >= 2 {
                result = GameResult::WhiteWin(GameEndReason::Resignation);
                break;
            }
            if consecutive_low_eval_white >= 2 {
                result = GameResult::BlackWin(GameEndReason::Resignation);
                break;
            }

            // 6. 記録
            plies.push(PlyRecord {
                ply: pos.ply,
                sfen,
                side_to_move: current_side,
                mv: chosen_move,
                score,
            });

            // 7. 着手の実行
            pos.do_move(chosen_move);
        }

        let total_plies = plies.len();
        GameRecord {
            game_id,
            plies,
            result,
            total_plies,
        }
    }
}
