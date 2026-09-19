use super::sprt::{Sprt, SprtConfig};
use crate::board::Position;
use crate::eval::EvalMode;
use crate::movegen::MoveGenerator;
use crate::search::SearchEngine;
use crate::selfplay::game::SimpleRng;
use crate::types::Color;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

pub struct MatchConfig {
    pub name_a: String,
    pub name_b: String,
    pub eval_a: EvalMode,
    pub eval_b: EvalMode,
    pub pairs: usize,
    pub depth: u8,
    pub threads: usize,
    pub random_opening: usize,
    pub max_plies: usize,
    pub tt_size_mb: usize,
    pub sprt_config: Option<SprtConfig>,
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self {
            name_a: "Model_A".to_string(),
            name_b: "Model_B".to_string(),
            eval_a: EvalMode::Hce,
            eval_b: EvalMode::Hce,
            pairs: 20,
            depth: 2,
            threads: 1,
            random_opening: 6,
            max_plies: 320,
            tt_size_mb: 16,
            sprt_config: Some(SprtConfig::default()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MatchResult {
    pub wins_a: usize,
    pub wins_b: usize,
    pub draws: usize,
    pub total_games: usize,
    pub win_rate_a: f64,
    pub elo_diff_a: f64,
    pub sprt: Option<Sprt>,
}

pub struct MatchRunner;

impl MatchRunner {
    /// 2つのモデル間の先後交代ペアマッチを並列実行
    pub fn run_match(config: &MatchConfig) -> MatchResult {
        Self::run_match_extended(config, None, config.pairs)
    }

    /// 既存の検定状態（勝敗、SPRT、進行ペア番号）を引き継ぎ、追加ペア分だけを実行して累積集計する
    pub fn run_match_extended(
        config: &MatchConfig,
        previous_result: Option<&MatchResult>,
        target_pairs: usize,
    ) -> MatchResult {
        let (start_pair, initial_wins_a, initial_wins_b, initial_draws, prev_sprt) =
            if let Some(prev) = previous_result {
                let s_pair = (prev.wins_a + prev.wins_b + prev.draws) / 2;
                (
                    s_pair,
                    prev.wins_a,
                    prev.wins_b,
                    prev.draws,
                    prev.sprt.clone(),
                )
            } else {
                (0, 0, 0, 0, None)
            };

        let target_pairs = target_pairs.max(start_pair);
        if start_pair >= target_pairs
            && let Some(prev) = previous_result
        {
            return prev.clone();
        }

        let num_threads = config.threads.max(1);
        let pairs_total = target_pairs;

        let pair_counter = Arc::new(AtomicUsize::new(start_pair));
        let finished_counter = Arc::new(AtomicUsize::new(start_pair));
        let early_stop_flag = Arc::new(AtomicBool::new(false));

        let wins_a_total = Arc::new(AtomicUsize::new(initial_wins_a));
        let wins_b_total = Arc::new(AtomicUsize::new(initial_wins_b));
        let draws_total = Arc::new(AtomicUsize::new(initial_draws));

        let sprt_tracker = if let Some(ps) = prev_sprt {
            if ps.is_decided() {
                early_stop_flag.store(true, Ordering::Relaxed);
            }
            Some(Arc::new(Mutex::new(ps)))
        } else {
            config
                .sprt_config
                .as_ref()
                .map(|sc| Arc::new(Mutex::new(Sprt::new(sc.clone()))))
        };

        if start_pair == 0 {
            println!("=== TabulaShogi Arena Match ===");
            println!(
                "{} vs {} | Pairs: {} (Games: {}), Threads: {}, Depth: {}",
                config.name_a,
                config.name_b,
                pairs_total,
                pairs_total * 2,
                num_threads,
                config.depth
            );
        } else {
            println!(
                "\n--- Incremental Overtime: Pairs {} -> {} (Games {} -> {}), Threads: {} ---",
                start_pair,
                pairs_total,
                start_pair * 2,
                pairs_total * 2,
                num_threads
            );
        }
        println!("--------------------------------------");

        thread::scope(|s| {
            for _thread_id in 0..num_threads {
                let pair_counter = Arc::clone(&pair_counter);
                let finished_counter = Arc::clone(&finished_counter);
                let early_stop_flag = Arc::clone(&early_stop_flag);
                let wins_a_total = Arc::clone(&wins_a_total);
                let wins_b_total = Arc::clone(&wins_b_total);
                let draws_total = Arc::clone(&draws_total);
                let sprt_tracker = sprt_tracker.as_ref().map(Arc::clone);

                s.spawn(move || {
                    let mut engine_a = SearchEngine::new(config.tt_size_mb)
                        .with_eval_mode(config.eval_a.clone());
                    let mut engine_b = SearchEngine::new(config.tt_size_mb)
                        .with_eval_mode(config.eval_b.clone());

                    loop {
                        if early_stop_flag.load(Ordering::Relaxed) {
                            break;
                        }
                        let p_idx = pair_counter.fetch_add(1, Ordering::Relaxed);
                        if p_idx >= pairs_total {
                            break;
                        }

                        // ペア用の共通初期局面を生成
                        let seed = 0x9e3779b97f4a7c15u64
                            .wrapping_add((p_idx as u64).wrapping_mul(0xbf58476d1ce4e5b9));
                        let initial_pos = Self::generate_opening_position(config.random_opening, seed);

                        // Game 1: Black = A, White = B
                        engine_a.clear();
                        engine_b.clear();
                        let mut pos1 = initial_pos.clone();
                        let res1 = Self::play_game(
                            &mut engine_a,
                            &mut engine_b,
                            &mut pos1,
                            config.depth,
                            config.max_plies,
                        );

                        let (win_a_1, win_b_1, draw_1) = match res1 {
                            Some(Color::Black) => (1, 0, 0),
                            Some(Color::White) => (0, 1, 0),
                            None => (0, 0, 1),
                        };

                        // Game 2: Black = B, White = A (先後入替)
                        engine_a.clear();
                        engine_b.clear();
                        let mut pos2 = initial_pos;
                        let res2 = Self::play_game(
                            &mut engine_b,
                            &mut engine_a,
                            &mut pos2,
                            config.depth,
                            config.max_plies,
                        );

                        let (win_a_2, win_b_2, draw_2) = match res2 {
                            Some(Color::Black) => (0, 1, 0),
                            Some(Color::White) => (1, 0, 0),
                            None => (0, 0, 1),
                        };

                        let w_a = win_a_1 + win_a_2;
                        let w_b = win_b_1 + win_b_2;
                        let dr = draw_1 + draw_2;

                        wins_a_total.fetch_add(w_a, Ordering::Relaxed);
                        wins_b_total.fetch_add(w_b, Ordering::Relaxed);
                        draws_total.fetch_add(dr, Ordering::Relaxed);

                        if let Some(ref tracker) = sprt_tracker {
                            let mut s = tracker.lock().unwrap();
                            s.record_batch(w_a, w_b, dr);
                            if s.is_decided() {
                                early_stop_flag.store(true, Ordering::Relaxed);
                            }
                        }

                        let fin = finished_counter.fetch_add(1, Ordering::Relaxed) + 1;
                        if fin.is_multiple_of(5) || fin == pairs_total {
                            let cur_wa = wins_a_total.load(Ordering::Relaxed);
                            let cur_wb = wins_b_total.load(Ordering::Relaxed);
                            let cur_dr = draws_total.load(Ordering::Relaxed);
                            let cur_tot = cur_wa + cur_wb + cur_dr;
                            let wr = if cur_tot > 0 {
                                (cur_wa as f64 + 0.5 * cur_dr as f64) / (cur_tot as f64) * 100.0
                            } else {
                                50.0
                            };
                            print!(
                                "\r[Progress] Pair {}/{} ({} games) | {}: {} | {}: {} | Draw: {} | WinRate: {:.1}%",
                                fin, pairs_total, cur_tot, config.name_a, cur_wa, config.name_b, cur_wb, cur_dr, wr
                            );
                        }
                    }
                });
            }
        });

        println!();
        println!("--------------------------------------");

        let wins_a = wins_a_total.load(Ordering::Relaxed);
        let wins_b = wins_b_total.load(Ordering::Relaxed);
        let draws = draws_total.load(Ordering::Relaxed);
        let total_games = wins_a + wins_b + draws;

        let final_sprt = sprt_tracker.map(|t| t.lock().unwrap().clone());
        let win_rate_a = if total_games > 0 {
            (wins_a as f64 + 0.5 * draws as f64) / (total_games as f64)
        } else {
            0.5
        };

        let elo_diff_a = if win_rate_a <= 0.001 {
            -1000.0
        } else if win_rate_a >= 0.999 {
            1000.0
        } else {
            -400.0 * (1.0 / win_rate_a - 1.0).log10()
        };

        println!(
            "Match Completed: {} games | {}: {} (wins) | {}: {} (wins) | Draws: {}",
            total_games, config.name_a, wins_a, config.name_b, wins_b, draws
        );
        println!(
            "WinRate ({}): {:.2}% | Elo Diff: {:+.1}",
            config.name_a,
            win_rate_a * 100.0,
            elo_diff_a
        );

        if let Some(ref sprt) = final_sprt {
            println!(
                "SPRT Status: {:?} | LLR: {:.2} [{:.2}, {:.2}]",
                sprt.status, sprt.llr, sprt.lower_bound, sprt.upper_bound
            );
        }
        println!("======================================");

        MatchResult {
            wins_a,
            wins_b,
            draws,
            total_games,
            win_rate_a,
            elo_diff_a,
            sprt: final_sprt,
        }
    }

    /// 共通のランダム序盤局面を生成
    pub fn generate_opening_position(random_plies: usize, seed: u64) -> Position {
        if random_plies == 0 {
            return Position::startpos();
        }

        let mut rng = SimpleRng::new(seed);
        for _attempt in 0..10 {
            let mut pos = Position::startpos();
            let mut valid = true;

            for _ in 0..random_plies {
                let moves = MoveGenerator::generate_legal_moves(&mut pos);
                if moves.is_empty() || pos.repetition_count() >= 4 {
                    valid = false;
                    break;
                }
                let choice = moves[rng.gen_range(moves.len())];
                pos.do_move(choice);
            }

            if valid && !pos.is_in_check(pos.side_to_move) {
                return pos;
            }
        }

        Position::startpos()
    }

    /// 1局の対戦シミュレーション
    /// 戻り値: Some(勝者色), None: 引き分け
    pub fn play_game(
        engine_black: &mut SearchEngine,
        engine_white: &mut SearchEngine,
        pos: &mut Position,
        depth: u8,
        max_plies: usize,
    ) -> Option<Color> {
        let mut plies = 0;

        while plies < max_plies {
            if pos.repetition_count() >= 4 {
                return None; // 千日手引き分け
            }

            let turn = pos.side_to_move;
            let (best_mv, score) = match turn {
                Color::Black => engine_black.search_fixed_depth(pos, depth),
                Color::White => engine_white.search_fixed_depth(pos, depth),
            };

            let mv = match best_mv {
                Some(m) => m,
                None => return Some(turn.opposite()), // 合法手なし (投了/詰み)
            };

            // 投了判定 (詰みスコア または -25,000 点以下)
            if score <= -25_000 {
                return Some(turn.opposite());
            }

            pos.do_move(mv);
            plies += 1;
        }

        None // 最大手数到達 (引き分け)
    }
}
