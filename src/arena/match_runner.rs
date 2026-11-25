use super::sprt::{
    PromotionVerdict, Sprt, SprtConfig, SprtStatus, elo_diff_from_win_rate,
    evaluate_promotion_verdict, win_rate,
};
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
    /// Generation namespace for deterministic, generation-varying opening seeds.
    pub generation: usize,
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
            generation: 0,
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

impl MatchResult {
    /// SPRT と最低対局数に基づく昇格判定結果
    pub fn promotion_verdict(&self, min_promotion_games: usize) -> PromotionVerdict {
        evaluate_promotion_verdict(self.sprt.as_ref(), self.total_games, min_promotion_games)
    }
}

/// SPRT が `Continue` のまま有望な勝率を維持している場合の動的延長ポリシー
#[derive(Debug, Clone, Copy)]
pub struct OvertimePolicy {
    /// 延長を許可する下限勝率 (例: 0.52)
    pub min_win_rate: f64,
    /// 1 回の延長ごとに追加するペア数
    pub step_pairs: usize,
    /// 延長を含めた最大ペア数上限
    pub max_pairs: usize,
}

impl OvertimePolicy {
    /// 自律ループ標準の動的延長ポリシー（勝率 52% 以上で `base_pairs` ずつ最大 3 倍まで延長）
    pub fn standard(base_pairs: usize) -> Self {
        Self {
            min_win_rate: 0.52,
            step_pairs: base_pairs,
            max_pairs: base_pairs.saturating_mul(3),
        }
    }

    /// 次の目標ペア数を算出（延長不要または上限到達時は `None`）
    pub fn next_target_pairs(&self, result: &MatchResult, current_pairs: usize) -> Option<usize> {
        let sprt = result.sprt.as_ref()?;
        if sprt.status == SprtStatus::Continue
            && result.win_rate_a >= self.min_win_rate
            && self.step_pairs > 0
            && current_pairs < self.max_pairs
        {
            Some(
                current_pairs
                    .saturating_add(self.step_pairs)
                    .min(self.max_pairs),
            )
        } else {
            None
        }
    }
}

pub struct MatchRunner;

impl MatchRunner {
    /// 2つのモデル間の先後交代ペアマッチを並列実行
    pub fn run_match(config: &MatchConfig) -> MatchResult {
        Self::run_match_extended(config, None, config.pairs)
    }

    /// 先後交代ペアマッチを実行し、必要に応じて SPRT 動的延長を累積実行する
    pub fn run_match_with_overtime(
        config: &MatchConfig,
        overtime: Option<OvertimePolicy>,
    ) -> MatchResult {
        let mut current_pairs = config.pairs;
        let mut match_res = Self::run_match_extended(config, None, current_pairs);

        if let Some(policy) = overtime {
            while let Some(next_pairs) = policy.next_target_pairs(&match_res, current_pairs) {
                if next_pairs <= current_pairs {
                    break;
                }
                current_pairs = next_pairs;
                if let Some(ref sprt) = match_res.sprt {
                    println!(
                        "\n[SPRT Overtime] Indecisive Continue with positive win rate {:.1}% (LLR: {:.2}). Incrementally extending to {} pairs...",
                        match_res.win_rate_a * 100.0,
                        sprt.llr,
                        current_pairs
                    );
                }
                match_res = Self::run_match_extended(config, Some(&match_res), current_pairs);
            }
        }

        match_res
    }

    /// 既存の検定状態（勝敗、SPRT、進行ペア番号）を引き継ぎ、追加ペア分だけを実行して累積集計する
    fn run_match_extended(
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
                        let seed = Self::opening_seed(config.generation, p_idx);
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
                            let wr = win_rate(cur_wa, cur_wb, cur_dr) * 100.0;
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
        let win_rate_a = win_rate(wins_a, wins_b, draws);
        let elo_diff_a = elo_diff_from_win_rate(win_rate_a);

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

    /// Deterministic seed namespace for one generation/pair.
    /// Both color-swapped games in a pair consume the same generated position.
    pub fn opening_seed(generation: usize, pair_index: usize) -> u64 {
        let mut z = 0x9e3779b97f4a7c15u64
            ^ (generation as u64).wrapping_mul(0x517cc1b727220a95)
            ^ (pair_index as u64).wrapping_mul(0xbf58476d1ce4e5b9);
        // SplitMix64 finalizer prevents adjacent generations/pairs from sharing correlated seeds.
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn continue_result(wins_a: usize, wins_b: usize) -> MatchResult {
        let mut sprt = Sprt::new(SprtConfig {
            elo0: 0.0,
            elo1: 50.0,
            alpha: 0.05,
            beta: 0.05,
            min_games: 100,
        });
        sprt.record_batch(wins_a, wins_b, 0);
        let wr = win_rate(wins_a, wins_b, 0);
        MatchResult {
            wins_a,
            wins_b,
            draws: 0,
            total_games: wins_a + wins_b,
            win_rate_a: wr,
            elo_diff_a: elo_diff_from_win_rate(wr),
            sprt: Some(sprt),
        }
    }

    #[test]
    fn overtime_policy_extends_only_when_continue_positive_and_below_cap() {
        let policy = OvertimePolicy::standard(15);
        assert_eq!(policy.max_pairs, 45);

        let promising = continue_result(16, 14); // 53.3% >= 52%
        assert_eq!(policy.next_target_pairs(&promising, 15), Some(30));
        assert_eq!(policy.next_target_pairs(&promising, 30), Some(45));
        assert_eq!(policy.next_target_pairs(&promising, 45), None);

        let weak = continue_result(15, 15); // 50.0% < 52%
        assert_eq!(policy.next_target_pairs(&weak, 15), None);

        let mut decided = promising.clone();
        decided.sprt.as_mut().unwrap().status = SprtStatus::Pass;
        assert_eq!(policy.next_target_pairs(&decided, 15), None);

        let zero_policy = OvertimePolicy::standard(0);
        assert_eq!(zero_policy.next_target_pairs(&promising, 0), None);
    }

    #[test]
    fn run_match_with_overtime_executes_incremental_extension_cumulatively() {
        let config = MatchConfig {
            name_a: "A".to_string(),
            name_b: "B".to_string(),
            eval_a: EvalMode::Hce,
            eval_b: EvalMode::Hce,
            pairs: 1,
            depth: 1,
            threads: 1,
            generation: 1,
            random_opening: 2,
            max_plies: 12,
            tt_size_mb: 2,
            sprt_config: Some(SprtConfig {
                elo0: 0.0,
                elo1: 50.0,
                alpha: 0.05,
                beta: 0.05,
                min_games: 100, // Keep SPRT in Continue so overtime triggers deterministically
            }),
        };
        let policy = OvertimePolicy {
            min_win_rate: 0.0,
            step_pairs: 1,
            max_pairs: 2,
        };

        let result = MatchRunner::run_match_with_overtime(&config, Some(policy));
        assert_eq!(result.total_games, 4);
        assert_eq!(result.wins_a + result.wins_b + result.draws, 4);
        let sprt = result.sprt.expect("SPRT tracker should be present");
        assert_eq!(sprt.total_games(), 4);
        assert_eq!(sprt.status, SprtStatus::Continue);
    }
}
