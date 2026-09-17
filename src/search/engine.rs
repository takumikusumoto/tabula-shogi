use super::ordering::MoveOrderer;
use super::time_mgr::{TimeControl, TimeManager};
use super::tt::{NodeType, TranspositionTable};
use crate::board::Position;
use crate::book::OpeningBook;
use crate::movegen::MoveGenerator;
use crate::types::Move;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) const INF: i32 = 30_000;
pub(crate) const MATE_SCORE: i32 = 28_000;

// --- 探索パラメータ定数 ---
const TIME_CHECK_INTERVAL: u64 = 1024;
const ASPIRATION_DELTA: i32 = 35;
const REVERSE_FUTILITY_MARGIN: i32 = 120; // depth あたり
const FUTILITY_MARGIN: i32 = 180;
pub(crate) const MATE_SCORE_TT_MARGIN: i32 = 500;

/// 探索結果の詳細種別（通常完了、探索中断、定跡ヒット、詰み証明、合法手なし）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchOutcome {
    /// 反復深化が指定深さまで正常に完了
    Completed {
        depth: u8,
        score: i32,
        best_move: Option<Move>,
    },
    /// ノード数制限または時間切れにより反復深化が途中で中断
    Aborted {
        completed_depth: u8,
        fallback_score: i32,
        best_move: Option<Move>,
    },
    /// 定跡データベースによる着手（探索未実施）
    Book { best_move: Move },
    /// df-pn 詰み探索による証明完了
    MateProven { score: i32, best_move: Option<Move> },
    /// 合法手なし（投了局面）
    NoLegalMoves,
}

impl SearchOutcome {
    #[inline]
    pub fn best_move(&self) -> Option<Move> {
        match *self {
            SearchOutcome::Completed { best_move, .. } => best_move,
            SearchOutcome::Aborted { best_move, .. } => best_move,
            SearchOutcome::Book { best_move } => Some(best_move),
            SearchOutcome::MateProven { best_move, .. } => best_move,
            SearchOutcome::NoLegalMoves => None,
        }
    }

    #[inline]
    pub fn score(&self) -> i32 {
        match *self {
            SearchOutcome::Completed { score, .. } => score,
            SearchOutcome::Aborted { fallback_score, .. } => fallback_score,
            SearchOutcome::Book { .. } => 0,
            SearchOutcome::MateProven { score, .. } => score,
            SearchOutcome::NoLegalMoves => -MATE_SCORE,
        }
    }

    /// 深読み再評価（ラベル生成）に安全に採用できる評価値を取得
    /// 定跡（Book）や探索中断（Aborted）は信頼できる深さの評価値ではないため絶対に採用しない
    #[inline]
    pub fn reliable_score_for_relabel(&self, min_depth: u8) -> Option<i32> {
        match *self {
            SearchOutcome::Completed { depth, score, .. } if depth >= min_depth => Some(score),
            SearchOutcome::MateProven { score, .. } => Some(score),
            _ => None,
        }
    }
}

/// 探索結果の簡易構造体（後方互換性用）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchResult {
    pub best_move: Option<Move>,
    pub score: i32,
    pub completed_depth: u8,
}

pub struct SearchContext<'a> {
    pub time_mgr: &'a TimeManager,
    pub stop_flag: &'a Arc<AtomicBool>,
}

pub struct SearchEngine {
    pub tt: Arc<TranspositionTable>,
    pub(crate) nodes: u64,
    pub(crate) killer_moves: [[Option<Move>; 2]; 64],
    pub(crate) counter_moves: [[Option<Move>; 81]; 81],
    pub(crate) history: [[i32; 81]; 81],
    pub eval_mode: crate::eval::EvalMode,
    pub max_nodes: Option<u64>,
    pub use_book: bool,
    pub halfkp_accumulators: Vec<crate::eval::HalfKPAccumulator>,
}

impl SearchEngine {
    pub fn new(tt_size_mb: usize) -> Self {
        let mut engine = SearchEngine {
            tt: Arc::new(TranspositionTable::new(tt_size_mb)),
            nodes: 0,
            killer_moves: [[None; 2]; 64],
            counter_moves: [[None; 81]; 81],
            history: [[0; 81]; 81],
            eval_mode: crate::eval::EvalMode::Hce,
            max_nodes: None,
            use_book: true,
            halfkp_accumulators: vec![crate::eval::HalfKPAccumulator::empty(); 128],
        };
        engine.reset_heuristics();
        engine
    }

    /// 探索ノード数を取得
    pub fn nodes(&self) -> u64 {
        self.nodes
    }

    pub fn with_shared_tt(tt: Arc<TranspositionTable>) -> Self {
        let mut engine = SearchEngine {
            tt,
            nodes: 0,
            killer_moves: [[None; 2]; 64],
            counter_moves: [[None; 81]; 81],
            history: [[0; 81]; 81],
            eval_mode: crate::eval::EvalMode::Hce,
            max_nodes: None,
            use_book: true,
            halfkp_accumulators: vec![crate::eval::HalfKPAccumulator::empty(); 128],
        };
        engine.reset_heuristics();
        engine
    }

    pub fn with_eval_mode(mut self, eval_mode: crate::eval::EvalMode) -> Self {
        self.eval_mode = eval_mode;
        self
    }

    /// ルート局面でのアキュムレータ初期化 (探索開始時に一度だけ全計算)
    pub fn init_root_accumulator(&mut self, pos: &Position) {
        if let crate::eval::EvalMode::HalfKP(ref halfkp) = self.eval_mode {
            self.halfkp_accumulators[0] = halfkp.compute_accumulators_full(pos);
        }
    }

    /// do_move 直後の高速アキュムレータ差分更新ヘルパー
    #[inline(always)]
    pub fn update_accumulator_after_move_at_ply(
        &mut self,
        pos: &Position,
        mv: Move,
        next_ply: usize,
    ) {
        if let crate::eval::EvalMode::HalfKP(ref halfkp) = self.eval_mode
            && next_ply < self.halfkp_accumulators.len()
            && next_ply > 0
        {
            self.halfkp_accumulators[next_ply] = self.halfkp_accumulators[next_ply - 1];
            halfkp.update_accumulator_after_move(&mut self.halfkp_accumulators[next_ply], pos, mv);
        }
    }

    /// ply指定による高速局面評価 (HalfKP差分アキュムレータ活用)
    #[inline(always)]
    pub fn evaluate_at_ply(&self, pos: &Position, ply: usize) -> i32 {
        match &self.eval_mode {
            crate::eval::EvalMode::Hce => crate::eval::Evaluator::evaluate(pos),
            crate::eval::EvalMode::Nnue(nnue) => nnue.evaluate(pos),
            crate::eval::EvalMode::HalfKP(halfkp) => {
                if ply < self.halfkp_accumulators.len() && self.halfkp_accumulators[ply].computed[0]
                {
                    halfkp.evaluate_with_accumulator(pos, &self.halfkp_accumulators[ply])
                } else {
                    halfkp.evaluate(pos)
                }
            }
        }
    }

    #[inline(always)]
    pub fn evaluate(&self, pos: &Position) -> i32 {
        self.eval_mode.evaluate(pos)
    }

    /// ヒューリスティクステーブルを全リセット
    fn reset_heuristics(&mut self) {
        self.nodes = 0;
        self.killer_moves = [[None; 2]; 64];
        self.counter_moves = [[None; 81]; 81];
        self.history = [[0; 81]; 81];
    }

    pub fn clear(&mut self) {
        self.tt.clear();
        self.reset_heuristics();
    }

    /// 自己対局・再評価用の固定深さ詳細探索 (結果種別 SearchOutcome を直接返却)
    pub fn search_fixed_depth_outcome(
        &mut self,
        pos: &mut Position,
        target_depth: u8,
    ) -> SearchOutcome {
        // 1. 定跡データベースの照会 (use_book が有効な場合のみ。探索スコアではなく純粋な定跡手として返却)
        if self.use_book
            && let Some(book_move) = OpeningBook::probe(pos)
        {
            let legal_moves = MoveGenerator::generate_legal_moves(pos);
            if legal_moves.contains(&book_move) {
                return SearchOutcome::Book {
                    best_move: book_move,
                };
            }
        }

        // 2. 手番側が相手玉を攻められる局面で詰み探索を実行する。
        let us = pos.side_to_move;
        let opp = us.opposite();
        let opp_king_sq = pos.king_sq[opp.index()];
        let near_king_threat = opp_king_sq.is_some_and(|ks| {
            (0..81).any(|sq_idx| {
                if let Some(p) = pos.board[sq_idx]
                    && p.color == us
                {
                    let sq = crate::types::Square::from_index(sq_idx);
                    let file_diff = (sq.file() as i8 - ks.file() as i8).abs();
                    let rank_diff = (sq.rank() as i8 - ks.rank() as i8).abs();
                    return file_diff <= 2 && rank_diff <= 2;
                }
                false
            })
        });

        if near_king_threat
            || pos.is_in_check(opp)
            || !MoveGenerator::generate_checks(pos).is_empty()
        {
            let mut dfpn = super::dfpn::DfpnSolver::new(5_000);
            let (is_mate, mate_move) = dfpn.solve(pos);
            if is_mate && let Some(mv) = mate_move {
                return SearchOutcome::MateProven {
                    best_move: Some(mv),
                    score: MATE_SCORE - 1,
                };
            }
        }

        self.nodes = 0;
        self.init_root_accumulator(pos);
        let stop_flag = Arc::new(AtomicBool::new(false));
        let tc = TimeControl {
            infinite: true,
            ..Default::default()
        };
        let time_mgr = TimeManager::new(&tc, pos.side_to_move);
        let ctx = SearchContext {
            time_mgr: &time_mgr,
            stop_flag: &stop_flag,
        };

        let mut root_moves = MoveGenerator::generate_legal_moves(pos);
        if root_moves.is_empty() {
            return SearchOutcome::NoLegalMoves;
        }

        let mut best_move = root_moves[0];
        // 深さ1すら完了しなかった場合の安全なフォールバック。
        let mut best_score = self.evaluate(pos);
        let mut completed_depth = 0;
        let mut was_aborted = false;

        'deepening: for depth in 1..=target_depth {
            let mut alpha = -INF;
            let mut beta = INF;

            if depth >= 4 && best_score.abs() < MATE_SCORE - 200 {
                alpha = (best_score - ASPIRATION_DELTA).max(-INF);
                beta = (best_score + ASPIRATION_DELTA).min(INF);
            }

            let tt_move = self.tt.probe(pos.hash).and_then(|e| e.best_move);
            let prev_mv = pos.history.last().map(|rec| rec.mv);
            let counter_mv = prev_mv.and_then(|pm| {
                pm.from()
                    .and_then(|from_sq| self.counter_moves[from_sq.index()][pm.to().index()])
            });
            MoveOrderer::order_moves(
                &mut root_moves,
                pos,
                tt_move,
                &[None, None],
                counter_mv,
                Some(&self.history),
            );

            let mut loop_best_move = None;
            loop {
                let mut current_alpha = alpha;
                for &mv in &root_moves {
                    pos.do_move(mv);
                    self.update_accumulator_after_move_at_ply(pos, mv, 1);
                    let score = -self.negamax(pos, depth - 1, -beta, -current_alpha, 1, true, &ctx);
                    pos.undo_move();

                    // 中断した深さの結果は採用せず、ルートTTにも保存しない。
                    if stop_flag.load(Ordering::Relaxed) {
                        was_aborted = true;
                        break 'deepening;
                    }

                    if score > current_alpha {
                        current_alpha = score;
                        loop_best_move = Some(mv);
                    }
                    if current_alpha >= beta {
                        break;
                    }
                }

                if current_alpha <= alpha {
                    if alpha == -INF {
                        best_score = current_alpha;
                        if let Some(bm) = loop_best_move {
                            best_move = bm;
                        }
                        break;
                    }
                    alpha = -INF;
                } else if current_alpha >= beta {
                    if beta == INF {
                        best_score = current_alpha;
                        if let Some(bm) = loop_best_move {
                            best_move = bm;
                        }
                        break;
                    }
                    beta = INF;
                } else {
                    best_score = current_alpha;
                    if let Some(bm) = loop_best_move {
                        best_move = bm;
                    }
                    break;
                }
            }

            self.tt.store(
                pos.hash,
                depth,
                Self::score_to_tt(best_score, 0),
                NodeType::Exact,
                Some(best_move),
            );

            completed_depth = depth;

            if best_score.abs() >= MATE_SCORE - 100 {
                break;
            }
        }

        if was_aborted {
            SearchOutcome::Aborted {
                completed_depth,
                fallback_score: best_score,
                best_move: Some(best_move),
            }
        } else {
            SearchOutcome::Completed {
                depth: completed_depth,
                score: best_score,
                best_move: Some(best_move),
            }
        }
    }

    /// 自己対局・再評価用の固定深さ詳細探索 (後方互換性ラッパー)
    pub fn search_fixed_depth_detail(
        &mut self,
        pos: &mut Position,
        target_depth: u8,
    ) -> SearchResult {
        let outcome = self.search_fixed_depth_outcome(pos, target_depth);
        let completed_depth = match outcome {
            SearchOutcome::Completed { depth, .. } => depth,
            SearchOutcome::Aborted {
                completed_depth, ..
            } => completed_depth,
            SearchOutcome::Book { .. } => 0, // 定跡は探索未実施のため完了深度ゼロ (偽装排除)
            SearchOutcome::MateProven { .. } => target_depth.max(1),
            SearchOutcome::NoLegalMoves => target_depth,
        };
        SearchResult {
            best_move: outcome.best_move(),
            score: outcome.score(),
            completed_depth,
        }
    }

    /// 自己対局用の固定深さ探索 (USI標準出力を行わず、最善手と探索スコアを返却)
    pub fn search_fixed_depth(
        &mut self,
        pos: &mut Position,
        target_depth: u8,
    ) -> (Option<Move>, i32) {
        let outcome = self.search_fixed_depth_outcome(pos, target_depth);
        (outcome.best_move(), outcome.score())
    }

    /// 各合法手の浅い探索評価値に基づき、ソフトマックス温度サンプリングによって着手を選択する
    pub fn search_with_temperature(
        &mut self,
        pos: &mut Position,
        target_depth: u8,
        temperature: f32,
        rng_seed: u64,
    ) -> (Option<Move>, i32) {
        // 1. 詰み探索 (df-pn Solver) の即時判定
        let mut dfpn = super::dfpn::DfpnSolver::new(20_000);
        let (is_mate, mate_move) = dfpn.solve(pos);
        if is_mate && let Some(mv) = mate_move {
            return (Some(mv), MATE_SCORE - 1);
        }

        let legal_moves = MoveGenerator::generate_legal_moves(pos);
        if legal_moves.is_empty() {
            return (None, -MATE_SCORE);
        }

        if legal_moves.len() == 1 || temperature <= 0.01 {
            return self.search_fixed_depth(pos, target_depth);
        }

        self.nodes = 0;
        self.init_root_accumulator(pos);
        let stop_flag = Arc::new(AtomicBool::new(false));
        let tc = TimeControl {
            infinite: true,
            ..Default::default()
        };
        let time_mgr = TimeManager::new(&tc, pos.side_to_move);
        let ctx = SearchContext {
            time_mgr: &time_mgr,
            stop_flag: &stop_flag,
        };

        // 各合法手を探索して評価値を取得
        let child_depth = target_depth.saturating_sub(1);
        let mut scored_moves: Vec<(Move, i32)> = Vec::with_capacity(legal_moves.len());

        for &mv in &legal_moves {
            pos.do_move(mv);
            self.update_accumulator_after_move_at_ply(pos, mv, 1);
            let score = -self.negamax(pos, child_depth, -INF, INF, 1, true, &ctx);
            pos.undo_move();
            scored_moves.push((mv, score));
        }

        // 最善スコアの検出
        let best_score = scored_moves.iter().map(|(_, s)| *s).max().unwrap_or(-INF);

        // 詰みスコアがある場合は最善手を即採用
        if best_score >= MATE_SCORE - 200
            && let Some(&(best_mv, score)) = scored_moves.iter().find(|(_, s)| *s == best_score)
        {
            return (Some(best_mv), score);
        }

        // 温度スケーリング (歩1枚 = 100cp を基準とする)
        let t_cp = (temperature * 100.0).max(1.0);

        // ボルツマン重みの計算 (最善手から 350cp 以上劣る大悪手は除外)
        let prune_threshold = best_score - 350;
        let mut weights: Vec<f32> = Vec::with_capacity(scored_moves.len());

        for &(_, score) in &scored_moves {
            if score < prune_threshold {
                weights.push(0.0);
            } else {
                let delta = (score - best_score) as f32 / t_cp;
                weights.push(delta.exp());
            }
        }

        let total_weight: f32 = weights.iter().sum();
        if total_weight <= 0.0 {
            let best_entry = scored_moves.into_iter().max_by_key(|(_, s)| *s);
            return (best_entry.map(|(m, _)| m), best_score);
        }

        // 乱数による確率サンプリング
        let r_val = (rng_seed as f64) / (u64::MAX as f64);
        let mut target = (r_val as f32) * total_weight;

        for (i, &(mv, _score)) in scored_moves.iter().enumerate() {
            target -= weights[i];
            if target <= 0.0 {
                return (Some(mv), best_score);
            }
        }

        let best_entry = scored_moves.into_iter().max_by_key(|(_, s)| *s);
        (best_entry.map(|(m, _)| m), best_score)
    }

    /// 反復深化探索のエントリポイント
    pub fn search(
        &mut self,
        pos: &mut Position,
        tc: &TimeControl,
        stop_flag: Arc<AtomicBool>,
    ) -> Option<Move> {
        // 1. 定跡データベース (Opening Book) の照会
        if let Some(book_move) = OpeningBook::probe(pos) {
            let legal_moves = MoveGenerator::generate_legal_moves(pos);
            if legal_moves.contains(&book_move) {
                println!("info string book move found: {}", book_move.to_usi());
                return Some(book_move);
            } else {
                eprintln!(
                    "Warning: Book move {} is not legal in current position! Falling back to search.",
                    book_move.to_usi()
                );
            }
        }

        // 2. 詰み探索 (df-pn Solver) の即時判定
        // 持ち時間や手数に余裕がある場合、自玉／敵玉の詰みを即座に解決
        let mut dfpn = super::dfpn::DfpnSolver::new(50_000);
        let (is_mate, mate_move) = dfpn.solve(pos);
        if is_mate && let Some(mv) = mate_move {
            println!("info score mate 1 pv {}", mv.to_usi());
            return Some(mv);
        }

        self.nodes = 0;
        self.init_root_accumulator(pos);
        let time_mgr = TimeManager::new(tc, pos.side_to_move);
        let ctx = SearchContext {
            time_mgr: &time_mgr,
            stop_flag: &stop_flag,
        };

        // 合法手を生成
        let mut root_moves = MoveGenerator::generate_legal_moves(pos);
        if root_moves.is_empty() {
            return None; // 投了
        }

        let mut best_move = root_moves[0];
        let mut best_score = -INF;
        let max_depth: u8 = 32;

        for depth in 1..=max_depth {
            // 時間切れチェック (反復深化の継続判断)
            if depth > 1
                && (!time_mgr.should_continue_deepening() || stop_flag.load(Ordering::Relaxed))
            {
                break;
            }

            let mut current_best_move = None;
            let mut alpha = -INF;
            let mut beta = INF;

            // Aspiration Windows (depth >= 4 で吸引窓探索)
            if depth >= 4 && best_score.abs() < MATE_SCORE - 200 {
                alpha = (best_score - ASPIRATION_DELTA).max(-INF);
                beta = (best_score + ASPIRATION_DELTA).min(INF);
            }

            // ルートの手のオーダリング
            let tt_move = self.tt.probe(pos.hash).and_then(|e| e.best_move);
            let prev_mv = pos.history.last().map(|rec| rec.mv);
            let counter_mv = prev_mv.and_then(|pm| {
                pm.from()
                    .and_then(|from_sq| self.counter_moves[from_sq.index()][pm.to().index()])
            });
            MoveOrderer::order_moves(
                &mut root_moves,
                pos,
                tt_move,
                &[None, None],
                counter_mv,
                Some(&self.history),
            );

            let mut abort = false;

            loop {
                let mut current_alpha = alpha;
                let mut loop_best_move = None;

                for &mv in &root_moves {
                    pos.do_move(mv);
                    self.update_accumulator_after_move_at_ply(pos, mv, 1);
                    let score = -self.negamax(pos, depth - 1, -beta, -current_alpha, 1, true, &ctx);
                    pos.undo_move();

                    if stop_flag.load(Ordering::Relaxed) {
                        abort = true;
                        break;
                    }

                    if score > current_alpha {
                        current_alpha = score;
                        loop_best_move = Some(mv);
                    }

                    // 設計レビュー F04: fail-high時は子の探索窓逆転を防ぐため即座にループを脱出
                    if current_alpha >= beta {
                        break;
                    }
                }

                if abort {
                    break;
                }

                // 窓探索の合致判定
                if current_alpha <= alpha {
                    if alpha == -INF {
                        current_best_move = loop_best_move;
                        best_score = current_alpha;
                        break;
                    }
                    // fail-low: 窓を下方に広げて再探索
                    alpha = -INF;
                } else if current_alpha >= beta {
                    if beta == INF {
                        current_best_move = loop_best_move;
                        best_score = current_alpha;
                        break;
                    }
                    // fail-high: 窓を上方に広げて再探索
                    beta = INF;
                } else {
                    current_best_move = loop_best_move;
                    best_score = current_alpha;
                    break;
                }
            }

            if abort {
                break;
            }

            if let Some(bm) = current_best_move {
                best_move = bm;

                // 置換表にルート局面の結果を保存
                self.tt.store(
                    pos.hash,
                    depth,
                    Self::score_to_tt(best_score, 0),
                    NodeType::Exact,
                    Some(best_move),
                );

                // PV(読み筋)の構築
                let pv = self.extract_pv(pos, best_move, depth);

                // USI info 出力
                let elapsed_ms = time_mgr.elapsed_ms().max(1);
                let nps = (self.nodes as u128 * 1000) / elapsed_ms as u128;
                let pv_str = pv.iter().map(|m| m.to_usi()).collect::<Vec<_>>().join(" ");

                let score_str = if best_score.abs() >= MATE_SCORE - 100 {
                    let mate_in = if best_score > 0 {
                        ((MATE_SCORE - best_score) + 1) / 2
                    } else {
                        -((MATE_SCORE + best_score) + 1) / 2
                    };
                    format!("score mate {mate_in}")
                } else {
                    format!("score cp {best_score}")
                };

                println!(
                    "info depth {} {} time {} nodes {} nps {} pv {}",
                    depth, score_str, elapsed_ms, self.nodes, nps, pv_str
                );

                // 詰みが見つかった場合は探索終了
                if best_score.abs() >= MATE_SCORE - 100 {
                    break;
                }
            }
        }

        Some(best_move)
    }

    /// Lazy SMP ヘルパースレッド用探索ルーチン
    /// メインスレッドとは異なる深さ・探索順序で共有置換表を更新する
    pub fn search_helper(
        &mut self,
        pos: &mut Position,
        tc: &TimeControl,
        stop_flag: Arc<AtomicBool>,
        thread_id: usize,
    ) {
        self.nodes = 0;
        self.init_root_accumulator(pos);
        let time_mgr = TimeManager::new(tc, pos.side_to_move);
        let ctx = SearchContext {
            time_mgr: &time_mgr,
            stop_flag: &stop_flag,
        };

        let mut root_moves = MoveGenerator::generate_legal_moves(pos);
        if root_moves.is_empty() {
            return;
        }

        // スレッドごとに初期深さオフセットを分散 (Lazy SMP standard)
        let start_depth = (thread_id % 2) as u8 + 1;
        let max_depth: u8 = 32;

        for depth in start_depth..=max_depth {
            if depth > 1
                && (!time_mgr.should_continue_deepening() || stop_flag.load(Ordering::Relaxed))
            {
                break;
            }

            let tt_move = self.tt.probe(pos.hash).and_then(|e| e.best_move);
            let prev_mv = pos.history.last().map(|rec| rec.mv);
            let counter_mv = prev_mv.and_then(|pm| {
                pm.from()
                    .and_then(|from_sq| self.counter_moves[from_sq.index()][pm.to().index()])
            });
            MoveOrderer::order_moves(
                &mut root_moves,
                pos,
                tt_move,
                &[None, None],
                counter_mv,
                Some(&self.history),
            );

            let mut best_score = -INF;
            let mut best_move = None;

            for &mv in &root_moves {
                if stop_flag.load(Ordering::Relaxed) {
                    return;
                }

                pos.do_move(mv);
                self.update_accumulator_after_move_at_ply(pos, mv, 1);
                let score = -self.negamax(pos, depth - 1, -INF, INF, 1, true, &ctx);
                pos.undo_move();

                if score > best_score {
                    best_score = score;
                    best_move = Some(mv);
                }
            }

            if let Some(bm) = best_move {
                self.tt.store(
                    pos.hash,
                    depth,
                    Self::score_to_tt(best_score, 0),
                    NodeType::Exact,
                    Some(bm),
                );
            }
        }
    }

    #[inline]
    fn score_to_tt(score: i32, ply: usize) -> i32 {
        if score >= MATE_SCORE - MATE_SCORE_TT_MARGIN {
            score + (ply as i32)
        } else if score <= -MATE_SCORE + MATE_SCORE_TT_MARGIN {
            score - (ply as i32)
        } else {
            score
        }
    }

    #[inline]
    fn score_from_tt(score: i32, ply: usize) -> i32 {
        if score >= MATE_SCORE - MATE_SCORE_TT_MARGIN {
            score - (ply as i32)
        } else if score <= -MATE_SCORE + MATE_SCORE_TT_MARGIN {
            score + (ply as i32)
        } else {
            score
        }
    }

    /// Negamax + Alpha-Beta探索 (Null Move, LMR, Check Extension)
    #[allow(clippy::too_many_arguments)]
    fn negamax(
        &mut self,
        pos: &mut Position,
        mut depth: u8,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        allow_null: bool,
        ctx: &SearchContext,
    ) -> i32 {
        // 定期的な時間チェック & ノード数制限チェック
        self.nodes += 1;
        if let Some(max_n) = self.max_nodes
            && self.nodes >= max_n
        {
            ctx.stop_flag.store(true, Ordering::Relaxed);
            return self.evaluate_at_ply(pos, ply);
        }
        if self.nodes.is_multiple_of(TIME_CHECK_INTERVAL) && ctx.time_mgr.is_time_up() {
            ctx.stop_flag.store(true, Ordering::Relaxed);
        }
        if ctx.stop_flag.load(Ordering::Relaxed) {
            return 0;
        }

        // 最大探索手数 (MAX_PLY) ガード
        if ply >= 64 {
            return self.evaluate_at_ply(pos, ply);
        }

        // 千日手判定
        if pos.repetition_count() >= 4 {
            return 0; // 引き分けスコア
        }

        let in_check = pos.is_in_check(pos.side_to_move);

        // Check Extension (王手延長: 王手がかかっているときは深さを維持、ただし手数・深さ制限)
        if in_check && depth < 20 && ply < 32 {
            depth += 1;
        }

        // 葉ノード判定: 静止探索へ
        if depth == 0 {
            return self.quiescence(pos, alpha, beta, ply, ctx);
        }

        let orig_alpha = alpha;

        // 置換表プローブ
        let mut tt_move = None;
        if let Some(entry) = self.tt.probe(pos.hash) {
            let tt_score = Self::score_from_tt(entry.score, ply);
            if entry.depth >= depth {
                match entry.node_type {
                    NodeType::Exact => return tt_score,
                    NodeType::LowerBound => {
                        if tt_score >= beta {
                            return tt_score;
                        }
                    }
                    NodeType::UpperBound => {
                        if tt_score <= alpha {
                            return tt_score;
                        }
                    }
                }
            }
            tt_move = entry.best_move;
        }

        // Null Move Pruning (パス枝刈り: R = 2)
        if allow_null && !in_check && depth >= 3 && ply > 0 {
            // do_null_move はハッシュ・手番・手数を一括更新する安全なインターフェース
            pos.do_null_move();
            if ply + 1 < self.halfkp_accumulators.len() {
                self.halfkp_accumulators[ply + 1] = self.halfkp_accumulators[ply];
            }

            let null_score = -self.negamax(
                pos,
                depth.saturating_sub(3),
                -beta,
                -beta + 1,
                ply + 1,
                false,
                ctx,
            );

            pos.undo_null_move();

            if ctx.stop_flag.load(Ordering::Relaxed) {
                return 0;
            }

            if null_score >= beta {
                return beta; // パスしても相手が勝てない優勢局面なのでカットオフ
            }
        }

        // 静的評価値の事前計算 (王手がかかっていない場合)
        let static_eval = if !in_check {
            Some(self.evaluate_at_ply(pos, ply))
        } else {
            None
        };

        // Reverse Futility Pruning (深さ1〜2で静的評価値が十分に高い場合はベータカット)
        if !in_check
            && depth <= 2
            && beta < MATE_SCORE - 200
            && let Some(eval) = static_eval
        {
            let margin = (depth as i32) * REVERSE_FUTILITY_MARGIN;
            if eval - margin >= beta {
                return beta;
            }
        }

        let mut moves = MoveGenerator::generate_legal_moves(pos);
        if moves.is_empty() {
            if in_check {
                // 詰み (手番側の負け)
                return -MATE_SCORE + (ply as i32);
            } else {
                // ステイルメイト（通常将棋では稀）
                return 0;
            }
        }

        // オーダリング
        let killer = if ply < 64 {
            self.killer_moves[ply]
        } else {
            [None, None]
        };
        let prev_rec = pos.history.last().copied();
        let prev_mv = prev_rec.map(|rec| rec.mv);
        let counter_mv = prev_mv.and_then(|pm| {
            pm.from()
                .and_then(|from_sq| self.counter_moves[from_sq.index()][pm.to().index()])
        });
        MoveOrderer::order_moves(
            &mut moves,
            pos,
            tt_move,
            &killer,
            counter_mv,
            Some(&self.history),
        );

        let mut best_move = None;
        let mut best_score = -INF;
        let mut move_count = 0;

        for mv in moves {
            move_count += 1;
            let is_capture = pos.board[mv.to().index()].is_some();
            let is_tactical = mv.is_promote()
                || is_capture
                || (mv.is_drop() && mv.to().is_promoted_zone(pos.side_to_move));

            // SEE Pruning (深さ2以下の静かな手または駒取りで、明らかな大損をスキップ)
            if depth <= 2 && !in_check && move_count > 1 {
                let see_val = super::see::SEE::evaluate(pos, mv);
                if see_val < -(depth as i32 * REVERSE_FUTILITY_MARGIN) {
                    continue;
                }
            }

            // Futility Pruning (深さ1の静かな手で追いつかない場合スキップ)
            if depth == 1
                && !in_check
                && !is_tactical
                && move_count > 1
                && let Some(eval) = static_eval
                && eval + FUTILITY_MARGIN <= alpha
            {
                continue;
            }

            // Recapture Extension (直前の相手の駒取りマスに対する取り返し手は探索深さを延長)
            let is_recapture = if let Some(pr) = prev_rec {
                pr.captured.is_some() && is_capture && mv.to() == pr.mv.to()
            } else {
                false
            };

            pos.do_move(mv);
            self.update_accumulator_after_move_at_ply(pos, mv, ply + 1);

            let mut score;
            let is_pv_move = move_count == 1;

            let ext = if is_recapture && depth <= 6 { 1 } else { 0 };

            if is_pv_move {
                // PVS: 1手目はフルウィンドウで探索
                score = -self.negamax(pos, depth - 1 + ext, -beta, -alpha, ply + 1, true, ctx);
            } else {
                // 2手目以降: LMR 適用深さの計算
                let mut search_depth = depth - 1 + ext;
                if depth >= 3 && move_count >= 4 && !is_tactical && !in_check {
                    search_depth = search_depth.saturating_sub(1);
                }

                // PVS: Null Window (Scout Search) で高速チェック
                score = -self.negamax(pos, search_depth, -alpha - 1, -alpha, ply + 1, true, ctx);

                // LMR 再探索の厳密化 (設計レビュー F03):
                // 1. 深さを削った探索が alpha を超えた場合、まず元の深さで確認再探索
                if score > alpha && search_depth < depth - 1 + ext {
                    score =
                        -self.negamax(pos, depth - 1 + ext, -alpha - 1, -alpha, ply + 1, true, ctx);
                }

                // 2. PVノードにおいて alpha を超えていれば、フルウィンドウで再探索
                if score > alpha && score < beta {
                    score = -self.negamax(pos, depth - 1 + ext, -beta, -alpha, ply + 1, true, ctx);
                }
            }

            pos.undo_move();

            if ctx.stop_flag.load(Ordering::Relaxed) {
                return 0;
            }

            if score > best_score {
                best_score = score;
                best_move = Some(mv);
            }

            if score > alpha {
                alpha = score;
            }

            if alpha >= beta {
                // ベータカット (Beta Cutoff)
                if ply < 64 && !mv.is_drop() && pos.board[mv.to().index()].is_none() {
                    // キラー手の更新
                    if self.killer_moves[ply][0] != Some(mv) {
                        self.killer_moves[ply][1] = self.killer_moves[ply][0];
                        self.killer_moves[ply][0] = Some(mv);
                    }
                    // 応手 (Countermove) の更新
                    if let Some(pm) = prev_mv
                        && let Some(p_from) = pm.from()
                    {
                        self.counter_moves[p_from.index()][pm.to().index()] = Some(mv);
                    }
                    // 歴史ヒューリスティックの加算 (設計レビュー F10: u8オーバーフロー修正)
                    if let Some(from_sq) = mv.from() {
                        let d = depth as i32;
                        self.history[from_sq.index()][mv.to().index()] += d * d;
                    }
                }
                break;
            }
        }

        // 置換表への保存
        let node_type = if best_score >= beta {
            NodeType::LowerBound
        } else if best_score > orig_alpha {
            NodeType::Exact
        } else {
            NodeType::UpperBound
        };

        self.tt.store(
            pos.hash,
            depth,
            Self::score_to_tt(best_score, ply),
            node_type,
            best_move,
        );

        best_score
    }

    /// 静止探索 (Quiescence Search) + SEE Pruning
    fn quiescence(
        &mut self,
        pos: &mut Position,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        ctx: &SearchContext,
    ) -> i32 {
        self.nodes += 1;
        if let Some(max_n) = self.max_nodes
            && self.nodes >= max_n
        {
            ctx.stop_flag.store(true, Ordering::Relaxed);
            return self.evaluate_at_ply(pos, ply);
        }
        if self.nodes.is_multiple_of(TIME_CHECK_INTERVAL) && ctx.time_mgr.is_time_up() {
            ctx.stop_flag.store(true, Ordering::Relaxed);
        }
        if ctx.stop_flag.load(Ordering::Relaxed) {
            return 0;
        }

        if ply >= 64 {
            return self.evaluate_at_ply(pos, ply);
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
                Some(&self.history),
            );

            for mv in evasions {
                pos.do_move(mv);
                self.update_accumulator_after_move_at_ply(pos, mv, ply + 1);
                let score = -self.quiescence(pos, -beta, -alpha, ply + 1, ctx);
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
            return alpha;
        }

        // 王手されていない通常局面: 静的評価（立合いスコア）
        let stand_pat = self.evaluate_at_ply(pos, ply);
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
            Some(&self.history),
        );

        for mv in tactical_moves {
            // SEE Pruning: 駒取り手でSEE < 0（損な取り合い）はスキップ
            if pos.board[mv.to().index()].is_some() && super::see::SEE::evaluate(pos, mv) < 0 {
                continue;
            }

            pos.do_move(mv);
            self.update_accumulator_after_move_at_ply(pos, mv, ply + 1);
            let score = -self.quiescence(pos, -beta, -alpha, ply + 1, ctx);
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

    /// 置換表を辿ってPV（読み筋）を取り出す
    fn extract_pv(&mut self, pos: &mut Position, first_move: Move, max_depth: u8) -> Vec<Move> {
        let mut pv = Vec::new();
        pv.push(first_move);

        let mut moves_done = Vec::new();
        pos.do_move(first_move);
        moves_done.push(first_move);

        // サイクル検出: 訪問済みハッシュを追跡して無限ループを防止 (#38)
        let mut visited = std::collections::HashSet::new();
        visited.insert(pos.hash);

        for _ in 1..max_depth {
            if let Some(entry) = self.tt.probe(pos.hash) {
                if let Some(mv) = entry.best_move {
                    pos.do_move(mv);
                    // ループが生じる局面はPVの打ち切り
                    if !visited.insert(pos.hash) {
                        pos.undo_move();
                        break;
                    }
                    pv.push(mv);
                    moves_done.push(mv);
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        // 盤面を元に戻す
        for _ in 0..moves_done.len() {
            pos.undo_move();
        }

        pv
    }
}
