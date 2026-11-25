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
const MATE_SCORE_TT_MARGIN: i32 = 500;
const MAX_TIMED_DEPTH: u8 = 32;

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

    /// 完了した反復深化の深さを返す（定跡・詰み証明・合法手なし等の非反復深化ケースは 0、ただし MateProven は 1）
    #[inline]
    pub fn completed_depth(&self) -> u8 {
        match *self {
            SearchOutcome::Completed { depth, .. } => depth,
            SearchOutcome::Aborted {
                completed_depth, ..
            } => completed_depth,
            SearchOutcome::Book { .. } | SearchOutcome::NoLegalMoves => 0,
            SearchOutcome::MateProven { .. } => 1,
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

/// 1 反復深化ステップ完了時の進捗情報（USI info 等の上位アダプタ向け）
#[derive(Debug, Clone)]
pub struct SearchInfo {
    pub depth: u8,
    pub score: i32,
    pub elapsed_ms: u64,
    pub nodes: u64,
    pub nps: u64,
    pub pv: Vec<Move>,
}

impl SearchInfo {
    /// 詰みスコアなら `Some(mate_in_plies_signed)`、通常評価値なら `None` を返す
    pub fn mate_in(&self) -> Option<i32> {
        if self.score.abs() >= MATE_SCORE - 100 {
            let mate_in = if self.score > 0 {
                ((MATE_SCORE - self.score) + 1) / 2
            } else {
                -((MATE_SCORE + self.score) + 1) / 2
            };
            Some(mate_in)
        } else {
            None
        }
    }
}

/// 探索中のイベント通知（SearchEngine から標準出力を分離するためのアダプタ境界）
#[derive(Debug, Clone)]
pub enum SearchEvent {
    BookMove(Move),
    IllegalBookMove(Move),
    MateFound(Move),
    Iteration(SearchInfo),
}

pub(super) struct SearchContext<'a> {
    pub(super) time_mgr: &'a TimeManager,
    pub(super) stop_flag: &'a Arc<AtomicBool>,
}

#[derive(Clone, Copy)]
struct DfpnPrecheck {
    max_nodes: u32,
    require_near_king_guard: bool,
}

struct RootSearchSpec {
    start_depth: u8,
    max_depth: u8,
    use_aspiration: bool,
    check_time_budget: bool,
    check_book: bool,
    dfpn_precheck: Option<DfpnPrecheck>,
    stop_on_mate: bool,
    emit_pv: bool,
}

pub struct SearchEngine {
    pub tt: Arc<TranspositionTable>,
    nodes: u64,
    killer_moves: [[Option<Move>; 2]; 64],
    counter_moves: [[Option<Move>; 81]; 81],
    pub(super) history: [[i32; 81]; 81],
    pub eval_mode: crate::eval::EvalMode,
    pub max_nodes: Option<u64>,
    pub use_book: bool,
    halfkp_accumulators: Vec<crate::eval::HalfKPAccumulator>,
}

impl SearchEngine {
    pub fn new(tt_size_mb: usize) -> Self {
        Self::with_shared_tt(Arc::new(TranspositionTable::new(tt_size_mb)))
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
    fn init_root_accumulator(&mut self, pos: &Position) {
        if let crate::eval::EvalMode::HalfKP(ref halfkp) = self.eval_mode {
            self.halfkp_accumulators[0] = halfkp.compute_accumulators_full(pos);
        }
    }

    /// do_move と HalfKP アキュムレータ差分更新・undo_move を組で保証する内部遷移ヘルパー
    #[inline(always)]
    pub(super) fn with_move<R>(
        &mut self,
        pos: &mut Position,
        mv: Move,
        next_ply: usize,
        f: impl FnOnce(&mut Self, &mut Position) -> R,
    ) -> R {
        pos.do_move(mv);
        if let crate::eval::EvalMode::HalfKP(ref halfkp) = self.eval_mode
            && next_ply < self.halfkp_accumulators.len()
            && next_ply > 0
        {
            self.halfkp_accumulators[next_ply] = self.halfkp_accumulators[next_ply - 1];
            halfkp.update_accumulator_after_move(&mut self.halfkp_accumulators[next_ply], pos, mv);
        }
        let result = f(self, pos);
        pos.undo_move();
        result
    }

    /// do_null_move と HalfKP アキュムレータ継承・undo_null_move を組で保証する内部遷移ヘルパー
    #[inline(always)]
    fn with_null_move<R>(
        &mut self,
        pos: &mut Position,
        next_ply: usize,
        f: impl FnOnce(&mut Self, &mut Position) -> R,
    ) -> R {
        pos.do_null_move();
        if matches!(self.eval_mode, crate::eval::EvalMode::HalfKP(_))
            && next_ply < self.halfkp_accumulators.len()
            && next_ply > 0
        {
            self.halfkp_accumulators[next_ply] = self.halfkp_accumulators[next_ply - 1];
        }
        let result = f(self, pos);
        pos.undo_null_move();
        result
    }

    /// ノード数上限・時間切れ・停止フラグ・最大手数ガードの共通チェック (negamax / qsearch 共用)
    #[inline(always)]
    pub(super) fn check_node_and_time(
        &mut self,
        pos: &Position,
        ply: usize,
        ctx: &SearchContext,
    ) -> Option<i32> {
        self.nodes += 1;
        if let Some(max_n) = self.max_nodes
            && self.nodes >= max_n
        {
            ctx.stop_flag.store(true, Ordering::Relaxed);
            return Some(self.evaluate_at_ply(pos, ply));
        }
        if self.nodes.is_multiple_of(TIME_CHECK_INTERVAL) && ctx.time_mgr.is_time_up() {
            ctx.stop_flag.store(true, Ordering::Relaxed);
        }
        if ctx.stop_flag.load(Ordering::Relaxed) {
            return Some(0);
        }
        if ply >= 64 {
            return Some(self.evaluate_at_ply(pos, ply));
        }
        None
    }

    /// ply指定による高速局面評価 (HalfKP差分アキュムレータ活用)
    #[inline(always)]
    pub(super) fn evaluate_at_ply(&self, pos: &Position, ply: usize) -> i32 {
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
        for acc in &mut self.halfkp_accumulators {
            *acc = crate::eval::HalfKPAccumulator::empty();
        }
    }

    /// 自己対局・再評価用の固定深さ詳細探索 (結果種別 SearchOutcome を直接返却)
    pub fn search_fixed_depth_outcome(
        &mut self,
        pos: &mut Position,
        target_depth: u8,
    ) -> SearchOutcome {
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
        let spec = RootSearchSpec {
            start_depth: 1,
            max_depth: target_depth,
            use_aspiration: true,
            check_time_budget: false,
            check_book: self.use_book,
            dfpn_precheck: Some(DfpnPrecheck {
                max_nodes: 5_000,
                require_near_king_guard: true,
            }),
            stop_on_mate: true,
            emit_pv: false,
        };
        self.run_root_search(pos, &spec, &ctx, |_| {})
    }

    /// 自己対局用の固定深さ探索 (最善手と探索スコアを返却)
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
        if let Some(mv) = Self::probe_dfpn_mate(
            pos,
            DfpnPrecheck {
                max_nodes: 20_000,
                require_near_king_guard: false,
            },
        ) {
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
            let score = self.with_move(pos, mv, 1, |eng, p| {
                -eng.negamax(p, child_depth, -INF, INF, 1, true, &ctx)
            });
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

    /// 時間制御付きの反復深化探索（進捗イベントをアダプタへ通知）
    pub fn search_with_events(
        &mut self,
        pos: &mut Position,
        tc: &TimeControl,
        stop_flag: Arc<AtomicBool>,
        on_event: impl FnMut(SearchEvent),
    ) -> SearchOutcome {
        let time_mgr = TimeManager::new(tc, pos.side_to_move);
        let ctx = SearchContext {
            time_mgr: &time_mgr,
            stop_flag: &stop_flag,
        };
        let spec = RootSearchSpec {
            start_depth: 1,
            max_depth: MAX_TIMED_DEPTH,
            use_aspiration: true,
            check_time_budget: true,
            check_book: self.use_book,
            dfpn_precheck: Some(DfpnPrecheck {
                max_nodes: 50_000,
                require_near_king_guard: false,
            }),
            stop_on_mate: true,
            emit_pv: true,
        };
        self.run_root_search(pos, &spec, &ctx, on_event)
    }

    /// 時間制御付きの反復深化探索（標準出力や不要な PV 抽出を行わず最善手のみ返却）
    pub fn search(
        &mut self,
        pos: &mut Position,
        tc: &TimeControl,
        stop_flag: Arc<AtomicBool>,
    ) -> Option<Move> {
        let time_mgr = TimeManager::new(tc, pos.side_to_move);
        let ctx = SearchContext {
            time_mgr: &time_mgr,
            stop_flag: &stop_flag,
        };
        let spec = RootSearchSpec {
            start_depth: 1,
            max_depth: MAX_TIMED_DEPTH,
            use_aspiration: true,
            check_time_budget: true,
            check_book: self.use_book,
            dfpn_precheck: Some(DfpnPrecheck {
                max_nodes: 50_000,
                require_near_king_guard: false,
            }),
            stop_on_mate: true,
            emit_pv: false,
        };
        self.run_root_search(pos, &spec, &ctx, |_| {}).best_move()
    }

    /// Lazy SMP ヘルパースレッド用探索ルーチン
    /// メインスレッドとは異なる初期深さで共有置換表を更新する
    pub fn search_helper(
        &mut self,
        pos: &mut Position,
        tc: &TimeControl,
        stop_flag: Arc<AtomicBool>,
        thread_id: usize,
    ) {
        let time_mgr = TimeManager::new(tc, pos.side_to_move);
        let ctx = SearchContext {
            time_mgr: &time_mgr,
            stop_flag: &stop_flag,
        };
        let spec = RootSearchSpec {
            start_depth: (thread_id % 2) as u8 + 1,
            max_depth: MAX_TIMED_DEPTH,
            use_aspiration: false,
            check_time_budget: true,
            check_book: false,
            dfpn_precheck: None,
            stop_on_mate: false,
            emit_pv: false,
        };
        let _ = self.run_root_search(pos, &spec, &ctx, |_| {});
    }

    /// 全探索エントリポイントで共有される単一のルート反復深化エンジン
    fn run_root_search(
        &mut self,
        pos: &mut Position,
        spec: &RootSearchSpec,
        ctx: &SearchContext,
        mut on_event: impl FnMut(SearchEvent),
    ) -> SearchOutcome {
        // 1. 定跡データベースの照会 (use_book が有効な場合のみ)
        if spec.check_book
            && let Some(book_move) = OpeningBook::probe(pos)
        {
            let legal_moves = MoveGenerator::generate_legal_moves(pos);
            if legal_moves.contains(&book_move) {
                on_event(SearchEvent::BookMove(book_move));
                return SearchOutcome::Book {
                    best_move: book_move,
                };
            } else {
                on_event(SearchEvent::IllegalBookMove(book_move));
            }
        }

        // 2. df-pn 詰み探索の事前チェック
        if let Some(precheck) = spec.dfpn_precheck
            && let Some(mv) = Self::probe_dfpn_mate(pos, precheck)
        {
            on_event(SearchEvent::MateFound(mv));
            return SearchOutcome::MateProven {
                best_move: Some(mv),
                score: MATE_SCORE - 1,
            };
        }

        self.nodes = 0;
        self.init_root_accumulator(pos);

        let mut root_moves = MoveGenerator::generate_legal_moves(pos);
        if root_moves.is_empty() {
            return SearchOutcome::NoLegalMoves;
        }

        let mut best_move = root_moves[0];
        let mut best_score = self.evaluate_at_ply(pos, 0);
        let mut completed_depth = 0u8;
        let mut aborted_mid_iteration = false;

        'deepening: for depth in spec.start_depth..=spec.max_depth {
            if spec.check_time_budget && depth > 1 {
                if ctx.stop_flag.load(Ordering::Relaxed) {
                    if completed_depth == 0 {
                        aborted_mid_iteration = true;
                    }
                    break;
                }
                if !ctx.time_mgr.should_continue_deepening() {
                    if completed_depth == 0 {
                        aborted_mid_iteration = true;
                    }
                    break;
                }
            }

            let mut alpha = -INF;
            let mut beta = INF;
            if spec.use_aspiration && depth >= 4 && best_score.abs() < MATE_SCORE - 200 {
                alpha = (best_score - ASPIRATION_DELTA).max(-INF);
                beta = (best_score + ASPIRATION_DELTA).min(INF);
            }

            self.order_root_moves(pos, &mut root_moves);

            let (iter_best_move, iter_score) = loop {
                let mut current_alpha = alpha;
                let mut loop_best_move = None;

                for &mv in &root_moves {
                    if ctx.stop_flag.load(Ordering::Relaxed) {
                        aborted_mid_iteration = true;
                        break 'deepening;
                    }

                    let score = self.with_move(pos, mv, 1, |eng, p| {
                        -eng.negamax(p, depth - 1, -beta, -current_alpha, 1, true, ctx)
                    });

                    // 中断した深さの結果は採用せず、ルートTTにも保存しない。
                    if ctx.stop_flag.load(Ordering::Relaxed) {
                        aborted_mid_iteration = true;
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
                        break (loop_best_move, current_alpha);
                    }
                    alpha = -INF;
                } else if current_alpha >= beta {
                    if beta == INF {
                        break (loop_best_move, current_alpha);
                    }
                    beta = INF;
                } else {
                    break (loop_best_move, current_alpha);
                }
            };

            if let Some(bm) = iter_best_move {
                best_move = bm;
                best_score = iter_score;

                self.tt.store(
                    pos.hash,
                    depth,
                    Self::score_to_tt(best_score, 0),
                    NodeType::Exact,
                    Some(best_move),
                );
                completed_depth = depth;

                if spec.emit_pv {
                    let elapsed_ms = ctx.time_mgr.elapsed_ms().max(1);
                    let nps = ((self.nodes as u128 * 1000) / elapsed_ms as u128) as u64;
                    let pv = self.extract_pv(pos, best_move, depth);
                    on_event(SearchEvent::Iteration(SearchInfo {
                        depth,
                        score: best_score,
                        elapsed_ms,
                        nodes: self.nodes,
                        nps,
                        pv,
                    }));
                }

                if spec.stop_on_mate && best_score.abs() >= MATE_SCORE - 100 {
                    break;
                }
            } else {
                best_score = iter_score;
                completed_depth = depth;
            }
        }

        if aborted_mid_iteration {
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

    fn order_root_moves(&self, pos: &Position, root_moves: &mut [Move]) {
        let tt_move = self.tt.probe(pos.hash).and_then(|e| e.best_move);
        let prev_mv = pos.history.last().map(|rec| rec.mv);
        let counter_mv = prev_mv.and_then(|pm| {
            pm.from()
                .and_then(|from_sq| self.counter_moves[from_sq.index()][pm.to().index()])
        });
        MoveOrderer::order_moves(
            root_moves,
            pos,
            tt_move,
            &[None, None],
            counter_mv,
            Some(&self.history),
        );
    }

    fn probe_dfpn_mate(pos: &mut Position, precheck: DfpnPrecheck) -> Option<Move> {
        if precheck.require_near_king_guard {
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
            if !near_king_threat
                && !pos.is_in_check(opp)
                && MoveGenerator::generate_checks(pos).is_empty()
            {
                return None;
            }
        }

        let mut dfpn = super::dfpn::DfpnSolver::new(precheck.max_nodes);
        let (is_mate, mate_move) = dfpn.solve(pos);
        if is_mate { mate_move } else { None }
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
        if let Some(early_score) = self.check_node_and_time(pos, ply, ctx) {
            return early_score;
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
            return super::qsearch::quiescence(self, pos, alpha, beta, ply, ctx);
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
            let null_score = self.with_null_move(pos, ply + 1, |eng, p| {
                -eng.negamax(
                    p,
                    depth.saturating_sub(3),
                    -beta,
                    -beta + 1,
                    ply + 1,
                    false,
                    ctx,
                )
            });

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
                return -MATE_SCORE + (ply as i32);
            } else {
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

            let is_pv_move = move_count == 1;
            let ext = if is_recapture && depth <= 6 { 1 } else { 0 };

            let score = self.with_move(pos, mv, ply + 1, |eng, p| {
                if is_pv_move {
                    -eng.negamax(p, depth - 1 + ext, -beta, -alpha, ply + 1, true, ctx)
                } else {
                    let mut search_depth = depth - 1 + ext;
                    if depth >= 3 && move_count >= 4 && !is_tactical && !in_check {
                        search_depth = search_depth.saturating_sub(1);
                    }

                    let mut s =
                        -eng.negamax(p, search_depth, -alpha - 1, -alpha, ply + 1, true, ctx);

                    if s > alpha && search_depth < depth - 1 + ext {
                        s = -eng.negamax(
                            p,
                            depth - 1 + ext,
                            -alpha - 1,
                            -alpha,
                            ply + 1,
                            true,
                            ctx,
                        );
                    }

                    if s > alpha && s < beta {
                        s = -eng.negamax(p, depth - 1 + ext, -beta, -alpha, ply + 1, true, ctx);
                    }
                    s
                }
            });

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
                if ply < 64 && !mv.is_drop() && pos.board[mv.to().index()].is_none() {
                    if self.killer_moves[ply][0] != Some(mv) {
                        self.killer_moves[ply][1] = self.killer_moves[ply][0];
                        self.killer_moves[ply][0] = Some(mv);
                    }
                    if let Some(pm) = prev_mv
                        && let Some(p_from) = pm.from()
                    {
                        self.counter_moves[p_from.index()][pm.to().index()] = Some(mv);
                    }
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

        for _ in 0..moves_done.len() {
            pos.undo_move();
        }

        pv
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{EvalMode, HalfKPEvaluator};
    use crate::selfplay::game::SimpleRng;

    #[test]
    fn with_move_and_with_null_move_keep_halfkp_accumulator_exact_across_plies() {
        let mut eval_inst = HalfKPEvaluator::new();
        for row in eval_inst.feature_weights.iter_mut().take(500) {
            row[0] = 25;
            row[63] = -18;
        }
        eval_inst.output_weights[0] = 40;
        eval_inst.output_weights[128] = -35;
        eval_inst.output_bias = 100;

        let eval = Arc::new(eval_inst);
        let mut engine = SearchEngine::new(4).with_eval_mode(EvalMode::HalfKP(Arc::clone(&eval)));
        let mut pos = Position::startpos();
        let mut rng = SimpleRng::new(0xABCDEF1234567890);

        engine.init_root_accumulator(&pos);
        assert_eq!(engine.evaluate_at_ply(&pos, 0), eval.evaluate(&pos));

        // Walk 4 nested plies plus a null-move + post-null child move and verify exact match at every ply.
        let moves1 = MoveGenerator::generate_legal_moves(&mut pos);
        let m1 = moves1[rng.gen_range(moves1.len())];
        engine.with_move(&mut pos, m1, 1, |eng1, p1| {
            assert_eq!(eng1.evaluate_at_ply(p1, 1), eval.evaluate(p1));
            eng1.with_null_move(p1, 2, |eng_n, pn| {
                assert_eq!(eng_n.evaluate_at_ply(pn, 2), eval.evaluate(pn));
                let moves_n = MoveGenerator::generate_legal_moves(pn);
                if !moves_n.is_empty() {
                    let mn = moves_n[rng.gen_range(moves_n.len())];
                    eng_n.with_move(pn, mn, 3, |eng_n3, pn3| {
                        assert_eq!(eng_n3.evaluate_at_ply(pn3, 3), eval.evaluate(pn3));
                    });
                }
            });
            let moves2 = MoveGenerator::generate_legal_moves(p1);
            let m2 = moves2[rng.gen_range(moves2.len())];
            eng1.with_move(p1, m2, 2, |eng2, p2| {
                assert_eq!(eng2.evaluate_at_ply(p2, 2), eval.evaluate(p2));
                let moves3 = MoveGenerator::generate_legal_moves(p2);
                let m3 = moves3[rng.gen_range(moves3.len())];
                eng2.with_move(p2, m3, 3, |eng3, p3| {
                    assert_eq!(eng3.evaluate_at_ply(p3, 3), eval.evaluate(p3));
                    let moves4 = MoveGenerator::generate_legal_moves(p3);
                    let m4 = moves4[rng.gen_range(moves4.len())];
                    eng3.with_move(p3, m4, 4, |eng4, p4| {
                        assert_eq!(eng4.evaluate_at_ply(p4, 4), eval.evaluate(p4));
                    });
                });
            });
        });
        assert_eq!(pos.to_sfen(), Position::startpos().to_sfen());
    }

    #[test]
    fn timed_search_respects_use_book_flag_and_emits_events() {
        let mut pos = Position::startpos();
        let tc = TimeControl {
            byoyomi: Some(50),
            ..Default::default()
        };

        // use_book = true hits opening book on startpos
        let mut book_engine = SearchEngine::new(1);
        book_engine.use_book = true;
        let mut saw_book = false;
        let out =
            book_engine.search_with_events(&mut pos, &tc, Arc::new(AtomicBool::new(false)), |ev| {
                if matches!(ev, SearchEvent::BookMove(_)) {
                    saw_book = true;
                }
            });
        assert!(matches!(out, SearchOutcome::Book { .. }));
        assert!(saw_book);

        // use_book = false skips opening book and runs actual search iterations
        let mut nobook_engine = SearchEngine::new(1);
        nobook_engine.use_book = false;
        nobook_engine.max_nodes = Some(500);
        let mut saw_iter = false;
        let out2 = nobook_engine.search_with_events(
            &mut pos,
            &tc,
            Arc::new(AtomicBool::new(false)),
            |ev| {
                if matches!(ev, SearchEvent::Iteration(_)) {
                    saw_iter = true;
                }
            },
        );
        assert!(!matches!(out2, SearchOutcome::Book { .. }));
        assert!(saw_iter || matches!(out2, SearchOutcome::Aborted { .. }));
    }
}
