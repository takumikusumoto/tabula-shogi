pub mod evaluator;
pub mod nnue;
pub mod trainer;

pub use evaluator::{EvalBreakdown, Evaluator};
pub use nnue::{MAX_EVAL_CP, NNUEEvaluator, RESIDUAL_BOUND_CP};
pub use trainer::NNUETrainer;

use crate::board::Position;

use std::sync::Arc;

/// 評価関数の動作モード
#[derive(Clone, Debug, Default)]
pub enum EvalMode {
    /// 手動評価関数 (Hand-Crafted Evaluation, デフォルト)
    #[default]
    Hce,
    /// スクラッチ NNUE 評価ネットワーク
    Nnue(Arc<NNUEEvaluator>),
}

impl EvalMode {
    #[inline(always)]
    pub fn evaluate(&self, pos: &Position) -> i32 {
        match self {
            EvalMode::Hce => Evaluator::evaluate(pos),
            EvalMode::Nnue(nnue) => nnue.evaluate(pos),
        }
    }
}
