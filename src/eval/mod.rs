pub mod evaluator;
pub mod halfkp;
pub mod halfkp_trainer;
pub mod nnue;
pub mod trainer;

pub use evaluator::{EvalBreakdown, Evaluator};
pub use halfkp::{
    HALFKP_HIDDEN_SIZE, HALFKP_INPUT_SIZE, HALFKP_MAGIC, HALFKP_PIECE_SIZE, HalfKPAccumulator,
    HalfKPEvaluator,
};
pub use halfkp_trainer::HalfKPTrainer;
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
    /// スクラッチ NNUE 評価ネットワーク (1駒特徴量)
    Nnue(Arc<NNUEEvaluator>),
    /// 本格 HalfKP 評価ネットワーク (自玉81マス×全駒2,520)
    HalfKP(Arc<HalfKPEvaluator>),
}

impl EvalMode {
    #[inline(always)]
    pub fn evaluate(&self, pos: &Position) -> i32 {
        match self {
            EvalMode::Hce => Evaluator::evaluate(pos),
            EvalMode::Nnue(nnue) => nnue.evaluate(pos),
            EvalMode::HalfKP(halfkp) => halfkp.evaluate(pos),
        }
    }
}
