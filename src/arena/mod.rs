pub mod loop_pipeline;
pub mod match_runner;
pub mod sprt;
pub mod workspace;

pub use loop_pipeline::{LoopArenaParams, LoopConfig, LoopTrainingParams, SelfImprovementLoop};
pub use match_runner::{MatchConfig, MatchResult, MatchRunner, OvertimePolicy};
pub use sprt::{
    PromotionVerdict, Sprt, SprtConfig, SprtStatus, elo_diff_from_win_rate,
    evaluate_promotion_verdict, win_rate,
};
pub use workspace::{GenerationMetrics, GenerationSummaryRow, LoopStoragePaths, LoopWorkspace};
