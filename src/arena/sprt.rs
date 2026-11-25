//! SPRT (Sequential Probability Ratio Test: 逐次確率比検定) ソルバー
//!
//! 外部ライブラリを一切使わず、Stockfish / Fishtest で標準的な Wald の逐次検定を実装。
//! 2つのモデル間の勝敗推移から、新モデルが統計的有意に強化されたかを判定します。

/// SPRT と最低対局数に基づく昇格判定結果
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromotionVerdict {
    /// SPRT が H1 合格 (`Pass`) かつ最低対局数を満たしたため昇格
    Promote,
    /// SPRT は H1 合格 (`Pass`) だが最低対局数に未達のため昇格保留
    DeferredInsufficientGames { played: usize, required: usize },
    /// SPRT が H1 合格に達しなかった (`Fail` / `Continue` / 無効) ため不採用
    Reject,
}

impl PromotionVerdict {
    pub fn is_promoted(self) -> bool {
        matches!(self, Self::Promote)
    }
}

/// 勝敗・引き分け数から勝率 (0.0..=1.0) を算出（0 局時は 0.5）
pub fn win_rate(wins: usize, losses: usize, draws: usize) -> f64 {
    let total = wins + losses + draws;
    if total == 0 {
        return 0.5;
    }
    (wins as f64 + 0.5 * draws as f64) / (total as f64)
}

/// 勝率 (0.0..=1.0) から推定 Elo 差を算出（極端な勝率は ±1000.0 にクランプ）
pub fn elo_diff_from_win_rate(wr: f64) -> f64 {
    if wr <= 0.001 {
        return -1000.0;
    }
    if wr >= 0.999 {
        return 1000.0;
    }
    -400.0 * (1.0 / wr - 1.0).log10()
}

/// SPRT 結果と総対局数から昇格可否を判定する単一ゲート
pub fn evaluate_promotion_verdict(
    sprt: Option<&Sprt>,
    total_games: usize,
    min_promotion_games: usize,
) -> PromotionVerdict {
    let Some(sprt) = sprt else {
        return PromotionVerdict::Reject;
    };
    if sprt.status != SprtStatus::Pass {
        return PromotionVerdict::Reject;
    }
    // SPRT 判定確定時点の対局数と総対局数の双方で最低対局数を満たすことを要求
    let evaluated_games = sprt.total_games().min(total_games);
    if evaluated_games < min_promotion_games {
        return PromotionVerdict::DeferredInsufficientGames {
            played: evaluated_games,
            required: min_promotion_games,
        };
    }
    PromotionVerdict::Promote
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SprtStatus {
    /// 帰無仮説を棄却し対立仮説を採択 (有意に強化された -> モデル昇格)
    Pass,
    /// 対立仮説を棄却 (強化されていない -> モデル破棄)
    Fail,
    /// 判定保留 (データ不足のため対局継続)
    Continue,
}

#[derive(Debug, Clone)]
pub struct SprtConfig {
    /// 帰無仮説 H0 の Elo 差 (通常 0.0)
    pub elo0: f64,
    /// 対立仮説 H1 の Elo 差 (例: +5.0 または +10.0)
    pub elo1: f64,
    /// 第1種過誤確率 alpha (偽陽性: 強くないのに合格と判定する確率)
    pub alpha: f64,
    /// 第2種過誤確率 beta (偽陰性: 強いのに不合格と判定する確率)
    pub beta: f64,
    /// 最低対局数 (Pass または Fail 判定を確定するまでの最小対局数)
    pub min_games: usize,
}

impl Default for SprtConfig {
    fn default() -> Self {
        Self {
            elo0: 0.0,
            elo1: 10.0,
            alpha: 0.05,
            beta: 0.05,
            min_games: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Sprt {
    pub config: SprtConfig,
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
    pub llr: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub status: SprtStatus,
}

impl Sprt {
    pub fn new(config: SprtConfig) -> Self {
        // Wald の境界値
        let lower_bound = (config.beta / (1.0 - config.alpha)).ln();
        let upper_bound = ((1.0 - config.beta) / config.alpha).ln();

        Self {
            config,
            wins: 0,
            losses: 0,
            draws: 0,
            llr: 0.0,
            lower_bound,
            upper_bound,
            status: SprtStatus::Continue,
        }
    }

    /// 対局結果を 1 局追加 (1.0: 勝ち, 0.5: 引分, 0.0: 負け)
    pub fn record_result(&mut self, score: f64) {
        // 終端状態 (Pass または Fail) に達した検定結果は不可逆（ラッチ）
        if self.status != SprtStatus::Continue {
            return;
        }

        if score >= 0.75 {
            self.wins += 1;
        } else if score <= 0.25 {
            self.losses += 1;
        } else {
            self.draws += 1;
        }

        self.update_llr();
    }

    /// 対局結果のバッチ追加
    pub fn record_batch(&mut self, wins: usize, losses: usize, draws: usize) {
        // 終端状態 (Pass または Fail) に達した検定結果は不可逆（ラッチ）
        if self.status != SprtStatus::Continue {
            return;
        }

        self.wins += wins;
        self.losses += losses;
        self.draws += draws;
        self.update_llr();
    }

    /// 検定が確定（Pass または Fail）したかを判定
    pub fn is_decided(&self) -> bool {
        self.status != SprtStatus::Continue
    }

    /// 対数尤度比 (LLR) の更新
    fn update_llr(&mut self) {
        if self.status != SprtStatus::Continue {
            return;
        }

        let total = (self.wins + self.losses + self.draws) as f64;
        if total < 2.0 {
            self.status = SprtStatus::Continue;
            return;
        }

        // Elo 差から勝率期待値への変換: P(win) = 1 / (1 + 10^(-elo / 400))
        let p0 = 1.0 / (1.0 + 10.0f64.powf(-self.config.elo0 / 400.0));
        let p1 = 1.0 / (1.0 + 10.0f64.powf(-self.config.elo1 / 400.0));

        let w = self.wins as f64;
        let d = self.draws as f64;
        let l = self.losses as f64;

        // 標本勝率
        let s = (w + 0.5 * d) / total;

        // 標本分散: sum((x_i - s)^2) / N
        // 分散下限: 全勝・全敗時の極小分散によるLLR爆発を防ぐため 0.05 を下限として設定
        let sample_var =
            (w * (1.0 - s).powi(2) + d * (0.5 - s).powi(2) + l * (0.0 - s).powi(2)) / total;
        let var = sample_var.max(0.05);

        // 正規近似に基づく LLR 計算 (Fishtest 準拠)
        // LLR = (p1 - p0) / var * sum(x_i - (p0 + p1) / 2)
        let delta_p = p1 - p0;
        let sum_diff = (w + 0.5 * d) - total * (p0 + p1) / 2.0;
        self.llr = (delta_p / var) * sum_diff;

        // 最低対局数 min_games を満たしている場合のみ、Pass または Fail の確定判定を行う
        if self.total_games() >= self.config.min_games {
            if self.llr >= self.upper_bound {
                self.status = SprtStatus::Pass;
            } else if self.llr <= self.lower_bound {
                self.status = SprtStatus::Fail;
            } else {
                self.status = SprtStatus::Continue;
            }
        } else {
            self.status = SprtStatus::Continue;
        }
    }

    /// 総対局数
    pub fn total_games(&self) -> usize {
        self.wins + self.losses + self.draws
    }

    /// 勝率 (0.0..=1.0)
    pub fn win_rate(&self) -> f64 {
        win_rate(self.wins, self.losses, self.draws)
    }

    /// 推定 Elo 差
    pub fn elo_diff(&self) -> f64 {
        elo_diff_from_win_rate(self.win_rate())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sprt_min_games_gating() {
        let config = SprtConfig {
            elo0: 0.0,
            elo1: 50.0,
            alpha: 0.05,
            beta: 0.05,
            min_games: 20,
        };

        let mut sprt = Sprt::new(config);

        // 6勝0敗: LLR は高いが min_games (20) 未満のため Continue を維持
        sprt.record_batch(6, 0, 0);
        assert_eq!(sprt.status, SprtStatus::Continue);
        assert!(!sprt.is_decided());
        assert!(sprt.llr > sprt.upper_bound);

        // 追加で14勝 (計20局 >= 20): ここで初めて Pass へ遷移
        sprt.record_batch(14, 0, 0);
        assert_eq!(sprt.status, SprtStatus::Pass);
        assert!(sprt.is_decided());
    }

    #[test]
    fn test_win_rate_and_elo_diff_helpers() {
        assert!((win_rate(0, 0, 0) - 0.5).abs() < 1e-12);
        assert!((win_rate(3, 1, 2) - 4.0 / 6.0).abs() < 1e-12);
        assert!((elo_diff_from_win_rate(0.5) - 0.0).abs() < 1e-12);
        assert_eq!(elo_diff_from_win_rate(0.0), -1000.0);
        assert_eq!(elo_diff_from_win_rate(1.0), 1000.0);
    }

    #[test]
    fn test_evaluate_promotion_verdict_enforces_sprt_and_min_games() {
        assert_eq!(
            evaluate_promotion_verdict(None, 100, 20),
            PromotionVerdict::Reject
        );

        let mut pass_early = Sprt::new(SprtConfig {
            elo0: 0.0,
            elo1: 50.0,
            alpha: 0.05,
            beta: 0.05,
            min_games: 0,
        });
        pass_early.record_batch(6, 0, 0);
        assert_eq!(pass_early.status, SprtStatus::Pass);
        assert_eq!(
            evaluate_promotion_verdict(Some(&pass_early), 6, 20),
            PromotionVerdict::DeferredInsufficientGames {
                played: 6,
                required: 20,
            }
        );
        // 6 局で早期 Pass ラッチ後に total_games だけ増えても、SPRT 評価済み 6 局 < 20 のため昇格不可
        assert_eq!(
            evaluate_promotion_verdict(Some(&pass_early), 24, 20),
            PromotionVerdict::DeferredInsufficientGames {
                played: 6,
                required: 20,
            }
        );

        let mut pass_enough = Sprt::new(SprtConfig {
            elo0: 0.0,
            elo1: 50.0,
            alpha: 0.05,
            beta: 0.05,
            min_games: 20,
        });
        pass_enough.record_batch(20, 0, 0);
        assert_eq!(pass_enough.status, SprtStatus::Pass);
        assert_eq!(
            evaluate_promotion_verdict(Some(&pass_enough), 20, 20),
            PromotionVerdict::Promote
        );

        let mut cont = Sprt::new(SprtConfig::default());
        cont.record_batch(5, 5, 0);
        assert_eq!(cont.status, SprtStatus::Continue);
        assert_eq!(
            evaluate_promotion_verdict(Some(&cont), 30, 20),
            PromotionVerdict::Reject
        );

        let mut fail = Sprt::new(SprtConfig::default());
        fail.record_batch(0, 30, 0);
        assert_eq!(fail.status, SprtStatus::Fail);
        assert_eq!(
            evaluate_promotion_verdict(Some(&fail), 30, 20),
            PromotionVerdict::Reject
        );
    }
}
