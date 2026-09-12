/// SPRT (Sequential Probability Ratio Test: 逐次確率比検定) ソルバー
///
/// 外部ライブラリを一切使わず、Stockfish / Fishtest で標準的な Wald の逐次検定を実装。
/// 2つのモデル間の勝敗推移から、新モデルが統計的有意に強化されたかを判定します。

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
}

impl Default for SprtConfig {
    fn default() -> Self {
        Self {
            elo0: 0.0,
            elo1: 10.0,
            alpha: 0.05,
            beta: 0.05,
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
        self.wins += wins;
        self.losses += losses;
        self.draws += draws;
        self.update_llr();
    }

    /// 対数尤度比 (LLR) の更新
    fn update_llr(&mut self) {
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
        let var = (w * (1.0 - s).powi(2) + d * (0.5 - s).powi(2) + l * (0.0 - s).powi(2)) / total;
        let var = var.max(0.01); // ゼロ除算防止

        // 正規近似に基づく LLR 計算 (Fishtest 準拠)
        // LLR = (p1 - p0) / var * sum(x_i - (p0 + p1) / 2)
        let delta_p = p1 - p0;
        let sum_diff = (w + 0.5 * d) - total * (p0 + p1) / 2.0;
        self.llr = (delta_p / var) * sum_diff;

        if self.llr >= self.upper_bound {
            self.status = SprtStatus::Pass;
        } else if self.llr <= self.lower_bound {
            self.status = SprtStatus::Fail;
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
        let total = self.total_games();
        if total == 0 {
            return 0.5;
        }
        (self.wins as f64 + 0.5 * self.draws as f64) / (total as f64)
    }

    /// 推定 Elo 差
    pub fn elo_diff(&self) -> f64 {
        let wr = self.win_rate();
        if wr <= 0.001 {
            return -1000.0;
        }
        if wr >= 0.999 {
            return 1000.0;
        }
        -400.0 * (1.0 / wr - 1.0).log10()
    }
}
