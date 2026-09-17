use super::halfkp::{HALFKP_HIDDEN_SIZE, HALFKP_INPUT_SIZE, HalfKPEvaluator, MAX_EVAL_CP};
use std::collections::HashMap;

/// ゼロ外部依存のスクラッチ HalfKP バックプロパゲーション学習器
/// - 特徴量次元: 204,120 (自玉81 × 全駒2,520)
/// - 隠れ層: 128 (先手128 + 後手128 = 256次元結合)
/// - 最適化手法: スパース AdamW (バッチ内出現特徴量のみ追跡・更新)
/// - 活性化関数: ClippedReLU (0.0..=64.0)
/// - 評価値スケール: Evaluator と完全一致 (output / 128 = cp)
pub struct HalfKPTrainer {
    pub feature_weights: Vec<[f32; HALFKP_HIDDEN_SIZE]>,
    pub feature_biases: [f32; HALFKP_HIDDEN_SIZE],
    pub output_weights: [f32; HALFKP_HIDDEN_SIZE * 2],
    pub output_bias: f32,

    // AdamW モーメンタム状態 (スパース特徴量重み用)
    m_feat: Vec<[f32; HALFKP_HIDDEN_SIZE]>,
    v_feat: Vec<[f32; HALFKP_HIDDEN_SIZE]>,
    m_f_bias: [f32; HALFKP_HIDDEN_SIZE],
    v_f_bias: [f32; HALFKP_HIDDEN_SIZE],
    m_out: [f32; HALFKP_HIDDEN_SIZE * 2],
    v_out: [f32; HALFKP_HIDDEN_SIZE * 2],
    m_bias: f32,
    v_bias: f32,
    beta1_pow: f32,
    beta2_pow: f32,
}

impl Default for HalfKPTrainer {
    fn default() -> Self {
        Self::new()
    }
}

impl HalfKPTrainer {
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn adamw_step(
        w: &mut f32,
        m: &mut f32,
        v: &mut f32,
        g: f32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        one_minus_b1: f32,
        one_minus_b2: f32,
        eps: f32,
        wd: f32,
    ) {
        *m = beta1 * *m + (1.0 - beta1) * g;
        *v = beta2 * *v + (1.0 - beta2) * g * g;
        let m_hat = *m / one_minus_b1;
        let v_hat = *v / one_minus_b2;
        *w -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * *w);
    }

    /// 新規初期化 (Evaluator の初期重みから構築)
    pub fn new() -> Self {
        let initial_eval = HalfKPEvaluator::new();
        Self::from_evaluator(&initial_eval)
    }

    /// 既存の HalfKPEvaluator から Trainer を初期化 (ウォームスタート)
    pub fn from_evaluator(eval: &HalfKPEvaluator) -> Self {
        let mut feature_weights = Vec::with_capacity(HALFKP_INPUT_SIZE);
        for row in &eval.feature_weights {
            let mut f_row = [0.0f32; HALFKP_HIDDEN_SIZE];
            for (w, &q) in f_row.iter_mut().zip(row.iter()) {
                *w = q as f32;
            }
            feature_weights.push(f_row);
        }

        let mut feature_biases = [0.0f32; HALFKP_HIDDEN_SIZE];
        for (b, &q) in feature_biases.iter_mut().zip(eval.feature_biases.iter()) {
            *b = q as f32;
        }

        let mut output_weights = [0.0f32; HALFKP_HIDDEN_SIZE * 2];
        for (w, &q) in output_weights.iter_mut().zip(eval.output_weights.iter()) {
            *w = q as f32;
        }

        let output_bias = eval.output_bias as f32;

        HalfKPTrainer {
            feature_weights,
            feature_biases,
            output_weights,
            output_bias,
            m_feat: vec![[0.0f32; HALFKP_HIDDEN_SIZE]; HALFKP_INPUT_SIZE],
            v_feat: vec![[0.0f32; HALFKP_HIDDEN_SIZE]; HALFKP_INPUT_SIZE],
            m_f_bias: [0.0f32; HALFKP_HIDDEN_SIZE],
            v_f_bias: [0.0f32; HALFKP_HIDDEN_SIZE],
            m_out: [0.0f32; HALFKP_HIDDEN_SIZE * 2],
            v_out: [0.0f32; HALFKP_HIDDEN_SIZE * 2],
            m_bias: 0.0,
            v_bias: 0.0,
            beta1_pow: 1.0,
            beta2_pow: 1.0,
        }
    }

    /// 学習済み Trainer から推論用 HalfKPEvaluator へエクスポート
    pub fn to_evaluator(&self) -> HalfKPEvaluator {
        let mut eval = HalfKPEvaluator {
            feature_weights: Vec::with_capacity(HALFKP_INPUT_SIZE),
            feature_biases: [0i16; HALFKP_HIDDEN_SIZE],
            output_weights: [0i16; HALFKP_HIDDEN_SIZE * 2],
            output_bias: self
                .output_bias
                .round()
                .clamp(i32::MIN as f32, i32::MAX as f32) as i32,
        };

        for row in &self.feature_weights {
            let mut i_row = [0i16; HALFKP_HIDDEN_SIZE];
            for (dest, &src) in i_row.iter_mut().zip(row.iter()) {
                *dest = src.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            }
            eval.feature_weights.push(i_row);
        }

        for (dest, &src) in eval
            .feature_biases
            .iter_mut()
            .zip(self.feature_biases.iter())
        {
            *dest = src.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        }

        for (dest, &src) in eval
            .output_weights
            .iter_mut()
            .zip(self.output_weights.iter())
        {
            *dest = src.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        }

        eval
    }

    /// フォワードパス (Mover-first 手番視点)
    /// 戻り値: (手番視点評価値cp, raw_output, m_acc, o_acc, m_hidden, o_hidden)
    pub fn forward(
        &self,
        mover_feats: &[usize],
        opp_feats: &[usize],
    ) -> (
        f32,
        f32,
        [f32; HALFKP_HIDDEN_SIZE],
        [f32; HALFKP_HIDDEN_SIZE],
        [f32; HALFKP_HIDDEN_SIZE],
        [f32; HALFKP_HIDDEN_SIZE],
    ) {
        let mut m_acc = self.feature_biases;
        for &f in mover_feats {
            if f < HALFKP_INPUT_SIZE {
                for (a, &w) in m_acc.iter_mut().zip(self.feature_weights[f].iter()) {
                    *a += w;
                }
            }
        }

        let mut o_acc = self.feature_biases;
        for &f in opp_feats {
            if f < HALFKP_INPUT_SIZE {
                for (a, &w) in o_acc.iter_mut().zip(self.feature_weights[f].iter()) {
                    *a += w;
                }
            }
        }

        // ClippedReLU (0.0..=64.0)
        let mut m_hidden = [0.0f32; HALFKP_HIDDEN_SIZE];
        for (h, &a) in m_hidden.iter_mut().zip(m_acc.iter()) {
            *h = a.clamp(0.0, 64.0);
        }

        let mut o_hidden = [0.0f32; HALFKP_HIDDEN_SIZE];
        for (h, &a) in o_hidden.iter_mut().zip(o_acc.iter()) {
            *h = a.clamp(0.0, 64.0);
        }

        // 出力層 (Mover-first 結合)
        let mut output = self.output_bias;
        for (i, &h) in m_hidden.iter().enumerate() {
            output += h * self.output_weights[i];
        }
        for (i, &h) in o_hidden.iter().enumerate() {
            output += h * self.output_weights[HALFKP_HIDDEN_SIZE + i];
        }

        let raw_cp = output / 128.0;
        let score_cp = raw_cp.clamp(-MAX_EVAL_CP as f32, MAX_EVAL_CP as f32);

        (score_cp, output, m_acc, o_acc, m_hidden, o_hidden)
    }

    /// シグモイド勝率予測関数
    #[inline(always)]
    pub fn sigmoid(score: f32, k: f32) -> f32 {
        1.0 / (1.0 + 10.0f32.powf(-score / k))
    }

    /// 単一バッチのフォワード・バックワード・スパース AdamW 更新
    /// - batch: `&[(mover_feats, opp_feats, target)]` (target: 0.0=敗北, 0.5=引分, 1.0=勝利)
    /// - lr: 学習率 (例: 0.001)
    /// - k: シグモイド感度係数 (通常 600.0)
    /// - 戻り値: 平均二乗誤差 (MSE Loss)
    pub fn train_batch(&mut self, batch: &[(Vec<usize>, Vec<usize>, f32)], lr: f32, k: f32) -> f32 {
        if batch.is_empty() {
            return 0.0;
        }

        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let epsilon = 1e-8f32;
        let weight_decay = 0.0001f32;
        let lambda_range = 0.001f32; // Preactivation Range Penalty 係数
        let ln10_div_k = 10.0f32.ln() / k;
        let max_eval = MAX_EVAL_CP as f32;

        let mut grad_out_w = [0.0f32; HALFKP_HIDDEN_SIZE * 2];
        let mut grad_out_b = 0.0f32;
        let mut grad_feature_biases = [0.0f32; HALFKP_HIDDEN_SIZE];
        let mut total_loss = 0.0f32;

        let mut active_feature_updates: HashMap<usize, [f32; HALFKP_HIDDEN_SIZE]> = HashMap::new();

        for (mover_feats, opp_feats, target) in batch {
            let (score_cp, raw_output, m_acc, o_acc, m_hidden, o_hidden) =
                self.forward(mover_feats, opp_feats);

            let pred = Self::sigmoid(score_cp, k);
            let error = pred - target; // (予測 - 教師信号)
            total_loss += error * error;

            // dL / d_score
            let d_sigmoid = pred * (1.0 - pred) * ln10_div_k;
            let d_loss_d_score = 2.0 * error * d_sigmoid;

            // 評価値クランプ勾配ガード: 飽和領域でも引き戻し勾配は通過させる
            let raw_cp = raw_output / 128.0;
            let d_output = if raw_cp >= max_eval {
                if d_loss_d_score > 0.0 {
                    (d_loss_d_score / 128.0).clamp(-10.0, 10.0)
                } else {
                    0.0
                }
            } else if raw_cp <= -max_eval {
                if d_loss_d_score < 0.0 {
                    (d_loss_d_score / 128.0).clamp(-10.0, 10.0)
                } else {
                    0.0
                }
            } else {
                (d_loss_d_score / 128.0).clamp(-10.0, 10.0)
            };

            grad_out_b += d_output;

            // 出力層重み勾配 & 隠れ層への逆伝播
            let mut d_m_acc = [0.0f32; HALFKP_HIDDEN_SIZE];
            for (i, &h) in m_hidden.iter().enumerate() {
                grad_out_w[i] += d_output * h;
                let task_grad = if m_acc[i] > 0.0 && m_acc[i] < 64.0 {
                    d_output * self.output_weights[i]
                } else {
                    0.0
                };
                let range_penalty = if m_acc[i] < 0.0 {
                    lambda_range * m_acc[i]
                } else if m_acc[i] > 64.0 {
                    lambda_range * (m_acc[i] - 64.0)
                } else {
                    0.0
                };
                d_m_acc[i] = task_grad + range_penalty;
            }

            let mut d_o_acc = [0.0f32; HALFKP_HIDDEN_SIZE];
            for (i, &h) in o_hidden.iter().enumerate() {
                grad_out_w[HALFKP_HIDDEN_SIZE + i] += d_output * h;
                let task_grad = if o_acc[i] > 0.0 && o_acc[i] < 64.0 {
                    d_output * self.output_weights[HALFKP_HIDDEN_SIZE + i]
                } else {
                    0.0
                };
                let range_penalty = if o_acc[i] < 0.0 {
                    lambda_range * o_acc[i]
                } else if o_acc[i] > 64.0 {
                    lambda_range * (o_acc[i] - 64.0)
                } else {
                    0.0
                };
                d_o_acc[i] = task_grad + range_penalty;
            }

            // 特徴量バイアス勾配蓄積
            for (gb, &dm) in grad_feature_biases.iter_mut().zip(d_m_acc.iter()) {
                *gb += dm;
            }
            for (gb, &do_val) in grad_feature_biases.iter_mut().zip(d_o_acc.iter()) {
                *gb += do_val;
            }

            // スパース特徴量重み勾配蓄積
            for &f in mover_feats {
                if f < HALFKP_INPUT_SIZE {
                    let entry = active_feature_updates
                        .entry(f)
                        .or_insert([0.0f32; HALFKP_HIDDEN_SIZE]);
                    for (g, &dm) in entry.iter_mut().zip(d_m_acc.iter()) {
                        *g += dm;
                    }
                }
            }
            for &f in opp_feats {
                if f < HALFKP_INPUT_SIZE {
                    let entry = active_feature_updates
                        .entry(f)
                        .or_insert([0.0f32; HALFKP_HIDDEN_SIZE]);
                    for (g, &do_val) in entry.iter_mut().zip(d_o_acc.iter()) {
                        *g += do_val;
                    }
                }
            }
        }

        // バッチ平均化
        let batch_size_f = batch.len() as f32;
        grad_out_b /= batch_size_f;
        for g in &mut grad_out_w {
            *g /= batch_size_f;
        }
        for g in &mut grad_feature_biases {
            *g /= batch_size_f;
        }

        // AdamW パワー更新
        self.beta1_pow *= beta1;
        self.beta2_pow *= beta2;
        let one_minus_beta1 = 1.0 - self.beta1_pow;
        let one_minus_beta2 = 1.0 - self.beta2_pow;

        // 1. 出力層バイアス更新
        // 1. 出力層バイアス更新
        Self::adamw_step(
            &mut self.output_bias,
            &mut self.m_bias,
            &mut self.v_bias,
            grad_out_b,
            lr,
            beta1,
            beta2,
            one_minus_beta1,
            one_minus_beta2,
            epsilon,
            weight_decay,
        );

        // 2. 出力層重み更新
        for (i, &g) in grad_out_w.iter().enumerate() {
            Self::adamw_step(
                &mut self.output_weights[i],
                &mut self.m_out[i],
                &mut self.v_out[i],
                g,
                lr,
                beta1,
                beta2,
                one_minus_beta1,
                one_minus_beta2,
                epsilon,
                weight_decay,
            );
        }

        // 3. 特徴量バイアス更新
        for (i, &g) in grad_feature_biases.iter().enumerate() {
            Self::adamw_step(
                &mut self.feature_biases[i],
                &mut self.m_f_bias[i],
                &mut self.v_f_bias[i],
                g,
                lr,
                beta1,
                beta2,
                one_minus_beta1,
                one_minus_beta2,
                epsilon,
                weight_decay,
            );
        }

        // 4. スパース特徴量重み更新
        for (f, grad_slice) in active_feature_updates {
            let m_slice = &mut self.m_feat[f];
            let v_slice = &mut self.v_feat[f];
            let w_slice = &mut self.feature_weights[f];
            for (i, &raw_g) in grad_slice.iter().enumerate() {
                let g = raw_g / batch_size_f;
                Self::adamw_step(
                    &mut w_slice[i],
                    &mut m_slice[i],
                    &mut v_slice[i],
                    g,
                    lr,
                    beta1,
                    beta2,
                    one_minus_beta1,
                    one_minus_beta2,
                    epsilon,
                    weight_decay,
                );
            }
        }

        total_loss / batch_size_f
    }
}
