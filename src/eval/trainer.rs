use super::nnue::{NNUE_HIDDEN_SIZE, NNUE_INPUT_SIZE, NNUEEvaluator, RESIDUAL_BOUND_CP};
use crate::board::Position;
use crate::selfplay::DatasetEntry;

/// ゼロ外部依存のスクラッチ NNUE バックプロパゲーション学習器
pub struct NNUETrainer {
    pub feature_weights: Vec<[f32; NNUE_HIDDEN_SIZE]>,
    pub feature_biases: [f32; NNUE_HIDDEN_SIZE],
    pub output_weights: [f32; NNUE_HIDDEN_SIZE * 2],
    pub output_bias: f32,

    // Adam モーメンタム追跡状態
    m_feat: Vec<[f32; NNUE_HIDDEN_SIZE]>,
    v_feat: Vec<[f32; NNUE_HIDDEN_SIZE]>,
    m_f_bias: [f32; NNUE_HIDDEN_SIZE],
    v_f_bias: [f32; NNUE_HIDDEN_SIZE],
    m_out: [f32; NNUE_HIDDEN_SIZE * 2],
    v_out: [f32; NNUE_HIDDEN_SIZE * 2],
    m_bias: f32,
    v_bias: f32,
    beta1_pow: f32,
    beta2_pow: f32,
}

impl Default for NNUETrainer {
    fn default() -> Self {
        Self::new()
    }
}

impl NNUETrainer {
    pub fn from_evaluator(eval: &NNUEEvaluator) -> Self {
        let mut feature_weights = Vec::with_capacity(NNUE_INPUT_SIZE);
        for row in &eval.feature_weights {
            let mut f_row = [0.0f32; NNUE_HIDDEN_SIZE];
            for (w, &q) in f_row.iter_mut().zip(row.iter()) {
                *w = q as f32 / 64.0;
            }
            feature_weights.push(f_row);
        }

        let mut feature_biases = [0.0f32; NNUE_HIDDEN_SIZE];
        for (b, &q) in feature_biases.iter_mut().zip(eval.feature_biases.iter()) {
            *b = q as f32 / 64.0;
        }

        let mut output_weights = [0.0f32; NNUE_HIDDEN_SIZE * 2];
        for (w, &q) in output_weights.iter_mut().zip(eval.output_weights.iter()) {
            *w = q as f32 / 64.0;
        }

        let output_bias = eval.output_bias as f32 / 4096.0;

        NNUETrainer {
            feature_weights,
            feature_biases,
            output_weights,
            output_bias,
            m_feat: vec![[0.0f32; NNUE_HIDDEN_SIZE]; NNUE_INPUT_SIZE],
            v_feat: vec![[0.0f32; NNUE_HIDDEN_SIZE]; NNUE_INPUT_SIZE],
            m_f_bias: [0.0f32; NNUE_HIDDEN_SIZE],
            v_f_bias: [0.0f32; NNUE_HIDDEN_SIZE],
            m_out: [0.0f32; NNUE_HIDDEN_SIZE * 2],
            v_out: [0.0f32; NNUE_HIDDEN_SIZE * 2],
            m_bias: 0.0,
            v_bias: 0.0,
            beta1_pow: 1.0,
            beta2_pow: 1.0,
        }
    }

    pub fn new() -> Self {
        let initial_eval = NNUEEvaluator::new();
        Self::from_evaluator(&initial_eval)
    }

    /// フォワードパス (Mover-first 手番視点)
    /// 戻り値: (手番視点残差cp, 生出力, m_acc, o_acc, m_hidden, o_hidden)
    pub fn forward(
        &self,
        mover_feats: &[usize],
        opp_feats: &[usize],
    ) -> (
        f32,
        f32,
        [f32; NNUE_HIDDEN_SIZE],
        [f32; NNUE_HIDDEN_SIZE],
        [f32; NNUE_HIDDEN_SIZE],
        [f32; NNUE_HIDDEN_SIZE],
    ) {
        let mut m_acc = self.feature_biases;
        for &f in mover_feats {
            if f < NNUE_INPUT_SIZE {
                for (a, &w) in m_acc.iter_mut().zip(self.feature_weights[f].iter()) {
                    *a += w;
                }
            }
        }

        let mut o_acc = self.feature_biases;
        for &f in opp_feats {
            if f < NNUE_INPUT_SIZE {
                for (a, &w) in o_acc.iter_mut().zip(self.feature_weights[f].iter()) {
                    *a += w;
                }
            }
        }

        // ClippedReLU (0.0..=1.0)
        let mut m_hidden = [0.0f32; NNUE_HIDDEN_SIZE];
        for (h, &a) in m_hidden.iter_mut().zip(m_acc.iter()) {
            *h = a.clamp(0.0, 1.0);
        }

        let mut o_hidden = [0.0f32; NNUE_HIDDEN_SIZE];
        for (h, &a) in o_hidden.iter_mut().zip(o_acc.iter()) {
            *h = a.clamp(0.0, 1.0);
        }

        // 出力層 (Mover-first 結合)
        let mut output = self.output_bias;
        for (i, &h) in m_hidden.iter().enumerate() {
            output += h * self.output_weights[i];
        }
        for (i, &h) in o_hidden.iter().enumerate() {
            output += h * self.output_weights[NNUE_HIDDEN_SIZE + i];
        }

        // センチポーンスケール & 有界残差
        let bound = RESIDUAL_BOUND_CP as f32;
        let raw_residual_cp = output * 256.0;
        let residual_cp = raw_residual_cp.clamp(-bound, bound);

        (residual_cp, output, m_acc, o_acc, m_hidden, o_hidden)
    }

    /// シグモイド勝率予測
    #[inline(always)]
    pub fn sigmoid(score: f32, k: f32) -> f32 {
        1.0 / (1.0 + 10.0f32.powf(-score / k))
    }

    /// 単一バッチのフォワード・バックワード・AdamW更新
    /// (Mover-first 結合, Preactivation Range Penalty, AdamW Weight Decay, 飽和残差ガード対応)
    pub fn train_batch(
        &mut self,
        batch: &[(Vec<usize>, Vec<usize>, i32, f32)],
        lr: f32,
        k: f32,
    ) -> f32 {
        if batch.is_empty() {
            return 0.0;
        }

        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let epsilon = 1e-8f32;
        let weight_decay = 0.001f32;
        let lambda_range = 0.01f32; // Preactivation Range Penalty 係数
        let ln10_div_k = 10.0f32.ln() / k;
        let bound = RESIDUAL_BOUND_CP as f32;

        let mut grad_out_w = [0.0f32; NNUE_HIDDEN_SIZE * 2];
        let mut grad_out_b = 0.0f32;
        let mut grad_feature_biases = [0.0f32; NNUE_HIDDEN_SIZE];
        let mut total_loss = 0.0f32;

        let mut active_feature_updates: std::collections::HashMap<usize, [f32; NNUE_HIDDEN_SIZE]> =
            std::collections::HashMap::new();

        for (mover_feats, opp_feats, mat_stm, target) in batch {
            let (residual_stm, raw_output, m_acc, o_acc, m_hidden, o_hidden) =
                self.forward(mover_feats, opp_feats);
            let score_stm = *mat_stm as f32 + residual_stm;

            let pred = Self::sigmoid(score_stm, k);
            let error = pred - target; // (予測 - 教師信号)
            total_loss += error * error;

            // dL / d_score
            let d_sigmoid = pred * (1.0 - pred) * ln10_div_k;
            let d_loss_d_score = 2.0 * error * d_sigmoid;

            // 残差飽和ガード:
            // 境界外であっても、正常領域へ引き戻す勾配 (過大時の引き下げ / 過小時の引き上げ) は確実に通す
            let raw_residual = raw_output * 256.0;
            let d_output = if raw_residual >= bound {
                if d_loss_d_score > 0.0 {
                    d_loss_d_score * 256.0
                } else {
                    0.0
                }
            } else if raw_residual <= -bound {
                if d_loss_d_score < 0.0 {
                    d_loss_d_score * 256.0
                } else {
                    0.0
                }
            } else {
                d_loss_d_score * 256.0
            };

            grad_out_b += d_output;

            // 出力層重み勾配 & 隠れ層への逆伝播 (タスク勾配 + Range Penalty)
            let mut d_m_acc = [0.0f32; NNUE_HIDDEN_SIZE];
            for (i, &h) in m_hidden.iter().enumerate() {
                grad_out_w[i] += d_output * h;
                let task_grad = if m_acc[i] > 0.0 && m_acc[i] < 1.0 {
                    d_output * self.output_weights[i]
                } else {
                    0.0
                };
                let range_penalty = if m_acc[i] < 0.0 {
                    lambda_range * m_acc[i]
                } else if m_acc[i] > 1.0 {
                    lambda_range * (m_acc[i] - 1.0)
                } else {
                    0.0
                };
                d_m_acc[i] = task_grad + range_penalty;
            }

            let mut d_o_acc = [0.0f32; NNUE_HIDDEN_SIZE];
            for (i, &h) in o_hidden.iter().enumerate() {
                grad_out_w[NNUE_HIDDEN_SIZE + i] += d_output * h;
                let task_grad = if o_acc[i] > 0.0 && o_acc[i] < 1.0 {
                    d_output * self.output_weights[NNUE_HIDDEN_SIZE + i]
                } else {
                    0.0
                };
                let range_penalty = if o_acc[i] < 0.0 {
                    lambda_range * o_acc[i]
                } else if o_acc[i] > 1.0 {
                    lambda_range * (o_acc[i] - 1.0)
                } else {
                    0.0
                };
                d_o_acc[i] = task_grad + range_penalty;
            }

            // 隠れ層バイアスの勾配累積 (Mover / Opponent アキュムレータ勾配の和)
            for i in 0..NNUE_HIDDEN_SIZE {
                grad_feature_biases[i] += d_m_acc[i] + d_o_acc[i];
            }

            // 特徴量層への逆伝播 (スパース累積)
            for &f in mover_feats {
                if f < NNUE_INPUT_SIZE {
                    let entry = active_feature_updates
                        .entry(f)
                        .or_insert([0.0f32; NNUE_HIDDEN_SIZE]);
                    for (acc_w, &d) in entry.iter_mut().zip(d_m_acc.iter()) {
                        *acc_w += d;
                    }
                }
            }

            for &f in opp_feats {
                if f < NNUE_INPUT_SIZE {
                    let entry = active_feature_updates
                        .entry(f)
                        .or_insert([0.0f32; NNUE_HIDDEN_SIZE]);
                    for (acc_w, &d) in entry.iter_mut().zip(d_o_acc.iter()) {
                        *acc_w += d;
                    }
                }
            }
        }

        let inv_n = 1.0f32 / (batch.len() as f32);
        self.beta1_pow *= beta1;
        self.beta2_pow *= beta2;

        // 出力層の AdamW 更新 (Weight Decay 適用)
        for (i, &gw) in grad_out_w.iter().enumerate() {
            let g = gw * inv_n;
            self.m_out[i] = beta1 * self.m_out[i] + (1.0 - beta1) * g;
            self.v_out[i] = beta2 * self.v_out[i] + (1.0 - beta2) * g * g;

            let m_hat = self.m_out[i] / (1.0 - self.beta1_pow);
            let v_hat = self.v_out[i] / (1.0 - self.beta2_pow);

            self.output_weights[i] = self.output_weights[i] * (1.0 - lr * weight_decay)
                - lr * m_hat / (v_hat.sqrt() + epsilon);
        }

        // 出力バイアスの Adam 更新
        let g_bias = grad_out_b * inv_n;
        self.m_bias = beta1 * self.m_bias + (1.0 - beta1) * g_bias;
        self.v_bias = beta2 * self.v_bias + (1.0 - beta2) * g_bias * g_bias;
        let m_bias_hat = self.m_bias / (1.0 - self.beta1_pow);
        let v_bias_hat = self.v_bias / (1.0 - self.beta2_pow);
        self.output_bias -= lr * m_bias_hat / (v_bias_hat.sqrt() + epsilon);

        // 隠れ層バイアスの Adam 更新 (Range Penalty による自己修復引き戻し)
        for (i, &gb) in grad_feature_biases.iter().enumerate() {
            let g = gb * inv_n;
            self.m_f_bias[i] = beta1 * self.m_f_bias[i] + (1.0 - beta1) * g;
            self.v_f_bias[i] = beta2 * self.v_f_bias[i] + (1.0 - beta2) * g * g;

            let m_hat = self.m_f_bias[i] / (1.0 - self.beta1_pow);
            let v_hat = self.v_f_bias[i] / (1.0 - self.beta2_pow);

            self.feature_biases[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
        }

        // 特徴量重みの AdamW 更新 (スパース更新 + Weight Decay 適用)
        for (f, grad_slice) in active_feature_updates {
            for (j, &gj) in grad_slice.iter().enumerate() {
                let g = gj * inv_n;
                self.m_feat[f][j] = beta1 * self.m_feat[f][j] + (1.0 - beta1) * g;
                self.v_feat[f][j] = beta2 * self.v_feat[f][j] + (1.0 - beta2) * g * g;

                let m_hat = self.m_feat[f][j] / (1.0 - self.beta1_pow);
                let v_hat = self.v_feat[f][j] / (1.0 - self.beta2_pow);

                self.feature_weights[f][j] = self.feature_weights[f][j] * (1.0 - lr * weight_decay)
                    - lr * m_hat / (v_hat.sqrt() + epsilon);
            }
        }

        total_loss * inv_n
    }

    /// データセットからエポック学習を実行
    pub fn train_dataset(
        &mut self,
        dataset: &[DatasetEntry],
        epochs: usize,
        lr: f32,
        batch_size: usize,
        k: f32,
    ) -> (NNUEEvaluator, f32, f32) {
        if dataset.is_empty() {
            return (self.quantize(), 0.0, 0.0);
        }

        // 局面特徴量および手番側マテリアルベースラインを事前パース (Mover-first)
        let mut parsed_data = Vec::with_capacity(dataset.len());
        for entry in dataset {
            if let Ok(pos) = Position::from_sfen(&entry.sfen) {
                let mover = pos.side_to_move;
                let opp = mover.opposite();
                let mover_feats = NNUEEvaluator::extract_features(&pos, mover);
                let opp_feats = NNUEEvaluator::extract_features(&pos, opp);
                let mat_stm = NNUEEvaluator::material_stm(&pos);

                // 探索評価値（深読み教師）に主軸を置き(90%)、浅い対局outcomeの過剰影響を抑制(10%)
                let score_prob = Self::sigmoid(entry.score as f32, k);
                let target = 0.90 * score_prob + 0.10 * entry.result;

                parsed_data.push((mover_feats, opp_feats, mat_stm, target));
            }
        }

        if parsed_data.is_empty() {
            return (self.quantize(), 0.0, 0.0);
        }

        // 初期損失の計算
        let mut initial_loss = 0.0f32;
        for (mover_feats, opp_feats, mat_stm, target) in &parsed_data {
            let (residual_stm, _, _, _, _, _) = self.forward(mover_feats, opp_feats);
            let score_stm = *mat_stm as f32 + residual_stm;
            let pred = Self::sigmoid(score_stm, k);
            let err = pred - target;
            initial_loss += err * err;
        }
        initial_loss /= parsed_data.len() as f32;

        let b_size = batch_size.max(1);
        let mut final_loss = initial_loss;
        let mut rng = crate::selfplay::SimpleRng::new(0xdeadbeefc0ffee);

        for _epoch in 0..epochs {
            // エポックごとの Fisher-Yates シャッフル
            let len = parsed_data.len();
            for i in (1..len).rev() {
                let j = rng.gen_range(i + 1);
                parsed_data.swap(i, j);
            }

            let mut epoch_loss = 0.0f32;
            let mut batches = 0;

            for chunk in parsed_data.chunks(b_size) {
                let loss = self.train_batch(chunk, lr, k);
                epoch_loss += loss;
                batches += 1;
            }

            if batches > 0 {
                final_loss = epoch_loss / (batches as f32);
            }
        }

        (self.quantize(), initial_loss, final_loss)
    }

    /// 量子化NNUE評価器へ変換
    pub fn quantize(&self) -> NNUEEvaluator {
        NNUEEvaluator::from_float_weights(
            &self.feature_weights,
            &self.feature_biases,
            &self.output_weights,
            self.output_bias,
        )
    }
}
