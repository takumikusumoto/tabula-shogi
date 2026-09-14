use super::nnue::{NNUE_HIDDEN_SIZE, NNUE_INPUT_SIZE, NNUEEvaluator};
use crate::board::Position;
use crate::selfplay::DatasetEntry;
use crate::types::Color;

/// ゼロ外部依存のスクラッチ NNUE バックプロパゲーション学習器
pub struct NNUETrainer {
    pub feature_weights: Vec<[f32; NNUE_HIDDEN_SIZE]>,
    pub feature_biases: [f32; NNUE_HIDDEN_SIZE],
    pub output_weights: [f32; NNUE_HIDDEN_SIZE * 2],
    pub output_bias: f32,

    // Adam モーメンタム追跡状態
    m_feat: Vec<[f32; NNUE_HIDDEN_SIZE]>,
    v_feat: Vec<[f32; NNUE_HIDDEN_SIZE]>,
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
    pub fn new() -> Self {
        let initial_eval = NNUEEvaluator::new();
        let mut feature_weights = Vec::with_capacity(NNUE_INPUT_SIZE);
        for row in &initial_eval.feature_weights {
            let mut f_row = [0.0f32; NNUE_HIDDEN_SIZE];
            for (w, &q) in f_row.iter_mut().zip(row.iter()) {
                *w = q as f32 / 64.0;
            }
            feature_weights.push(f_row);
        }

        let mut feature_biases = [0.0f32; NNUE_HIDDEN_SIZE];
        for (b, &q) in feature_biases
            .iter_mut()
            .zip(initial_eval.feature_biases.iter())
        {
            *b = q as f32 / 64.0;
        }

        let mut output_weights = [0.0f32; NNUE_HIDDEN_SIZE * 2];
        for (w, &q) in output_weights
            .iter_mut()
            .zip(initial_eval.output_weights.iter())
        {
            *w = q as f32 / 64.0;
        }

        let output_bias = initial_eval.output_bias as f32 / 256.0;

        NNUETrainer {
            feature_weights,
            feature_biases,
            output_weights,
            output_bias,
            m_feat: vec![[0.0f32; NNUE_HIDDEN_SIZE]; NNUE_INPUT_SIZE],
            v_feat: vec![[0.0f32; NNUE_HIDDEN_SIZE]; NNUE_INPUT_SIZE],
            m_out: [0.0f32; NNUE_HIDDEN_SIZE * 2],
            v_out: [0.0f32; NNUE_HIDDEN_SIZE * 2],
            m_bias: 0.0,
            v_bias: 0.0,
            beta1_pow: 1.0,
            beta2_pow: 1.0,
        }
    }

    /// フォワードパス
    /// 戻り値: (先手視点スコア, b_acc, w_acc, b_hidden, w_hidden)
    pub fn forward(
        &self,
        b_feats: &[usize],
        w_feats: &[usize],
    ) -> (
        f32,
        [f32; NNUE_HIDDEN_SIZE],
        [f32; NNUE_HIDDEN_SIZE],
        [f32; NNUE_HIDDEN_SIZE],
        [f32; NNUE_HIDDEN_SIZE],
    ) {
        let mut b_acc = self.feature_biases;
        for &f in b_feats {
            if f < NNUE_INPUT_SIZE {
                for (a, &w) in b_acc.iter_mut().zip(self.feature_weights[f].iter()) {
                    *a += w;
                }
            }
        }

        let mut w_acc = self.feature_biases;
        for &f in w_feats {
            if f < NNUE_INPUT_SIZE {
                for (a, &w) in w_acc.iter_mut().zip(self.feature_weights[f].iter()) {
                    *a += w;
                }
            }
        }

        // ClippedReLU (0.0..=1.0)
        let mut b_hidden = [0.0f32; NNUE_HIDDEN_SIZE];
        for (h, &a) in b_hidden.iter_mut().zip(b_acc.iter()) {
            *h = a.clamp(0.0, 1.0);
        }

        let mut w_hidden = [0.0f32; NNUE_HIDDEN_SIZE];
        for (h, &a) in w_hidden.iter_mut().zip(w_acc.iter()) {
            *h = a.clamp(0.0, 1.0);
        }

        // 出力層
        let mut output = self.output_bias;
        for (i, &h) in b_hidden.iter().enumerate() {
            output += h * self.output_weights[i];
        }
        for (i, &h) in w_hidden.iter().enumerate() {
            output += h * self.output_weights[NNUE_HIDDEN_SIZE + i];
        }

        // センチポーンスケール
        let score_cp = output * 256.0;

        (score_cp, b_acc, w_acc, b_hidden, w_hidden)
    }

    /// シグモイド勝率予測
    #[inline(always)]
    pub fn sigmoid(score: f32, k: f32) -> f32 {
        1.0 / (1.0 + 10.0f32.powf(-score / k))
    }

    /// 単一バッチのフォワード・バックワード・Adam更新
    #[allow(clippy::too_many_arguments)]
    pub fn train_batch(
        &mut self,
        batch: &[(Vec<usize>, Vec<usize>, Color, f32)],
        lr: f32,
        k: f32,
    ) -> f32 {
        if batch.is_empty() {
            return 0.0;
        }

        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let epsilon = 1e-8f32;
        let ln10_div_k = 10.0f32.ln() / k;

        let mut grad_out_w = [0.0f32; NNUE_HIDDEN_SIZE * 2];
        let mut grad_out_b = 0.0f32;
        let mut total_loss = 0.0f32;

        let mut active_feature_updates: std::collections::HashMap<usize, [f32; NNUE_HIDDEN_SIZE]> =
            std::collections::HashMap::new();

        for (b_feats, w_feats, turn, result) in batch {
            let (score_black, b_acc, w_acc, b_hidden, w_hidden) = self.forward(b_feats, w_feats);
            let score = match turn {
                Color::Black => score_black,
                Color::White => -score_black,
            };

            let pred = Self::sigmoid(score, k);
            let error = pred - result; // (予測 - 実績)
            total_loss += error * error;

            // dL / d_score
            let d_sigmoid = pred * (1.0 - pred) * ln10_div_k;
            let d_loss_d_score = 2.0 * error * d_sigmoid;

            // dL / d_output (手番による符号考慮)
            let turn_sign = match turn {
                Color::Black => 1.0f32,
                Color::White => -1.0f32,
            };
            let d_output = d_loss_d_score * turn_sign * 256.0;

            grad_out_b += d_output;

            // 出力層重み勾配 & 隠れ層への逆伝播
            let mut d_b_acc = [0.0f32; NNUE_HIDDEN_SIZE];
            for (i, &h) in b_hidden.iter().enumerate() {
                grad_out_w[i] += d_output * h;
                // ClippedReLU gradient: 0.0 < acc < 1.0 のみ通過
                if b_acc[i] > 0.0 && b_acc[i] < 1.0 {
                    d_b_acc[i] = d_output * self.output_weights[i];
                }
            }

            let mut d_w_acc = [0.0f32; NNUE_HIDDEN_SIZE];
            for (i, &h) in w_hidden.iter().enumerate() {
                grad_out_w[NNUE_HIDDEN_SIZE + i] += d_output * h;
                if w_acc[i] > 0.0 && w_acc[i] < 1.0 {
                    d_w_acc[i] = d_output * self.output_weights[NNUE_HIDDEN_SIZE + i];
                }
            }

            // 特徴量層への逆伝播 (スパース累積)
            for &f in b_feats {
                if f < NNUE_INPUT_SIZE {
                    let entry = active_feature_updates
                        .entry(f)
                        .or_insert([0.0f32; NNUE_HIDDEN_SIZE]);
                    for (acc_w, &d) in entry.iter_mut().zip(d_b_acc.iter()) {
                        *acc_w += d;
                    }
                }
            }

            for &f in w_feats {
                if f < NNUE_INPUT_SIZE {
                    let entry = active_feature_updates
                        .entry(f)
                        .or_insert([0.0f32; NNUE_HIDDEN_SIZE]);
                    for (acc_w, &d) in entry.iter_mut().zip(d_w_acc.iter()) {
                        *acc_w += d;
                    }
                }
            }
        }

        let inv_n = 1.0f32 / (batch.len() as f32);
        self.beta1_pow *= beta1;
        self.beta2_pow *= beta2;

        // 出力層の Adam 更新
        for (i, &gw) in grad_out_w.iter().enumerate() {
            let g = gw * inv_n;
            self.m_out[i] = beta1 * self.m_out[i] + (1.0 - beta1) * g;
            self.v_out[i] = beta2 * self.v_out[i] + (1.0 - beta2) * g * g;

            let m_hat = self.m_out[i] / (1.0 - self.beta1_pow);
            let v_hat = self.v_out[i] / (1.0 - self.beta2_pow);

            self.output_weights[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
        }

        let g_bias = grad_out_b * inv_n;
        self.m_bias = beta1 * self.m_bias + (1.0 - beta1) * g_bias;
        self.v_bias = beta2 * self.v_bias + (1.0 - beta2) * g_bias * g_bias;
        let m_bias_hat = self.m_bias / (1.0 - self.beta1_pow);
        let v_bias_hat = self.v_bias / (1.0 - self.beta2_pow);
        self.output_bias -= lr * m_bias_hat / (v_bias_hat.sqrt() + epsilon);

        // 特徴量重みの Adam 更新 (スパース更新)
        for (f, grad_slice) in active_feature_updates {
            for (j, &gj) in grad_slice.iter().enumerate() {
                let g = gj * inv_n;
                self.m_feat[f][j] = beta1 * self.m_feat[f][j] + (1.0 - beta1) * g;
                self.v_feat[f][j] = beta2 * self.v_feat[f][j] + (1.0 - beta2) * g * g;

                let m_hat = self.m_feat[f][j] / (1.0 - self.beta1_pow);
                let v_hat = self.v_feat[f][j] / (1.0 - self.beta2_pow);

                self.feature_weights[f][j] -= lr * m_hat / (v_hat.sqrt() + epsilon);
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

        // 局面特徴量を事前パース
        let mut parsed_data = Vec::with_capacity(dataset.len());
        for entry in dataset {
            if let Ok(pos) = Position::from_sfen(&entry.sfen) {
                let b_feats = NNUEEvaluator::extract_features(&pos, Color::Black);
                let w_feats = NNUEEvaluator::extract_features(&pos, Color::White);
                parsed_data.push((b_feats, w_feats, pos.side_to_move, entry.result));
            }
        }

        if parsed_data.is_empty() {
            return (self.quantize(), 0.0, 0.0);
        }

        // 初期損失の計算
        let mut initial_loss = 0.0f32;
        for (b_feats, w_feats, turn, result) in &parsed_data {
            let (score_black, _, _, _, _) = self.forward(b_feats, w_feats);
            let score = match turn {
                Color::Black => score_black,
                Color::White => -score_black,
            };
            let pred = Self::sigmoid(score, k);
            let err = pred - result;
            initial_loss += err * err;
        }
        initial_loss /= parsed_data.len() as f32;

        let b_size = batch_size.max(1);
        let mut final_loss = initial_loss;
        let mut rng = crate::selfplay::SimpleRng::new(0xdeadbeefc0ffee);

        for _epoch in 0..epochs {
            // エポックごとの Fisher-Yates シャッフル（ミニバッチ間の相関を解消）
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
