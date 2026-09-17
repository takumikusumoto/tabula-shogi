use super::halfkp::{HALFKP_HIDDEN_SIZE, HALFKP_INPUT_SIZE, HalfKPEvaluator, MAX_EVAL_CP};
use std::collections::HashMap;
use std::io::{self, Read, Write};

/// バッチ学習ステップの各種損失統計
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainStepLoss {
    /// 予測勝率と勝敗教師信号の平均二乗誤差 (MSE)
    pub mse_loss: f32,
    /// 前活性 (Preactivation) の [0, 64] 逸脱に対する Range Penalty
    pub range_loss: f32,
    /// 総損失 (mse_loss + range_loss)
    pub total_loss: f32,
}

impl TrainStepLoss {
    pub fn zero() -> Self {
        Self {
            mse_loss: 0.0,
            range_loss: 0.0,
            total_loss: 0.0,
        }
    }
}

/// ゼロ外部依存のスクラッチ HalfKP バックプロパゲーション学習器
/// - 特徴量次元: 204,120 (自玉81 × 全駒2,520)
/// - 隠れ層: 128 (先手128 + 後手128 = 256次元結合)
/// - 最適化手法: スパース AdamW (PyTorch SparseAdam 準拠のローカル更新ステップ)
/// - 活性化関数: ClippedReLU (0.0..=64.0)
/// - 評価値スケール: Evaluator と完全一致 (output / 128 = cp)
pub struct HalfKPTrainer {
    pub feature_weights: Vec<[f32; HALFKP_HIDDEN_SIZE]>,
    pub feature_biases: [f32; HALFKP_HIDDEN_SIZE],
    pub output_weights: [f32; HALFKP_HIDDEN_SIZE * 2],
    pub output_bias: f32,

    // AdamW モーメンタム状態 (スパース特徴量重み用)
    pub m_feat: Vec<[f32; HALFKP_HIDDEN_SIZE]>,
    pub v_feat: Vec<[f32; HALFKP_HIDDEN_SIZE]>,
    /// 特徴量ごとのローカル更新回数 (スパース AdamW のバイアス補正用)
    pub step_feat: Vec<u32>,
    pub m_f_bias: [f32; HALFKP_HIDDEN_SIZE],
    pub v_f_bias: [f32; HALFKP_HIDDEN_SIZE],
    pub m_out: [f32; HALFKP_HIDDEN_SIZE * 2],
    pub v_out: [f32; HALFKP_HIDDEN_SIZE * 2],
    pub m_bias: f32,
    pub v_bias: f32,
    pub beta1_pow: f32,
    pub beta2_pow: f32,
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
            step_feat: vec![0u32; HALFKP_INPUT_SIZE],
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
    /// - 戻り値: `TrainStepLoss` (MSE損失, Range Penalty損失, 総損失)
    pub fn train_batch(
        &mut self,
        batch: &[(Vec<usize>, Vec<usize>, f32)],
        lr: f32,
        k: f32,
    ) -> TrainStepLoss {
        if batch.is_empty() {
            return TrainStepLoss::zero();
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
        let mut total_mse_loss = 0.0f32;
        let mut total_range_loss = 0.0f32;

        let mut active_feature_updates: HashMap<usize, [f32; HALFKP_HIDDEN_SIZE]> = HashMap::new();

        for (mover_feats, opp_feats, target) in batch {
            let (score_cp, raw_output, m_acc, o_acc, m_hidden, o_hidden) =
                self.forward(mover_feats, opp_feats);

            let pred = Self::sigmoid(score_cp, k);
            let error = pred - target; // (予測 - 教師信号)
            total_mse_loss += error * error;

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
                let (range_grad, range_loss) = if m_acc[i] < 0.0 {
                    (
                        lambda_range * m_acc[i],
                        0.5 * lambda_range * m_acc[i] * m_acc[i],
                    )
                } else if m_acc[i] > 64.0 {
                    let diff = m_acc[i] - 64.0;
                    (lambda_range * diff, 0.5 * lambda_range * diff * diff)
                } else {
                    (0.0, 0.0)
                };
                total_range_loss += range_loss;
                d_m_acc[i] = task_grad + range_grad;
            }

            let mut d_o_acc = [0.0f32; HALFKP_HIDDEN_SIZE];
            for (i, &h) in o_hidden.iter().enumerate() {
                grad_out_w[HALFKP_HIDDEN_SIZE + i] += d_output * h;
                let task_grad = if o_acc[i] > 0.0 && o_acc[i] < 64.0 {
                    d_output * self.output_weights[HALFKP_HIDDEN_SIZE + i]
                } else {
                    0.0
                };
                let (range_grad, range_loss) = if o_acc[i] < 0.0 {
                    (
                        lambda_range * o_acc[i],
                        0.5 * lambda_range * o_acc[i] * o_acc[i],
                    )
                } else if o_acc[i] > 64.0 {
                    let diff = o_acc[i] - 64.0;
                    (lambda_range * diff, 0.5 * lambda_range * diff * diff)
                } else {
                    (0.0, 0.0)
                };
                total_range_loss += range_loss;
                d_o_acc[i] = task_grad + range_grad;
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

        // AdamW 密パラメータ更新用バイアス補正係数 (出力層・バイアス)
        self.beta1_pow *= beta1;
        self.beta2_pow *= beta2;
        let one_minus_beta1 = 1.0 - self.beta1_pow;
        let one_minus_beta2 = 1.0 - self.beta2_pow;

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

        // 4. スパース特徴量重み更新 (PyTorch SparseAdam 準拠のローカルタイムステップ)
        for (f, grad_slice) in active_feature_updates {
            let step = self.step_feat[f].saturating_add(1);
            self.step_feat[f] = step;
            let one_minus_b1_f = if step > 1000 {
                1.0
            } else {
                1.0 - beta1.powi(step as i32)
            };
            let one_minus_b2_f = if step > 10000 {
                1.0
            } else {
                1.0 - beta2.powi(step as i32)
            };

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
                    one_minus_b1_f,
                    one_minus_b2_f,
                    epsilon,
                    weight_decay,
                );
            }
        }

        let mse = total_mse_loss / batch_size_f;
        let range = total_range_loss / batch_size_f;
        TrainStepLoss {
            mse_loss: mse,
            range_loss: range,
            total_loss: mse + range,
        }
    }

    /// Trainer 状態（浮動小数点重み、Adam モーメンタム、ローカル更新回数）を安全にアトミック保存
    /// 一時ファイル (.tmp) への書き出し、置換前の直前世代バックアップ (.bak) 確保、およびアトミックリネームにより
    /// 314MBの書き込み途中でのクラッシュや停電による既存チェックポイントの道連れ破壊を100%防止
    pub fn save_checkpoint(&self, path: &str) -> io::Result<()> {
        let tmp_path = format!("{path}.tmp");
        let bak_path = format!("{path}.bak");

        let file = std::fs::File::create(&tmp_path)?;
        let mut writer = io::BufWriter::new(file);

        writer.write_all(b"TB_HKPCK")?;
        writer.write_all(&self.beta1_pow.to_le_bytes())?;
        writer.write_all(&self.beta2_pow.to_le_bytes())?;
        writer.write_all(&self.output_bias.to_le_bytes())?;
        writer.write_all(&self.m_bias.to_le_bytes())?;
        writer.write_all(&self.v_bias.to_le_bytes())?;

        for &b in &self.feature_biases {
            writer.write_all(&b.to_le_bytes())?;
        }
        for &m in &self.m_f_bias {
            writer.write_all(&m.to_le_bytes())?;
        }
        for &v in &self.v_f_bias {
            writer.write_all(&v.to_le_bytes())?;
        }

        for &w in &self.output_weights {
            writer.write_all(&w.to_le_bytes())?;
        }
        for &m in &self.m_out {
            writer.write_all(&m.to_le_bytes())?;
        }
        for &v in &self.v_out {
            writer.write_all(&v.to_le_bytes())?;
        }

        for row in &self.feature_weights {
            for &w in row {
                writer.write_all(&w.to_le_bytes())?;
            }
        }
        for row in &self.m_feat {
            for &m in row {
                writer.write_all(&m.to_le_bytes())?;
            }
        }
        for row in &self.v_feat {
            for &v in row {
                writer.write_all(&v.to_le_bytes())?;
            }
        }
        for &s in &self.step_feat {
            writer.write_all(&s.to_le_bytes())?;
        }

        writer.flush()?;
        drop(writer);

        // 置換前に既存の正常なチェックポイントを直前世代バックアップ (.bak) として確保
        // (バックアップ作成に失敗した場合はエラーを返して既存ファイルを保護)
        if std::path::Path::new(path).exists() {
            std::fs::copy(path, &bak_path)?;
            let _ = std::fs::remove_file(path);
        }

        // 一時ファイルを本番パスへアトミックリネーム
        std::fs::rename(&tmp_path, path)?;

        Ok(())
    }

    /// Trainer 状態をバイナリから完全復元
    pub fn load_checkpoint(path: &str) -> Result<Self, String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("Failed to open HalfKP checkpoint '{path}': {e}"))?;

        // チェックポイントファイルサイズの事前検証 (破損・切り詰めの早期検出)
        let expected_len: u64 = 8  // magic "TB_HKPCK"
            + 5 * 4  // beta1_pow, beta2_pow, output_bias, m_bias, v_bias
            + 3 * (HALFKP_HIDDEN_SIZE as u64) * 4  // feature_biases, m_f_bias, v_f_bias
            + 3 * (HALFKP_HIDDEN_SIZE as u64 * 2) * 4  // output_weights, m_out, v_out
            + 3 * (HALFKP_INPUT_SIZE as u64) * (HALFKP_HIDDEN_SIZE as u64) * 4  // feature_weights, m_feat, v_feat
            + (HALFKP_INPUT_SIZE as u64) * 4; // step_feat
        let file_len = file
            .metadata()
            .map_err(|e| format!("Failed to read checkpoint file metadata: {e}"))?
            .len();
        if file_len != expected_len {
            return Err(format!(
                "HalfKP checkpoint file size mismatch: got {} bytes, expected {} bytes",
                file_len, expected_len
            ));
        }

        let mut reader = io::BufReader::new(file);

        let mut magic = [0u8; 8];
        reader
            .read_exact(&mut magic)
            .map_err(|e| format!("Failed to read checkpoint magic: {e}"))?;
        if &magic != b"TB_HKPCK" {
            return Err("Invalid HalfKP checkpoint magic header".to_string());
        }

        let read_f32 = |r: &mut io::BufReader<std::fs::File>| -> Result<f32, String> {
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf)
                .map_err(|e| format!("Checkpoint read f32 error: {e}"))?;
            Ok(f32::from_le_bytes(buf))
        };
        let read_u32 = |r: &mut io::BufReader<std::fs::File>| -> Result<u32, String> {
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf)
                .map_err(|e| format!("Checkpoint read u32 error: {e}"))?;
            Ok(u32::from_le_bytes(buf))
        };

        let beta1_pow = read_f32(&mut reader)?;
        let beta2_pow = read_f32(&mut reader)?;
        let output_bias = read_f32(&mut reader)?;
        let m_bias = read_f32(&mut reader)?;
        let v_bias = read_f32(&mut reader)?;

        let mut feature_biases = [0.0f32; HALFKP_HIDDEN_SIZE];
        for b in &mut feature_biases {
            *b = read_f32(&mut reader)?;
        }
        let mut m_f_bias = [0.0f32; HALFKP_HIDDEN_SIZE];
        for m in &mut m_f_bias {
            *m = read_f32(&mut reader)?;
        }
        let mut v_f_bias = [0.0f32; HALFKP_HIDDEN_SIZE];
        for v in &mut v_f_bias {
            *v = read_f32(&mut reader)?;
        }

        let mut output_weights = [0.0f32; HALFKP_HIDDEN_SIZE * 2];
        for w in &mut output_weights {
            *w = read_f32(&mut reader)?;
        }
        let mut m_out = [0.0f32; HALFKP_HIDDEN_SIZE * 2];
        for m in &mut m_out {
            *m = read_f32(&mut reader)?;
        }
        let mut v_out = [0.0f32; HALFKP_HIDDEN_SIZE * 2];
        for v in &mut v_out {
            *v = read_f32(&mut reader)?;
        }

        let mut feature_weights = Vec::with_capacity(HALFKP_INPUT_SIZE);
        for _ in 0..HALFKP_INPUT_SIZE {
            let mut row = [0.0f32; HALFKP_HIDDEN_SIZE];
            for w in &mut row {
                *w = read_f32(&mut reader)?;
            }
            feature_weights.push(row);
        }

        let mut m_feat = Vec::with_capacity(HALFKP_INPUT_SIZE);
        for _ in 0..HALFKP_INPUT_SIZE {
            let mut row = [0.0f32; HALFKP_HIDDEN_SIZE];
            for m in &mut row {
                *m = read_f32(&mut reader)?;
            }
            m_feat.push(row);
        }

        let mut v_feat = Vec::with_capacity(HALFKP_INPUT_SIZE);
        for _ in 0..HALFKP_INPUT_SIZE {
            let mut row = [0.0f32; HALFKP_HIDDEN_SIZE];
            for v in &mut row {
                *v = read_f32(&mut reader)?;
            }
            v_feat.push(row);
        }

        let mut step_feat = Vec::with_capacity(HALFKP_INPUT_SIZE);
        for _ in 0..HALFKP_INPUT_SIZE {
            step_feat.push(read_u32(&mut reader)?);
        }

        Ok(HalfKPTrainer {
            feature_weights,
            feature_biases,
            output_weights,
            output_bias,
            m_feat,
            v_feat,
            step_feat,
            m_f_bias,
            v_f_bias,
            m_out,
            v_out,
            m_bias,
            v_bias,
            beta1_pow,
            beta2_pow,
        })
    }
}
