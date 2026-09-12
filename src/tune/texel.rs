use super::params::{PARAM_COUNT, TunableParams};
use crate::board::Position;
use crate::types::{Color, PieceType};

/// 1局面の特徴量表現 (高速勾配計算用)
#[derive(Debug, Clone)]
pub struct PositionFeatures {
    /// 各パラメータに対応する特徴量カウント (手番視点)
    pub x: [f64; PARAM_COUNT],
    /// 終局結果 (1.0 = 勝ち, 0.5 = 引分, 0.0 = 負け)
    pub result: f64,
}

impl PositionFeatures {
    /// 局面から特徴量ベクトルを抽出
    pub fn extract(pos: &Position, result: f64) -> Self {
        let mut x = [0.0; PARAM_COUNT];
        let s = pos.side_to_move.sign() as f64;

        // 1. 盤上駒特徴量
        for sq_idx in 0..81 {
            if let Some(piece) = pos.board[sq_idx] {
                let idx = match piece.piece_type {
                    PieceType::Pawn => 0,
                    PieceType::Lance => 1,
                    PieceType::Knight => 2,
                    PieceType::Silver => 3,
                    PieceType::Gold => 4,
                    PieceType::Bishop => 5,
                    PieceType::Rook => 6,
                    PieceType::ProPawn => 7,
                    PieceType::ProLance => 8,
                    PieceType::ProKnight => 9,
                    PieceType::ProSilver => 10,
                    PieceType::Horse => 11,
                    PieceType::Dragon => 12,
                    PieceType::King => continue,
                };
                let piece_sign = if piece.color == Color::Black {
                    1.0
                } else {
                    -1.0
                };
                x[idx] += piece_sign * s;
            }
        }

        // 2. 持ち駒特徴量
        for (h_idx, &pt) in PieceType::HAND_PIECES.iter().enumerate() {
            let b_count = pos.hand[Color::Black.index()][h_idx] as f64;
            let w_count = pos.hand[Color::White.index()][h_idx] as f64;
            x[13 + h_idx] = (b_count - w_count) * s;
            let _ = pt;
        }

        // 3. 手番ボーナス特徴量
        x[20] = 1.0;

        PositionFeatures { x, result }
    }

    /// 特徴量ベクトルと重みの内積 (手番側視点の評価値)
    #[inline(always)]
    pub fn dot(&self, weights: &[f64]) -> f64 {
        self.x.iter().zip(weights.iter()).map(|(a, b)| a * b).sum()
    }
}

pub struct TexelTuner;

impl TexelTuner {
    pub const DEFAULT_K: f64 = 400.0;

    /// シグモイド関数 (Elo勝率モデル)
    #[inline(always)]
    pub fn sigmoid(score: f64, k: f64) -> f64 {
        1.0 / (1.0 + 10.0_f64.powf(-score / k))
    }

    /// 平均二乗誤差 (MSE) を計算
    pub fn compute_mse(features: &[PositionFeatures], weights: &[f64], k: f64) -> f64 {
        if features.is_empty() {
            return 0.0;
        }

        let mut total_loss = 0.0;
        for pf in features {
            let score = pf.dot(weights);
            let pred = Self::sigmoid(score, k);
            let diff = pf.result - pred;
            total_loss += diff * diff;
        }

        total_loss / (features.len() as f64)
    }

    /// Adamオプティマイザによるパラメータ最適化
    pub fn train_adam(
        features: &[PositionFeatures],
        initial_params: &TunableParams,
        epochs: usize,
        lr: f64,
        k: f64,
    ) -> (TunableParams, f64, f64) {
        let n = features.len();
        if n == 0 {
            return (initial_params.clone(), 0.0, 0.0);
        }

        let mut w = initial_params.to_vec();
        let initial_loss = Self::compute_mse(features, &w, k);

        let ln10_div_k = 10.0_f64.ln() / k;

        // Adam モーメンタム追跡変数
        let beta1 = 0.9_f64;
        let beta2 = 0.999_f64;
        let epsilon = 1e-8_f64;
        let mut m = [0.0; PARAM_COUNT];
        let mut v = [0.0; PARAM_COUNT];

        let mut beta1_pow = 1.0_f64;
        let mut beta2_pow = 1.0_f64;

        for _epoch in 0..epochs {
            let mut grad = [0.0; PARAM_COUNT];

            for pf in features {
                let score = pf.dot(&w);
                let pred = Self::sigmoid(score, k);
                let error = pred - pf.result; // (予測 - 実測)

                // 勾配計算: dE/dw_j = 2 * (pred - result) * pred * (1 - pred) * (ln(10) / K) * x_j
                let d_sigmoid = pred * (1.0 - pred) * ln10_div_k;
                let factor = 2.0 * error * d_sigmoid;

                for (g, &xj) in grad.iter_mut().zip(pf.x.iter()) {
                    *g += factor * xj;
                }
            }

            // 平均勾配
            let inv_n = 1.0 / (n as f64);
            for g in &mut grad {
                *g *= inv_n;
            }

            // Adam パラメータ更新
            beta1_pow *= beta1;
            beta2_pow *= beta2;

            for j in 0..PARAM_COUNT {
                m[j] = beta1 * m[j] + (1.0 - beta1) * grad[j];
                v[j] = beta2 * v[j] + (1.0 - beta2) * grad[j] * grad[j];

                let m_hat = m[j] / (1.0 - beta1_pow);
                let v_hat = v[j] / (1.0 - beta2_pow);

                w[j] -= lr * m_hat / (v_hat.sqrt() + epsilon);

                // 駒価値が負にならないようガード (最低限の物理的整合性)
                if j < 13 || (13..20).contains(&j) {
                    w[j] = w[j].max(10.0);
                }
            }
        }

        let final_loss = Self::compute_mse(features, &w, k);
        let mut tuned = initial_params.clone();
        tuned.from_vec(&w);

        (tuned, initial_loss, final_loss)
    }
}
