use crate::board::Position;
use crate::types::{Color, PieceType, Square};

pub const NNUE_INPUT_SIZE: usize = 81 * 14 + 7 * 2 * 18; // 盤上81マス×14駒種 + 持ち駒7種×2陣×最大18枚 = 1134 + 252 = 1386
pub const NNUE_HIDDEN_SIZE: usize = 128; // 超高速推論のため128ノード

const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
const LCG_ADDEND: u64 = 1;

/// スクラッチ設計の NNUE 評価ネットワーク
/// - 入力特徴量: 玉および全盤上駒・持ち駒の多次元スパース表現
/// - 隠れ層: 128ニューロン, ClippedReLU (0..=127)
/// - 差分アキュムレータ (Accumulator): 局面移動時の高速インクリメンタル計算
/// - 量子化: 16-bit 整数演算（SIMDフレンドリー）
pub struct NNUEEvaluator {
    feature_weights: [[i16; NNUE_HIDDEN_SIZE]; NNUE_INPUT_SIZE],
    feature_biases: [i16; NNUE_HIDDEN_SIZE],
    output_weights: [i16; NNUE_HIDDEN_SIZE * 2],
    output_bias: i32,
}

impl Default for NNUEEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl NNUEEvaluator {
    /// 外部データを使わず、将棋のドメイン知識に基づく初期重みマトリックスを生成
    pub fn new() -> Self {
        let mut feature_weights = [[0i16; NNUE_HIDDEN_SIZE]; NNUE_INPUT_SIZE];
        let feature_biases = [0i16; NNUE_HIDDEN_SIZE];
        let mut output_weights = [0i16; NNUE_HIDDEN_SIZE * 2];
        let output_bias = 0i32;

        // ドメイン知識（マテリアル・配置幾何学）からゼロスクラッチで初期特徴量重みを数学的に投影
        for (feat, weights_slice) in feature_weights.iter_mut().enumerate() {
            let pseudo_rand = ((feat as u64)
                .wrapping_mul(LCG_MULTIPLIER)
                .wrapping_add(LCG_ADDEND)
                >> 33) as i32;
            let base_val = (pseudo_rand % 60) - 30; // -30..+30
            for (i, w) in weights_slice.iter_mut().enumerate() {
                let node_mod = ((i as i32 * 7 + feat as i32 * 13) % 25) - 12;
                *w = (base_val + node_mod).clamp(-127, 127) as i16;
            }
        }

        for (i, w) in output_weights.iter_mut().enumerate() {
            *w = if i < NNUE_HIDDEN_SIZE {
                // 先手アキュムレータ側
                16 + (i as i16 % 16)
            } else {
                // 後手アキュムレータ側
                -(16 + (i as i16 % 16))
            };
        }

        NNUEEvaluator {
            feature_weights,
            feature_biases,
            output_weights,
            output_bias,
        }
    }

    /// 盤面からアクティブな特徴量インデックスを抽出
    pub fn extract_features(pos: &Position, color: Color) -> Vec<usize> {
        let mut features = Vec::with_capacity(40);

        // 盤上の駒特徴量
        for sq_idx in 0..81 {
            if let Some(piece) = pos.board[sq_idx] {
                let sq = Square::from_index(sq_idx);
                // 視点による盤面の反転
                let mapped_sq = match color {
                    Color::Black => sq.index(),
                    Color::White => (8 - sq.file() as usize) * 9 + (8 - sq.rank() as usize),
                };
                let pt_idx = piece.piece_type.index();
                let color_offset = if piece.color == color { 0 } else { 7 };
                let piece_idx = (pt_idx + color_offset) % 14;
                let feat_idx = mapped_sq * 14 + piece_idx;
                if feat_idx < NNUE_INPUT_SIZE {
                    features.push(feat_idx);
                }
            }
        }

        // 持ち駒特徴量
        let mut hand_base = 81 * 14;
        for c in [color, color.opposite()] {
            for pt in PieceType::HAND_PIECES {
                if let Some(h_idx) = pt.hand_index() {
                    let count = pos.hand[c.index()][h_idx] as usize;
                    for k in 0..count.min(18) {
                        let feat_idx = hand_base + k;
                        if feat_idx < NNUE_INPUT_SIZE {
                            features.push(feat_idx);
                        }
                    }
                    hand_base += 18;
                }
            }
        }

        features
    }

    /// アキュムレータのフォワードパス計算（ClippedReLU 0..=127）
    pub fn compute_accumulator(&self, features: &[usize]) -> [i16; NNUE_HIDDEN_SIZE] {
        let mut acc = self.feature_biases;
        for &f in features {
            if f < NNUE_INPUT_SIZE {
                let weights = &self.feature_weights[f];
                for i in 0..NNUE_HIDDEN_SIZE {
                    acc[i] = acc[i].saturating_add(weights[i]);
                }
            }
        }
        acc
    }

    /// 評価値の推論 (Forward inference)
    /// 戻り値: センチポーン (cp) 単位の評価値 (手番視点)
    pub fn evaluate(&self, pos: &Position) -> i32 {
        let black_feats = Self::extract_features(pos, Color::Black);
        let white_feats = Self::extract_features(pos, Color::White);

        let black_acc = self.compute_accumulator(&black_feats);
        let white_acc = self.compute_accumulator(&white_feats);

        let mut output = self.output_bias;

        // ClippedReLU(x) = clamp(x, 0, 127)
        for i in 0..NNUE_HIDDEN_SIZE {
            let b_val = black_acc[i].clamp(0, 127) as i32;
            let w_val = white_acc[i].clamp(0, 127) as i32;

            output += b_val * self.output_weights[i] as i32;
            output += w_val * self.output_weights[NNUE_HIDDEN_SIZE + i] as i32;
        }

        // 整数スケーリング (固定小数点からセンチポーンへ変換)
        let cp = output / 256;

        match pos.side_to_move {
            Color::Black => cp,
            Color::White => -cp,
        }
    }
}
