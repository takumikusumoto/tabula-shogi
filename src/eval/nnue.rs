use crate::board::Position;
use crate::types::{Color, PieceType, Square};

pub const NNUE_INPUT_SIZE: usize = 81 * 28 + 7 * 2 * 18; // 盤上81マス×28駒種(自駒14+敵駒14) + 持ち駒7種×2陣×最大18枚 = 2268 + 252 = 2520
pub const NNUE_HIDDEN_SIZE: usize = 128; // 超高速推論のため128ノード

const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
const LCG_ADDEND: u64 = 1;

pub const NNUE_MAGIC: &[u8; 8] = b"TABU_NN5";
pub const RESIDUAL_BOUND_CP: i32 = 600;

/// スクラッチ設計の Residual Baseline NNUE 評価ネットワーク
/// - ベースライン: 完全な盤上・持ち駒の駒割り (Material Balance)
/// - 残差ネットワーク: 配置・玉の堅さ・手番等の高度な評価 (Residual)
/// - 入力特徴量: 玉および全盤上駒(自軍14+敵軍14)・持ち駒の多次元スパース表現 (2520次元)
/// - 隠れ層: 128ニューロン, ClippedReLU (0..=64)
/// - 差分アキュムレータ (Accumulator): 局面移動時の高速インクリメンタル計算
/// - 量子化: 16-bit 整数演算（SIMDフレンドリー）
#[derive(Clone, Debug)]
pub struct NNUEEvaluator {
    pub feature_weights: Vec<[i16; NNUE_HIDDEN_SIZE]>,
    pub feature_biases: [i16; NNUE_HIDDEN_SIZE],
    pub output_weights: [i16; NNUE_HIDDEN_SIZE * 2],
    pub output_bias: i32,
}

impl Default for NNUEEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl NNUEEvaluator {
    /// 残差ベースライン方式に基づく健全な初期化
    /// - 初期出力重み/バイアス: 0 (初期状態の評価値は 100% 正確な駒割りに一致)
    /// - 隠れ層バイアス: 32 (リニア活性領域 [0, 64] の中心に配置し、死滅ニューロンゼロを保証)
    /// - 特徴量重み: {-1, 0, +1} の多様な決定論的サンプリング (相関を崩し学習準備)
    pub fn new() -> Self {
        let mut feature_weights = vec![[0i16; NNUE_HIDDEN_SIZE]; NNUE_INPUT_SIZE];
        let feature_biases = [32i16; NNUE_HIDDEN_SIZE];
        let output_weights = [0i16; NNUE_HIDDEN_SIZE * 2];
        let output_bias = 0i32;

        for (feat, weights_slice) in feature_weights.iter_mut().enumerate() {
            for (i, w) in weights_slice.iter_mut().enumerate() {
                // 周期性のない多様なサンプリング (-1, 0, +1)
                let h = ((feat as u64)
                    .wrapping_mul(LCG_MULTIPLIER)
                    .wrapping_add((i as u64).wrapping_mul(0x9e3779b97f4a7c15))
                    .wrapping_add(LCG_ADDEND)
                    >> 33) as i32;
                *w = ((h % 3) - 1) as i16;
            }
        }

        NNUEEvaluator {
            feature_weights,
            feature_biases,
            output_weights,
            output_bias,
        }
    }

    /// 浮動小数点学習重みから量子化NNUE評価器を構築
    pub fn from_float_weights(
        feature_weights: &[[f32; NNUE_HIDDEN_SIZE]],
        feature_biases: &[f32; NNUE_HIDDEN_SIZE],
        output_weights: &[f32; NNUE_HIDDEN_SIZE * 2],
        output_bias: f32,
    ) -> Self {
        let mut quantized_feats = Vec::with_capacity(NNUE_INPUT_SIZE);
        for row in feature_weights {
            let mut q_row = [0i16; NNUE_HIDDEN_SIZE];
            for (w, &f) in q_row.iter_mut().zip(row.iter()) {
                *w = (f * 64.0).clamp(-127.0, 127.0).round() as i16;
            }
            quantized_feats.push(q_row);
        }

        let mut quantized_biases = [0i16; NNUE_HIDDEN_SIZE];
        for (b, &f) in quantized_biases.iter_mut().zip(feature_biases.iter()) {
            *b = (f * 64.0).clamp(-127.0, 127.0).round() as i16;
        }

        let mut quantized_out = [0i16; NNUE_HIDDEN_SIZE * 2];
        for (w, &f) in quantized_out.iter_mut().zip(output_weights.iter()) {
            *w = (f * 64.0).clamp(-127.0, 127.0).round() as i16;
        }

        let quantized_out_bias = (output_bias * 4096.0).round() as i32;

        NNUEEvaluator {
            feature_weights: quantized_feats,
            feature_biases: quantized_biases,
            output_weights: quantized_out,
            output_bias: quantized_out_bias,
        }
    }

    /// バイナリバイト列へシリアライズ (約355 KB)
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(
            8 + 8
                + NNUE_INPUT_SIZE * NNUE_HIDDEN_SIZE * 2
                + NNUE_HIDDEN_SIZE * 2
                + NNUE_HIDDEN_SIZE * 4
                + 4,
        );
        bytes.extend_from_slice(NNUE_MAGIC);
        bytes.extend_from_slice(&(NNUE_INPUT_SIZE as u32).to_le_bytes());
        bytes.extend_from_slice(&(NNUE_HIDDEN_SIZE as u32).to_le_bytes());

        for row in &self.feature_weights {
            for &w in row {
                bytes.extend_from_slice(&w.to_le_bytes());
            }
        }
        for &b in &self.feature_biases {
            bytes.extend_from_slice(&b.to_le_bytes());
        }
        for &w in &self.output_weights {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        bytes.extend_from_slice(&self.output_bias.to_le_bytes());

        bytes
    }

    /// バイナリバイト列からデシリアライズ
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let expected_min = 8
            + 8
            + NNUE_INPUT_SIZE * NNUE_HIDDEN_SIZE * 2
            + NNUE_HIDDEN_SIZE * 2
            + (NNUE_HIDDEN_SIZE * 2) * 2
            + 4;
        if bytes.len() < expected_min {
            return Err(format!(
                "NNUE binary too short: {} bytes (expected at least {})",
                bytes.len(),
                expected_min
            ));
        }
        if &bytes[0..8] != NNUE_MAGIC {
            return Err("Invalid NNUE magic header".to_string());
        }
        let input_size = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let hidden_size = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        if input_size != NNUE_INPUT_SIZE || hidden_size != NNUE_HIDDEN_SIZE {
            return Err(format!(
                "NNUE size mismatch: got {}x{}, expected {}x{}",
                input_size, hidden_size, NNUE_INPUT_SIZE, NNUE_HIDDEN_SIZE
            ));
        }

        let mut offset = 16;
        let mut feature_weights = Vec::with_capacity(NNUE_INPUT_SIZE);
        for _ in 0..NNUE_INPUT_SIZE {
            let mut row = [0i16; NNUE_HIDDEN_SIZE];
            for item in row.iter_mut() {
                *item = i16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
                offset += 2;
            }
            feature_weights.push(row);
        }

        let mut feature_biases = [0i16; NNUE_HIDDEN_SIZE];
        for item in feature_biases.iter_mut() {
            *item = i16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
            offset += 2;
        }

        let mut output_weights = [0i16; NNUE_HIDDEN_SIZE * 2];
        for item in output_weights.iter_mut() {
            *item = i16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
            offset += 2;
        }

        let output_bias = i32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());

        Ok(NNUEEvaluator {
            feature_weights,
            feature_biases,
            output_weights,
            output_bias,
        })
    }

    /// ファイルに保存
    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, self.to_bytes())
    }

    /// ファイルから読み込み
    pub fn load_from_file(path: &str) -> Result<Self, String> {
        let bytes =
            std::fs::read(path).map_err(|e| format!("Failed to read NNUE file '{path}': {e}"))?;
        Self::from_bytes(&bytes)
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
                let color_offset = if piece.color == color { 0 } else { 14 };
                let piece_idx = pt_idx + color_offset;
                let feat_idx = mapped_sq * 28 + piece_idx;
                if feat_idx < NNUE_INPUT_SIZE {
                    features.push(feat_idx);
                }
            }
        }

        // 持ち駒特徴量
        let mut hand_base = 81 * 28;
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

    /// 先手視点のマテリアル（駒割り）バランスを計算 (センチポーン単位)
    /// 玉(King)は詰み探索側で処理されるため除外
    pub fn material_black(pos: &Position) -> i32 {
        let mut mat = 0;
        for piece in pos.board.iter().flatten() {
            if piece.piece_type != PieceType::King {
                let val = piece.piece_type.base_value();
                match piece.color {
                    Color::Black => mat += val,
                    Color::White => mat -= val,
                }
            }
        }
        for pt in PieceType::HAND_PIECES {
            if let Some(h_idx) = pt.hand_index() {
                let b_count = pos.hand[Color::Black.index()][h_idx] as i32;
                let w_count = pos.hand[Color::White.index()][h_idx] as i32;
                mat += (b_count - w_count) * pt.base_value();
            }
        }
        mat
    }

    /// 手番視点のマテリアル（駒割り）バランスを計算 (センチポーン単位)
    pub fn material_stm(pos: &Position) -> i32 {
        match pos.side_to_move {
            Color::Black => Self::material_black(pos),
            Color::White => -Self::material_black(pos),
        }
    }

    /// 特徴量インデックス列から先手視点のマテリアルバランスを復元
    /// (盤面 Position が直接利用できない場合のフォールバック・検証用)
    pub fn material_from_features(black_feats: &[usize]) -> i32 {
        let mut mat = 0;
        for &f in black_feats {
            if f < 81 * 28 {
                // 盤上駒
                let piece_idx = f % 28;
                let is_self = piece_idx < 14;
                let pt_idx = if is_self { piece_idx } else { piece_idx - 14 };
                let pt = PieceType::ALL[pt_idx];
                if pt != PieceType::King {
                    let val = pt.base_value();
                    if is_self {
                        mat += val;
                    } else {
                        mat -= val;
                    }
                }
            } else if f < NNUE_INPUT_SIZE {
                // 持ち駒
                let hand_offset = f - 81 * 28;
                let hand_slot = hand_offset / 18;
                let is_self = hand_slot < 7;
                let h_idx = if is_self { hand_slot } else { hand_slot - 7 };
                let pt = PieceType::HAND_PIECES[h_idx];
                let val = pt.base_value();
                if is_self {
                    mat += val;
                } else {
                    mat -= val;
                }
            }
        }
        mat
    }

    /// 盤面から先手・後手視点のアキュムレータを単一パス・ゼロアロケーションで直接計算
    #[inline]
    pub fn compute_accumulators_direct(
        &self,
        pos: &Position,
    ) -> ([i16; NNUE_HIDDEN_SIZE], [i16; NNUE_HIDDEN_SIZE]) {
        let mut b_acc = self.feature_biases;
        let mut w_acc = self.feature_biases;

        // 盤上の駒
        for sq_idx in 0..81 {
            if let Some(piece) = pos.board[sq_idx] {
                let sq = Square::from_index(sq_idx);
                let pt_idx = piece.piece_type.index();

                // 先手視点 (Black)
                let b_mapped_sq = sq.index();
                let b_color_offset = if piece.color == Color::Black { 0 } else { 14 };
                let b_feat = b_mapped_sq * 28 + pt_idx + b_color_offset;
                if b_feat < NNUE_INPUT_SIZE {
                    let w_slice = &self.feature_weights[b_feat];
                    for i in 0..NNUE_HIDDEN_SIZE {
                        b_acc[i] = b_acc[i].saturating_add(w_slice[i]);
                    }
                }

                // 後手視点 (White, 点対称反転)
                let w_mapped_sq = (8 - sq.file() as usize) * 9 + (8 - sq.rank() as usize);
                let w_color_offset = if piece.color == Color::White { 0 } else { 14 };
                let w_feat = w_mapped_sq * 28 + pt_idx + w_color_offset;
                if w_feat < NNUE_INPUT_SIZE {
                    let w_slice = &self.feature_weights[w_feat];
                    for i in 0..NNUE_HIDDEN_SIZE {
                        w_acc[i] = w_acc[i].saturating_add(w_slice[i]);
                    }
                }
            }
        }

        // 持ち駒 (先手視点: 自軍=Black, 敵軍=White / 後手視点: 自軍=White, 敵軍=Black)
        let mut b_hand_base = 81 * 28;
        let mut w_hand_base = 81 * 28;

        for c_self in [true, false] {
            let (b_color, w_color) = if c_self {
                (Color::Black, Color::White)
            } else {
                (Color::White, Color::Black)
            };

            for pt in PieceType::HAND_PIECES {
                if let Some(h_idx) = pt.hand_index() {
                    let b_count = pos.hand[b_color.index()][h_idx] as usize;
                    for k in 0..b_count.min(18) {
                        let feat = b_hand_base + k;
                        if feat < NNUE_INPUT_SIZE {
                            let w_slice = &self.feature_weights[feat];
                            for i in 0..NNUE_HIDDEN_SIZE {
                                b_acc[i] = b_acc[i].saturating_add(w_slice[i]);
                            }
                        }
                    }

                    let w_count = pos.hand[w_color.index()][h_idx] as usize;
                    for k in 0..w_count.min(18) {
                        let feat = w_hand_base + k;
                        if feat < NNUE_INPUT_SIZE {
                            let w_slice = &self.feature_weights[feat];
                            for i in 0..NNUE_HIDDEN_SIZE {
                                w_acc[i] = w_acc[i].saturating_add(w_slice[i]);
                            }
                        }
                    }

                    b_hand_base += 18;
                    w_hand_base += 18;
                }
            }
        }

        (b_acc, w_acc)
    }

    /// 評価値の推論 (Forward inference)
    /// 戻り値: センチポーン (cp) 単位の評価値 (手番視点)
    /// 評価値 = 手番側駒割りベースライン + NNUE 有界残差 (Residual)
    /// 手番中心 (Mover-first) 結合により、完全な手番対称性を保証
    pub fn evaluate(&self, pos: &Position) -> i32 {
        let (b_acc, w_acc) = self.compute_accumulators_direct(pos);
        let (mover_acc, opp_acc) = match pos.side_to_move {
            Color::Black => (&b_acc, &w_acc),
            Color::White => (&w_acc, &b_acc),
        };

        let mut output = self.output_bias;

        // ClippedReLU(x) = clamp(x, 0, 64) (Float側 clamp(0.0, 1.0) * 64 と厳密整合)
        for i in 0..NNUE_HIDDEN_SIZE {
            let m_val = mover_acc[i].clamp(0, 64) as i32;
            let o_val = opp_acc[i].clamp(0, 64) as i32;

            output += m_val * self.output_weights[i] as i32;
            output += o_val * self.output_weights[NNUE_HIDDEN_SIZE + i] as i32;
        }

        // 整数スケーリング: 重み項 (64 * 64) とバイアス項 (4096) を 16 で割ることで
        // Float 学習側 (score = output * 256.0) と数学的に 1 対 1 で完全一致
        let raw_residual_cp = output / 16;
        let residual_cp = raw_residual_cp.clamp(-RESIDUAL_BOUND_CP, RESIDUAL_BOUND_CP);

        Self::material_stm(pos) + residual_cp
    }
}
