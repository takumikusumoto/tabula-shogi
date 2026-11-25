use crate::board::Position;
use crate::types::{Color, Move, Piece, PieceType, Square};
use std::io::{self, BufReader, Read, Write};

pub const HALFKP_PIECE_SIZE: usize = 81 * 28 + 7 * 2 * 18; // 2,520
pub const HALFKP_INPUT_SIZE: usize = 81 * HALFKP_PIECE_SIZE; // 81 * 2,520 = 204,120
pub const HALFKP_HIDDEN_SIZE: usize = 128;

const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
const LCG_ADDEND: u64 = 1;

/// Q6 feature weights, Q9 output weights, and their product-scale output bias.
///
/// The fixed-point forward pass uses these units:
/// - accumulator/activation: `float * 64`
/// - output weight: `float * 512`
/// - output sum/bias: `float * 64 * 512`
pub const HALFKP_FEATURE_SCALE: i32 = 64;
pub const HALFKP_OUTPUT_SCALE: i32 = 512;
pub const HALFKP_BIAS_SCALE: i32 = HALFKP_FEATURE_SCALE * HALFKP_OUTPUT_SCALE;
pub const HALFKP_ACTIVATION_MAX: i32 = 64 * HALFKP_FEATURE_SCALE;
pub const HALFKP_OUTPUT_DIVISOR: i64 =
    128 * HALFKP_FEATURE_SCALE as i64 * HALFKP_OUTPUT_SCALE as i64;

pub const HALFKP_MAGIC: &[u8; 8] = b"TABU_HK2";
pub const HALFKP_LEGACY_MAGIC: &[u8; 8] = b"TABU_HKP";
pub const MAX_EVAL_CP: i32 = 27_000;
pub const RESIDUAL_BOUND_CP: i32 = 25_000;

/// 前活性を i32 で保持する堅牢な差分アキュムレータ
/// - i16 飽和加算の不可逆性を完全に排除し、可逆な線形加減算を保証
/// - Q6 ClippedReLU [0, 4096] は評価値推論時に適用
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HalfKPAccumulator {
    pub accumulation: [[i32; HALFKP_HIDDEN_SIZE]; 2],
    pub computed: [bool; 2],
}

impl Default for HalfKPAccumulator {
    fn default() -> Self {
        Self::empty()
    }
}

impl HalfKPAccumulator {
    pub fn empty() -> Self {
        HalfKPAccumulator {
            accumulation: [[0i32; HALFKP_HIDDEN_SIZE]; 2],
            computed: [false, false],
        }
    }

    pub fn new(biases: &[i16; HALFKP_HIDDEN_SIZE]) -> Self {
        let mut acc = [[0i32; HALFKP_HIDDEN_SIZE]; 2];
        for i in 0..HALFKP_HIDDEN_SIZE {
            acc[0][i] = biases[i] as i32;
            acc[1][i] = biases[i] as i32;
        }
        HalfKPAccumulator {
            accumulation: acc,
            computed: [false, false],
        }
    }
}

/// 本格 HalfKP 評価ネットワーク (自玉81マス × 全駒特徴量 2,520 = 204,120次元)
/// - 駒割り基底の二重加算を完全撤廃し、純粋なスカラー局面評価値を直接推論
/// - 前活性 i32 による完全可逆な差分アキュムレータ更新
#[derive(Clone, Debug)]
pub struct HalfKPEvaluator {
    pub feature_weights: Vec<[i16; HALFKP_HIDDEN_SIZE]>,
    pub feature_biases: [i16; HALFKP_HIDDEN_SIZE],
    pub output_weights: [i16; HALFKP_HIDDEN_SIZE * 2],
    pub output_bias: i32,
}

impl Default for HalfKPEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl HalfKPEvaluator {
    #[inline(always)]
    pub(crate) fn initial_feature_weight(feat: usize, hidden: usize) -> i16 {
        let h = ((feat as u64)
            .wrapping_mul(LCG_MULTIPLIER)
            .wrapping_add((hidden as u64).wrapping_mul(0x9e3779b97f4a7c15))
            .wrapping_add(LCG_ADDEND)
            >> 33) as i32;
        (((h % 3) - 1) * HALFKP_FEATURE_SCALE) as i16
    }

    #[inline]
    fn quantize_i16(value: f32, scale: i32) -> i16 {
        (value * scale as f32)
            .round()
            .clamp(i16::MIN as f32, i16::MAX as f32) as i16
    }

    #[inline]
    fn quantize_i32(value: f32, scale: i32) -> i32 {
        (value * scale as f32)
            .round()
            .clamp(i32::MIN as f32, i32::MAX as f32) as i32
    }

    /// 決定論的サンプリングによる初期化
    pub fn new() -> Self {
        let mut feature_weights = vec![[0i16; HALFKP_HIDDEN_SIZE]; HALFKP_INPUT_SIZE];
        let feature_biases = [(32 * HALFKP_FEATURE_SCALE) as i16; HALFKP_HIDDEN_SIZE];
        let output_weights = [0i16; HALFKP_HIDDEN_SIZE * 2];
        let output_bias = 0i32;

        for (feat, weights_slice) in feature_weights.iter_mut().enumerate() {
            for (i, w) in weights_slice.iter_mut().enumerate() {
                *w = Self::initial_feature_weight(feat, i);
            }
        }

        HalfKPEvaluator {
            feature_weights,
            feature_biases,
            output_weights,
            output_bias,
        }
    }

    /// 浮動小数点重みから量子化モデルを構築 (HalfKPTrainer::to_evaluator と完全一致)
    pub fn from_float_weights(
        feature_weights: &[[f32; HALFKP_HIDDEN_SIZE]],
        feature_biases: &[f32; HALFKP_HIDDEN_SIZE],
        output_weights: &[f32; HALFKP_HIDDEN_SIZE * 2],
        output_bias: f32,
    ) -> Self {
        let mut quantized_feats = Vec::with_capacity(HALFKP_INPUT_SIZE);
        for row in feature_weights {
            let mut q_row = [0i16; HALFKP_HIDDEN_SIZE];
            for (w, &f) in q_row.iter_mut().zip(row.iter()) {
                *w = Self::quantize_i16(f, HALFKP_FEATURE_SCALE);
            }
            quantized_feats.push(q_row);
        }

        let mut quantized_biases = [0i16; HALFKP_HIDDEN_SIZE];
        for (b, &f) in quantized_biases.iter_mut().zip(feature_biases.iter()) {
            *b = Self::quantize_i16(f, HALFKP_FEATURE_SCALE);
        }

        let mut quantized_out = [0i16; HALFKP_HIDDEN_SIZE * 2];
        for (w, &f) in quantized_out.iter_mut().zip(output_weights.iter()) {
            *w = Self::quantize_i16(f, HALFKP_OUTPUT_SCALE);
        }

        let quantized_out_bias = Self::quantize_i32(output_bias, HALFKP_BIAS_SCALE);

        HalfKPEvaluator {
            feature_weights: quantized_feats,
            feature_biases: quantized_biases,
            output_weights: quantized_out,
            output_bias: quantized_out_bias,
        }
    }

    /// 視点側の自玉の正規化マスを取得
    #[inline(always)]
    pub fn get_king_sq(pos: &Position, color: Color) -> Square {
        match color {
            Color::Black => pos.king_sq[Color::Black.index()].unwrap_or(Square::from_index(0)),
            Color::White => {
                let raw_sq = pos.king_sq[Color::White.index()].unwrap_or(Square::from_index(80));
                let flipped_idx = (8 - raw_sq.file() as usize) * 9 + (8 - raw_sq.rank() as usize);
                Square::from_index(flipped_idx)
            }
        }
    }

    /// 盤上の単一駒に対する駒特徴量インデックス (0..2268)
    #[inline(always)]
    pub fn piece_to_feature(sq: Square, piece: Piece, perspective: Color) -> usize {
        let mapped_sq = match perspective {
            Color::Black => sq.index(),
            Color::White => (8 - sq.file() as usize) * 9 + (8 - sq.rank() as usize),
        };
        let color_offset = if piece.color == perspective { 0 } else { 14 };
        mapped_sq * 28 + piece.piece_type.index() + color_offset
    }

    /// 持ち駒に対する駒特徴量インデックス (2268..2520)
    #[inline(always)]
    pub fn hand_to_feature(pt: PieceType, count_idx: usize, is_self: bool) -> usize {
        let base = 81 * 28;
        let color_offset = if is_self { 0 } else { 7 * 18 };
        let h_idx = pt.hand_index().unwrap_or(0);
        base + color_offset + h_idx * 18 + count_idx
    }

    /// 局面から指定視点のアクティブな HalfKP 特徴量インデックス一覧を抽出
    pub fn extract_halfkp_features(pos: &Position, color: Color) -> Vec<usize> {
        let mut features = Vec::with_capacity(40);
        let king_sq = Self::get_king_sq(pos, color);
        let k_offset = king_sq.index() * HALFKP_PIECE_SIZE;

        // 盤上駒
        for sq_idx in 0..81 {
            if let Some(piece) = pos.board[sq_idx] {
                let sq = Square::from_index(sq_idx);
                let p_feat = Self::piece_to_feature(sq, piece, color);
                features.push(k_offset + p_feat);
            }
        }

        // 持ち駒 (自軍および敵軍)
        for (is_self, c) in [(true, color), (false, color.opposite())] {
            for pt in PieceType::HAND_PIECES {
                if let Some(h_idx) = pt.hand_index() {
                    let count = pos.hand[c.index()][h_idx] as usize;
                    for k in 0..count.min(18) {
                        let h_feat = Self::hand_to_feature(pt, k, is_self);
                        features.push(k_offset + h_feat);
                    }
                }
            }
        }

        features
    }

    #[inline(always)]
    fn add_weights(acc: &mut [i32; HALFKP_HIDDEN_SIZE], weights: &[i16; HALFKP_HIDDEN_SIZE]) {
        for (a, &w) in acc.iter_mut().zip(weights.iter()) {
            *a += w as i32;
        }
    }

    #[inline(always)]
    fn sub_weights(acc: &mut [i32; HALFKP_HIDDEN_SIZE], weights: &[i16; HALFKP_HIDDEN_SIZE]) {
        for (a, &w) in acc.iter_mut().zip(weights.iter()) {
            *a -= w as i32;
        }
    }

    /// 指定視点のアキュムレータをゼロから全加算計算 (前活性 i32)
    pub fn compute_accumulator_full(
        &self,
        pos: &Position,
        color: Color,
    ) -> [i32; HALFKP_HIDDEN_SIZE] {
        let mut acc = self.feature_biases.map(|b| b as i32);

        let king_sq = Self::get_king_sq(pos, color);
        let k_offset = king_sq.index() * HALFKP_PIECE_SIZE;

        // 盤上駒
        for sq_idx in 0..81 {
            if let Some(piece) = pos.board[sq_idx] {
                let sq = Square::from_index(sq_idx);
                let p_feat = Self::piece_to_feature(sq, piece, color);
                let feat_idx = k_offset + p_feat;
                if feat_idx < HALFKP_INPUT_SIZE {
                    Self::add_weights(&mut acc, &self.feature_weights[feat_idx]);
                }
            }
        }

        // 持ち駒
        for (is_self, c) in [(true, color), (false, color.opposite())] {
            for pt in PieceType::HAND_PIECES {
                if let Some(h_idx) = pt.hand_index() {
                    let count = pos.hand[c.index()][h_idx] as usize;
                    for k in 0..count.min(18) {
                        let h_feat = Self::hand_to_feature(pt, k, is_self);
                        let feat_idx = k_offset + h_feat;
                        if feat_idx < HALFKP_INPUT_SIZE {
                            Self::add_weights(&mut acc, &self.feature_weights[feat_idx]);
                        }
                    }
                }
            }
        }

        acc
    }

    /// 盤面から先手・後手両視点のアキュムレータを全再計算
    pub fn compute_accumulators_full(&self, pos: &Position) -> HalfKPAccumulator {
        HalfKPAccumulator {
            accumulation: [
                self.compute_accumulator_full(pos, Color::Black),
                self.compute_accumulator_full(pos, Color::White),
            ],
            computed: [true, true],
        }
    }

    /// do_move 後におけるアキュムレータの高速差分更新 (Position クローン完全ゼロ)
    pub fn update_accumulator_after_move(
        &self,
        acc: &mut HalfKPAccumulator,
        pos_after: &Position,
        mv: Move,
    ) {
        let us = pos_after.side_to_move.opposite();
        let last_rec = pos_after.history.last().expect("History must have record");
        let captured = last_rec.captured;

        for color in [Color::Black, Color::White] {
            let c_idx = color.index();
            let k_curr = Self::get_king_sq(pos_after, color);

            // 当該視点の玉が動いた場合 (直前の手が当該視点の玉の移動)
            if mv.from().is_some()
                && mv.drop_piece().is_none()
                && let Some(piece) = pos_after.board[mv.to().index()]
                && piece.piece_type == PieceType::King
                && color == us
            {
                acc.accumulation[c_idx] = self.compute_accumulator_full(pos_after, color);
                acc.computed[c_idx] = true;
                continue;
            }

            let k_offset = k_curr.index() * HALFKP_PIECE_SIZE;

            if mv.is_drop() {
                let drop_pt = mv.drop_piece().expect("Drop move must have piece type");
                let to_sq = mv.to();
                let placed_piece = Piece::new(drop_pt, us);

                // 1. 持ち駒から1枚減算 (使用前の手駒枚数は、現在の枚数 + 1)
                let h_idx = drop_pt.hand_index().expect("Valid hand piece");
                let prev_count = pos_after.hand[us.index()][h_idx] as usize + 1;
                let is_self = us == color;
                let count_slot = (prev_count - 1).min(17);
                let h_feat = Self::hand_to_feature(drop_pt, count_slot, is_self);
                let feat_idx = k_offset + h_feat;
                Self::sub_weights(
                    &mut acc.accumulation[c_idx],
                    &self.feature_weights[feat_idx],
                );

                // 2. 盤上に打たれた駒を加算
                let p_feat = Self::piece_to_feature(to_sq, placed_piece, color);
                let feat_idx = k_offset + p_feat;
                Self::add_weights(
                    &mut acc.accumulation[c_idx],
                    &self.feature_weights[feat_idx],
                );
            } else {
                let from_sq = mv.from().expect("Non-drop move must have from");
                let to_sq = mv.to();
                let piece_after = pos_after.board[to_sq.index()].expect("Moved piece at to_sq");

                // 移動前の元駒
                let orig_pt = if mv.is_promote() {
                    piece_after.piece_type.unpromote()
                } else {
                    piece_after.piece_type
                };
                let moved_piece = Piece::new(orig_pt, us);

                // 1. 移動元マス駒の減算
                let p_from_feat = Self::piece_to_feature(from_sq, moved_piece, color);
                let feat_from = k_offset + p_from_feat;
                Self::sub_weights(
                    &mut acc.accumulation[c_idx],
                    &self.feature_weights[feat_from],
                );

                // 2. 捕獲駒があれば盤上から減算し、手駒へ加算
                if let Some(cap) = captured {
                    let p_cap_feat = Self::piece_to_feature(to_sq, cap, color);
                    let feat_cap = k_offset + p_cap_feat;
                    Self::sub_weights(
                        &mut acc.accumulation[c_idx],
                        &self.feature_weights[feat_cap],
                    );

                    // 捕獲駒は手駒へ加算 (直前の手の前の手駒枚数は、現在の手駒枚数 - 1)
                    let unpromoted_pt = cap.piece_type.unpromote();
                    let h_idx = unpromoted_pt.hand_index().expect("Valid hand piece");
                    let prev_hand_count = (pos_after.hand[us.index()][h_idx] as usize)
                        .saturating_sub(1)
                        .min(17);
                    let is_self = us == color;
                    let h_feat = Self::hand_to_feature(unpromoted_pt, prev_hand_count, is_self);
                    let feat_hand = k_offset + h_feat;
                    Self::add_weights(
                        &mut acc.accumulation[c_idx],
                        &self.feature_weights[feat_hand],
                    );
                }

                // 3. 移動先マス駒の加算
                let p_to_feat = Self::piece_to_feature(to_sq, piece_after, color);
                let feat_to = k_offset + p_to_feat;
                Self::add_weights(&mut acc.accumulation[c_idx], &self.feature_weights[feat_to]);
            }

            acc.computed[c_idx] = true;
        }
    }

    /// アキュムレータを用いた高速局面評価 (Q6 ClippedReLU 0..=4096)
    pub fn evaluate_with_accumulator(&self, pos: &Position, acc: &HalfKPAccumulator) -> i32 {
        let (mover_acc, opp_acc) = match pos.side_to_move {
            Color::Black => (&acc.accumulation[0], &acc.accumulation[1]),
            Color::White => (&acc.accumulation[1], &acc.accumulation[0]),
        };

        let mut output = self.output_bias as i64;
        for i in 0..HALFKP_HIDDEN_SIZE {
            let m_val = mover_acc[i].clamp(0, HALFKP_ACTIVATION_MAX) as i64;
            let o_val = opp_acc[i].clamp(0, HALFKP_ACTIVATION_MAX) as i64;
            output += m_val * (self.output_weights[i] as i64);
            output += o_val * (self.output_weights[HALFKP_HIDDEN_SIZE + i] as i64);
        }

        // Symmetric round-to-nearest keeps the final integer score within 0.5 cp of
        // the fixed-point value (Rust's signed division otherwise truncates to zero).
        let rounded_cp = if output >= 0 {
            (output + HALFKP_OUTPUT_DIVISOR / 2) / HALFKP_OUTPUT_DIVISOR
        } else {
            -((-output + HALFKP_OUTPUT_DIVISOR / 2) / HALFKP_OUTPUT_DIVISOR)
        };
        let raw_cp = rounded_cp as i32;
        raw_cp.clamp(-MAX_EVAL_CP, MAX_EVAL_CP)
    }

    /// 局面からの即時評価 (全再計算経由)
    pub fn evaluate(&self, pos: &Position) -> i32 {
        let acc = self.compute_accumulators_full(pos);
        self.evaluate_with_accumulator(pos, &acc)
    }

    /// バイナリファイルへ安全にアトミック保存 (TABU_HK2)
    /// 52MBのメモリ一括確保を完全排除し、一時ファイルへのBufWriter逐次書き出しとアトミックリネームで既存ファイルを保護
    pub fn save_to_file(&self, path: &str) -> io::Result<()> {
        super::atomic_io::write_atomic_with_backup(path, |writer| {
            writer.write_all(HALFKP_MAGIC)?;
            writer.write_all(&(HALFKP_INPUT_SIZE as u32).to_le_bytes())?;
            writer.write_all(&(HALFKP_HIDDEN_SIZE as u32).to_le_bytes())?;

            for row in &self.feature_weights {
                for &w in row {
                    writer.write_all(&w.to_le_bytes())?;
                }
            }
            for &b in &self.feature_biases {
                writer.write_all(&b.to_le_bytes())?;
            }
            for &w in &self.output_weights {
                writer.write_all(&w.to_le_bytes())?;
            }
            writer.write_all(&self.output_bias.to_le_bytes())?;
            Ok(())
        })
    }

    /// バイナリファイルからストリーミング読み込み (ピークメモリ ~50 MiB)
    pub fn load_from_file(path: &str) -> Result<Self, String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("Failed to read HalfKP file '{path}': {e}"))?;
        let file_len = file
            .metadata()
            .map_err(|e| format!("Failed to read HalfKP file metadata '{path}': {e}"))?
            .len();

        let mut reader = BufReader::new(file);
        let mut magic = [0u8; 8];
        if file_len >= HALFKP_LEGACY_MAGIC.len() as u64 {
            reader
                .read_exact(&mut magic)
                .map_err(|e| format!("Failed to read HalfKP magic header from '{path}': {e}"))?;
            if &magic == HALFKP_LEGACY_MAGIC {
                return Err(
                    "Legacy HalfKP format TABU_HKP is not supported by the scaled evaluator; re-export the original float checkpoint with `tabula-shogi export-halfkp`"
                        .to_string(),
                );
            }
        }

        let expected_len = (16
            + HALFKP_INPUT_SIZE * HALFKP_HIDDEN_SIZE * 2
            + HALFKP_HIDDEN_SIZE * 2
            + (HALFKP_HIDDEN_SIZE * 2) * 2
            + 4) as u64;
        if file_len != expected_len {
            return Err(format!(
                "HalfKP file size mismatch: got {} bytes, expected {} bytes",
                file_len, expected_len
            ));
        }
        if &magic != HALFKP_MAGIC {
            return Err(format!(
                "Invalid HalfKP magic header: expected {}",
                String::from_utf8_lossy(HALFKP_MAGIC)
            ));
        }

        let read_u32 = |r: &mut BufReader<std::fs::File>| -> Result<u32, String> {
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf)
                .map_err(|e| format!("Failed to read HalfKP u32 from '{path}': {e}"))?;
            Ok(u32::from_le_bytes(buf))
        };
        let read_i32 = |r: &mut BufReader<std::fs::File>| -> Result<i32, String> {
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf)
                .map_err(|e| format!("Failed to read HalfKP i32 from '{path}': {e}"))?;
            Ok(i32::from_le_bytes(buf))
        };

        let in_size = read_u32(&mut reader)? as usize;
        let hid_size = read_u32(&mut reader)? as usize;
        if in_size != HALFKP_INPUT_SIZE || hid_size != HALFKP_HIDDEN_SIZE {
            return Err(format!(
                "HalfKP size mismatch: got {in_size}x{hid_size}, expected {HALFKP_INPUT_SIZE}x{HALFKP_HIDDEN_SIZE}"
            ));
        }

        let mut row_bytes = [0u8; HALFKP_HIDDEN_SIZE * 2];
        let mut feature_weights = Vec::with_capacity(HALFKP_INPUT_SIZE);
        for _ in 0..HALFKP_INPUT_SIZE {
            reader
                .read_exact(&mut row_bytes)
                .map_err(|e| format!("Failed to read HalfKP feature weights from '{path}': {e}"))?;
            let mut row = [0i16; HALFKP_HIDDEN_SIZE];
            let (chunks, _) = row_bytes.as_chunks::<2>();
            for (dst, &chunk) in row.iter_mut().zip(chunks) {
                *dst = i16::from_le_bytes(chunk);
            }
            feature_weights.push(row);
        }

        reader
            .read_exact(&mut row_bytes)
            .map_err(|e| format!("Failed to read HalfKP feature biases from '{path}': {e}"))?;
        let mut feature_biases = [0i16; HALFKP_HIDDEN_SIZE];
        let (bias_chunks, _) = row_bytes.as_chunks::<2>();
        for (dst, &chunk) in feature_biases.iter_mut().zip(bias_chunks) {
            *dst = i16::from_le_bytes(chunk);
        }

        let mut out_bytes = [0u8; HALFKP_HIDDEN_SIZE * 4];
        reader
            .read_exact(&mut out_bytes)
            .map_err(|e| format!("Failed to read HalfKP output weights from '{path}': {e}"))?;
        let mut output_weights = [0i16; HALFKP_HIDDEN_SIZE * 2];
        let (out_chunks, _) = out_bytes.as_chunks::<2>();
        for (dst, &chunk) in output_weights.iter_mut().zip(out_chunks) {
            *dst = i16::from_le_bytes(chunk);
        }

        let output_bias = read_i32(&mut reader)?;

        Ok(HalfKPEvaluator {
            feature_weights,
            feature_biases,
            output_weights,
            output_bias,
        })
    }
}
