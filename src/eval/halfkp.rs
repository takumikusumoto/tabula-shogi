use crate::board::Position;
use crate::types::{Color, Move, Piece, PieceType, Square};
use std::io::{self, BufWriter, Write};

pub const HALFKP_PIECE_SIZE: usize = 81 * 28 + 7 * 2 * 18; // 2,520
pub const HALFKP_INPUT_SIZE: usize = 81 * HALFKP_PIECE_SIZE; // 81 * 2,520 = 204,120
pub const HALFKP_HIDDEN_SIZE: usize = 128;

const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
const LCG_ADDEND: u64 = 1;

pub const HALFKP_MAGIC: &[u8; 8] = b"TABU_HKP";
pub const MAX_EVAL_CP: i32 = 27_000;
pub const RESIDUAL_BOUND_CP: i32 = 25_000;

/// 前活性を i32 で保持する堅牢な差分アキュムレータ
/// - i16 飽和加算の不可逆性を完全に排除し、可逆な線形加減算を保証
/// - ClippedReLU [0, 64] は評価値推論時に適用
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
    /// 決定論的サンプリングによる初期化
    pub fn new() -> Self {
        let mut feature_weights = vec![[0i16; HALFKP_HIDDEN_SIZE]; HALFKP_INPUT_SIZE];
        let feature_biases = [32i16; HALFKP_HIDDEN_SIZE];
        let output_weights = [0i16; HALFKP_HIDDEN_SIZE * 2];
        let output_bias = 0i32;

        for (feat, weights_slice) in feature_weights.iter_mut().enumerate() {
            for (i, w) in weights_slice.iter_mut().enumerate() {
                let h = ((feat as u64)
                    .wrapping_mul(LCG_MULTIPLIER)
                    .wrapping_add((i as u64).wrapping_mul(0x9e3779b97f4a7c15))
                    .wrapping_add(LCG_ADDEND)
                    >> 33) as i32;
                *w = ((h % 3) - 1) as i16;
            }
        }

        HalfKPEvaluator {
            feature_weights,
            feature_biases,
            output_weights,
            output_bias,
        }
    }

    /// 浮動小数点重みから量子化モデルを構築
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
                *w = (f * 64.0).clamp(-127.0, 127.0).round() as i16;
            }
            quantized_feats.push(q_row);
        }

        let mut quantized_biases = [0i16; HALFKP_HIDDEN_SIZE];
        for (b, &f) in quantized_biases.iter_mut().zip(feature_biases.iter()) {
            *b = (f * 64.0).clamp(-127.0, 127.0).round() as i16;
        }

        let mut quantized_out = [0i16; HALFKP_HIDDEN_SIZE * 2];
        for (w, &f) in quantized_out.iter_mut().zip(output_weights.iter()) {
            *w = (f * 512.0).clamp(-32767.0, 32767.0).round() as i16;
        }

        let quantized_out_bias = (output_bias * 32768.0).round() as i32;

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
                self.compute_accumulator_full(pos, Color::White)
            ],
            computed: [true, true],
        }
    }

    /// 指し手適用時の差分アキュムレータ更新
    /// - 玉移動: 該当視点を全再計算
    /// - 非玉移動: 移動元/先の加減算、捕獲駒の盤上減算および手駒加算、持ち駒打ちの減算
    pub fn update_accumulator_move(
        &self,
        acc: &mut HalfKPAccumulator,
        pos_before: &Position,
        mv: Move,
        pos_after: &Position,
    ) {
        for color in [Color::Black, Color::White] {
            let c_idx = color.index();
            let k_before = Self::get_king_sq(pos_before, color);
            let k_after = Self::get_king_sq(pos_after, color);

            // 当該視点の玉が動いた場合は全再計算
            if k_before != k_after {
                acc.accumulation[c_idx] = self.compute_accumulator_full(pos_after, color);
                acc.computed[c_idx] = true;
                continue;
            }

            let k_offset = k_before.index() * HALFKP_PIECE_SIZE;
            let us = pos_before.side_to_move;

            if mv.is_drop() {
                // 持ち駒打ち (Drop)
                let drop_pt = mv.drop_piece().expect("Drop move must have piece type");
                let to_sq = mv.to();
                let placed_piece = Piece::new(drop_pt, us);

                // 1. 持ち駒から1枚減算 (使用前の手駒枚数 - 1 のスロットを減算)
                let h_idx = drop_pt.hand_index().expect("Valid hand piece");
                let old_count = pos_before.hand[us.index()][h_idx] as usize;
                if old_count > 0 {
                    let is_self = us == color;
                    let h_feat = Self::hand_to_feature(drop_pt, old_count - 1, is_self);
                    let feat_idx = k_offset + h_feat;
                    Self::sub_weights(
                        &mut acc.accumulation[c_idx],
                        &self.feature_weights[feat_idx],
                    );
                }

                // 2. 盤上に打たれた駒を加算
                let p_feat = Self::piece_to_feature(to_sq, placed_piece, color);
                let feat_idx = k_offset + p_feat;
                Self::add_weights(
                    &mut acc.accumulation[c_idx],
                    &self.feature_weights[feat_idx],
                );
            } else {
                // 通常移動 (Move)
                let from_sq = mv.from().expect("Non-drop move must have from");
                let to_sq = mv.to();
                let moved_piece = pos_before.board[from_sq.index()].expect("Piece at from");

                // 1. 移動元マス駒の減算
                let p_from_feat = Self::piece_to_feature(from_sq, moved_piece, color);
                let feat_from = k_offset + p_from_feat;
                Self::sub_weights(
                    &mut acc.accumulation[c_idx],
                    &self.feature_weights[feat_from],
                );

                // 2. 捕獲駒があれば盤上から減算し、自軍持ち駒へ加算
                if let Some(captured) = pos_before.board[to_sq.index()] {
                    let p_cap_feat = Self::piece_to_feature(to_sq, captured, color);
                    let feat_cap = k_offset + p_cap_feat;
                    Self::sub_weights(
                        &mut acc.accumulation[c_idx],
                        &self.feature_weights[feat_cap],
                    );

                    // 捕獲駒は生駒として手番側 (us) の手駒へ加算
                    let unpromoted_pt = captured.piece_type.unpromote();
                    let h_idx = unpromoted_pt.hand_index().expect("Valid hand piece");
                    let prev_hand_count = pos_before.hand[us.index()][h_idx] as usize;
                    let is_self = us == color;
                    let h_feat = Self::hand_to_feature(unpromoted_pt, prev_hand_count, is_self);
                    let feat_hand = k_offset + h_feat;
                    Self::add_weights(
                        &mut acc.accumulation[c_idx],
                        &self.feature_weights[feat_hand],
                    );
                }

                // 3. 移動先マス駒の加算 (成りの場合は成駒)
                let final_piece = if mv.is_promote() {
                    let promoted_pt = moved_piece
                        .piece_type
                        .promote()
                        .unwrap_or(moved_piece.piece_type);
                    Piece::new(promoted_pt, us)
                } else {
                    moved_piece
                };
                let p_to_feat = Self::piece_to_feature(to_sq, final_piece, color);
                let feat_to = k_offset + p_to_feat;
                Self::add_weights(&mut acc.accumulation[c_idx], &self.feature_weights[feat_to]);
            }

            acc.computed[c_idx] = true;
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
                let h_feat = Self::hand_to_feature(drop_pt, prev_count - 1, is_self);
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
                    let prev_hand_count =
                        (pos_after.hand[us.index()][h_idx] as usize).saturating_sub(1);
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

    /// アキュムレータを用いた高速局面評価 (ClippedReLU 0..=64 & 純粋スカラー評価)
    pub fn evaluate_with_accumulator(&self, pos: &Position, acc: &HalfKPAccumulator) -> i32 {
        let (mover_acc, opp_acc) = match pos.side_to_move {
            Color::Black => (&acc.accumulation[0], &acc.accumulation[1]),
            Color::White => (&acc.accumulation[1], &acc.accumulation[0]),
        };

        let mut output = self.output_bias as i64;
        for i in 0..HALFKP_HIDDEN_SIZE {
            let m_val = mover_acc[i].clamp(0, 64) as i64;
            let o_val = opp_acc[i].clamp(0, 64) as i64;
            output += m_val * (self.output_weights[i] as i64);
            output += o_val * (self.output_weights[HALFKP_HIDDEN_SIZE + i] as i64);
        }

        let raw_cp = (output / 128) as i32;
        raw_cp.clamp(-MAX_EVAL_CP, MAX_EVAL_CP)
    }

    /// 局面からの即時評価 (全再計算経由)
    pub fn evaluate(&self, pos: &Position) -> i32 {
        let acc = self.compute_accumulators_full(pos);
        self.evaluate_with_accumulator(pos, &acc)
    }

    /// バイナリファイルへ安全にアトミック保存 (TABU_HKP)
    /// 52MBのメモリ一括確保を完全排除し、一時ファイルへのBufWriter逐次書き出しとアトミックリネームで既存ファイルを保護
    pub fn save_to_file(&self, path: &str) -> io::Result<()> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp_path = format!("{path}.tmp");
        let file = std::fs::File::create(&tmp_path)?;
        let mut writer = BufWriter::new(file);

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

        writer.flush()?;
        drop(writer);

        // アトミックリネームによる置換 (Windows では既存ファイルがあるとエラーになる場合があるため事前に置換)
        if std::path::Path::new(path).exists() {
            let _ = std::fs::remove_file(path);
        }
        std::fs::rename(&tmp_path, path)?;

        Ok(())
    }

    /// バイナリファイルから読み込み
    pub fn load_from_file(path: &str) -> Result<Self, String> {
        let bytes =
            std::fs::read(path).map_err(|e| format!("Failed to read HalfKP file '{path}': {e}"))?;
        let expected_len = 16
            + HALFKP_INPUT_SIZE * HALFKP_HIDDEN_SIZE * 2
            + HALFKP_HIDDEN_SIZE * 2
            + (HALFKP_HIDDEN_SIZE * 2) * 2
            + 4;
        if bytes.len() != expected_len {
            return Err(format!(
                "HalfKP file size mismatch: got {} bytes, expected {} bytes",
                bytes.len(),
                expected_len
            ));
        }
        if &bytes[0..8] != HALFKP_MAGIC {
            return Err("Invalid HalfKP magic header".to_string());
        }
        let in_size = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let hid_size = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        if in_size != HALFKP_INPUT_SIZE || hid_size != HALFKP_HIDDEN_SIZE {
            return Err(format!(
                "HalfKP size mismatch: got {in_size}x{hid_size}, expected {HALFKP_INPUT_SIZE}x{HALFKP_HIDDEN_SIZE}"
            ));
        }

        let mut offset = 16;
        let mut feature_weights = Vec::with_capacity(HALFKP_INPUT_SIZE);
        for _ in 0..HALFKP_INPUT_SIZE {
            let mut row = [0i16; HALFKP_HIDDEN_SIZE];
            for item in row.iter_mut() {
                *item = i16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
                offset += 2;
            }
            feature_weights.push(row);
        }

        let mut feature_biases = [0i16; HALFKP_HIDDEN_SIZE];
        for item in feature_biases.iter_mut() {
            *item = i16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
            offset += 2;
        }

        let mut output_weights = [0i16; HALFKP_HIDDEN_SIZE * 2];
        for item in output_weights.iter_mut() {
            *item = i16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
            offset += 2;
        }

        let output_bias = i32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());

        Ok(HalfKPEvaluator {
            feature_weights,
            feature_biases,
            output_weights,
            output_bias,
        })
    }
}
