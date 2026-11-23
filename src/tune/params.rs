use crate::board::Position;
use crate::types::{Color, PieceType};

/// チューニング対象の評価パラメータ
#[derive(Debug, Clone, PartialEq)]
pub struct TunableParams {
    /// 盤上駒価値 (歩, 香, 桂, 銀, 金, 角, 飛, と, 成香, 成桂, 成銀, 馬, 龍)
    pub piece_values: [f64; 13],
    /// 持ち駒価値 (歩, 香, 桂, 銀, 金, 角, 飛)
    pub hand_values: [f64; 7],
    /// 手番ボーナス
    pub tempo: f64,
}

pub const PARAM_COUNT: usize = 13 + 7 + 1; // 21 parameters

impl Default for TunableParams {
    fn default() -> Self {
        TunableParams {
            piece_values: [
                100.0,  // Pawn
                320.0,  // Lance
                350.0,  // Knight
                500.0,  // Silver
                550.0,  // Gold
                850.0,  // Bishop
                1000.0, // Rook
                530.0,  // ProPawn
                530.0,  // ProLance
                530.0,  // ProKnight
                550.0,  // ProSilver
                1150.0, // Horse
                1300.0, // Dragon
            ],
            hand_values: [
                120.0,  // Pawn
                350.0,  // Lance
                390.0,  // Knight
                550.0,  // Silver
                600.0,  // Gold
                950.0,  // Bishop
                1100.0, // Rook
            ],
            tempo: 25.0,
        }
    }
}

impl TunableParams {
    /// パラメータを1次元ベクトルに変換
    pub fn to_vec(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(PARAM_COUNT);
        v.extend_from_slice(&self.piece_values);
        v.extend_from_slice(&self.hand_values);
        v.push(self.tempo);
        v
    }

    /// 1次元ベクトルからパラメータを復元
    pub fn from_vec(&mut self, v: &[f64]) {
        if v.len() >= PARAM_COUNT {
            self.piece_values.copy_from_slice(&v[0..13]);
            self.hand_values.copy_from_slice(&v[13..20]);
            self.tempo = v[20];
        }
    }

    /// パラメータベクトルを適用した高速局面評価 (手番側視点の評価スコア)
    pub fn evaluate(&self, pos: &Position) -> f64 {
        let mut b_score = 0.0;
        let mut w_score = 0.0;

        // 1. 盤上駒の評価
        for sq_idx in 0..81 {
            if let Some(piece) = pos.board[sq_idx] {
                let idx = Self::piece_type_to_idx(piece.piece_type);
                let val = self.piece_values[idx];
                if piece.color == Color::Black {
                    b_score += val;
                } else {
                    w_score += val;
                }
            }
        }

        // 2. 持ち駒の評価
        for (h_idx, &pt) in PieceType::HAND_PIECES.iter().enumerate() {
            let b_count = pos.hand[Color::Black.index()][h_idx] as f64;
            let w_count = pos.hand[Color::White.index()][h_idx] as f64;
            let h_val = self.hand_values[h_idx];
            b_score += b_count * h_val;
            w_score += w_count * h_val;
            let _ = pt;
        }

        let diff = b_score - w_score;
        let s = pos.side_to_move.sign() as f64;
        diff * s + self.tempo
    }

    fn piece_type_to_idx(pt: PieceType) -> usize {
        match pt {
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
            PieceType::King => 0,
        }
    }
}
