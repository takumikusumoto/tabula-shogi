use super::piece::PieceType;
use super::square::Square;
use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Move {
    from: Option<Square>,
    to: Square,
    promote: bool,
    drop_piece: Option<PieceType>,
}

impl Move {
    pub const NONE: Move = Move {
        from: None,
        to: Square(0),
        promote: false,
        drop_piece: None,
    };

    #[inline(always)]
    pub fn is_none(self) -> bool {
        self.from.is_none() && self.drop_piece.is_none()
    }

    /// Encode Move as u16 for atomic TT storage
    #[inline(always)]
    pub fn as_u16(self) -> u16 {
        if self.is_none() {
            return 0;
        }
        let to_idx = self.to.index() as u16; // 0..80 (7 bits)
        let prom_bit = if self.promote { 1 << 7 } else { 0 };
        if let Some(dp) = self.drop_piece {
            // Drop move: flag bit 15 = 1, piece type in bits 8..11, to in bits 0..6 (7 bits, 0..=80)
            let pt_idx = dp.index() as u16;
            (1 << 15) | (pt_idx << 8) | prom_bit | to_idx
        } else if let Some(from_sq) = self.from {
            // Normal move: flag bit 15 = 0, from in bits 8..14, to in bits 0..6 (7 bits, 0..=80)
            let from_idx = from_sq.index() as u16;
            (from_idx << 8) | prom_bit | to_idx
        } else {
            0
        }
    }

    /// Decode Move from u16
    #[inline(always)]
    pub fn from_u16(val: u16) -> Option<Self> {
        if val == 0 {
            return None;
        }
        let to_idx = (val & 0x7F) as usize;
        if to_idx >= 81 {
            return None;
        }
        let to_sq = Square::from_index(to_idx);
        let promote = (val & (1 << 7)) != 0;
        let is_drop = (val & (1 << 15)) != 0;

        if is_drop {
            let pt_idx = ((val >> 8) & 0x0F) as usize;
            let pt = PieceType::from_hand_index(pt_idx)?;
            Some(Move::drop(to_sq, pt))
        } else {
            let from_idx = ((val >> 8) & 0x7F) as usize;
            if from_idx >= 81 {
                return None;
            }
            let from_sq = Square::from_index(from_idx);
            Some(Move::normal(from_sq, to_sq, promote))
        }
    }

    #[inline(always)]
    pub fn normal(from: Square, to: Square, promote: bool) -> Self {
        Move {
            from: Some(from),
            to,
            promote,
            drop_piece: None,
        }
    }

    #[inline(always)]
    pub fn drop(to: Square, piece_type: PieceType) -> Self {
        Move {
            from: None,
            to,
            promote: false,
            drop_piece: Some(piece_type),
        }
    }

    #[inline(always)]
    pub fn is_drop(self) -> bool {
        self.from.is_none()
    }

    #[inline(always)]
    pub fn from(self) -> Option<Square> {
        self.from
    }

    #[inline(always)]
    pub fn to(self) -> Square {
        self.to
    }

    #[inline(always)]
    pub fn is_promote(self) -> bool {
        self.promote
    }

    #[inline(always)]
    pub fn drop_piece(self) -> Option<PieceType> {
        self.drop_piece
    }

    /// USI文字列 ("7g7f", "8h2b+", "P*5b" 等) を返す
    pub fn to_usi(self) -> String {
        if let Some(dp) = self.drop_piece {
            format!("{}*{}", dp.to_usi_char(), self.to.to_usi())
        } else if let Some(from) = self.from {
            if self.promote {
                format!("{}{}+", from.to_usi(), self.to.to_usi())
            } else {
                format!("{}{}", from.to_usi(), self.to.to_usi())
            }
        } else {
            "none".to_string()
        }
    }

    /// USI文字列から Move を復元
    pub fn from_usi(s: &str) -> Option<Self> {
        let trimmed = s.trim();
        if trimmed.len() < 4 {
            return None;
        }

        // 駒打ち形式: "P*7f"
        if trimmed.chars().nth(1) == Some('*') {
            let pt_char = trimmed.chars().next()?;
            let (pt, _) = PieceType::from_sfen_char(pt_char)?;
            pt.hand_index()?;
            let to_sq = Square::from_usi(&trimmed[2..4])?;
            return Some(Move::drop(to_sq, pt));
        }

        // 通常手: "7g7f" または 成り "8h2b+"
        let from_sq = Square::from_usi(&trimmed[0..2])?;
        let to_sq = Square::from_usi(&trimmed[2..4])?;
        let promote = trimmed.len() >= 5 && trimmed.ends_with('+');
        Some(Move::normal(from_sq, to_sq, promote))
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Move({})", self.to_usi())
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_usi())
    }
}
