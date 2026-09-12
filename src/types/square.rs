use super::color::Color;
use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Square(pub u8);

impl Square {
    pub const NUM_SQUARES: usize = 81;

    #[inline(always)]
    pub fn new(file: u8, rank: u8) -> Self {
        debug_assert!(file < 9 && rank < 9);
        Square(file * 9 + rank)
    }

    #[inline(always)]
    pub fn from_index(index: usize) -> Self {
        debug_assert!(index < Self::NUM_SQUARES);
        Square(index as u8)
    }

    #[inline(always)]
    pub fn index(self) -> usize {
        self.0 as usize
    }

    #[inline(always)]
    pub fn file(self) -> u8 {
        self.0 / 9
    }

    #[inline(always)]
    pub fn rank(self) -> u8 {
        self.0 % 9
    }

    /// 敵陣かどうか (先手は0..=2段目, 後手は6..=8段目)
    #[inline(always)]
    pub fn is_promoted_zone(self, color: Color) -> bool {
        match color {
            Color::Black => self.rank() <= 2,
            Color::White => self.rank() >= 6,
        }
    }

    /// 相対的な移動が盤面内にあるかをチェックし、Squareを返す
    #[inline(always)]
    pub fn offset(self, df: i8, dr: i8) -> Option<Square> {
        let f = self.file() as i8 + df;
        let r = self.rank() as i8 + dr;
        if (0..9).contains(&f) && (0..9).contains(&r) {
            Some(Square::new(f as u8, r as u8))
        } else {
            None
        }
    }

    /// USI形式の文字列 ("7g" 等) に変換
    pub fn to_usi(self) -> String {
        let file_char = (b'1' + self.file()) as char;
        let rank_char = (b'a' + self.rank()) as char;
        format!("{file_char}{rank_char}")
    }

    /// USI文字列 ("7g" 等) からのパース
    pub fn from_usi(s: &str) -> Option<Self> {
        if s.len() < 2 {
            return None;
        }
        let bytes = s.as_bytes();
        let file_byte = bytes[0];
        let rank_byte = bytes[1];
        if !(b'1'..=b'9').contains(&file_byte) || !(b'a'..=b'i').contains(&rank_byte) {
            return None;
        }
        let file = file_byte - b'1';
        let rank = rank_byte - b'a';
        Some(Square::new(file, rank))
    }
}

impl fmt::Debug for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Square({})", self.to_usi())
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_usi())
    }
}
