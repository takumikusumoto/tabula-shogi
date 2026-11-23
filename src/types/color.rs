#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Color {
    Black, // 先手 (▲)
    White, // 後手 (△)
}

impl Color {
    #[inline(always)]
    pub fn opposite(self) -> Self {
        match self {
            Color::Black => Color::White,
            Color::White => Color::Black,
        }
    }

    #[inline(always)]
    pub fn index(self) -> usize {
        match self {
            Color::Black => 0,
            Color::White => 1,
        }
    }

    /// 評価値の符号 (+1 = Black, -1 = White)
    #[inline(always)]
    pub fn sign(self) -> i32 {
        match self {
            Color::Black => 1,
            Color::White => -1,
        }
    }

    /// 前方方向の rank オフセット (Black = -1, White = +1)
    #[inline(always)]
    pub fn forward_dir(self) -> i8 {
        match self {
            Color::Black => -1,
            Color::White => 1,
        }
    }

    pub fn to_usi(self) -> char {
        match self {
            Color::Black => 'b',
            Color::White => 'w',
        }
    }

    pub fn from_usi(c: char) -> Option<Self> {
        match c {
            'b' | 'B' => Some(Color::Black),
            'w' | 'W' => Some(Color::White),
            _ => None,
        }
    }
}
