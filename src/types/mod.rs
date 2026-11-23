pub mod color;
pub mod moves;
pub mod piece;
pub mod square;

pub use color::Color;
pub use moves::Move;
pub use piece::{Piece, PieceType};
pub use square::Square;

pub const NUM_SQUARES: usize = Square::NUM_SQUARES;

/// 王の8方向の相対座標 (df, dr)
pub const KING_DIRS: [(i8, i8); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1)
];

/// 飛車・香車の4方向直交ベクトル (df, dr)
pub const ORTHOGONAL_DIRS: [(i8, i8); 4] = [(0, 1), (0, -1), (1, 0), (-1, 0)];

/// 角の4方向斜めベクトル (df, dr)
pub const DIAGONAL_DIRS: [(i8, i8); 4] = [(-1, -1), (-1, 1), (1, -1), (1, 1)];
