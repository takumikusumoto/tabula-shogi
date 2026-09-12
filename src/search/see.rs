use crate::board::Position;
use crate::types::{Color, Move, Piece, PieceType, Square};

pub struct SEE;

pub const PROMOTION_SEE_VALUE: i32 = 200;
pub const MAX_SEE_DEPTH: usize = 32;

impl SEE {
    /// 静的駒交換評価 (Static Exchange Evaluation)
    /// 指定された手 `mv` を指した結果、最終的な駒得失点（センチポーン）を返す。
    /// 正の値なら駒得、負の値なら駒損（Bad Capture）。
    pub fn evaluate(pos: &Position, mv: Move) -> i32 {
        let to_sq = mv.to();
        let target_piece = match pos.board[to_sq.index()] {
            Some(p) => p,
            None => {
                // 駒取りでない場合
                return if mv.is_promote() {
                    PROMOTION_SEE_VALUE // 成りのみ
                } else {
                    0
                };
            }
        };

        // 駒打ちで取ることはできない（将棋のルール上）
        let moved_piece_type = if let Some(from_sq) = mv.from() {
            match pos.board[from_sq.index()] {
                Some(p) => p.piece_type,
                None => return 0,
            }
        } else {
            return 0;
        };

        // 各キャプチャ段階での被捕獲駒の価値
        let mut gain = [0i32; MAX_SEE_DEPTH];
        let mut d = 0;

        // 0段階目で得る駒
        gain[0] = target_piece.piece_type.base_value();

        // 仮想的な盤面で攻撃駒を順番に辿る
        let mut attackers = Self::get_attackers_to(pos, to_sq);
        let mut current_pt = moved_piece_type;
        let mut side = pos.side_to_move.opposite();

        // 最初の手の攻撃駒を除去
        if let Some(from_sq) = mv.from() {
            attackers.retain(|&(sq, _)| sq != from_sq);
        }

        while !attackers.is_empty() {
            // 最小価値の攻撃駒 (Least Valuable Attacker) を探す
            let lva_idx = Self::find_lva(&attackers, side);
            if let Some(idx) = lva_idx {
                d += 1;
                // d手目で手番側が取る駒は直前の current_pt
                gain[d] = current_pt.base_value();
                let (_, piece) = attackers.remove(idx);
                current_pt = piece.piece_type;
                side = side.opposite();
            } else {
                break;
            }
        }

        // ミニマックス的に巻き戻し: 相手は不利ならパス（交換打ち切り）できる
        // gain[d - 1] = gain[d - 1] - max(0, gain[d])
        while d > 0 {
            gain[d - 1] -= gain[d].max(0);
            d -= 1;
        }

        gain[0]
    }

    /// 対象マス `target_sq` に利きを持つ全駒 (Square, Piece) を収集
    fn get_attackers_to(pos: &Position, target_sq: Square) -> Vec<(Square, Piece)> {
        let mut attackers = Vec::with_capacity(16);
        for sq_idx in 0..81 {
            if let Some(p) = pos.board[sq_idx] {
                let sq = Square::from_index(sq_idx);
                if pos.attacks_to(target_sq, p.color) {
                    // 自駒の利きがこのsqから出ているかチェック
                    if Self::can_attack(pos, sq, target_sq, p) {
                        attackers.push((sq, p));
                    }
                }
            }
        }
        attackers
    }

    /// sqにある駒pが直接target_sqに利いているか
    fn can_attack(pos: &Position, from: Square, to: Square, p: Piece) -> bool {
        let df = to.file() as i8 - from.file() as i8;
        let dr = to.rank() as i8 - from.rank() as i8;
        let fwd_dr: i8 = if p.color == Color::Black { -1 } else { 1 };

        match p.piece_type {
            PieceType::Pawn => df == 0 && dr == fwd_dr,
            PieceType::Knight => (df == -1 || df == 1) && dr == fwd_dr * 2,
            PieceType::Silver => (df == 0 && dr == fwd_dr) || (df.abs() == 1 && dr.abs() == 1),
            PieceType::Gold
            | PieceType::ProPawn
            | PieceType::ProLance
            | PieceType::ProKnight
            | PieceType::ProSilver => (df.abs() + dr.abs() == 1) || (df.abs() == 1 && dr == fwd_dr),
            PieceType::King => df.abs() <= 1 && dr.abs() <= 1 && (df != 0 || dr != 0),
            PieceType::Lance => {
                if df != 0 {
                    return false;
                }
                if (p.color == Color::Black && dr >= 0) || (p.color == Color::White && dr <= 0) {
                    return false;
                }
                Self::is_ray_clear(pos, from, to, 0, dr.signum())
            }
            PieceType::Rook | PieceType::Dragon => {
                if df == 0 || dr == 0 {
                    Self::is_ray_clear(pos, from, to, df.signum(), dr.signum())
                } else if p.piece_type == PieceType::Dragon {
                    df.abs() == 1 && dr.abs() == 1
                } else {
                    false
                }
            }
            PieceType::Bishop | PieceType::Horse => {
                if df.abs() == dr.abs() {
                    Self::is_ray_clear(pos, from, to, df.signum(), dr.signum())
                } else if p.piece_type == PieceType::Horse {
                    df.abs() + dr.abs() == 1
                } else {
                    false
                }
            }
        }
    }

    /// 直線上の間に障害物がないか
    fn is_ray_clear(pos: &Position, from: Square, to: Square, step_f: i8, step_r: i8) -> bool {
        let mut cf = from.file() as i8 + step_f;
        let mut cr = from.rank() as i8 + step_r;
        let tf = to.file() as i8;
        let tr = to.rank() as i8;

        while cf != tf || cr != tr {
            let sq = Square::new(cf as u8, cr as u8);
            if pos.board[sq.index()].is_some() {
                return false;
            }
            cf += step_f;
            cr += step_r;
        }
        true
    }

    /// 指定サイドの最小価値の攻撃駒のインデックスを探す
    fn find_lva(attackers: &[(Square, Piece)], side: Color) -> Option<usize> {
        let mut best_idx = None;
        let mut min_val = i32::MAX;

        for (i, &(_, piece)) in attackers.iter().enumerate() {
            if piece.color == side {
                let val = piece.piece_type.base_value();
                if val < min_val {
                    min_val = val;
                    best_idx = Some(i);
                }
            }
        }
        best_idx
    }
}
