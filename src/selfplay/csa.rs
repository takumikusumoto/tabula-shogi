use super::game::GameRecord;
use crate::board::Position;
use crate::types::{Color, Move, PieceType};

pub struct CsaSerializer;

impl CsaSerializer {
    pub fn format_move(color: Color, mv: Move, pt_after: PieceType) -> String {
        let sign = match color {
            Color::Black => '+',
            Color::White => '-',
        };
        let from_str = if let Some(from_sq) = mv.from() {
            format!("{}{}", from_sq.file() + 1, from_sq.rank() + 1)
        } else {
            "00".to_string()
        };
        let to_str = format!("{}{}", mv.to().file() + 1, mv.to().rank() + 1);
        let pt_str = match pt_after {
            PieceType::Pawn => "FU",
            PieceType::Lance => "KY",
            PieceType::Knight => "KE",
            PieceType::Silver => "GI",
            PieceType::Gold => "KI",
            PieceType::Bishop => "KA",
            PieceType::Rook => "HI",
            PieceType::King => "OU",
            PieceType::ProPawn => "TO",
            PieceType::ProLance => "NY",
            PieceType::ProKnight => "NK",
            PieceType::ProSilver => "NG",
            PieceType::Horse => "UM",
            PieceType::Dragon => "RY",
        };
        format!("{sign}{from_str}{to_str}{pt_str}")
    }

    /// 1対局の記録を標準CSA形式 (Version 2.2) の文字列へ変換
    pub fn serialize_game(game: &GameRecord) -> String {
        let mut out = String::new();
        out.push_str("V2.2\n");
        out.push_str("N+TabulaShogi\n");
        out.push_str("N-TabulaShogi\n");
        out.push_str("PI\n");
        out.push_str("+\n");

        // 盤面を再生して着手後の駒種（成りの反映）を厳密に特定
        let mut pos = Position::startpos();
        for ply_rec in &game.plies {
            let mv = ply_rec.mv;
            let pt_after = if let Some(dp) = mv.drop_piece() {
                dp
            } else if let Some(from_sq) = mv.from() {
                if let Some(p) = pos.board[from_sq.index()] {
                    if mv.is_promote() {
                        p.piece_type.promote().unwrap_or(p.piece_type)
                    } else {
                        p.piece_type
                    }
                } else {
                    PieceType::Pawn
                }
            } else {
                PieceType::Pawn
            };

            let csa_move = Self::format_move(ply_rec.side_to_move, mv, pt_after);
            out.push_str(&csa_move);
            out.push('\n');
            out.push_str("T1\n");

            pos.do_move(mv);
        }

        out.push_str(game.result.to_csa_end_comment());
        out.push('\n');
        out
    }
}
