use crate::board::Position;
use crate::types::{Color, DIAGONAL_DIRS, KING_DIRS, Move, ORTHOGONAL_DIRS, PieceType, Square};

pub struct MoveGenerator;

impl MoveGenerator {
    /// 現在の局面のすべての厳密な合法手 (Strict Legal Moves) を生成
    pub fn generate_legal_moves(pos: &mut Position) -> Vec<Move> {
        let pseudo_moves = Self::generate_pseudo_legal_moves(pos);
        let us = pos.side_to_move;
        let mut legal_moves = Vec::with_capacity(pseudo_moves.len());

        for mv in pseudo_moves {
            pos.do_move(mv);

            // 1. 自殺手（自分の玉に王手がかかったまま）のチェック
            if pos.is_in_check(us) {
                pos.undo_move();
                continue;
            }

            // 2. 打ち歩詰めのチェック
            // 打った駒が歩で、相手玉に王手がかかっている場合、相手に回避手があるか確認
            if mv.is_drop() && mv.drop_piece() == Some(PieceType::Pawn) {
                let opp = us.opposite();
                if pos.is_in_check(opp) {
                    // 相手の合法手を生成して1つでもあれば詰みではない
                    let opp_pseudo = Self::generate_pseudo_legal_moves(pos);
                    let mut has_escape = false;
                    for opp_mv in opp_pseudo {
                        pos.do_move(opp_mv);
                        if !pos.is_in_check(opp) {
                            has_escape = true;
                            pos.undo_move();
                            break;
                        }
                        pos.undo_move();
                    }

                    if !has_escape {
                        // 相手に回避手がない = 打ち歩詰めなので非合法
                        pos.undo_move();
                        continue;
                    }
                }
            }

            pos.undo_move();
            legal_moves.push(mv);
        }

        legal_moves
    }

    /// 王手を回避する合法手のみを生成（王手がかかっている局面用）
    pub fn generate_evasions(pos: &mut Position) -> Vec<Move> {
        // Astra 6 レビュー F07 指摘: 王手回避手においても打ち歩詰め禁止等の完全合法手規則を厳格に適用
        Self::generate_legal_moves(pos)
    }

    /// 王手となる合法手のみを生成（詰み探索用）
    pub fn generate_checks(pos: &mut Position) -> Vec<Move> {
        let legal = Self::generate_legal_moves(pos);
        let opp = pos.side_to_move.opposite();
        let mut checks = Vec::with_capacity(legal.len());

        for mv in legal {
            pos.do_move(mv);
            if pos.is_in_check(opp) {
                checks.push(mv);
            }
            pos.undo_move();
        }

        checks
    }

    /// 疑似合法手（王手放置・打ち歩詰めチェック前の移動ルールを満たす手）の生成
    pub fn generate_pseudo_legal_moves(pos: &Position) -> Vec<Move> {
        let mut moves = Vec::with_capacity(128);
        let us = pos.side_to_move;

        // 1. 盤上の駒の移動
        for sq_idx in 0..81 {
            if let Some(piece) = pos.board[sq_idx]
                && piece.color == us
            {
                let from = Square::from_index(sq_idx);
                Self::generate_piece_moves(pos, from, piece.piece_type, us, &mut moves);
            }
        }

        // 2. 持ち駒の打込み
        Self::generate_drop_moves(pos, us, &mut moves);

        moves
    }

    /// 駒打ち手の生成
    fn generate_drop_moves(pos: &Position, us: Color, moves: &mut Vec<Move>) {
        // 二歩チェック用: 自軍の歩が存在する筋を一度だけ事前計算 (#27)
        let mut pawn_files = [false; 9];
        for sq_idx in 0..81 {
            if let Some(p) = pos.board[sq_idx]
                && p.color == us
                && p.piece_type == PieceType::Pawn
            {
                pawn_files[Square::from_index(sq_idx).file() as usize] = true;
            }
        }

        for (h_idx, &pt) in PieceType::HAND_PIECES.iter().enumerate() {
            if pos.hand[us.index()][h_idx] == 0 {
                continue;
            }

            for file in 0..9 {
                // 二歩チェック (歩の場合、同一筋に生歩があればスキップ)
                if pt == PieceType::Pawn && pawn_files[file as usize] {
                    continue;
                }

                for rank in 0..9 {
                    let to = Square::new(file, rank);
                    if pos.board[to.index()].is_some() {
                        continue; // 空マスのみ
                    }

                    // 行き所のない駒の打ち禁止
                    match pt {
                        PieceType::Pawn | PieceType::Lance => {
                            if (us == Color::Black && rank == 0)
                                || (us == Color::White && rank == 8)
                            {
                                continue;
                            }
                        }
                        PieceType::Knight
                            if ((us == Color::Black && rank <= 1)
                                || (us == Color::White && rank >= 7)) =>
                        {
                            continue;
                        }
                        _ => {}
                    }

                    moves.push(Move::drop(to, pt));
                }
            }
        }
    }

    /// 盤上駒の移動手の生成
    fn generate_piece_moves(
        pos: &Position,
        from: Square,
        pt: PieceType,
        us: Color,
        moves: &mut Vec<Move>,
    ) {
        let fwd_dr = us.forward_dir();

        match pt {
            PieceType::Pawn => {
                if let Some(to) = from.offset(0, fwd_dr) {
                    Self::add_move_with_promotion(from, to, pt, us, moves, pos);
                }
            }
            PieceType::Lance => {
                let mut cr = from.rank() as i8 + fwd_dr;
                while (0..9).contains(&cr) {
                    let to = Square::new(from.file(), cr as u8);
                    if let Some(target_p) = pos.board[to.index()] {
                        if target_p.color != us {
                            Self::add_move_with_promotion(from, to, pt, us, moves, pos);
                        }
                        break;
                    } else {
                        Self::add_move_with_promotion(from, to, pt, us, moves, pos);
                    }
                    cr += fwd_dr;
                }
            }
            PieceType::Knight => {
                for df in [-1, 1] {
                    if let Some(to) = from.offset(df, fwd_dr * 2) {
                        Self::add_move_with_promotion(from, to, pt, us, moves, pos);
                    }
                }
            }
            PieceType::Silver => {
                // 前方1 + 斜め4
                if let Some(to) = from.offset(0, fwd_dr) {
                    Self::add_move_with_promotion(from, to, pt, us, moves, pos);
                }
                for (df, dr) in DIAGONAL_DIRS {
                    if let Some(to) = from.offset(df, dr) {
                        Self::add_move_with_promotion(from, to, pt, us, moves, pos);
                    }
                }
            }
            PieceType::Gold
            | PieceType::ProPawn
            | PieceType::ProLance
            | PieceType::ProKnight
            | PieceType::ProSilver => {
                // 前後左右4 + 前斜め2
                for (df, dr) in ORTHOGONAL_DIRS {
                    if let Some(to) = from.offset(df, dr) {
                        Self::add_move_with_promotion(from, to, pt, us, moves, pos);
                    }
                }
                for &df in &[-1, 1] {
                    if let Some(to) = from.offset(df, fwd_dr) {
                        Self::add_move_with_promotion(from, to, pt, us, moves, pos);
                    }
                }
            }
            PieceType::King => {
                // 全8方向
                for (df, dr) in KING_DIRS {
                    if let Some(to) = from.offset(df, dr) {
                        Self::add_move_with_promotion(from, to, pt, us, moves, pos);
                    }
                }
            }
            PieceType::Bishop => {
                // 斜め4方向レイスキャン
                Self::ray_moves(pos, from, &DIAGONAL_DIRS, pt, us, moves);
            }
            PieceType::Horse => {
                // 斜め4方向レイスキャン + 上下左右1マス
                Self::ray_moves(pos, from, &DIAGONAL_DIRS, pt, us, moves);
                for (df, dr) in ORTHOGONAL_DIRS {
                    if let Some(to) = from.offset(df, dr) {
                        Self::add_move_with_promotion(from, to, pt, us, moves, pos);
                    }
                }
            }
            PieceType::Rook => {
                // 上下左右4方向レイスキャン
                Self::ray_moves(pos, from, &ORTHOGONAL_DIRS, pt, us, moves);
            }
            PieceType::Dragon => {
                // 上下左右4方向レイスキャン + 斜め4マス
                Self::ray_moves(pos, from, &ORTHOGONAL_DIRS, pt, us, moves);
                for (df, dr) in DIAGONAL_DIRS {
                    if let Some(to) = from.offset(df, dr) {
                        Self::add_move_with_promotion(from, to, pt, us, moves, pos);
                    }
                }
            }
        }
    }

    /// レイスキャン用のヘルパー
    fn ray_moves(
        pos: &Position,
        from: Square,
        dirs: &[(i8, i8)],
        pt: PieceType,
        us: Color,
        moves: &mut Vec<Move>,
    ) {
        for &(df, dr) in dirs {
            let mut cf = from.file() as i8 + df;
            let mut cr = from.rank() as i8 + dr;
            while (0..9).contains(&cf) && (0..9).contains(&cr) {
                let to = Square::new(cf as u8, cr as u8);
                if let Some(target_p) = pos.board[to.index()] {
                    if target_p.color != us {
                        Self::add_move_with_promotion(from, to, pt, us, moves, pos);
                    }
                    break;
                } else {
                    Self::add_move_with_promotion(from, to, pt, us, moves, pos);
                }
                cf += df;
                cr += dr;
            }
        }
    }

    /// 成り・不成りの判定と手の追加
    fn add_move_with_promotion(
        from: Square,
        to: Square,
        pt: PieceType,
        us: Color,
        moves: &mut Vec<Move>,
        pos: &Position,
    ) {
        if let Some(target_p) = pos.board[to.index()]
            && target_p.color == us
        {
            return; // 味方の駒があるマスには移動不可
        }

        let can_promote =
            pt.can_promote() && (from.is_promoted_zone(us) || to.is_promoted_zone(us));

        // 成りの強制チェック（行き所のない駒）
        let must_promote = match pt {
            PieceType::Pawn | PieceType::Lance => {
                (us == Color::Black && to.rank() == 0) || (us == Color::White && to.rank() == 8)
            }
            PieceType::Knight => {
                (us == Color::Black && to.rank() <= 1) || (us == Color::White && to.rank() >= 7)
            }
            _ => false,
        };

        if must_promote {
            moves.push(Move::normal(from, to, true));
        } else {
            if can_promote {
                moves.push(Move::normal(from, to, true));
            }
            moves.push(Move::normal(from, to, false));
        }
    }
}
