use crate::board::zobrist::get_zobrist_keys;
use crate::types::{
    Color, DIAGONAL_DIRS, KING_DIRS, Move, ORTHOGONAL_DIRS, Piece, PieceType, Square,
};

#[derive(Clone, Copy, Debug)]
pub struct PositionRecord {
    pub mv: Move,
    pub captured: Option<Piece>,
    pub prev_hash: u64,
}

#[derive(Clone)]
pub struct Position {
    pub board: [Option<Piece>; 81],
    pub hand: [[u8; 7]; 2],
    pub side_to_move: Color,
    pub king_sq: [Option<Square>; 2],
    pub ply: usize,
    pub hash: u64,
    pub history: Vec<PositionRecord>,
    pub hash_history: Vec<u64>,
}

/// ゲームの最大手数（十分大きい値）
pub const MAX_GAME_PLIES: usize = 512;

impl Position {
    pub fn empty() -> Self {
        Position {
            board: [None; 81],
            hand: [[0; 7]; 2],
            side_to_move: Color::Black,
            king_sq: [None; 2],
            ply: 0,
            hash: 0,
            history: Vec::with_capacity(MAX_GAME_PLIES / 2),
            hash_history: Vec::with_capacity(MAX_GAME_PLIES / 2),
        }
    }

    /// 平手初期局面の生成
    pub fn startpos() -> Self {
        Self::from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
            .expect("Valid startpos sfen")
    }

    /// SFEN文字列から局面を構築
    pub fn from_sfen(sfen: &str) -> Result<Self, String> {
        let parts: Vec<&str> = sfen.split_whitespace().collect();
        if parts.len() < 3 {
            return Err("SFEN needs at least 3 parts (board, turn, hand)".to_string());
        }

        let mut pos = Position::empty();
        let rows: Vec<&str> = parts[0].split('/').collect();
        if rows.len() != 9 {
            return Err(format!("SFEN board must have 9 rows, got {}", rows.len()));
        }

        for (rank_idx, row) in rows.iter().enumerate() {
            let mut file_idx: i8 = 8; // 9筋から1筋 (8 down to 0)
            let mut chars = row.chars().peekable();

            while let Some(c) = chars.next() {
                if file_idx < 0 {
                    return Err(format!("Row {rank_idx} has too many squares"));
                }

                if let Some(digit) = c.to_digit(10) {
                    file_idx -= digit as i8;
                } else {
                    let is_promoted = c == '+';
                    let piece_char = if is_promoted {
                        chars
                            .next()
                            .ok_or_else(|| "Dangling + in SFEN".to_string())?
                    } else {
                        c
                    };

                    let (base_pt, color) = PieceType::from_sfen_char(piece_char)
                        .ok_or_else(|| format!("Invalid piece char '{piece_char}'"))?;
                    let pt = if is_promoted {
                        base_pt
                            .promote()
                            .ok_or_else(|| format!("Cannot promote {base_pt:?}"))?
                    } else {
                        base_pt
                    };

                    let sq = Square::new(file_idx as u8, rank_idx as u8);
                    pos.board[sq.index()] = Some(Piece::new(pt, color));
                    if pt == PieceType::King {
                        pos.king_sq[color.index()] = Some(sq);
                    }

                    file_idx -= 1;
                }
            }
        }

        // 手番
        pos.side_to_move = match parts[1] {
            "b" => Color::Black,
            "w" => Color::White,
            _ => return Err(format!("Invalid turn '{}'", parts[1])),
        };

        // 持ち駒
        if parts[2] != "-" {
            let mut count: usize = 0;
            for c in parts[2].chars() {
                if let Some(d) = c.to_digit(10) {
                    count = count * 10 + d as usize;
                } else {
                    let num = if count == 0 { 1 } else { count };
                    let (pt, color) = PieceType::from_sfen_char(c)
                        .ok_or_else(|| format!("Invalid hand char '{c}'"))?;
                    let h_idx = pt
                        .hand_index()
                        .ok_or_else(|| format!("{pt:?} cannot be in hand"))?;
                    pos.hand[color.index()][h_idx] += num as u8;
                    count = 0;
                }
            }
        }

        // 手数
        if parts.len() >= 4 {
            pos.ply = parts[3].parse().unwrap_or(1);
        } else {
            pos.ply = 1;
        }

        pos.hash = pos.compute_hash();
        pos.hash_history.push(pos.hash);

        Ok(pos)
    }

    /// 現在の局面をSFEN文字列に変換
    pub fn to_sfen(&self) -> String {
        let mut s = String::new();
        // 1. 盤面 (1段目から9段目)
        for rank in 0..9 {
            if rank > 0 {
                s.push('/');
            }
            let mut empty_count = 0;
            for file in (0..9).rev() {
                let sq = Square::new(file, rank);
                if let Some(piece) = self.board[sq.index()] {
                    if empty_count > 0 {
                        s.push_str(&empty_count.to_string());
                        empty_count = 0;
                    }
                    s.push_str(&piece.to_sfen());
                } else {
                    empty_count += 1;
                }
            }
            if empty_count > 0 {
                s.push_str(&empty_count.to_string());
            }
        }

        // 2. 手番
        s.push(' ');
        s.push(match self.side_to_move {
            Color::Black => 'b',
            Color::White => 'w',
        });

        // 3. 持ち駒 (飛, 角, 金, 銀, 桂, 香, 歩 の順)
        s.push(' ');
        let mut hand_str = String::new();
        for color in [Color::Black, Color::White] {
            for pt in [
                PieceType::Rook,
                PieceType::Bishop,
                PieceType::Gold,
                PieceType::Silver,
                PieceType::Knight,
                PieceType::Lance,
                PieceType::Pawn
            ] {
                if let Some(h_idx) = pt.hand_index() {
                    let count = self.hand[color.index()][h_idx];
                    if count > 0 {
                        if count > 1 {
                            hand_str.push_str(&count.to_string());
                        }
                        let mut c = match pt {
                            PieceType::Rook => 'r',
                            PieceType::Bishop => 'b',
                            PieceType::Gold => 'g',
                            PieceType::Silver => 's',
                            PieceType::Knight => 'n',
                            PieceType::Lance => 'l',
                            PieceType::Pawn => 'p',
                            _ => '?',
                        };
                        if color == Color::Black {
                            c = c.to_ascii_uppercase();
                        }
                        hand_str.push(c);
                    }
                }
            }
        }
        if hand_str.is_empty() {
            s.push('-');
        } else {
            s.push_str(&hand_str);
        }

        // 4. 手数
        s.push(' ');
        s.push_str(&self.ply.to_string());

        s
    }

    /// 局面全体のハッシュを新規計算
    pub fn compute_hash(&self) -> u64 {
        let keys = get_zobrist_keys();
        let mut h: u64 = 0;

        for sq_idx in 0..81 {
            if let Some(piece) = self.board[sq_idx] {
                let sq = Square::from_index(sq_idx);
                h ^= keys.piece_key(sq, piece);
            }
        }

        for c in [Color::Black, Color::White] {
            for pt in PieceType::HAND_PIECES {
                if let Some(idx) = pt.hand_index() {
                    let count = self.hand[c.index()][idx] as usize;
                    if count > 0 {
                        h ^= keys.hand_key(c, pt, count);
                    }
                }
            }
        }

        if self.side_to_move == Color::White {
            h ^= keys.side_to_move;
        }

        h
    }

    /// 指定のマスに相手側の利きがあるか
    pub fn attacks_to(&self, target_sq: Square, attacker_color: Color) -> bool {
        self.attacks_to_internal(target_sq, attacker_color, true)
    }

    /// 玉以外の駒による利きがあるか（玉頭防御やピンなどの判定用）
    pub fn attacks_to_without_king(&self, target_sq: Square, attacker_color: Color) -> bool {
        self.attacks_to_internal(target_sq, attacker_color, false)
    }

    fn attacks_to_internal(
        &self,
        target_sq: Square,
        attacker_color: Color,
        include_king: bool,
    ) -> bool {
        let keys_sq = target_sq;
        let tf = keys_sq.file() as i8;
        let tr = keys_sq.rank() as i8;

        // 1. 歩の利きチェック (逆方向に歩があるか)
        let pawn_dr = -attacker_color.forward_dir();
        if let Some(sq) = keys_sq.offset(0, pawn_dr)
            && let Some(p) = self.board[sq.index()]
            && p.color == attacker_color
            && p.piece_type == PieceType::Pawn
        {
            return true;
        }

        // 2. 桂馬の利きチェック
        let knight_dr = -2 * attacker_color.forward_dir();
        for df in [-1, 1] {
            if let Some(sq) = keys_sq.offset(df, knight_dr)
                && let Some(p) = self.board[sq.index()]
                && p.color == attacker_color
                && p.piece_type == PieceType::Knight
            {
                return true;
            }
        }

        // 3. 銀の利きチェック (前方1 + 斜め4)
        // 攻撃側視点で銀の利き位置 = 防御側(target_sq)から見て逆方向
        let silver_fwd_dr = -attacker_color.forward_dir();
        // 直前方からの利き (target_sq の fwd_dr)
        if let Some(sq) = keys_sq.offset(0, silver_fwd_dr)
            && let Some(p) = self.board[sq.index()]
            && p.color == attacker_color
            && p.piece_type == PieceType::Silver
        {
            return true;
        }
        // 斜め4方向
        for (df, dr) in DIAGONAL_DIRS {
            if let Some(sq) = keys_sq.offset(df, dr)
                && let Some(p) = self.board[sq.index()]
                && p.color == attacker_color
                && p.piece_type == PieceType::Silver
            {
                return true;
            }
        }

        // 4. 金および金相当の成駒 (前後左右4 + 前斜め2)
        let gold_fwd_dr = -attacker_color.forward_dir();
        let gold_dirs = [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
            (-1, gold_fwd_dr),
            (1, gold_fwd_dr)
        ];
        for (df, dr) in gold_dirs {
            if let Some(sq) = keys_sq.offset(df, dr)
                && let Some(p) = self.board[sq.index()]
                && p.color == attacker_color
            {
                match p.piece_type {
                    PieceType::Gold
                    | PieceType::ProPawn
                    | PieceType::ProLance
                    | PieceType::ProKnight
                    | PieceType::ProSilver => return true,
                    _ => {}
                }
            }
        }

        // 5. 王の利きチェック (周囲8マス)
        if include_king {
            for (df, dr) in KING_DIRS {
                if let Some(sq) = keys_sq.offset(df, dr)
                    && let Some(p) = self.board[sq.index()]
                    && p.color == attacker_color
                    && p.piece_type == PieceType::King
                {
                    return true;
                }
            }
        }

        // 6. 香車の利き (レイスキャン: 攻撃側先手ならtargetの下方向から上がってくる)
        let lance_dr = -attacker_color.forward_dir();
        let mut curr_r = tr + lance_dr;
        while (0..9).contains(&curr_r) {
            let sq = Square::new(tf as u8, curr_r as u8);
            if let Some(p) = self.board[sq.index()] {
                if p.color == attacker_color && p.piece_type == PieceType::Lance {
                    return true;
                }
                break; // 遮る駒がある
            }
            curr_r += lance_dr;
        }

        // 7. 飛車・竜の利き (上下左右4方向レイスキャン)
        for (df, dr) in ORTHOGONAL_DIRS {
            let mut cf = tf + df;
            let mut cr = tr + dr;
            while (0..9).contains(&cf) && (0..9).contains(&cr) {
                let sq = Square::new(cf as u8, cr as u8);
                if let Some(p) = self.board[sq.index()] {
                    if p.color == attacker_color
                        && (p.piece_type == PieceType::Rook || p.piece_type == PieceType::Dragon)
                    {
                        return true;
                    }
                    break;
                }
                cf += df;
                cr += dr;
            }
        }

        // 8. 角・馬の利き (斜め4方向レイスキャン)
        for (df, dr) in DIAGONAL_DIRS {
            let mut cf = tf + df;
            let mut cr = tr + dr;
            while (0..9).contains(&cf) && (0..9).contains(&cr) {
                let sq = Square::new(cf as u8, cr as u8);
                if let Some(p) = self.board[sq.index()] {
                    if p.color == attacker_color
                        && (p.piece_type == PieceType::Bishop || p.piece_type == PieceType::Horse)
                    {
                        return true;
                    }
                    break;
                }
                cf += df;
                cr += dr;
            }
        }

        // 9. 馬の近接4方向 (上下左右1マス)
        for (df, dr) in ORTHOGONAL_DIRS {
            if let Some(sq) = keys_sq.offset(df, dr)
                && let Some(p) = self.board[sq.index()]
                && p.color == attacker_color
                && p.piece_type == PieceType::Horse
            {
                return true;
            }
        }

        // 10. 竜の近接4方向 (斜め4マス)
        for (df, dr) in DIAGONAL_DIRS {
            if let Some(sq) = keys_sq.offset(df, dr)
                && let Some(p) = self.board[sq.index()]
                && p.color == attacker_color
                && p.piece_type == PieceType::Dragon
            {
                return true;
            }
        }

        false
    }

    /// 自玉に王手がかかっているか
    pub fn is_in_check(&self, color: Color) -> bool {
        if let Some(ksq) = self.king_sq[color.index()] {
            self.attacks_to(ksq, color.opposite())
        } else {
            false
        }
    }

    /// 指し手を適用 (do_move)
    pub fn do_move(&mut self, mv: Move) {
        let keys = get_zobrist_keys();
        let us = self.side_to_move;
        let opp = us.opposite();
        let prev_hash = self.hash;

        let mut captured = None;

        if let Some(from_sq) = mv.from() {
            // 通常移動
            let piece = self.board[from_sq.index()].expect("Move from piece must exist");
            self.board[from_sq.index()] = None;
            self.hash ^= keys.piece_key(from_sq, piece);

            // 駒の捕獲チェック
            let to_sq = mv.to();
            if let Some(cap) = self.board[to_sq.index()] {
                captured = Some(cap);
                self.hash ^= keys.piece_key(to_sq, cap);

                // 取った駒を持ち駒に追加 (成駒は生駒に戻す、玉は持ち駒に入らない)
                let hand_pt = cap.piece_type.unpromote();
                if let Some(h_idx) = hand_pt.hand_index() {
                    let old_count = self.hand[us.index()][h_idx] as usize;
                    if old_count > 0 {
                        self.hash ^= keys.hand_key(us, hand_pt, old_count);
                    }
                    self.hand[us.index()][h_idx] += 1;
                    self.hash ^= keys.hand_key(us, hand_pt, old_count + 1);
                }
            }

            // 移動先の駒
            let placed_pt = if mv.is_promote() {
                piece.piece_type.promote().unwrap_or(piece.piece_type)
            } else {
                piece.piece_type
            };
            let new_piece = Piece::new(placed_pt, us);
            self.board[to_sq.index()] = Some(new_piece);
            self.hash ^= keys.piece_key(to_sq, new_piece);

            // 玉の位置追跡
            if piece.piece_type == PieceType::King {
                self.king_sq[us.index()] = Some(to_sq);
            }
        } else {
            // 駒打ち
            let drop_pt = mv.drop_piece().expect("Drop move must have piece type");
            let to_sq = mv.to();
            let h_idx = drop_pt.hand_index().expect("Drop piece has hand index");

            let old_count = self.hand[us.index()][h_idx] as usize;
            self.hash ^= keys.hand_key(us, drop_pt, old_count);
            self.hand[us.index()][h_idx] -= 1;
            if old_count > 1 {
                self.hash ^= keys.hand_key(us, drop_pt, old_count - 1);
            }

            let new_piece = Piece::new(drop_pt, us);
            self.board[to_sq.index()] = Some(new_piece);
            self.hash ^= keys.piece_key(to_sq, new_piece);
        }

        // 手番交代
        self.side_to_move = opp;
        self.hash ^= keys.side_to_move;
        self.ply += 1;

        self.history.push(PositionRecord {
            mv,
            captured,
            prev_hash,
        });
        self.hash_history.push(self.hash);
    }

    /// 指し手を巻き戻す (undo_move)
    pub fn undo_move(&mut self) {
        let record = self.history.pop().expect("Undo with empty history");
        let mv = record.mv;
        let opp = self.side_to_move;
        let us = opp.opposite();

        self.side_to_move = us;
        self.ply -= 1;
        self.hash_history.pop();
        self.hash = record.prev_hash;

        let to_sq = mv.to();

        if let Some(from_sq) = mv.from() {
            // 通常移動の巻き戻し
            let moved_piece = self.board[to_sq.index()].expect("Piece at to_sq must exist");
            let orig_pt = if mv.is_promote() {
                moved_piece.piece_type.unpromote()
            } else {
                moved_piece.piece_type
            };
            self.board[from_sq.index()] = Some(Piece::new(orig_pt, us));

            // 取られた駒の復元
            self.board[to_sq.index()] = record.captured;
            if let Some(cap) = record.captured {
                let hand_pt = cap.piece_type.unpromote();
                if let Some(h_idx) = hand_pt.hand_index() {
                    self.hand[us.index()][h_idx] -= 1;
                }
            }

            // 玉の位置復元
            if orig_pt == PieceType::King {
                self.king_sq[us.index()] = Some(from_sq);
            }
        } else {
            // 駒打ちの巻き戻し
            let drop_pt = mv.drop_piece().expect("Drop move must have piece type");
            self.board[to_sq.index()] = None;
            let h_idx = drop_pt.hand_index().expect("Drop piece has hand index");
            self.hand[us.index()][h_idx] += 1;
        }
    }

    /// 同一局面の出現回数 (千日手判定)
    pub fn repetition_count(&self) -> usize {
        let mut count = 0;
        for &h in self.hash_history.iter().rev() {
            if h == self.hash {
                count += 1;
            }
        }
        count
    }

    /// ヌルムーブ実行 (パスして相手番へ) — hash_history を更新しないため千日手判定は行わない
    pub fn do_null_move(&mut self) {
        let keys = get_zobrist_keys();
        self.side_to_move = self.side_to_move.opposite();
        self.hash ^= keys.side_to_move;
        self.ply += 1;
    }

    /// ヌルムーブ巻き戻し
    pub fn undo_null_move(&mut self) {
        let keys = get_zobrist_keys();
        self.side_to_move = self.side_to_move.opposite();
        self.hash ^= keys.side_to_move;
        self.ply -= 1;
    }
}
