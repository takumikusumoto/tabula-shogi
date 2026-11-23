use crate::board::Position;
use crate::types::{Color, DIAGONAL_DIRS, KING_DIRS, PieceType, Square};

pub struct Evaluator;

/// 評価内訳構造体（設計レビュー 7.4 準拠）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EvalBreakdown {
    pub material_board: i32,
    pub material_hand: i32,
    pub piece_square: i32,
    pub king_safety: i32,
    pub king_danger: i32,
    pub mobility: i32,
    pub coordination: i32,
    pub outposts: i32,
    pub tempo: i32,
    pub total: i32,
}

// 持ち駒価値 (打つ柔軟性があるため生駒より高め)
const HAND_VAL: [i32; 7] = [
    120,  // Pawn
    350,  // Lance
    390,  // Knight
    550,  // Silver
    600,  // Gold
    950,  // Bishop
    1100, // Rook
];

// 手番ボーナス
const TEMPO_BONUS: i32 = 25;

// マス別位置ボーナス (先手視点: rank 0=1段目/敵陣奥, rank 8=9段目/自陣底)
static PAWN_PST: [[i16; 9]; 9] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],          // 1段目 (行き所なし)
    [25, 25, 25, 30, 30, 30, 25, 25, 25], // 2段目 (敵陣)
    [15, 20, 20, 25, 25, 25, 20, 20, 15], // 3段目 (敵陣)
    [10, 15, 18, 20, 22, 20, 18, 15, 10], // 4段目
    [5, 18, 15, 18, 22, 18, 18, 10, 5],   // 5段目 (天王山・位取り)
    [0, 20, 10, 12, 16, 12, 22, 5, 0],    // 6段目 (7六, 2六の角道・飛先好手)
    [0, 0, 0, 0, 0, 0, 0, 0, 0],          // 7段目 (初期配置)
    [0, 0, 0, 0, 0, 0, 0, 0, 0],          // 8段目
    [0, 0, 0, 0, 0, 0, 0, 0, 0],          // 9段目
];

static SILVER_PST: [[i16; 9]; 9] = [
    [10, 15, 20, 20, 20, 20, 20, 15, 10],
    [15, 25, 30, 35, 35, 35, 30, 25, 15],
    [10, 20, 25, 30, 30, 30, 25, 20, 10],
    [5, 15, 20, 25, 25, 25, 20, 15, 5],
    [0, 12, 18, 22, 22, 22, 18, 12, 0],
    [-5, 8, 15, 18, 18, 18, 15, 8, -5],
    [-10, 5, 10, 15, 12, 15, 10, 5, -10],
    [-15, 0, 5, 10, 8, 10, 5, 0, -15],
    [-20, -10, -5, 0, 0, 0, -5, -10, -20]
];

static GOLD_PST: [[i16; 9]; 9] = [
    [10, 15, 20, 20, 20, 20, 20, 15, 10],
    [10, 20, 25, 30, 30, 30, 25, 20, 10],
    [5, 15, 20, 25, 25, 25, 20, 15, 5],
    [0, 10, 15, 20, 20, 20, 15, 10, 0],
    [0, 10, 15, 20, 20, 20, 15, 10, 0],
    [0, 12, 16, 20, 20, 20, 16, 12, 0],
    [5, 18, 22, 22, 22, 22, 22, 18, 5],
    [10, 25, 28, 25, 22, 25, 28, 25, 10],
    [5, 15, 20, 20, 15, 20, 20, 15, 5]
];

static KING_PST: [[i16; 9]; 9] = [
    [-100, -100, -100, -100, -100, -100, -100, -100, -100],
    [-80, -80, -80, -80, -80, -80, -80, -80, -80],
    [-60, -60, -60, -60, -60, -60, -60, -60, -60],
    [-50, -50, -50, -50, -50, -50, -50, -50, -50],
    [-40, -40, -40, -40, -40, -40, -40, -40, -40],
    [-30, -30, -30, -30, -30, -30, -30, -30, -30],
    [-15, 0, 8, 12, -10, 12, 8, 0, -15],
    [15, 35, 40, 25, -20, 25, 40, 35, 15], // 囲い位置 (7八, 8八, 2八, 3八)
    [10, 30, 35, 15, -40, 15, 35, 30, 10], // 5九居玉は-40点の大幅ペナルティ
];

impl Evaluator {
    /// 局面の静的評価値を計算 (手番側視点のスコア)
    #[inline(always)]
    pub fn evaluate(pos: &Position) -> i32 {
        Self::evaluate_detailed(pos).total
    }

    /// 局面の静的評価値および特徴量内訳を計算 (手番側視点)
    pub fn evaluate_detailed(pos: &Position) -> EvalBreakdown {
        // 先手視点(Black)と後手視点(White)の各特徴量を集計
        let mut b_mat_board = 0;
        let mut w_mat_board = 0;
        let mut b_pst = 0;
        let mut w_pst = 0;
        let mut b_mob = 0;
        let mut w_mob = 0;
        let mut b_coord = 0;
        let mut w_coord = 0;
        let mut b_outpost = 0;
        let mut w_outpost = 0;

        // 各筋(file 0..9)の歩の存在フラグを事前集計 (オープン筋・セミオープン筋判定用)
        let mut b_pawns_in_file = [false; 9];
        let mut w_pawns_in_file = [false; 9];
        for sq_idx in 0..81 {
            if let Some(piece) = pos.board[sq_idx]
                && piece.piece_type == PieceType::Pawn
            {
                let file = Square::from_index(sq_idx).file() as usize;
                if piece.color == Color::Black {
                    b_pawns_in_file[file] = true;
                } else {
                    w_pawns_in_file[file] = true;
                }
            }
        }

        // 1. 盤上の駒得およびPST、大駒モビリティ、駒の連携、拠点
        for sq_idx in 0..81 {
            if let Some(piece) = pos.board[sq_idx] {
                let sq = Square::from_index(sq_idx);
                let pt = piece.piece_type;
                let c = piece.color;
                let val = pt.base_value();

                let pst_bonus = Self::pst_value(pt, sq, c);
                let mobility =
                    Self::piece_mobility(pos, sq, pt, c, &b_pawns_in_file, &w_pawns_in_file);
                let coordination = Self::piece_coordination(pos, sq, pt, c);
                let outpost = Self::outpost_bonus(pos, sq, pt, c);

                if c == Color::Black {
                    b_mat_board += val;
                    b_pst += pst_bonus;
                    b_mob += mobility;
                    b_coord += coordination;
                    b_outpost += outpost;
                } else {
                    w_mat_board += val;
                    w_pst += pst_bonus;
                    w_mob += mobility;
                    w_coord += coordination;
                    w_outpost += outpost;
                }
            }
        }

        // 2. 持ち駒の評価 (先後それぞれ独立計算)
        let (b_mat_hand, w_mat_hand) = Self::evaluate_hands(pos);

        // 3. 玉の安全度・囲いおよび危険度
        let b_safety = Self::king_safety_and_castle(pos, Color::Black);
        let w_safety = Self::king_safety_and_castle(pos, Color::White);
        let b_danger = Self::king_danger_and_flight(pos, Color::Black);
        let w_danger = Self::king_danger_and_flight(pos, Color::White);

        // 先手視点(Black)での差分スコア
        let diff_mat_board = b_mat_board - w_mat_board;
        let diff_mat_hand = b_mat_hand - w_mat_hand;
        let diff_pst = b_pst - w_pst;
        let diff_safety = b_safety - w_safety;
        let diff_danger = b_danger - w_danger;
        let diff_mob = b_mob - w_mob;
        let diff_coord = b_coord - w_coord;
        let diff_outpost = b_outpost - w_outpost;

        let s = pos.side_to_move.sign();
        let tempo = TEMPO_BONUS;

        let total = (diff_mat_board
            + diff_mat_hand
            + diff_pst
            + diff_safety
            + diff_danger
            + diff_mob
            + diff_coord
            + diff_outpost)
            * s
            + tempo;

        EvalBreakdown {
            material_board: diff_mat_board * s,
            material_hand: diff_mat_hand * s,
            piece_square: diff_pst * s,
            king_safety: diff_safety * s,
            king_danger: diff_danger * s,
            mobility: diff_mob * s,
            coordination: diff_coord * s,
            outposts: diff_outpost * s,
            tempo,
            total,
        }
    }

    /// 持ち駒の評価（価値逓減 Diminishing Returns ＆ 歩切れ評価）
    fn evaluate_hands(pos: &Position) -> (i32, i32) {
        const PAWN_HAND_IDX: usize = 0;
        let mut b_score = 0;
        let mut w_score = 0;

        let b_pawn_count = pos.hand[Color::Black.index()][PAWN_HAND_IDX];
        let w_pawn_count = pos.hand[Color::White.index()][PAWN_HAND_IDX];

        // 歩切れペナルティ / 歩保持ボーナス
        if b_pawn_count == 0 {
            b_score -= 20;
        } else {
            b_score += 15;
        }
        if w_pawn_count == 0 {
            w_score -= 20;
        } else {
            w_score += 15;
        }

        // 各持駒の枚数に応じた価値逓減計算
        for (h_idx, &pt) in PieceType::HAND_PIECES.iter().enumerate() {
            let b_count = pos.hand[Color::Black.index()][h_idx] as usize;
            let w_count = pos.hand[Color::White.index()][h_idx] as usize;
            let base_val = HAND_VAL[h_idx];

            match pt {
                PieceType::Pawn => {
                    // 歩: 1〜2枚目は130点、3〜4枚目は110点、5枚目以降は80点
                    b_score += Self::diminishing_pawn_value(b_count);
                    w_score += Self::diminishing_pawn_value(w_count);
                }
                PieceType::Silver | PieceType::Gold => {
                    // 金銀: 1〜2枚目は標準、3枚目以降は盤上の駒が減るため少し逓減(-30点)
                    b_score += Self::diminishing_general_value(b_count, base_val);
                    w_score += Self::diminishing_general_value(w_count, base_val);
                }
                _ => {
                    b_score += (b_count as i32) * base_val;
                    w_score += (w_count as i32) * base_val;
                }
            }
        }

        (b_score, w_score)
    }

    /// 歩の枚数に応じた価値逓減
    fn diminishing_pawn_value(count: usize) -> i32 {
        match count {
            0 => 0,
            1 => 130,
            2 => 260,
            3 => 370,
            4 => 480,
            _ => 480 + (count as i32 - 4) * 80,
        }
    }

    /// 金・銀の枚数に応じた価値逓減
    fn diminishing_general_value(count: usize, base_val: i32) -> i32 {
        if count <= 2 {
            (count as i32) * base_val
        } else {
            2 * base_val + (count as i32 - 2) * (base_val - 30)
        }
    }

    /// 仮想的な玉の危険度評価（相手の持ち駒圧力、玉頭の直通、退路数）
    fn king_danger_and_flight(pos: &Position, color: Color) -> i32 {
        let ksq = match pos.king_sq[color.index()] {
            Some(sq) => sq,
            None => return 0,
        };
        let opp = color.opposite();
        let opp_idx = opp.index();
        let fwd_dr = color.forward_dir();
        let mut danger_score = 0;

        // 1. 玉頭の圧力・危険度チェック (自玉自身の利きは除外)
        if let Some(head_sq) = ksq.offset(0, fwd_dr) {
            if pos.attacks_to(head_sq, opp) {
                danger_score -= 35; // 玉頭が狙われている危険
            }
            if pos.attacks_to_without_king(head_sq, color) {
                danger_score += 15; // 自玉頭を味方の駒がしっかり守っている
            }
        }

        // 2. 相手の手駒に応じた仮想王手・近接打ち込み圧力 (Virtual King Danger)
        let gold_idx = PieceType::Gold.hand_index().unwrap_or(4);
        let silver_idx = PieceType::Silver.hand_index().unwrap_or(3);
        let knight_idx = PieceType::Knight.hand_index().unwrap_or(2);
        let opp_golds = pos.hand[opp_idx][gold_idx] as i32;
        let opp_silvers = pos.hand[opp_idx][silver_idx] as i32;
        let opp_knights = pos.hand[opp_idx][knight_idx] as i32;
        let major_drop_power = opp_golds * 2 + opp_silvers + opp_knights;

        let mut dangerous_drop_squares = 0;
        let mut flight_squares = 0;

        for &(df, dr) in &KING_DIRS {
            if let Some(adj) = ksq.offset(df, dr) {
                let has_opp_attack = pos.attacks_to(adj, opp);
                let is_empty = pos.board[adj.index()].is_none();

                // 相手の持ち駒が豊富で、空きマスまたは敵の利きがあるマスは危険地帯
                if (is_empty || has_opp_attack) && major_drop_power > 0 {
                    dangerous_drop_squares += 1;
                }

                // 玉の前方マスへの安全な退路 (上部脱出)
                if dr == fwd_dr && is_empty && !has_opp_attack {
                    flight_squares += 1;
                }
            }
        }

        // 相手持駒が豊富で玉の周りがガラ空き・敵利きだらけのときのペナルティ
        danger_score -= dangerous_drop_squares * major_drop_power * 3;

        // 3. 上部脱出（Flight Squares）の評価
        if flight_squares >= 2 {
            danger_score += 25; // 上部脱出可能な好形（入玉志向・粘り）
        } else if flight_squares == 0 {
            danger_score -= 20; // 前方の退路がない閉塞状態
        }

        danger_score
    }

    /// 拠点の歩・と金（Outpost Pawns）の質的評価
    fn outpost_bonus(pos: &Position, sq: Square, pt: PieceType, color: Color) -> i32 {
        let rank = if color == Color::Black {
            sq.rank()
        } else {
            8 - sq.rank()
        };

        // 敵陣 (1〜3段目) の歩およびと金
        if rank <= 2 {
            let is_supported = pos.attacks_to_without_king(sq, color);
            match pt {
                PieceType::Pawn => {
                    if is_supported {
                        25 // 味方の駒が後ろから支えている敵陣のクサビの歩
                    } else {
                        0
                    }
                }
                PieceType::ProPawn => {
                    let mut bonus = 20;
                    if is_supported {
                        bonus += 20; // 支えのあると金
                    }
                    // 相手玉の近傍（3x3）に食い込んでいると金はさらに大加点
                    if let Some(opp_ksq) = pos.king_sq[color.opposite().index()] {
                        let df = (sq.file() as i8 - opp_ksq.file() as i8).abs();
                        let dr = (sq.rank() as i8 - opp_ksq.rank() as i8).abs();
                        if df <= 1 && dr <= 1 {
                            bonus += 35; // 相手玉の喉元に張り付くと金
                        }
                    }
                    bonus
                }
                _ => 0,
            }
        } else {
            0
        }
    }

    /// 大駒（角・飛車）の利き・モビリティ評価 & オープン筋ボーナス
    #[inline(always)]
    fn piece_mobility(
        pos: &Position,
        sq: Square,
        pt: PieceType,
        color: Color,
        b_pawns_in_file: &[bool; 9],
        w_pawns_in_file: &[bool; 9],
    ) -> i32 {
        match pt {
            PieceType::Bishop | PieceType::Horse => {
                // 斜め4方向のオープンマス数をカウント (角道開放の重視)
                let mut open_count = 0;
                let f = sq.file() as i8;
                let r = sq.rank() as i8;
                for (df, dr) in DIAGONAL_DIRS {
                    let mut cf = f + df;
                    let mut cr = r + dr;
                    while (0..9).contains(&cf) && (0..9).contains(&cr) {
                        let target_sq = Square::new(cf as u8, cr as u8);
                        if pos.board[target_sq.index()].is_some() {
                            break;
                        }
                        open_count += 1;
                        cf += df;
                        cr += dr;
                    }
                }
                open_count * 5 // 1マス通るごとに+5点
            }
            PieceType::Rook | PieceType::Dragon => {
                let mut score = 0;
                let file = sq.file() as usize;

                // オープン筋・セミオープン筋支配ボーナス
                let own_pawn = if color == Color::Black {
                    b_pawns_in_file[file]
                } else {
                    w_pawns_in_file[file]
                };
                let opp_pawn = if color == Color::Black {
                    w_pawns_in_file[file]
                } else {
                    b_pawns_in_file[file]
                };

                if !own_pawn {
                    if !opp_pawn {
                        score += 35; // 完全なオープン筋（自軍・敵軍ともに歩なし）
                    } else {
                        score += 20; // セミオープン筋（自軍の歩がない直通筋）
                    }
                }

                // 飛車の前方オープンマス数 (先手は上方向、後手は下方向)
                let fwd_dr = color.forward_dir();
                let mut fwd_open = 0;
                let f = sq.file();
                let mut cr = sq.rank() as i8 + fwd_dr;
                while (0..9).contains(&cr) {
                    let target_sq = Square::new(f, cr as u8);
                    if pos.board[target_sq.index()].is_some() {
                        break;
                    }
                    fwd_open += 1;
                    cr += fwd_dr;
                }
                score += fwd_open * 7; // 前方に直通しているマスごとに+7点

                // 飛車・龍の横利き評価 (左右のオープンマス数をカウント)
                let r_rank = sq.rank();
                let mut horiz_open = 0;
                for step_f in &[-1i8, 1i8] {
                    let mut cf = sq.file() as i8 + step_f;
                    while (0..9).contains(&cf) {
                        let target_sq = Square::new(cf as u8, r_rank);
                        if pos.board[target_sq.index()].is_some() {
                            break;
                        }
                        horiz_open += 1;
                        cf += step_f;
                    }
                }
                // 敵陣での横利き、または龍の横利きは加点を強化
                let is_in_enemy_territory = if color == Color::Black {
                    r_rank <= 2
                } else {
                    r_rank >= 6
                };
                let horiz_mult = if is_in_enemy_territory || pt == PieceType::Dragon {
                    6
                } else {
                    3
                };
                score += horiz_open * horiz_mult;

                score
            }
            PieceType::Lance => {
                // 香車の前方オープンマス数 (先手は上方向、後手は下方向)
                let fwd_dr = color.forward_dir();
                let mut fwd_open = 0;
                let f = sq.file();
                let mut cr = sq.rank() as i8 + fwd_dr;
                while (0..9).contains(&cr) {
                    let target_sq = Square::new(f, cr as u8);
                    if pos.board[target_sq.index()].is_some() {
                        break;
                    }
                    fwd_open += 1;
                    cr += fwd_dr;
                }
                fwd_open * 4 // 前方の直通マスごとに+4点
            }
            PieceType::Knight => {
                // 桂馬の高跳び歩の餌食ペナルティ（敵の歩に睨まれる5段目・4段目の危険度）
                let rank = if color == Color::Black {
                    sq.rank()
                } else {
                    8 - sq.rank()
                };
                if rank <= 4 { -15 } else { 10 }
            }
            _ => 0,
        }
    }

    /// 駒の連携（味方の紐付き保護 & 浮き駒ペナルティ）
    #[inline(always)]
    fn piece_coordination(pos: &Position, sq: Square, pt: PieceType, color: Color) -> i32 {
        let opp = color.opposite();

        // 浮き駒判定: 王以外の駒で、味方の利きによる支え（紐）がない場合
        if pt != PieceType::King {
            let is_defended = pos.attacks_to_without_king(sq, color);
            let is_attacked = pos.attacks_to(sq, opp);

            if is_attacked && !is_defended {
                // 敵に狙われているのに紐がついていない致命的な浮き駒 (Hanging Piece)
                let penalty = match pt {
                    PieceType::Rook | PieceType::Dragon => -45,
                    PieceType::Bishop | PieceType::Horse => -40,
                    PieceType::Gold | PieceType::Silver => -30,
                    _ => -15,
                };
                return penalty;
            } else if is_defended {
                // 味方の利きで守られている（紐付き保護）
                return 15;
            }
        }

        // 金銀スクラム連携ボーナス（味方金銀の近接守り合い）
        match pt {
            PieceType::Gold | PieceType::Silver => {
                let back_dr = -color.forward_dir();
                let mut connected = false;

                // 後方3方向 (真後ろ、斜め後ろ) または真横に味方の金・銀・歩がいるか
                for &(df, dr) in &[(0, back_dr), (-1, back_dr), (1, back_dr), (-1, 0), (1, 0)] {
                    if let Some(adj) = sq.offset(df, dr)
                        && let Some(p) = pos.board[adj.index()]
                        && p.color == color
                    {
                        match p.piece_type {
                            PieceType::Gold
                            | PieceType::Silver
                            | PieceType::Pawn
                            | PieceType::ProPawn
                            | PieceType::ProSilver => {
                                connected = true;
                                break;
                            }
                            _ => {}
                        }
                    }
                }

                if connected {
                    15 // 金銀スクラム連携ボーナス
                } else {
                    -10 // 孤立ペナルティ
                }
            }
            PieceType::Pawn => {
                // 歩の後ろに味方の金・銀・飛・香があるか (歩の支え)
                let back_dr = -color.forward_dir();
                if let Some(adj) = sq.offset(0, back_dr)
                    && let Some(p) = pos.board[adj.index()]
                    && p.color == color
                {
                    return 10;
                }
                0
            }
            _ => 0,
        }
    }

    /// PSTボーナス計算
    #[inline(always)]
    fn pst_value(pt: PieceType, sq: Square, color: Color) -> i32 {
        let (file, rank) = if color == Color::Black {
            (sq.file() as usize, sq.rank() as usize)
        } else {
            (8 - sq.file() as usize, 8 - sq.rank() as usize)
        };

        match pt {
            PieceType::Pawn => PAWN_PST[rank][file] as i32,
            PieceType::Silver => SILVER_PST[rank][file] as i32,
            PieceType::Gold
            | PieceType::ProPawn
            | PieceType::ProLance
            | PieceType::ProKnight
            | PieceType::ProSilver => GOLD_PST[rank][file] as i32,
            PieceType::King => KING_PST[rank][file] as i32,
            PieceType::Rook => {
                if rank <= 2 {
                    35
                } else {
                    10
                }
            }
            PieceType::Dragon => 45,
            PieceType::Bishop => {
                let center_dist = (file as i32 - 4).abs() + (rank as i32 - 4).abs();
                (8 - center_dist) * 5
            }
            PieceType::Horse => 40,
            _ => 0,
        }
    }

    /// 玉の周囲の守りおよび囲い完成ボーナス
    fn king_safety_and_castle(pos: &Position, color: Color) -> i32 {
        let ksq = match pos.king_sq[color.index()] {
            Some(sq) => sq,
            None => return 0,
        };

        let mut safety = 0;

        // 1. 玉周囲の守備駒スコア
        for &(df, dr) in &KING_DIRS {
            if let Some(adj_sq) = ksq.offset(df, dr)
                && let Some(p) = pos.board[adj_sq.index()]
                && p.color == color
            {
                match p.piece_type {
                    PieceType::Gold | PieceType::ProPawn | PieceType::ProSilver => {
                        safety += 25;
                    }
                    PieceType::Silver => {
                        safety += 20;
                    }
                    PieceType::Knight | PieceType::Lance | PieceType::Pawn => {
                        safety += 10;
                    }
                    _ => {}
                }
            }
        }

        // 2. 囲いの完成ボーナス (美濃、矢倉、穴熊、舟囲い、雁木等の好形)
        let (f, r) = if color == Color::Black {
            (ksq.file(), ksq.rank())
        } else {
            (8 - ksq.file(), 8 - ksq.rank())
        };

        // 穴熊 (玉が9九=file 8, rank 8 に深く潜り、8八や7八に金銀がいる堅陣)
        if f >= 7 && r >= 7 {
            let mut anaguma_guards = 0;
            for &(gf, gr) in &[(f - 1, r), (f, r - 1), (f - 1, r - 1)] {
                let actual_sq = if color == Color::Black {
                    Square::new(gf, gr)
                } else {
                    Square::new(8 - gf, 8 - gr)
                };
                if let Some(p) = pos.board[actual_sq.index()]
                    && p.color == color
                    && (p.piece_type == PieceType::Gold
                        || p.piece_type == PieceType::Silver
                        || p.piece_type == PieceType::Knight
                        || p.piece_type == PieceType::Lance)
                {
                    anaguma_guards += 1;
                }
            }
            if anaguma_guards >= 2 {
                safety += 70; // 穴熊の堅陣ボーナス
            }
        }

        // 美濃囲い・矢倉系 (玉が7八/8八にいて金銀が連携)
        if (f == 6 || f == 7) && r >= 6 {
            let sq_guard = if color == Color::Black {
                Square::new(6, 7) // 7八
            } else {
                Square::new(2, 1) // 3二
            };
            if let Some(p) = pos.board[sq_guard.index()]
                && p.color == color
                && (p.piece_type == PieceType::Silver || p.piece_type == PieceType::Gold)
            {
                safety += 45; // 美濃・矢倉ボーナス
            }
        }

        // 舟囲い・雁木系 (玉が6八/7九にいて前線に銀・金が立つ)
        if (f == 5 || f == 6) && (r == 7 || r == 8) {
            safety += 20;
        }

        // 居玉ペナルティ (中央5九/5一の無防備な玉)
        // 設計レビュー 7.3: TTキャッシュのキー不整合を防ぐため pos.ply 依存を排除
        if f == 4 && r == 8 {
            safety -= 35;
        }

        safety
    }
}
