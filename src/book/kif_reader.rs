use crate::board::Position;
use crate::movegen::MoveGenerator;
use crate::types::{Move, PieceType, Square};
use std::collections::HashMap;

/// KIF形式（変化記法を含む）の棋譜ツリー読込器
#[derive(Debug, Default, Clone)]
pub struct KifBook {
    /// 局面ハッシュ -> (指し手, 採用重み) のリスト
    pub book_moves: HashMap<u64, Vec<(Move, u32)>>,
}

impl KifBook {
    pub fn new() -> Self {
        Self {
            book_moves: HashMap::new(),
        }
    }

    /// KIFファイルから定跡ツリーを読み込み（UTF-8 および Shift-JIS/CP932 自動判別）
    pub fn load_from_file(path: &str) -> Result<Self, String> {
        let bytes =
            std::fs::read(path).map_err(|e| format!("Failed to read KIF file '{path}': {e}"))?;
        let content = if let Ok(utf8_str) = std::str::from_utf8(&bytes) {
            utf8_str.to_string()
        } else {
            let (cow, _, _) = encoding_rs::SHIFT_JIS.decode(&bytes);
            cow.into_owned()
        };
        Self::from_kif_string(&content)
    }

    /// KIF文字列から定跡ツリーを構築
    pub fn from_kif_string(content: &str) -> Result<Self, String> {
        let mut book = Self::new();
        // history[n] はアクティブ経路上の n 手進めた局面
        let mut history = vec![Position::startpos()];
        let mut pos = history[0].clone();
        let mut ended = false;

        let mut is_mainline = true;

        for (index, raw) in content.lines().enumerate() {
            let line_no = index + 1;
            let line = raw.trim();

            if let Some(branch) = line.strip_prefix("変化：") {
                is_mainline = false;
                let ply = branch
                    .strip_suffix('手')
                    .and_then(|s| s.trim().parse::<usize>().ok())
                    .ok_or_else(|| format!("Line {line_no}: invalid branch '{line}'"))?;

                if ply == 0 || ply > history.len() {
                    return Err(format!(
                        "Line {line_no}: branch ply {ply} exceeds active history length {}",
                        history.len()
                    ));
                }

                history.truncate(ply);
                pos = history[ply - 1].clone();
                ended = false;
                continue;
            }

            if let Some(handicap) = line.strip_prefix("手合割：")
                && handicap.trim() != "平手"
            {
                return Err(format!("Line {line_no}: only 平手 is supported"));
            }

            if line.is_empty()
                || line.starts_with(['#', '*', '&'])
                || line.contains('：')
                || line.starts_with("手数")
            {
                continue;
            }

            if line.starts_with("まで") {
                ended = true;
                continue;
            }

            let (number, body) = match line.split_once(char::is_whitespace) {
                Some(pair) => pair,
                None => continue,
            };
            let ply = match number.parse::<usize>() {
                Ok(p) => p,
                Err(_) => continue,
            };

            if ended || ply != history.len() {
                return Err(format!(
                    "Line {line_no}: unexpected ply {ply}; expected {} in an active section",
                    history.len()
                ));
            }

            if matches!(
                body.split_whitespace().next(),
                Some(
                    "中断"
                        | "投了"
                        | "持将棋"
                        | "千日手"
                        | "切れ負け"
                        | "反則勝ち"
                        | "反則負け"
                        | "入玉勝ち"
                        | "不戦勝"
                        | "不戦敗"
                        | "詰み"
                        | "不詰"
                )
            ) {
                ended = true;
                continue;
            }

            let last_to = pos.history.last().map(|record| record.mv.to());
            let mv = parse_kif_move_line(line, &mut pos, last_to).ok_or_else(|| {
                format!("Line {line_no}, ply {ply}: invalid or illegal move '{line}'")
            })?;

            let weight = if is_mainline { 200 } else { 100 };
            let entries = book.book_moves.entry(pos.hash).or_default();
            if let Some(existing) = entries.iter_mut().find(|(m, _)| *m == mv) {
                if weight > existing.1 {
                    existing.1 = weight;
                }
            } else {
                entries.push((mv, weight));
            }

            pos.do_move(mv);
            history.push(pos.clone());
        }

        Ok(book)
    }

    /// 現在の局面の定跡手一覧を取得
    pub fn probe_moves(&self, pos: &Position) -> Option<&[(Move, u32)]> {
        self.book_moves.get(&pos.hash).map(|v| v.as_slice())
    }

    /// 最も重みの高い定跡手を取得
    pub fn probe_best(&self, pos: &Position) -> Option<Move> {
        let moves = self.probe_moves(pos)?;
        moves.iter().max_by_key(|(_, w)| *w).map(|(m, _)| *m)
    }

    /// 確率サンプリング（重みに応じてランダム選択）
    pub fn probe_sample(&self, pos: &Position, rng_val: u32) -> Option<Move> {
        let moves = self.probe_moves(pos)?;
        if moves.is_empty() {
            return None;
        }
        let total_weight: u32 = moves.iter().map(|(_, w)| *w).sum();
        if total_weight == 0 {
            return Some(moves[0].0);
        }

        let mut roll = rng_val % total_weight;
        for (mv, weight) in moves {
            if roll < *weight {
                return Some(*mv);
            }
            roll -= *weight;
        }
        Some(moves[0].0)
    }
}

/// 単一のKIF指し手行をパース
fn parse_kif_move_line(line: &str, pos: &mut Position, last_to: Option<Square>) -> Option<Move> {
    let (number, body) = line.trim().split_once(char::is_whitespace)?;
    number.parse::<usize>().ok()?;
    let body = body.trim_start();

    let (to, rest) = if let Some(rest) = body.strip_prefix('同') {
        (last_to?, rest.trim_start())
    } else {
        let mut chars = body.chars();
        let file = parse_zenkaku_digit(chars.next()?)?;
        let rank = parse_kanji_digit(chars.next()?)?;
        (Square::new(file, rank), chars.as_str())
    };

    let (piece_type, rest) = parse_piece_type(rest)?;
    let candidate = if let Some(tail) = rest.strip_prefix('打') {
        piece_type.hand_index()?;

        let tail = tail.trim().trim_end_matches(['+', '*', '!', '?']).trim();
        if !tail.is_empty() && !(tail.starts_with('(') && tail.ends_with(')') && tail.contains(':'))
        {
            return None;
        }

        Move::drop(to, piece_type)
    } else {
        let (promote, rest) = if let Some(rest) = rest.strip_prefix("不成") {
            (false, rest)
        } else if let Some(rest) = rest.strip_prefix('成') {
            (true, rest)
        } else {
            (false, rest)
        };

        // 盤上の駒の移動は必ず (from) 移動元座標を検証
        let (origin, tail) = rest.trim_start().strip_prefix('(')?.split_once(')')?;
        let bytes = origin.as_bytes();

        if bytes.len() != 2 || !bytes.iter().all(|b| (b'1'..=b'9').contains(b)) {
            return None;
        }

        let tail = tail.trim().trim_end_matches(['+', '*', '!', '?']).trim();
        if !tail.is_empty() && !(tail.starts_with('(') && tail.ends_with(')') && tail.contains(':'))
        {
            return None;
        }

        let from = Square::new(bytes[0] - b'1', bytes[1] - b'1');
        let piece = pos.board[from.index()]?;
        if piece.color != pos.side_to_move || piece.piece_type != piece_type {
            return None;
        }

        Move::normal(from, to, promote)
    };

    let legal = MoveGenerator::generate_legal_moves(pos);
    if legal.contains(&candidate) {
        Some(candidate)
    } else {
        None
    }
}

fn parse_piece_type(text: &str) -> Option<(PieceType, &str)> {
    use PieceType::*;

    for (name, piece_type) in [
        ("成銀", ProSilver),
        ("成桂", ProKnight),
        ("成香", ProLance),
        ("全", ProSilver),
        ("圭", ProKnight),
        ("杏", ProLance),
        ("と", ProPawn),
        ("馬", Horse),
        ("龍", Dragon),
        ("竜", Dragon),
        ("歩", Pawn),
        ("香", Lance),
        ("桂", Knight),
        ("銀", Silver),
        ("金", Gold),
        ("角", Bishop),
        ("飛", Rook),
        ("玉", King),
        ("王", King)
    ] {
        if let Some(rest) = text.strip_prefix(name) {
            return Some((piece_type, rest));
        }
    }

    None
}

fn parse_zenkaku_digit(c: char) -> Option<u8> {
    match c {
        '１' => Some(0),
        '２' => Some(1),
        '３' => Some(2),
        '４' => Some(3),
        '５' => Some(4),
        '６' => Some(5),
        '７' => Some(6),
        '８' => Some(7),
        '９' => Some(8),
        '1'..='9' => Some(c as u8 - b'1'),
        _ => None,
    }
}

fn parse_kanji_digit(c: char) -> Option<u8> {
    match c {
        '一' => Some(0),
        '二' => Some(1),
        '三' => Some(2),
        '四' => Some(3),
        '五' => Some(4),
        '六' => Some(5),
        '七' => Some(6),
        '八' => Some(7),
        '九' => Some(8),
        '1'..='9' => Some(c as u8 - b'1'),
        _ => None,
    }
}
