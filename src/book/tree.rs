use crate::board::Position;
use crate::types::Move;

pub struct OpeningBook;

/// 定跡適用最大手数（26手以降は深層探索へ）
pub const OPENING_BOOK_MAX_PLY: usize = 26;

impl OpeningBook {
    /// 現在の局面または手数から定跡手を取得 (定跡にヒットすればSome(Move))
    pub fn probe(pos: &Position) -> Option<Move> {
        if pos.ply > OPENING_BOOK_MAX_PLY {
            return None;
        }

        // 過去の手順をUSI文字列リストとして抽出
        let history_str: Vec<String> = pos.history.iter().map(|rec| rec.mv.to_usi()).collect();
        let path = history_str.join(" ");

        let book_move_str = match path.as_str() {
            // ==========================================
            // 初手 (平手)
            // ==========================================
            "" => Some("7g7f"),

            // ==========================================
            // 1手目 7g7f に対する応手
            // ==========================================
            "7g7f" => Some("3c3d"),
            "2g2f" => Some("8c8d"),

            // ==========================================
            // 2手進んだ局面
            // ==========================================
            "7g7f 3c3d" => Some("2g2f"),
            "7g7f 8c8d" => Some("2g2f"),
            "2g2f 8c8d" => Some("2f2e"),

            // ==========================================
            // 3手進んだ局面
            // ==========================================
            "7g7f 3c3d 2g2f" => Some("8c8d"),
            "7g7f 8c8d 2g2f" => Some("8d8e"),
            "2g2f 8c8d 2f2e" => Some("8d8e"),

            // ==========================================
            // 4手進んだ局面
            // ==========================================
            "7g7f 3c3d 2g2f 8c8d" => Some("2f2e"),
            "7g7f 8c8d 2g2f 8d8e" => Some("6i7h"),
            "2g2f 8c8d 2f2e 8d8e" => Some("6i7h"),

            // ==========================================
            // 5手進んだ局面
            // ==========================================
            "7g7f 3c3d 2g2f 8c8d 2f2e" => Some("8d8e"),
            "7g7f 8c8d 2g2f 8d8e 6i7h" => Some("4a3b"),
            "2g2f 8c8d 2f2e 8d8e 6i7h" => Some("4a3b"),

            // ==========================================
            // 6手進んだ局面
            // ==========================================
            "7g7f 3c3d 2g2f 8c8d 2f2e 8d8e" => Some("6i7h"),
            "7g7f 8c8d 2g2f 8d8e 6i7h 4a3b" => Some("2f2e"),
            "2g2f 8c8d 2f2e 8d8e 6i7h 4a3b" => Some("2e2d"),

            // ==========================================
            // 7手進んだ局面 (角換わり・相掛かり分岐)
            // ==========================================
            "7g7f 3c3d 2g2f 8c8d 2f2e 8d8e 6i7h" => Some("4a3b"),

            // ==========================================
            // 【角換わり】本線 (8手目〜20手目)
            // ==========================================
            "7g7f 3c3d 2g2f 8c8d 2f2e 8d8e 6i7h 4a3b" => Some("8h2b+"),
            "7g7f 3c3d 2g2f 8c8d 2f2e 8d8e 6i7h 4a3b 8h2b+" => Some("3a2b"),
            "7g7f 3c3d 2g2f 8c8d 2f2e 8d8e 6i7h 4a3b 8h2b+ 3a2b" => Some("7i8h"),
            "7g7f 3c3d 2g2f 8c8d 2f2e 8d8e 6i7h 4a3b 8h2b+ 3a2b 7i8h" => Some("5a4b"),
            "7g7f 3c3d 2g2f 8c8d 2f2e 8d8e 6i7h 4a3b 8h2b+ 3a2b 7i8h 5a4b" => Some("5i6i"),
            "7g7f 3c3d 2g2f 8c8d 2f2e 8d8e 6i7h 4a3b 8h2b+ 3a2b 7i8h 5a4b 5i6i" => Some("2b3c"),
            "7g7f 3c3d 2g2f 8c8d 2f2e 8d8e 6i7h 4a3b 8h2b+ 3a2b 7i8h 5a4b 5i6i 2b3c" => {
                Some("6i7i")
            }
            "7g7f 3c3d 2g2f 8c8d 2f2e 8d8e 6i7h 4a3b 8h2b+ 3a2b 7i8h 5a4b 5i6i 2b3c 6i7i" => {
                Some("6a7b")
            }
            "7g7f 3c3d 2g2f 8c8d 2f2e 8d8e 6i7h 4a3b 8h2b+ 3a2b 7i8h 5a4b 5i6i 2b3c 6i7i 6a7b" => {
                Some("3i4h")
            }

            // ==========================================
            // 【相掛かり】本線 (飛先交換)
            // ==========================================
            "2g2f 8c8d 2f2e 8d8e 6i7h 4a3b 2e2d" => Some("2c2d"),
            "2g2f 8c8d 2f2e 8d8e 6i7h 4a3b 2e2d 2c2d" => Some("2h2d"),

            // ==========================================
            // 【矢倉】本線 (角道を止めて矢倉戦)
            // ==========================================
            "7g7f 8c8d 7i6h" => Some("3c3d"),
            "7g7f 8c8d 7i6h 3c3d" => Some("6g6f"),
            "7g7f 8c8d 7i6h 3c3d 6g6f" => Some("7a6b"),

            // ==========================================
            // 【横歩取り】本線
            // ==========================================
            "7g7f 3c3d 2g2f 8c8d 2f2e 8d8e 2e2d" => Some("2c2d"),
            "7g7f 3c3d 2g2f 8c8d 2f2e 8d8e 2e2d 2c2d" => Some("2h2d"),
            "7g7f 3c3d 2g2f 8c8d 2f2e 8d8e 2e2d 2c2d 2h2d" => Some("8e8f"),
            "7g7f 3c3d 2g2f 8c8d 2f2e 8d8e 2e2d 2c2d 2h2d 8e8f" => Some("8g8f"),
            "7g7f 3c3d 2g2f 8c8d 2f2e 8d8e 2e2d 2c2d 2h2d 8e8f 8g8f" => Some("8b8f"),
            "7g7f 3c3d 2g2f 8c8d 2f2e 8d8e 2e2d 2c2d 2h2d 8e8f 8g8f 8b8f" => Some("2d3d"),

            // ==========================================
            // 【四間飛車・三間飛車・中飛車】(対抗形)
            // ==========================================
            // 四間飛車 (角道を止めて4筋へ振る: 4c4d -> 8b4b)
            "7g7f 3c3d 2g2f 4c4d" => Some("2f2e"),
            "7g7f 3c3d 2g2f 4c4d 2f2e" => Some("8b4b"),
            "7g7f 3c3d 2g2f 4c4d 2f2e 8b4b" => Some("6i7h"),

            // 三間飛車
            "7g7f 3c3d 2g2f 3a4b" => Some("2f2e"),

            // ゴキゲン中飛車 (後手 5c5d -> 8b5b)
            "7g7f 3c3d 2g2f 5c5d" => Some("2f2e"),
            "7g7f 3c3d 2g2f 5c5d 2f2e" => Some("8b5b"),
            "7g7f 3c3d 2g2f 5c5d 2f2e 8b5b" => Some("4i3h"),
            "7g7f 3c3d 2g2f 5c5d 2f2e 8b5b 4i3h" => Some("5a4b"),

            _ => None,
        };

        if let Some(mv_str) = book_move_str {
            Move::from_usi(mv_str)
        } else {
            None
        }
    }
}
