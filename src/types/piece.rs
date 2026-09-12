use super::color::Color;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PieceType {
    Pawn,      // 歩 (p)
    Lance,     // 香 (l)
    Knight,    // 桂 (n)
    Silver,    // 銀 (s)
    Gold,      // 金 (g)
    Bishop,    // 角 (b)
    Rook,      // 飛 (r)
    King,      // 玉 (k)
    ProPawn,   // と (+p)
    ProLance,  // 成香 (+l)
    ProKnight, // 成桂 (+n)
    ProSilver, // 成銀 (+s)
    Horse,     // 馬 (+b)
    Dragon,    // 竜 (+r)
}

impl PieceType {
    pub const ALL: [PieceType; 14] = [
        PieceType::Pawn,
        PieceType::Lance,
        PieceType::Knight,
        PieceType::Silver,
        PieceType::Gold,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::King,
        PieceType::ProPawn,
        PieceType::ProLance,
        PieceType::ProKnight,
        PieceType::ProSilver,
        PieceType::Horse,
        PieceType::Dragon,
    ];

    pub const HAND_PIECES: [PieceType; 7] = [
        PieceType::Pawn,
        PieceType::Lance,
        PieceType::Knight,
        PieceType::Silver,
        PieceType::Gold,
        PieceType::Bishop,
        PieceType::Rook,
    ];

    #[inline(always)]
    pub fn index(self) -> usize {
        match self {
            PieceType::Pawn => 0,
            PieceType::Lance => 1,
            PieceType::Knight => 2,
            PieceType::Silver => 3,
            PieceType::Gold => 4,
            PieceType::Bishop => 5,
            PieceType::Rook => 6,
            PieceType::King => 7,
            PieceType::ProPawn => 8,
            PieceType::ProLance => 9,
            PieceType::ProKnight => 10,
            PieceType::ProSilver => 11,
            PieceType::Horse => 12,
            PieceType::Dragon => 13,
        }
    }

    #[inline(always)]
    pub fn hand_index(self) -> Option<usize> {
        match self {
            PieceType::Pawn => Some(0),
            PieceType::Lance => Some(1),
            PieceType::Knight => Some(2),
            PieceType::Silver => Some(3),
            PieceType::Gold => Some(4),
            PieceType::Bishop => Some(5),
            PieceType::Rook => Some(6),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn from_hand_index(idx: usize) -> Option<PieceType> {
        match idx {
            0 => Some(PieceType::Pawn),
            1 => Some(PieceType::Lance),
            2 => Some(PieceType::Knight),
            3 => Some(PieceType::Silver),
            4 => Some(PieceType::Gold),
            5 => Some(PieceType::Bishop),
            6 => Some(PieceType::Rook),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn is_promoted(self) -> bool {
        matches!(
            self,
            PieceType::ProPawn
                | PieceType::ProLance
                | PieceType::ProKnight
                | PieceType::ProSilver
                | PieceType::Horse
                | PieceType::Dragon
        )
    }

    #[inline(always)]
    pub fn can_promote(self) -> bool {
        matches!(
            self,
            PieceType::Pawn
                | PieceType::Lance
                | PieceType::Knight
                | PieceType::Silver
                | PieceType::Bishop
                | PieceType::Rook
        )
    }

    #[inline(always)]
    pub fn promote(self) -> Option<PieceType> {
        match self {
            PieceType::Pawn => Some(PieceType::ProPawn),
            PieceType::Lance => Some(PieceType::ProLance),
            PieceType::Knight => Some(PieceType::ProKnight),
            PieceType::Silver => Some(PieceType::ProSilver),
            PieceType::Bishop => Some(PieceType::Horse),
            PieceType::Rook => Some(PieceType::Dragon),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn unpromote(self) -> PieceType {
        match self {
            PieceType::ProPawn => PieceType::Pawn,
            PieceType::ProLance => PieceType::Lance,
            PieceType::ProKnight => PieceType::Knight,
            PieceType::ProSilver => PieceType::Silver,
            PieceType::Horse => PieceType::Bishop,
            PieceType::Dragon => PieceType::Rook,
            other => other,
        }
    }

    /// 基本的な駒の価値 (センチポーン単位, 歩=100)
    #[inline(always)]
    pub fn base_value(self) -> i32 {
        match self {
            PieceType::Pawn => 100,
            PieceType::Lance => 320,
            PieceType::Knight => 350,
            PieceType::Silver => 500,
            PieceType::Gold => 550,
            PieceType::Bishop => 850,
            PieceType::Rook => 1000,
            PieceType::King => 20000,
            PieceType::ProPawn => 530,
            PieceType::ProLance => 530,
            PieceType::ProKnight => 530,
            PieceType::ProSilver => 550,
            PieceType::Horse => 1150,
            PieceType::Dragon => 1300,
        }
    }

    pub fn to_usi_char(self) -> char {
        match self.unpromote() {
            PieceType::Pawn => 'P',
            PieceType::Lance => 'L',
            PieceType::Knight => 'N',
            PieceType::Silver => 'S',
            PieceType::Gold => 'G',
            PieceType::Bishop => 'B',
            PieceType::Rook => 'R',
            PieceType::King => 'K',
            _ => '?',
        }
    }

    pub fn from_sfen_char(c: char) -> Option<(PieceType, Color)> {
        let color = if c.is_uppercase() {
            Color::Black
        } else {
            Color::White
        };
        let pt = match c.to_ascii_uppercase() {
            'P' => PieceType::Pawn,
            'L' => PieceType::Lance,
            'N' => PieceType::Knight,
            'S' => PieceType::Silver,
            'G' => PieceType::Gold,
            'B' => PieceType::Bishop,
            'R' => PieceType::Rook,
            'K' => PieceType::King,
            _ => return None,
        };
        Some((pt, color))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Piece {
    pub piece_type: PieceType,
    pub color: Color,
}

impl Piece {
    #[inline(always)]
    pub fn new(piece_type: PieceType, color: Color) -> Self {
        Piece { piece_type, color }
    }

    pub fn to_sfen(self) -> String {
        let is_promoted = self.piece_type.is_promoted();
        let unpromoted = self.piece_type.unpromote();
        let mut c = match unpromoted {
            PieceType::Pawn => 'p',
            PieceType::Lance => 'l',
            PieceType::Knight => 'n',
            PieceType::Silver => 's',
            PieceType::Gold => 'g',
            PieceType::Bishop => 'b',
            PieceType::Rook => 'r',
            PieceType::King => 'k',
            _ => '?',
        };
        if self.color == Color::Black {
            c = c.to_ascii_uppercase();
        }
        if is_promoted {
            format!("+{c}")
        } else {
            c.to_string()
        }
    }
}
