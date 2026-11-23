use crate::types::{Color, Piece, PieceType, Square};
use std::sync::OnceLock;

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        XorShift64 {
            state: if seed == 0 { 0x123456789ABCDEF0 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

pub struct ZobristKeys {
    // 81マス * 14駒種 * 2色
    pub board: [[[u64; 2]; 14]; 81],
    // 2色 * 7持ち駒種 * 19枚 (0..=18)
    pub hand: [[[u64; 19]; 7]; 2],
    // 後手番のときにXOR
    pub side_to_move: u64,
}

impl ZobristKeys {
    pub fn init() -> Self {
        let mut rng = XorShift64::new(0x2026_0906_1234_5678);
        let mut board = [[[0u64; 2]; 14]; 81];
        for slot in &mut board {
            for pt_slot in slot.iter_mut() {
                for c_slot in pt_slot.iter_mut() {
                    *c_slot = rng.next_u64();
                }
            }
        }

        let mut hand = [[[0u64; 19]; 7]; 2];
        for color_slot in &mut hand {
            for pt_slot in color_slot.iter_mut() {
                for count_slot in pt_slot.iter_mut() {
                    *count_slot = rng.next_u64();
                }
            }
        }

        let side_to_move = rng.next_u64();

        ZobristKeys {
            board,
            hand,
            side_to_move,
        }
    }

    #[inline(always)]
    pub fn piece_key(&self, sq: Square, piece: Piece) -> u64 {
        self.board[sq.index()][piece.piece_type.index()][piece.color.index()]
    }

    #[inline(always)]
    pub fn hand_key(&self, color: Color, pt: PieceType, count: usize) -> u64 {
        if let Some(idx) = pt.hand_index() {
            let count_clamped = count.min(18);
            self.hand[color.index()][idx][count_clamped]
        } else {
            0
        }
    }
}

static KEYS: OnceLock<ZobristKeys> = OnceLock::new();

pub fn get_zobrist_keys() -> &'static ZobristKeys {
    KEYS.get_or_init(ZobristKeys::init)
}
