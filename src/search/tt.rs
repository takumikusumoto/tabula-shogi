use crate::types::Move;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NodeType {
    Exact = 0,
    LowerBound = 1, // ベータカット発生 (score >= beta)
    UpperBound = 2, // 全探索でアルファ更新なし (score <= alpha)
}

impl NodeType {
    #[inline(always)]
    pub fn from_u8(val: u8) -> Self {
        match val {
            1 => NodeType::LowerBound,
            2 => NodeType::UpperBound,
            _ => NodeType::Exact,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TTEntry {
    pub hash: u64,
    pub depth: u8,
    pub score: i32,
    pub node_type: NodeType,
    pub best_move: Option<Move>,
}

/// Lock-free 64-bit packed entry
/// data: [ best_move(16) | score(32) | depth(8) | node_type(8) ]
struct AtomicTTEntry {
    hash: AtomicU64,
    data: AtomicU64,
}

impl AtomicTTEntry {
    pub fn new() -> Self {
        AtomicTTEntry {
            hash: AtomicU64::new(0),
            data: AtomicU64::new(0),
        }
    }
}

impl Default for AtomicTTEntry {
    fn default() -> Self {
        Self::new()
    }
}

pub struct TranspositionTable {
    entries: Vec<AtomicTTEntry>,
    mask: usize,
}

impl TranspositionTable {
    /// 指定のサイズ(MB)で共有置換表を確保
    pub fn new(size_mb: usize) -> Self {
        let entry_size = std::mem::size_of::<AtomicTTEntry>();
        let num_entries = (size_mb * 1024 * 1024) / entry_size;
        // 2の累乗以下の最大サイズ (要求サイズを超えない範囲で最大化)
        let capacity = if num_entries.is_power_of_two() {
            num_entries
        } else {
            (num_entries.next_power_of_two() / 2).max(1024)
        };

        let entries = (0..capacity).map(|_| AtomicTTEntry::default()).collect();

        TranspositionTable {
            entries,
            mask: capacity - 1,
        }
    }

    #[inline(always)]
    pub fn probe(&self, hash: u64) -> Option<TTEntry> {
        let idx = (hash as usize) & self.mask;
        let entry = &self.entries[idx];

        let stored_hash = entry.hash.load(Ordering::Acquire);
        if stored_hash != hash {
            return None;
        }

        let raw_data = entry.data.load(Ordering::Acquire);
        // 設計レビュー F05: key と data を XOR 結合することで、他局面の混在した書き込みを検知・排除
        let data = raw_data ^ stored_hash;

        if entry.hash.load(Ordering::Acquire) != hash {
            return None;
        }

        let node_type = NodeType::from_u8((data & 0xFF) as u8);
        let depth = ((data >> 8) & 0xFF) as u8;
        let score = ((data >> 16) & 0xFFFF_FFFF) as i32;
        let move_u16 = ((data >> 48) & 0xFFFF) as u16;
        let best_move = Move::from_u16(move_u16);

        Some(TTEntry {
            hash,
            depth,
            score,
            node_type,
            best_move,
        })
    }

    #[inline(always)]
    pub fn store(
        &self,
        hash: u64,
        depth: u8,
        score: i32,
        node_type: NodeType,
        best_move: Option<Move>,
    ) {
        let idx = (hash as usize) & self.mask;
        let entry = &self.entries[idx];

        let existing_hash = entry.hash.load(Ordering::Relaxed);
        let existing_raw = entry.data.load(Ordering::Relaxed);
        let existing_data = existing_raw ^ existing_hash;
        if existing_hash == hash {
            let existing_depth = ((existing_data >> 8) & 0xFF) as u8;
            if existing_depth > depth && best_move.is_none() {
                return;
            }
        }

        let move_u16 = best_move.map(|m| m.as_u16()).unwrap_or(0);
        let data = (node_type as u64 & 0xFF)
            | ((depth as u64 & 0xFF) << 8)
            | (((score as u32) as u64) << 16)
            | ((move_u16 as u64) << 48);

        // XOR 結合したデータを保存
        entry.data.store(data ^ hash, Ordering::Release);
        entry.hash.store(hash, Ordering::Release);
    }

    pub fn clear(&self) {
        for entry in &self.entries {
            entry.hash.store(0, Ordering::Relaxed);
            entry.data.store(0, Ordering::Relaxed);
        }
    }
}
