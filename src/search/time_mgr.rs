use crate::types::Color;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TimeControl {
    pub btime: Option<u64>,
    pub wtime: Option<u64>,
    pub byoyomi: Option<u64>,
    pub binc: Option<u64>,
    pub winc: Option<u64>,
    pub infinite: bool,
}

pub struct TimeManager {
    start_time: Instant,
    target_duration: Duration,
    hard_limit: Duration,
    is_infinite: bool,
}

const DEFAULT_AVAILABLE_MS: u64 = 2000;
const SAFETY_MARGIN_RATIO: u64 = 15;
const MIN_SAFETY_MARGIN_MS: u64 = 20;
const MAX_SAFETY_MARGIN_MS: u64 = 150;
const MIN_USABLE_MS: u64 = 10;
const TIME_DIVISOR_BYOYOMI: u64 = 30;
const TIME_DIVISOR_NORMAL: u64 = 35;
const MAX_ALLOC_RATIO: u64 = 3;

impl TimeManager {
    pub fn new(tc: &TimeControl, us: Color) -> Self {
        let start_time = Instant::now();

        if tc.infinite {
            return TimeManager {
                start_time,
                target_duration: Duration::from_secs(3600),
                hard_limit: Duration::from_secs(3600),
                is_infinite: true,
            };
        }

        let (remaining_time, inc) = match us {
            Color::Black => (tc.btime, tc.binc.unwrap_or(0)),
            Color::White => (tc.wtime, tc.winc.unwrap_or(0)),
        };

        // この手で利用可能な最大時間 (持ち時間 + 加算, または秒読み)
        // Astra 6 レビュー 9.3: 利用可能時間より長い時間を割り当てて時間切れ負けするバグを防止
        let available_ms = if let Some(byo) = tc.byoyomi {
            if byo > 0 {
                byo
            } else if let Some(rem) = remaining_time {
                rem + inc
            } else {
                DEFAULT_AVAILABLE_MS
            }
        } else if let Some(rem) = remaining_time {
            rem + inc
        } else {
            DEFAULT_AVAILABLE_MS
        };

        // 通信バッファ・安全マージン (最小20ms, 最大150ms)
        let safety_margin =
            (available_ms / SAFETY_MARGIN_RATIO).clamp(MIN_SAFETY_MARGIN_MS, MAX_SAFETY_MARGIN_MS);
        let usable_ms = available_ms
            .saturating_sub(safety_margin)
            .max(MIN_USABLE_MS);

        let target_ms = if let Some(byo) = tc.byoyomi {
            if byo > 0 {
                // 秒読み重視: 安全マージンを引いた秒読み時間
                byo.saturating_sub(safety_margin).max(MIN_USABLE_MS)
            } else if let Some(rem) = remaining_time {
                // 持ち時間制: 残り時間の1/30 + 加算
                ((rem / TIME_DIVISOR_BYOYOMI) + inc)
                    .min(usable_ms)
                    .max(MIN_USABLE_MS)
            } else {
                usable_ms.min(DEFAULT_AVAILABLE_MS)
            }
        } else if let Some(rem) = remaining_time {
            // 秒読みなしの持ち時間制
            let mut alloc = (rem / TIME_DIVISOR_NORMAL) + inc;
            if alloc > rem / MAX_ALLOC_RATIO {
                alloc = rem / MAX_ALLOC_RATIO;
            }
            alloc.min(usable_ms).max(MIN_USABLE_MS)
        } else {
            usable_ms.min(DEFAULT_AVAILABLE_MS)
        };

        // ハードリミット: どんなに延長しても利用可能時間 (usable_ms) を超えない
        let hard_ms = (target_ms * 2).clamp(target_ms, usable_ms);

        let target_duration = Duration::from_millis(target_ms);
        let hard_limit = Duration::from_millis(hard_ms);

        TimeManager {
            start_time,
            target_duration,
            hard_limit,
            is_infinite: false,
        }
    }

    /// 反復深化で「次の深さの探索を開始すべきか」の判定
    #[inline(always)]
    pub fn should_continue_deepening(&self) -> bool {
        if self.is_infinite {
            return true;
        }
        self.start_time.elapsed() < self.target_duration
    }

    /// 探索ノード内で「直ちに探索を打ち切るべきか（ハードリミット）」の判定
    #[inline(always)]
    pub fn is_time_up(&self) -> bool {
        if self.is_infinite {
            return false;
        }
        self.start_time.elapsed() >= self.hard_limit
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }
}
