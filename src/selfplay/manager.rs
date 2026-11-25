use super::config::SelfPlayConfig;
use super::csa::CsaSerializer;
use super::dataset::{DatasetEntry, DatasetHandler};
use super::game::{DrawReason, GameRecord, GameResult, GameRunner, SimpleRng};
use crate::search::SearchEngine;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

#[derive(Debug, Default, Clone)]
pub struct SelfPlayStats {
    pub completed_games: usize,
    pub black_wins: usize,
    pub white_wins: usize,
    pub draws_sennichite: usize,
    pub draws_max_plies: usize,
    pub total_plies: usize,
    /// ファイル書き込みまたはスレッド実行中に発生した異常エラー件数
    pub io_errors: usize,
}

impl SelfPlayStats {
    pub fn update(&mut self, record: &GameRecord) {
        self.completed_games += 1;
        self.total_plies += record.total_plies;
        match record.result {
            GameResult::BlackWin(_) => self.black_wins += 1,
            GameResult::WhiteWin(_) => self.white_wins += 1,
            GameResult::Draw(DrawReason::Sennichite) => self.draws_sennichite += 1,
            GameResult::Draw(DrawReason::MaxPliesExceeded)
            | GameResult::Draw(DrawReason::Jishogi) => self.draws_max_plies += 1,
        }
    }
}

pub struct SelfPlayManager;

impl SelfPlayManager {
    /// 自己対局バッチセッションを fail-closed に実行
    /// 出力ファイルオープン障害、書き込み/検証エラー、ワーカースレッド異常時は `Err` を返す。
    pub fn run(config: SelfPlayConfig) -> Result<SelfPlayStats, String> {
        let start_time = Instant::now();
        let num_games = config.num_games;
        let num_threads = config.threads.clamp(1, 64);

        let csa_file = match config.csa_output.as_deref() {
            Some(path) => Some(Arc::new(Mutex::new(open_append_file(path, "CSA output")?))),
            None => None,
        };

        let data_file = match config.data_output.as_deref() {
            Some(path) => Some(Arc::new(Mutex::new(open_append_file(
                path,
                "dataset output",
            )?))),
            None => None,
        };

        if num_games == 0 {
            return Ok(SelfPlayStats::default());
        }

        println!("=== TabulaShogi Self-Play Pipeline ===");
        println!(
            "Games: {}, Threads: {}, Depth: {}, RandomOpening: {} plies, StartGameId: {}",
            num_games, num_threads, config.depth, config.random_opening_plies, config.start_game_id
        );
        if let Some(ref csa) = config.csa_output {
            println!("CSA Output: {csa}");
        }
        if let Some(ref data) = config.data_output {
            println!("Dataset Output: {data}");
        }
        println!("--------------------------------------");

        let game_counter = Arc::new(AtomicUsize::new(0));
        let abort_flag = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(Mutex::new(SelfPlayStats::default()));
        let mut handles = Vec::with_capacity(num_threads);

        for _thread_id in 0..num_threads {
            let counter = Arc::clone(&game_counter);
            let abort = Arc::clone(&abort_flag);
            let stats_lock = Arc::clone(&stats);
            let csa_lock = csa_file.clone();
            let data_lock = data_file.clone();
            let cfg = config.clone();

            let handle = thread::spawn(move || {
                let mut engine =
                    SearchEngine::new(cfg.tt_size_mb).with_eval_mode(cfg.eval_mode.clone());
                engine.use_book = cfg.use_book;
                loop {
                    if abort.load(Ordering::Relaxed) {
                        break;
                    }
                    let game_idx = counter.fetch_add(1, Ordering::SeqCst);
                    if game_idx >= num_games {
                        break;
                    }
                    let global_game_id = cfg.start_game_id + game_idx;

                    // 対局ごとに決定論的なユニークシードを生成 (スレッド数・割当てに依存せず global_game_id のみで完全再現)
                    let id_val = global_game_id as u64 + 1;
                    let mixed = id_val.wrapping_mul(0x9e3779b97f4a7c15)
                        ^ (id_val.wrapping_mul(0x517cc1b727220a95) >> 32);
                    let game_seed = cfg.seed.wrapping_add(mixed);
                    let mut game_rng = SimpleRng::new(if game_seed == 0 {
                        0xdeadbeefcafe
                    } else {
                        game_seed
                    });

                    let record = {
                        engine.clear();
                        GameRunner::play_game(global_game_id + 1, &cfg, &mut engine, &mut game_rng)
                    };

                    // CSA書き出し (エラー検知・即座に全体中断)
                    if let Some(ref lock) = csa_lock {
                        let csa_str = CsaSerializer::serialize_game(&record);
                        let mut file = lock.lock().unwrap_or_else(|e| e.into_inner());
                        if let Err(e) = file
                            .write_all(csa_str.as_bytes())
                            .and_then(|_| file.flush())
                        {
                            eprintln!("\n[Error] CSA write/flush failed: {e}");
                            let mut s = stats_lock.lock().unwrap_or_else(|e| e.into_inner());
                            s.io_errors += 1;
                            abort.store(true, Ordering::Relaxed);
                            break;
                        }
                    }

                    // データセット書き出し (エラー検知・即座に全体中断)
                    if let Some(ref lock) = data_lock {
                        match DatasetHandler::extract_entries_checked(
                            &record,
                            cfg.random_opening_plies + 1,
                        ) {
                            Ok(entries) => {
                                if let Err(e) = write_dataset_entries(lock, &entries) {
                                    eprintln!("\n[Error] Dataset write/flush failed: {e}");
                                    let mut s =
                                        stats_lock.lock().unwrap_or_else(|e| e.into_inner());
                                    s.io_errors += 1;
                                    abort.store(true, Ordering::Relaxed);
                                    break;
                                }
                            }
                            Err(e) => {
                                eprintln!("\n[Error] Dataset record validation failed: {e}");
                                let mut s = stats_lock.lock().unwrap_or_else(|e| e.into_inner());
                                s.io_errors += 1;
                                abort.store(true, Ordering::Relaxed);
                                break;
                            }
                        }
                    }

                    // 統計更新
                    {
                        let mut s = stats_lock.lock().unwrap_or_else(|e| e.into_inner());
                        s.update(&record);
                        let avg_plies = s.total_plies.checked_div(s.completed_games).unwrap_or(0);
                        print!(
                            "\r[Progress] Game {}/{} | B:{} W:{} D:{} | AvgPlies: {}    ",
                            s.completed_games,
                            num_games,
                            s.black_wins,
                            s.white_wins,
                            s.draws_sennichite + s.draws_max_plies,
                            avg_plies
                        );
                        let _ = std::io::stdout().flush();
                    }
                }
            });

            handles.push(handle);
        }

        for h in handles {
            if let Err(e) = h.join() {
                eprintln!("\n[Error] Self-play worker thread panicked: {:?}", e);
                abort_flag.store(true, Ordering::Relaxed);
                let mut s = stats.lock().unwrap_or_else(|e| e.into_inner());
                s.io_errors += 1;
            }
        }

        println!();
        let elapsed = start_time.elapsed();
        let final_stats = stats.lock().unwrap_or_else(|e| e.into_inner()).clone();

        if final_stats.io_errors > 0 || final_stats.completed_games < num_games {
            return Err(format!(
                "Self-play session failed with {} I/O or worker errors ({}/{} games completed)",
                final_stats.io_errors, final_stats.completed_games, num_games
            ));
        }

        print_session_summary(&final_stats, elapsed);
        Ok(final_stats)
    }
}

fn print_session_summary(final_stats: &SelfPlayStats, elapsed: std::time::Duration) {
    let total_time_sec = elapsed.as_secs_f64().max(0.001);
    let games_per_sec = final_stats.completed_games as f64 / total_time_sec;

    println!("--------------------------------------");
    println!(
        "Self-Play Session Completed in {:.2}s ({:.2} games/sec)",
        total_time_sec, games_per_sec
    );
    println!(
        "Results: Black: {} ({:.1}%), White: {} ({:.1}%), Draws: {} ({:.1}%)",
        final_stats.black_wins,
        (final_stats.black_wins as f64 / final_stats.completed_games.max(1) as f64) * 100.0,
        final_stats.white_wins,
        (final_stats.white_wins as f64 / final_stats.completed_games.max(1) as f64) * 100.0,
        (final_stats.draws_sennichite + final_stats.draws_max_plies),
        ((final_stats.draws_sennichite + final_stats.draws_max_plies) as f64
            / final_stats.completed_games.max(1) as f64)
            * 100.0,
    );
    println!(
        "Draw breakdown: Sennichite: {}, MaxPlies: {}",
        final_stats.draws_sennichite, final_stats.draws_max_plies
    );
    println!(
        "Total Plies: {}, Average Plies: {}",
        final_stats.total_plies,
        final_stats
            .total_plies
            .checked_div(final_stats.completed_games)
            .unwrap_or(0)
    );
    println!("======================================");
}

fn open_append_file(path: &str, label: &str) -> Result<File, String> {
    if path.trim().is_empty() {
        return Err(format!("{label} path must not be empty"));
    }
    if let Some(parent) = Path::new(path).parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create parent directory for {label} '{path}': {e}"))?;
    }
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("Failed to open {label} file '{path}': {e}"))
}

fn write_dataset_entries(lock: &Mutex<File>, entries: &[DatasetEntry]) -> std::io::Result<()> {
    let mut file = lock.lock().unwrap_or_else(|e| e.into_inner());
    for entry in entries {
        file.write_all(DatasetHandler::format_entry(entry).as_bytes())?;
    }
    file.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TempDirGuard(PathBuf);

    impl TempDirGuard {
        fn new(prefix: &str) -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "tabula_shogi_sp_{prefix}_{}_{}",
                std::process::id(),
                id
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn selfplay_manager_returns_err_instead_of_panicking_on_unwritable_output() {
        let tmp = TempDirGuard::new("unwritable");
        let dir_as_file = tmp.0.to_string_lossy().into_owned();

        let cfg = SelfPlayConfig {
            num_games: 1,
            threads: 1,
            depth: 1,
            max_plies: 4,
            data_output: Some(dir_as_file),
            ..Default::default()
        };
        let err = SelfPlayManager::run(cfg);
        assert!(
            err.is_err(),
            "Passing a directory as data_output must return Err fail-closed, not panic"
        );
    }
}
