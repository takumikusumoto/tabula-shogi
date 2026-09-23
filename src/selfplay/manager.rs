use super::config::SelfPlayConfig;
use super::csa::CsaSerializer;
use super::dataset::{DatasetEntry, DatasetHandler};
use super::game::{DrawReason, GameRecord, GameResult, GameRunner, SimpleRng};
use crate::search::SearchEngine;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
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
    /// 自己対局バッチセッションを実行
    pub fn run(config: SelfPlayConfig) -> SelfPlayStats {
        let start_time = Instant::now();
        let num_games = config.num_games;
        let num_threads = config.threads.clamp(1, 64);

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
        let stats = Arc::new(Mutex::new(SelfPlayStats::default()));

        let csa_file = config.csa_output.as_ref().map(|path| {
            if let Some(parent) = std::path::Path::new(path).parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            Arc::new(Mutex::new(
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .expect("Failed to open CSA output file"),
            ))
        });

        let data_file = config.data_output.as_ref().map(|path| {
            if let Some(parent) = std::path::Path::new(path).parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            Arc::new(Mutex::new(
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .expect("Failed to open Dataset output file"),
            ))
        });

        let mut handles = Vec::with_capacity(num_threads);

        for _thread_id in 0..num_threads {
            let counter = Arc::clone(&game_counter);
            let stats_lock = Arc::clone(&stats);
            let csa_lock = csa_file.clone();
            let data_lock = data_file.clone();
            let cfg = config.clone();

            let handle = thread::spawn(move || {
                let mut engine =
                    SearchEngine::new(cfg.tt_size_mb).with_eval_mode(cfg.eval_mode.clone());
                engine.use_book = cfg.use_book;
                loop {
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

                    // CSA書き出し (エラー検知・伝播)
                    if let Some(ref lock) = csa_lock {
                        let csa_str = CsaSerializer::serialize_game(&record);
                        let mut file = lock.lock().unwrap();
                        if let Err(e) = file
                            .write_all(csa_str.as_bytes())
                            .and_then(|_| file.flush())
                        {
                            eprintln!("\n[Error] CSA write/flush failed: {e}");
                            let mut s = stats_lock.lock().unwrap();
                            s.io_errors += 1;
                        }
                    }

                    // データセット書き出し (エラー検知・伝播)
                    if let Some(ref lock) = data_lock {
                        let entries: Vec<DatasetEntry> =
                            match DatasetHandler::extract_entries_checked(
                                &record,
                                cfg.random_opening_plies + 1,
                            ) {
                                Ok(entries) => entries,
                                Err(e) => {
                                    eprintln!("\n[Error] Dataset record validation failed: {e}");
                                    let mut s = stats_lock.lock().unwrap();
                                    s.io_errors += 1;
                                    Vec::new()
                                }
                            };
                        let mut file = lock.lock().unwrap();
                        let mut write_err = false;
                        for entry in entries {
                            if let Err(e) =
                                file.write_all(DatasetHandler::format_entry(&entry).as_bytes())
                            {
                                eprintln!("\n[Error] Dataset write failed: {e}");
                                write_err = true;
                                break;
                            }
                        }
                        if !write_err {
                            match file.flush() {
                                Ok(()) => {}
                                Err(e) => {
                                    eprintln!("\n[Error] Dataset flush failed: {e}");
                                    write_err = true;
                                }
                            }
                        }
                        if write_err {
                            let mut s = stats_lock.lock().unwrap();
                            s.io_errors += 1;
                        }
                    }

                    // 統計更新
                    {
                        let mut s = stats_lock.lock().unwrap();
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
                let mut s = stats.lock().unwrap();
                s.io_errors += 1;
            }
        }

        println!();
        let elapsed = start_time.elapsed();
        let final_stats = stats.lock().unwrap();
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

        final_stats.clone()
    }
}
