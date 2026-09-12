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

#[derive(Debug, Default)]
pub struct SelfPlayStats {
    pub completed_games: usize,
    pub black_wins: usize,
    pub white_wins: usize,
    pub draws_sennichite: usize,
    pub draws_max_plies: usize,
    pub total_plies: usize,
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
            "Games: {}, Threads: {}, Depth: {}, RandomOpening: {} plies",
            num_games, num_threads, config.depth, config.random_opening_plies
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
            Arc::new(Mutex::new(
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .expect("Failed to open CSA output file"),
            ))
        });

        let data_file = config.data_output.as_ref().map(|path| {
            Arc::new(Mutex::new(
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .expect("Failed to open Dataset output file"),
            ))
        });

        let mut handles = Vec::with_capacity(num_threads);

        for thread_id in 0..num_threads {
            let counter = Arc::clone(&game_counter);
            let stats_lock = Arc::clone(&stats);
            let cfg = config.clone();
            let csa_lock = csa_file.clone();
            let data_lock = data_file.clone();

            let handle = thread::spawn(move || {
                let mut engine =
                    SearchEngine::new(cfg.tt_size_mb).with_eval_mode(cfg.eval_mode.clone());
                // 各スレッドに独立した乱数シードを供給
                let thread_seed =
                    cfg.seed ^ ((thread_id as u64 + 1).wrapping_mul(0x517cc1b727220a95));
                let mut rng = SimpleRng::new(thread_seed);

                loop {
                    let game_id = counter.fetch_add(1, Ordering::SeqCst);
                    if game_id >= num_games {
                        break;
                    }

                    let record = GameRunner::play_game(game_id + 1, &cfg, &mut engine, &mut rng);

                    // CSA書き出し
                    if let Some(ref lock) = csa_lock {
                        let csa_str = CsaSerializer::serialize_game(&record);
                        let mut file = lock.lock().unwrap();
                        let _ = file.write_all(csa_str.as_bytes());
                        let _ = file.flush();
                    }

                    // データセット書き出し
                    if let Some(ref lock) = data_lock {
                        let entries: Vec<DatasetEntry> =
                            DatasetHandler::extract_entries(&record, cfg.random_opening_plies + 1);
                        let mut file = lock.lock().unwrap();
                        for entry in entries {
                            let _ = file.write_all(DatasetHandler::format_entry(&entry).as_bytes());
                        }
                        let _ = file.flush();
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
            let _ = h.join();
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

        SelfPlayStats {
            completed_games: final_stats.completed_games,
            black_wins: final_stats.black_wins,
            white_wins: final_stats.white_wins,
            draws_sennichite: final_stats.draws_sennichite,
            draws_max_plies: final_stats.draws_max_plies,
            total_plies: final_stats.total_plies,
        }
    }
}
