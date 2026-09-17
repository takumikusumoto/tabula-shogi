pub mod config;
pub mod csa;
pub mod dataset;
pub mod game;
pub mod manager;
pub mod partition;

pub use config::SelfPlayConfig;
pub use csa::CsaSerializer;
pub use dataset::{DatasetEntry, DatasetHandler, StreamingBatchReader};
pub use game::{GameRecord, GameResult, GameRunner, PlyRecord, SimpleRng};
pub use manager::{SelfPlayManager, SelfPlayStats};
pub use partition::{
    MultiPartitionStreamingReader, PartitionConfig, PartitionedSelfPlayManager,
    PartitionedSessionStats,
};

/// CLI引数をパースして自己対局を実行
pub fn run_cli(args: &[String]) {
    let mut config = SelfPlayConfig::default();
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "--games" | "-g" => {
                if i + 1 < args.len() {
                    config.num_games = args[i + 1].parse().unwrap_or(config.num_games);
                    i += 1;
                }
            }
            "--threads" | "-t" => {
                if i + 1 < args.len() {
                    config.threads = args[i + 1].parse().unwrap_or(config.threads);
                    i += 1;
                }
            }
            "--depth" | "-d" => {
                if i + 1 < args.len() {
                    config.depth = args[i + 1].parse().unwrap_or(config.depth);
                    i += 1;
                }
            }
            "--random-opening" | "-r" => {
                if i + 1 < args.len() {
                    config.random_opening_plies =
                        args[i + 1].parse().unwrap_or(config.random_opening_plies);
                    i += 1;
                }
            }
            "--max-plies" | "-m" => {
                if i + 1 < args.len() {
                    config.max_plies = args[i + 1].parse().unwrap_or(config.max_plies);
                    i += 1;
                }
            }
            "--csa" => {
                if i + 1 < args.len() {
                    config.csa_output = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--data" => {
                if i + 1 < args.len() {
                    config.data_output = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--tt-size" => {
                if i + 1 < args.len() {
                    config.tt_size_mb = args[i + 1].parse().unwrap_or(config.tt_size_mb);
                    i += 1;
                }
            }
            "--seed" => {
                if i + 1 < args.len() {
                    config.seed = args[i + 1].parse().unwrap_or(config.seed);
                    i += 1;
                }
            }
            "--help" | "-h" => {
                print_help();
                return;
            }
            _ => {}
        }
        i += 1;
    }

    SelfPlayManager::run(config);
}

fn print_help() {
    println!(
        r#"TabulaShogi Self-Play Generator
USAGE:
    tabula-shogi selfplay [OPTIONS]

OPTIONS:
    -g, --games <N>           Number of games to generate [default: 10]
    -t, --threads <T>         Number of parallel worker threads [default: 1]
    -d, --depth <D>           Search depth per move [default: 3]
    -r, --random-opening <K>  Number of initial plies to choose randomly [default: 8]
    -m, --max-plies <M>       Maximum plies per game before declared draw [default: 320]
        --csa <PATH>          Path to append generated CSA game records [default: none]
        --data <PATH>         Path to append training dataset (TSV) [default: none]
        --tt-size <MB>        Transposition table size in MB [default: 16]
        --seed <SEED>         Pseudo-random number generator seed
    -h, --help                Print this help message
"#
    );
}
