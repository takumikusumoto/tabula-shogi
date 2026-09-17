use std::sync::Arc;
use tabula_shogi::eval::EvalMode;
use tabula_shogi::eval::halfkp::HalfKPEvaluator;
use tabula_shogi::eval::halfkp_stream_trainer::{HalfKPStreamTrainer, StreamTrainConfig};
use tabula_shogi::selfplay::config::SelfPlayConfig;
use tabula_shogi::selfplay::partition::{PartitionConfig, PartitionedSelfPlayManager};

fn print_usage() {
    println!(
        r#"TabulaShogi Titan HalfKP Pipeline (100M Dataset & Stream Training)

USAGE:
    titan_halfkp <SUBCOMMAND> [OPTIONS]

SUBCOMMANDS:
    generate    Run partitioned self-play data generation (multi-thread, resume-capable)
    train       Run memory-bounded streaming HalfKP training on partitioned datasets
    run         Execute end-to-end pipeline (generate + train)
    help        Print this help message

GENERATE OPTIONS:
    --games <N>              Total games to generate [default: 100]
    --games-per-part <N>     Games per partition file [default: 20]
    --threads <T>            Worker threads [default: 4]
    --depth <D>              Search depth per move [default: 2]
    --dir <DIR>              Dataset output directory [default: data/titan]
    --seed <SEED>            Base PRNG seed [default: 0x9E3779B97F4A7C15]
    --eval-mode <MODE>       Eval mode: 'hce' or 'halfkp:<path>' [default: hce]

TRAIN OPTIONS:
    --dir <DIR>              Dataset input directory [default: data/titan]
    --epochs <E>             Training epochs [default: 1]
    --batch-size <B>         Batch size [default: 1024]
    --lr <LR>                Learning rate [default: 0.001]
    --model <PATH>           Output model path [default: models/halfkp_titan.bin]
    --checkpoint <PATH>      Checkpoint path [default: models/halfkp_titan_ckpt.bin]
    --no-resume              Do not resume from existing checkpoint
"#
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "generate" => run_generate(&args[2..]),
        "train" => run_train(&args[2..]),
        "run" => {
            run_generate(&args[2..]);
            run_train(&args[2..]);
        }
        "help" | "-h" | "--help" => print_usage(),
        other => {
            eprintln!("Unknown subcommand: '{other}'");
            print_usage();
            std::process::exit(1);
        }
    }
}

fn parse_eval_mode(val: &str) -> EvalMode {
    if val == "hce" {
        EvalMode::Hce
    } else if let Some(path) = val.strip_prefix("halfkp:") {
        println!("Loading HalfKP model from '{path}' for self-play...");
        match HalfKPEvaluator::load_from_file(path) {
            Ok(eval) => EvalMode::HalfKP(Arc::new(eval)),
            Err(e) => {
                eprintln!("Failed to load HalfKP model from '{path}': {e}. Falling back to Hce.");
                EvalMode::Hce
            }
        }
    } else {
        eprintln!("Unknown eval mode '{val}'. Using Hce.");
        EvalMode::Hce
    }
}

fn run_generate(args: &[String]) {
    let mut total_games = 100usize;
    let mut games_per_part = 20usize;
    let mut threads = 4usize;
    let mut depth = 2u8;
    let mut dir = "data/titan".to_string();
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut eval_mode_str = "hce".to_string();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--games" if i + 1 < args.len() => {
                total_games = args[i + 1].parse().unwrap_or(total_games);
                i += 1;
            }
            "--games-per-part" if i + 1 < args.len() => {
                games_per_part = args[i + 1].parse().unwrap_or(games_per_part);
                i += 1;
            }
            "--threads" if i + 1 < args.len() => {
                threads = args[i + 1].parse().unwrap_or(threads);
                i += 1;
            }
            "--depth" if i + 1 < args.len() => {
                depth = args[i + 1].parse().unwrap_or(depth);
                i += 1;
            }
            "--dir" if i + 1 < args.len() => {
                dir = args[i + 1].clone();
                i += 1;
            }
            "--seed" if i + 1 < args.len() => {
                seed = args[i + 1].parse().unwrap_or(seed);
                i += 1;
            }
            "--eval-mode" if i + 1 < args.len() => {
                eval_mode_str = args[i + 1].clone();
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }

    let eval_mode = parse_eval_mode(&eval_mode_str);

    let config = PartitionConfig {
        total_games,
        games_per_partition: games_per_part,
        output_dir: dir,
        base_config: SelfPlayConfig {
            num_games: games_per_part,
            threads,
            depth,
            random_opening_plies: 8,
            max_plies: 300,
            resign_threshold: -2500,
            csa_output: None,
            data_output: None,
            tt_size_mb: 16,
            seed,
            eval_mode,
            temperature_plies: 24,
            start_game_id: 0,
        },
    };

    PartitionedSelfPlayManager::run(config);
}

fn run_train(args: &[String]) {
    let mut data_dir = "data/titan".to_string();
    let mut epochs = 1usize;
    let mut batch_size = 1024usize;
    let mut lr = 0.001f32;
    let mut model_output_path = "models/halfkp_titan.bin".to_string();
    let mut checkpoint_path = Some("models/halfkp_titan_ckpt.bin".to_string());
    let mut resume_from_checkpoint = true;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--dir" if i + 1 < args.len() => {
                data_dir = args[i + 1].clone();
                i += 1;
            }
            "--epochs" if i + 1 < args.len() => {
                epochs = args[i + 1].parse().unwrap_or(epochs);
                i += 1;
            }
            "--batch-size" if i + 1 < args.len() => {
                batch_size = args[i + 1].parse().unwrap_or(batch_size);
                i += 1;
            }
            "--lr" if i + 1 < args.len() => {
                lr = args[i + 1].parse().unwrap_or(lr);
                i += 1;
            }
            "--model" if i + 1 < args.len() => {
                model_output_path = args[i + 1].clone();
                i += 1;
            }
            "--checkpoint" if i + 1 < args.len() => {
                checkpoint_path = Some(args[i + 1].clone());
                i += 1;
            }
            "--no-resume" => {
                resume_from_checkpoint = false;
            }
            _ => {}
        }
        i += 1;
    }

    let config = StreamTrainConfig {
        data_dir,
        batch_size,
        lr,
        k: 600.0,
        epochs,
        checkpoint_interval_batches: 2000,
        checkpoint_path,
        model_output_path,
        resume_from_checkpoint,
    };

    if let Err(e) = HalfKPStreamTrainer::train(config) {
        eprintln!("[Error] Stream training failed: {e}");
        std::process::exit(1);
    }
}
