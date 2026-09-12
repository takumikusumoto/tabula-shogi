use std::env;
use tabula_shogi::UsiHandler;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() <= 1 {
        // デフォルト: USIプロトコル通信ループ起動 (将棋所/ShogiGUI等のGUI互換)
        let mut handler = UsiHandler::new();
        handler.run();
        return;
    }

    match args[1].as_str() {
        "usi" => {
            let mut handler = UsiHandler::new();
            handler.run();
        }
        "selfplay" => {
            tabula_shogi::selfplay::run_cli(&args[2..]);
        }
        "tune" => {
            tabula_shogi::tune::run_cli(&args[2..]);
        }
        "train-nnue" => {
            run_train_nnue(&args[2..]);
        }
        "bench" => {
            run_benchmark();
        }
        "--help" | "-h" | "help" => {
            print_main_help();
        }
        "--version" | "-v" | "version" => {
            println!("TabulaShogi {}", env!("CARGO_PKG_VERSION"));
        }
        _ => {
            eprintln!("Unknown command: '{}'. Use --help for usage.", args[1]);
            std::process::exit(1);
        }
    }
}

fn run_train_nnue(args: &[String]) {
    use tabula_shogi::eval::NNUETrainer;
    use tabula_shogi::selfplay::DatasetHandler;

    let mut data_path = String::new();
    let mut out_path = "nnue.bin".to_string();
    let mut epochs = 20;
    let mut lr = 0.001f32;
    let mut batch_size = 64;
    let mut k = 400.0f32;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => {
                if i + 1 < args.len() {
                    data_path = args[i + 1].clone();
                    i += 1;
                }
            }
            "--out" | "-o" => {
                if i + 1 < args.len() {
                    out_path = args[i + 1].clone();
                    i += 1;
                }
            }
            "--epochs" | "-e" => {
                if i + 1 < args.len() {
                    epochs = args[i + 1].parse().unwrap_or(epochs);
                    i += 1;
                }
            }
            "--lr" => {
                if i + 1 < args.len() {
                    lr = args[i + 1].parse().unwrap_or(lr);
                    i += 1;
                }
            }
            "--batch-size" | "-b" => {
                if i + 1 < args.len() {
                    batch_size = args[i + 1].parse().unwrap_or(batch_size);
                    i += 1;
                }
            }
            "--k" => {
                if i + 1 < args.len() {
                    k = args[i + 1].parse().unwrap_or(k);
                    i += 1;
                }
            }
            "--help" | "-h" => {
                println!(
                    r#"TabulaShogi NNUE Trainer
USAGE:
    tabula-shogi train-nnue --data <PATH> [OPTIONS]

OPTIONS:
        --data <PATH>        Path to self-play training dataset (TSV) [REQUIRED]
    -o, --out <PATH>         Path to output quantized NNUE weights binary [default: nnue.bin]
    -e, --epochs <N>         Number of training epochs [default: 20]
        --lr <FLOAT>         Learning rate for Adam optimizer [default: 0.001]
    -b, --batch-size <N>     Mini-batch size [default: 64]
        --k <FLOAT>          Logistic scale factor K [default: 400.0]
    -h, --help               Print this help message
"#
                );
                return;
            }
            _ => {}
        }
        i += 1;
    }

    if data_path.is_empty() {
        eprintln!("Error: --data <FILE> is required.");
        eprintln!("Run 'tabula-shogi train-nnue --help' for usage.");
        return;
    }

    println!("=== TabulaShogi Scratch NNUE Trainer ===");
    println!("Dataset: {data_path}");
    println!("Epochs: {epochs}, BatchSize: {batch_size}, LR: {lr}, K: {k}");
    println!("Output File: {out_path}");
    println!("Loading dataset...");

    let entries = match DatasetHandler::load_from_file(&data_path) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("Failed to read dataset: {err}");
            return;
        }
    };

    if entries.is_empty() {
        eprintln!("Dataset contains 0 entries.");
        return;
    }

    println!(
        "Loaded {} entries. Initializing NNUE backpropagation network...",
        entries.len()
    );
    let mut trainer = NNUETrainer::new();

    let start_time = std::time::Instant::now();
    let (trained_eval, init_loss, final_loss) =
        trainer.train_dataset(&entries, epochs, lr, batch_size, k);
    let elapsed = start_time.elapsed();

    let reduction = ((init_loss - final_loss) / init_loss.max(1e-7)) * 100.0;
    println!("--------------------------------------");
    println!("Training Complete in {:.2}s!", elapsed.as_secs_f64());
    println!("Initial MSE Loss: {:.6}", init_loss);
    println!(
        "Final MSE Loss:   {:.6} (Reduction: {:.2}%)",
        final_loss, reduction
    );

    println!("Quantizing and saving weights to '{out_path}'...");
    if let Err(e) = trained_eval.save_to_file(&out_path) {
        eprintln!("Failed to save weights file: {e}");
    } else {
        println!("Successfully saved quantized NNUE model to '{out_path}' (355 KB)!");
    }
    println!("======================================");
}

fn run_benchmark() {
    use std::time::Instant;
    use tabula_shogi::board::Position;
    use tabula_shogi::search::SearchEngine;

    println!("TabulaShogi Benchmark (Depth 6 search on startpos)...");
    let mut engine = SearchEngine::new(64);
    let mut pos = Position::startpos();
    let start = Instant::now();
    let (best_mv, score) = engine.search_fixed_depth(&mut pos, 6);
    let elapsed = start.elapsed();
    let nodes = engine.nodes();
    let nps = if elapsed.as_millis() > 0 {
        (nodes as u128 * 1000) / elapsed.as_millis()
    } else {
        0
    };

    println!(
        "BestMove: {}, Score: {} cp, Nodes: {}, Time: {:.2}s, NPS: {}",
        best_mv
            .map(|m| m.to_usi())
            .unwrap_or_else(|| "none".to_string()),
        score,
        nodes,
        elapsed.as_secs_f64(),
        nps
    );
}

fn print_main_help() {
    println!(
        r#"TabulaShogi {} — An Autonomous Rust Shogi Engine (Tabula Rasa)

USAGE:
    tabula-shogi [COMMAND] [OPTIONS]

COMMANDS:
    usi         Run USI (Universal Shogi Interface) engine loop [DEFAULT]
    selfplay    Autonomous self-play generation pipeline (games, CSA, dataset)
    tune        Texel Tuning solver for evaluation parameter optimization
    train-nnue  Scratch NNUE neural network trainer (backprop + Adam)
    bench       Run search performance benchmark on standard positions

OPTIONS:
    -h, --help       Print this help message
    -v, --version    Print version information

Run 'tabula-shogi <COMMAND> --help' for details on a specific command.
"#,
        env!("CARGO_PKG_VERSION")
    );
}
