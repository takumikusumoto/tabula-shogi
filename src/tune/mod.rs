pub mod params;
pub mod texel;

pub use params::{PARAM_COUNT, TunableParams};
pub use texel::{PositionFeatures, TexelTuner};

use crate::board::Position;
use crate::selfplay::DatasetHandler;

/// CLI引数をパースしてTexel Tuningを実行
pub fn run_cli(args: &[String]) {
    let mut data_path = String::new();
    let mut epochs = 50;
    let mut lr = 1.0;
    let mut k = TexelTuner::DEFAULT_K;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => {
                if i + 1 < args.len() {
                    data_path = args[i + 1].clone();
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
            "--k" => {
                if i + 1 < args.len() {
                    k = args[i + 1].parse().unwrap_or(k);
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

    if data_path.is_empty() {
        eprintln!("Error: --data <FILE> is required.");
        print_help();
        return;
    }

    println!("=== TabulaShogi Texel Tuning Solver ===");
    println!("Dataset: {data_path}");
    println!("Epochs: {epochs}, Learning Rate: {lr}, K: {k}");
    println!("Loading dataset...");

    let entries = match DatasetHandler::load_from_file(&data_path) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("Failed to read dataset file '{data_path}': {err}");
            return;
        }
    };

    if entries.is_empty() {
        eprintln!("Dataset contains 0 valid entries.");
        return;
    }

    println!("Loaded {} entries. Extracting features...", entries.len());

    let mut features = Vec::with_capacity(entries.len());
    for entry in &entries {
        if let Ok(pos) = Position::from_sfen(&entry.sfen) {
            features.push(PositionFeatures::extract(&pos, entry.result as f64));
        }
    }

    println!("Extracted {} valid positions for training.", features.len());
    println!("Optimizing parameters via Adam...");

    let initial = TunableParams::default();
    let (tuned, initial_loss, final_loss) =
        TexelTuner::train_adam(&features, &initial, epochs, lr, k);

    let reduction = ((initial_loss - final_loss) / initial_loss.max(1e-9)) * 100.0;
    println!("--------------------------------------");
    println!("Optimization Complete!");
    println!("Initial MSE Loss: {:.6}", initial_loss);
    println!(
        "Final MSE Loss:   {:.6} (Reduction: {:.2}%)",
        final_loss, reduction
    );
    println!("--------------------------------------");
    println!("--- Tuned Piece Values ---");
    let names = [
        "Pawn",
        "Lance",
        "Knight",
        "Silver",
        "Gold",
        "Bishop",
        "Rook",
        "ProPawn",
        "ProLance",
        "ProKnight",
        "ProSilver",
        "Horse",
        "Dragon",
    ];
    for (name, (&old_v, &new_v)) in names
        .iter()
        .zip(initial.piece_values.iter().zip(tuned.piece_values.iter()))
    {
        println!(
            "{:10}: {:6.1} -> {:6.1} ({:+5.1})",
            name,
            old_v,
            new_v,
            new_v - old_v
        );
    }
    println!("--- Tuned Hand Values ---");
    let hand_names = [
        "Pawn", "Lance", "Knight", "Silver", "Gold", "Bishop", "Rook",
    ];
    for (name, (&old_v, &new_v)) in hand_names
        .iter()
        .zip(initial.hand_values.iter().zip(tuned.hand_values.iter()))
    {
        println!(
            "{:10}: {:6.1} -> {:6.1} ({:+5.1})",
            name,
            old_v,
            new_v,
            new_v - old_v
        );
    }
    println!(
        "Tempo     : {:6.1} -> {:6.1} ({:+5.1})",
        initial.tempo,
        tuned.tempo,
        tuned.tempo - initial.tempo
    );
    println!("======================================");
}

fn print_help() {
    println!(
        r#"TabulaShogi Texel Tuning Solver
USAGE:
    tabula-shogi tune --data <PATH> [OPTIONS]

OPTIONS:
        --data <PATH>    Path to training dataset TSV file [REQUIRED]
    -e, --epochs <N>     Optimization epochs [default: 50]
        --lr <FLOAT>     Learning rate [default: 1.0]
        --k <FLOAT>      Logistic scale factor K [default: 400.0]
    -h, --help           Print this help message
"#
    );
}
