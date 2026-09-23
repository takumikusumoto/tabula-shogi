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
        "export-halfkp" => {
            run_export_halfkp(&args[2..]);
        }
        "match" => {
            run_match(&args[2..]);
        }
        "loop" => {
            run_loop(&args[2..]);
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

fn run_export_halfkp(args: &[String]) {
    use tabula_shogi::board::Position;
    use tabula_shogi::eval::halfkp::{
        HALFKP_BIAS_SCALE, HALFKP_FEATURE_SCALE, HALFKP_MAGIC, HALFKP_OUTPUT_SCALE, HalfKPEvaluator,
    };
    use tabula_shogi::eval::halfkp_trainer::HalfKPTrainer;
    use tabula_shogi::movegen::MoveGenerator;
    use tabula_shogi::selfplay::game::SimpleRng;

    let parser = ArgParser::new(args);
    if parser.has_flag("--help", Some("-h")) {
        println!(
            r#"TabulaShogi HalfKP Checkpoint Exporter
USAGE:
    tabula-shogi export-halfkp [OPTIONS]

OPTIONS:
        --checkpoint <PATH>   Float AdamW checkpoint [default: models/candidate_halfkp_ckpt.bin]
    -o, --out <PATH>          TABU_HK2 quantized model [default: models/candidate_halfkp.bin]
        --samples <N>         Deterministic positions for float/int verification [default: 128]
    -h, --help                Print this help message
"#
        );
        return;
    }

    let checkpoint_path = parser
        .get_string("--checkpoint", None)
        .unwrap_or_else(|| "models/candidate_halfkp_ckpt.bin".to_string());
    let output_path = parser
        .get_string("--out", Some("-o"))
        .unwrap_or_else(|| "models/candidate_halfkp.bin".to_string());
    let sample_count = parser.get_value("--samples", None).unwrap_or(128usize);

    let checkpoint_canonical = std::fs::canonicalize(&checkpoint_path).unwrap_or_else(|e| {
        eprintln!("Error: cannot resolve checkpoint '{checkpoint_path}': {e}");
        std::process::exit(1);
    });
    if let Ok(output_canonical) = std::fs::canonicalize(&output_path)
        && checkpoint_canonical == output_canonical
    {
        eprintln!(
            "Error: checkpoint and output paths must be different; refusing to overwrite the checkpoint"
        );
        std::process::exit(1);
    }

    println!("=== TabulaShogi HalfKP TABU_HK2 Export ===");
    println!("Checkpoint: {checkpoint_path}");
    println!("Output:     {output_path}");
    println!(
        "Scales: feature={}, output={}, bias={}",
        HALFKP_FEATURE_SCALE, HALFKP_OUTPUT_SCALE, HALFKP_BIAS_SCALE
    );

    let trainer = HalfKPTrainer::load_checkpoint(&checkpoint_path).unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });
    let evaluator = trainer.to_evaluator();
    let stats = trainer.quantization_stats(&evaluator);

    let mut pos = Position::startpos();
    let mut rng = SimpleRng::new(0x243f6a8885a308d3);
    let mut total_abs_error = 0.0f64;
    let mut max_abs_error = 0.0f32;
    let samples = sample_count.max(1);
    for _ in 0..samples {
        let mover = HalfKPEvaluator::extract_halfkp_features(&pos, pos.side_to_move);
        let opponent = HalfKPEvaluator::extract_halfkp_features(&pos, pos.side_to_move.opposite());
        let (float_cp, _, _, _, _, _) = trainer.forward(&mover, &opponent);
        let integer_cp = evaluator.evaluate(&pos);
        let abs_error = (float_cp - integer_cp as f32).abs();
        total_abs_error += abs_error as f64;
        max_abs_error = max_abs_error.max(abs_error);

        let legal_moves = MoveGenerator::generate_legal_moves(&mut pos);
        if legal_moves.is_empty() || pos.repetition_count() >= 4 {
            pos = Position::startpos();
        } else {
            let mv = legal_moves[rng.gen_range(legal_moves.len())];
            pos.do_move(mv);
        }
    }
    let mae = total_abs_error / samples as f64;

    evaluator.save_to_file(&output_path).unwrap_or_else(|e| {
        eprintln!("Error: failed to save HalfKP model '{output_path}': {e}");
        std::process::exit(1);
    });

    let feature_total = stats.feature_values.max(1) as f64;
    println!(
        "Feature non-zero: float={}/{} ({:.4}%), TABU_HK2={}/{} ({:.4}%)",
        stats.float_nonzero_features,
        stats.feature_values,
        stats.float_nonzero_features as f64 * 100.0 / feature_total,
        stats.quantized_nonzero_features,
        stats.feature_values,
        stats.quantized_nonzero_features as f64 * 100.0 / feature_total
    );
    println!(
        "Changed from deterministic initialization: float={}/{} ({:.4}%), legacy-scale-survivors={}/{} ({:.4}%), TABU_HK2-survivors={}/{} ({:.4}%)",
        stats.float_changed_from_initial,
        stats.feature_values,
        stats.float_changed_from_initial as f64 * 100.0 / feature_total,
        stats.legacy_changed_from_initial,
        stats.feature_values,
        stats.legacy_changed_from_initial as f64 * 100.0 / feature_total,
        stats.scaled_changed_from_initial,
        stats.feature_values,
        stats.scaled_changed_from_initial as f64 * 100.0 / feature_total
    );
    println!(
        "Saturation: feature={}, feature_bias={}, output={}, output_bias={}",
        stats.feature_saturations,
        stats.feature_bias_saturations,
        stats.output_saturations,
        usize::from(stats.output_bias_saturated)
    );
    println!(
        "Output weights non-zero: {}/{}",
        stats.output_nonzero,
        evaluator.output_weights.len()
    );
    println!(
        "Float/int verification: samples={}, MAE={:.6} cp, max_abs_error={:.6} cp",
        samples, mae, max_abs_error
    );

    drop(evaluator);
    let loaded = HalfKPEvaluator::load_from_file(&output_path).unwrap_or_else(|e| {
        eprintln!("Error: exported model failed reload validation: {e}");
        std::process::exit(1);
    });
    let bytes = std::fs::metadata(&output_path)
        .map(|m| m.len())
        .unwrap_or(0);
    println!(
        "Reload validation: magic={}, bytes={}, startpos_eval={} cp",
        String::from_utf8_lossy(HALFKP_MAGIC),
        bytes,
        loaded.evaluate(&Position::startpos())
    );
    println!("Export completed successfully.");
}

struct ArgParser<'a> {
    args: &'a [String],
}

impl<'a> ArgParser<'a> {
    fn new(args: &'a [String]) -> Self {
        Self { args }
    }

    fn has_flag(&self, long: &str, short: Option<&str>) -> bool {
        self.args
            .iter()
            .any(|arg| arg == long || short.map(|s| arg == s).unwrap_or(false))
    }

    fn get_value<T: std::str::FromStr>(&self, long: &str, short: Option<&str>) -> Option<T> {
        let mut i = 0;
        while i < self.args.len() {
            let is_match =
                self.args[i] == long || short.map(|s| self.args[i] == s).unwrap_or(false);
            if is_match && i + 1 < self.args.len() {
                return match self.args[i + 1].parse::<T>() {
                    Ok(val) => Some(val),
                    Err(_) => {
                        eprintln!(
                            "Warning: Invalid value for '{}': '{}'",
                            long,
                            self.args[i + 1]
                        );
                        None
                    }
                };
            }
            i += 1;
        }
        None
    }

    fn get_string(&self, long: &str, short: Option<&str>) -> Option<String> {
        let mut i = 0;
        while i < self.args.len() {
            let is_match =
                self.args[i] == long || short.map(|s| self.args[i] == s).unwrap_or(false);
            if is_match && i + 1 < self.args.len() {
                return Some(self.args[i + 1].clone());
            }
            i += 1;
        }
        None
    }
}

fn run_train_nnue(args: &[String]) {
    use tabula_shogi::eval::NNUETrainer;
    use tabula_shogi::selfplay::DatasetHandler;

    let parser = ArgParser::new(args);
    if parser.has_flag("--help", Some("-h")) {
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

    let data_path = parser.get_string("--data", None).unwrap_or_default();
    let out_path = parser
        .get_string("--out", Some("-o"))
        .unwrap_or_else(|| "nnue.bin".to_string());
    let epochs = parser.get_value("--epochs", Some("-e")).unwrap_or(20);
    let lr = parser.get_value("--lr", None).unwrap_or(0.001f32);
    let batch_size = parser.get_value("--batch-size", Some("-b")).unwrap_or(64);
    let k = parser.get_value("--k", None).unwrap_or(400.0f32);

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

fn parse_eval_mode(desc: &str) -> (String, tabula_shogi::eval::EvalMode) {
    if desc.eq_ignore_ascii_case("hce") {
        ("HCE".to_string(), tabula_shogi::eval::EvalMode::Hce)
    } else if desc.eq_ignore_ascii_case("nnue") {
        (
            "NNUE(built-in)".to_string(),
            tabula_shogi::eval::EvalMode::Nnue(std::sync::Arc::new(
                tabula_shogi::eval::NNUEEvaluator::new(),
            )),
        )
    } else if desc.eq_ignore_ascii_case("halfkp") {
        (
            "HalfKP(built-in)".to_string(),
            tabula_shogi::eval::EvalMode::HalfKP(std::sync::Arc::new(
                tabula_shogi::eval::HalfKPEvaluator::new(),
            )),
        )
    } else {
        let path = std::path::Path::new(desc);
        if !path.exists() {
            eprintln!("Error: Evaluation model file '{desc}' does not exist (fail-closed).");
            std::process::exit(1);
        }

        // 1. Try HalfKP
        match tabula_shogi::eval::HalfKPEvaluator::load_from_file(desc) {
            Ok(halfkp) => (
                desc.to_string(),
                tabula_shogi::eval::EvalMode::HalfKP(std::sync::Arc::new(halfkp)),
            ),
            Err(e_hkp) => {
                // 2. Try NNUE
                match tabula_shogi::eval::NNUEEvaluator::load_from_file(desc) {
                    Ok(nnue) => (
                        desc.to_string(),
                        tabula_shogi::eval::EvalMode::Nnue(std::sync::Arc::new(nnue)),
                    ),
                    Err(e_nnue) => {
                        eprintln!(
                            "Error: Failed to load evaluation model from '{desc}'. Neither HalfKP ({e_hkp}) nor NNUE ({e_nnue}) could be loaded. Aborting (fail-closed)."
                        );
                        std::process::exit(1);
                    }
                }
            }
        }
    }
}

fn run_match(args: &[String]) {
    use tabula_shogi::arena::{MatchConfig, MatchRunner, SprtConfig};

    let parser = ArgParser::new(args);
    if parser.has_flag("--help", Some("-h")) {
        println!("TabulaShogi Arena Match");
        println!("USAGE:\n    tabula-shogi match [OPTIONS]");
        println!("OPTIONS:");
        println!("    --engine1 <HCE|NNUE|HALFKP|PATH>   First engine model [default: HCE]");
        println!("    --engine2 <HCE|NNUE|HALFKP|PATH>   Second engine model [default: HalfKP]");
        println!(
            "    -p, --pairs <N>                    Number of game pairs [default: 20] (total = 2*pairs)"
        );
        println!("    -d, --depth <D>                    Search depth [default: 2]");
        println!("    -t, --threads <T>                  Worker threads [default: 2]");
        println!("    -o, --opening <K>                  Random opening plies [default: 6]");
        println!("        --generation <N>               Opening seed generation [default: 0]");
        return;
    }

    let engine1_desc = parser
        .get_value("--engine1", None)
        .unwrap_or_else(|| "HCE".to_string());
    let engine2_desc = parser
        .get_value("--engine2", None)
        .unwrap_or_else(|| "HalfKP".to_string());
    let pairs = parser.get_value("--pairs", Some("-p")).unwrap_or(20);
    let depth = parser.get_value("--depth", Some("-d")).unwrap_or(2);
    let threads = parser.get_value("--threads", Some("-t")).unwrap_or(2);
    let opening = parser.get_value("--opening", Some("-o")).unwrap_or(6);
    let generation = parser.get_value("--generation", None).unwrap_or(0);

    let (name_a, eval_a) = parse_eval_mode(&engine1_desc);
    let (name_b, eval_b) = parse_eval_mode(&engine2_desc);

    let config = MatchConfig {
        name_a,
        name_b,
        eval_a,
        eval_b,
        pairs,
        depth,
        threads,
        generation,
        random_opening: opening,
        max_plies: 320,
        tt_size_mb: 16,
        sprt_config: Some(SprtConfig::default()),
    };

    MatchRunner::run_match(&config);
}

fn run_loop(args: &[String]) {
    use tabula_shogi::arena::{LoopConfig, SelfImprovementLoop};

    let parser = ArgParser::new(args);
    if parser.has_flag("--help", Some("-h")) {
        println!("TabulaShogi Autonomous Self-Improvement Loop (HalfKP)");
        println!("USAGE:\n    tabula-shogi loop [OPTIONS]");
        println!("OPTIONS:");
        println!("    -i, --iterations <N>    Number of improvement generations [default: 3]");
        println!("    -g, --games <N>         Self-play games per generation [default: 120]");
        println!("    -p, --eval-pairs <N>    Evaluation game pairs per generation [default: 15]");
        println!("    -t, --threads <T>       Worker threads [default: 2]");
        println!("    -d, --depth <D>         Search depth [default: 2]");
        println!("    -e, --epochs <E>        HalfKP training epochs per generation [default: 3]");
        println!("        --lr <FLOAT>        Learning rate [default: 0.001]");
        println!("    -b, --batch-size <N>    Mini-batch size [default: 1024]");
        println!(
            "        --data <PATH>       Path to cumulative training dataset [default: data/loop_dataset.tsv]"
        );
        println!(
            "        --deep-data <PATH>  Path to deep distilled position pool [default: data/deep_dataset.tsv]"
        );
        println!(
            "        --best <PATH>       Path to best model binary [default: models/best_halfkp.bin]"
        );
        println!(
            "        --candidate <PATH>  Path to candidate model binary [default: models/candidate_halfkp.bin]"
        );
        println!(
            "        --candidate-ckpt <PATH> Path to candidate checkpoint [default: models/candidate_halfkp_ckpt.bin]"
        );
        println!(
            "        --min-games <N>     Minimum evaluation games for promotion [default: 20]"
        );
        println!(
            "        --summary <PATH>    Path to CSV progress summary log [default: data/loop_summary.csv]"
        );
        println!("        --start-iter <N>    Explicit generation starting number");
        println!(
            "        --state <PATH>      Path to generation state file [default: data/loop_state.txt]"
        );
        return;
    }

    let mut config = LoopConfig::default();
    if let Some(val) = parser.get_value("--iterations", Some("-i")) {
        config.iterations = val;
    }
    if let Some(val) = parser.get_value("--games", Some("-g")) {
        config.games_per_iteration = val;
    }
    if let Some(val) = parser.get_value("--eval-pairs", Some("-p")) {
        config.arena.eval_pairs = val;
    }
    if let Some(val) = parser.get_value("--threads", Some("-t")) {
        config.arena.threads = val;
    }
    if let Some(val) = parser.get_value("--depth", Some("-d")) {
        config.arena.depth = val;
    }
    if let Some(val) = parser.get_value("--epochs", Some("-e")) {
        config.training.epochs = val;
    }
    if let Some(val) = parser.get_value("--lr", None) {
        config.training.lr = val;
    }
    if let Some(val) = parser.get_value("--batch-size", Some("-b")) {
        config.training.batch_size = val;
    }
    if let Some(val) = parser.get_string("--data", None) {
        config.paths.data_path = val;
    }
    if let Some(val) = parser.get_string("--deep-data", None) {
        config.paths.deep_data_path = val;
    }
    if let Some(val) = parser.get_string("--best", None) {
        config.paths.best_model_path = val;
    }
    if let Some(val) = parser.get_string("--candidate", None) {
        config.paths.candidate_model_path = val;
    }
    if let Some(val) = parser.get_string("--candidate-ckpt", None) {
        config.paths.candidate_ckpt_path = val;
    }
    if let Some(val) = parser.get_value("--min-games", None) {
        config.arena.min_promotion_games = val;
    }
    if let Some(val) = parser.get_string("--summary", None) {
        config.paths.summary_path = val;
    }
    if let Some(val) = parser.get_value("--start-iter", None) {
        config.start_iteration = Some(val);
    }
    if let Some(val) = parser.get_string("--state", None) {
        config.paths.state_path = val;
    }

    if let Err(e) = SelfImprovementLoop::run(&config) {
        eprintln!("{e}");
        std::process::exit(1);
    }
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
    export-halfkp Export a float HalfKP checkpoint to scaled TABU_HK2 weights
    match       Arena game-pair match between two models with SPRT testing
    loop        Full autonomous self-improvement loop (selfplay -> train -> match -> promote)
    bench       Run search performance benchmark on standard positions

OPTIONS:
    -h, --help       Print this help message
    -v, --version    Print version information

Run 'tabula-shogi <COMMAND> --help' for details on a specific command.
"#,
        env!("CARGO_PKG_VERSION")
    );
}
