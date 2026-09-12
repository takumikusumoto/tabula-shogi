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
    usi        Run USI (Universal Shogi Interface) engine loop [DEFAULT]
    selfplay   Autonomous self-play generation pipeline (games, CSA, dataset)
    tune       Texel Tuning solver for evaluation parameter optimization
    bench      Run search performance benchmark on standard positions

OPTIONS:
    -h, --help      Print this help message
    -v, --version   Print version information

Run 'tabula-shogi <COMMAND> --help' for details on a specific command.
"#,
        env!("CARGO_PKG_VERSION")
    );
}
