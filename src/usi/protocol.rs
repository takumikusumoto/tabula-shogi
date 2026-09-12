use super::parse::UsiCommand;
use crate::board::Position;
use crate::eval::NNUEEvaluator;
use crate::search::{SearchEngine, TimeControl};
use std::io::{self, BufRead};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

pub struct UsiHandler {
    pos: Position,
    engine: SearchEngine,
    threads: usize,
    tt_size_mb: usize,
    stop_flag: Arc<AtomicBool>,
    search_handle: Option<thread::JoinHandle<()>>,
}

impl Default for UsiHandler {
    fn default() -> Self {
        Self::new()
    }
}

pub const DEFAULT_TT_SIZE_MB: usize = 64;
pub const DEFAULT_THREADS: usize = 1;

impl UsiHandler {
    pub fn new() -> Self {
        UsiHandler {
            pos: Position::startpos(),
            engine: SearchEngine::new(DEFAULT_TT_SIZE_MB),
            threads: DEFAULT_THREADS,
            tt_size_mb: DEFAULT_TT_SIZE_MB,
            stop_flag: Arc::new(AtomicBool::new(false)),
            search_handle: None,
        }
    }

    /// メインUSIループ
    pub fn run(&mut self) {
        let stdin = io::stdin();
        let mut lines = stdin.lock().lines();

        while let Some(Ok(line)) = lines.next() {
            let cmd = UsiCommand::parse(&line);
            if !self.process_command(cmd) {
                break;
            }
        }
    }

    /// 単一のUSIコマンドを処理 (継続する場合はtrue, 終了時はfalse)
    pub fn process_command(&mut self, cmd: UsiCommand) -> bool {
        match cmd {
            UsiCommand::Usi => {
                println!("id name TabulaShogi {}", env!("CARGO_PKG_VERSION"));
                println!("id author Takumi Kusumoto");
                println!("option name USI_Hash type spin default 64 min 1 max 8192");
                println!(
                    "option name Threads type spin default {} min 1 max 64",
                    self.threads
                );
                println!("option name Eval_Type type combo default HCE var HCE var NNUE");
                println!("option name NNUE_File type string default <empty>");
                println!("usiok");
            }
            UsiCommand::IsReady => {
                println!("readyok");
            }
            UsiCommand::SetOption { name, value } => {
                if name.eq_ignore_ascii_case("usi_hash") {
                    if let Ok(mb) = value.parse::<usize>() {
                        self.tt_size_mb = mb;
                        let current_eval = self.engine.eval_mode.clone();
                        self.engine = SearchEngine::new(mb).with_eval_mode(current_eval);
                    }
                } else if name.eq_ignore_ascii_case("threads")
                    && let Ok(t) = value.parse::<usize>()
                {
                    self.threads = t.clamp(1, 64);
                } else if name.eq_ignore_ascii_case("eval_type") {
                    if value.eq_ignore_ascii_case("nnue") {
                        let nnue = NNUEEvaluator::new();
                        self.engine.eval_mode = crate::eval::EvalMode::Nnue(Arc::new(nnue));
                        println!("info string Evaluation mode switched to NNUE (built-in)");
                    } else {
                        self.engine.eval_mode = crate::eval::EvalMode::Hce;
                        println!("info string Evaluation mode switched to HCE");
                    }
                } else if name.eq_ignore_ascii_case("nnue_file") {
                    match NNUEEvaluator::load_from_file(&value) {
                        Ok(nnue) => {
                            self.engine.eval_mode = crate::eval::EvalMode::Nnue(Arc::new(nnue));
                            println!("info string Loaded NNUE weights from {value}");
                        }
                        Err(e) => {
                            eprintln!("Error loading NNUE file '{value}': {e}");
                        }
                    }
                }
            }
            UsiCommand::UsiNewGame => {
                self.stop_search_if_running();
                self.engine.clear();
                self.pos = Position::startpos();
            }
            UsiCommand::Position { sfen, moves } => {
                self.stop_search_if_running();

                if let Some(sfen_str) = sfen {
                    match Position::from_sfen(&sfen_str) {
                        Ok(p) => self.pos = p,
                        Err(e) => eprintln!("Error parsing SFEN: {e}"),
                    }
                } else {
                    self.pos = Position::startpos();
                }

                for mv in moves {
                    self.pos.do_move(mv);
                }
            }
            UsiCommand::Go(tc) => {
                self.start_search(tc);
            }
            UsiCommand::GoMate(_limit) => {
                self.start_mate_search();
            }
            UsiCommand::Stop => {
                self.stop_search_if_running();
            }
            UsiCommand::Quit => {
                self.stop_search_if_running();
                return false;
            }
            UsiCommand::GameOver(_) => {
                self.stop_search_if_running();
            }
            UsiCommand::Eval => match &self.engine.eval_mode {
                crate::eval::EvalMode::Hce => {
                    let breakdown = crate::eval::Evaluator::evaluate_detailed(&self.pos);
                    println!("info string --- HCE Evaluation Breakdown ---");
                    println!(
                        "info string material_board: {:+6} cp",
                        breakdown.material_board
                    );
                    println!(
                        "info string material_hand:  {:+6} cp",
                        breakdown.material_hand
                    );
                    println!(
                        "info string piece_square:   {:+6} cp",
                        breakdown.piece_square
                    );
                    println!(
                        "info string king_safety:    {:+6} cp",
                        breakdown.king_safety
                    );
                    println!(
                        "info string king_danger:    {:+6} cp",
                        breakdown.king_danger
                    );
                    println!("info string mobility:       {:+6} cp", breakdown.mobility);
                    println!(
                        "info string coordination:   {:+6} cp",
                        breakdown.coordination
                    );
                    println!("info string outposts:       {:+6} cp", breakdown.outposts);
                    println!("info string tempo:          {:+6} cp", breakdown.tempo);
                    println!("info string -------------------------------");
                    println!("info string total:          {:+6} cp", breakdown.total);
                    println!("eval {}", breakdown.total);
                }
                crate::eval::EvalMode::Nnue(nnue) => {
                    let score = nnue.evaluate(&self.pos);
                    println!("info string --- NNUE Evaluation ---");
                    println!("info string total:          {:+6} cp", score);
                    println!("eval {}", score);
                }
            },
            UsiCommand::Unknown(_) => {}
        }
        true
    }

    /// 現在の評価モードを取得
    pub fn eval_mode(&self) -> &crate::eval::EvalMode {
        &self.engine.eval_mode
    }

    fn start_search(&mut self, tc: TimeControl) {
        self.stop_search_if_running();

        self.stop_flag.store(false, Ordering::Relaxed);
        let stop_flag = Arc::clone(&self.stop_flag);
        let pos = self.pos.clone();
        let num_threads = self.threads;
        let eval_mode = self.engine.eval_mode.clone();

        // Astra 6 レビュー F11: 探索ごとにTTを作り直さず、対局中永続化した共有TTを再利用
        let shared_tt = Arc::clone(&self.engine.tt);

        // マルチスレッド Lazy SMP 探索
        let handle = thread::spawn(move || {
            let mut handles = Vec::with_capacity(num_threads);

            // メイン探索スレッド (Thread 0)
            let main_tt = Arc::clone(&shared_tt);
            let mut main_pos = pos.clone();
            let main_tc = tc;
            let main_stop = Arc::clone(&stop_flag);
            let main_eval = eval_mode.clone();

            let main_handle = thread::spawn(move || {
                let mut searcher = SearchEngine::with_shared_tt(main_tt).with_eval_mode(main_eval);
                searcher.search(&mut main_pos, &main_tc, main_stop)
            });

            // ヘルパースレッド (Thread 1..num_threads)
            if num_threads > 1 {
                for thread_id in 1..num_threads {
                    let helper_tt = Arc::clone(&shared_tt);
                    let mut helper_pos = pos.clone();
                    let helper_tc = tc;
                    let helper_stop = Arc::clone(&stop_flag);
                    let helper_eval = eval_mode.clone();

                    let h = thread::spawn(move || {
                        let mut searcher =
                            SearchEngine::with_shared_tt(helper_tt).with_eval_mode(helper_eval);
                        searcher.search_helper(&mut helper_pos, &helper_tc, helper_stop, thread_id);
                    });
                    handles.push(h);
                }
            }

            let best_move = main_handle.join().unwrap_or(None);

            // メイン探索が終了したらヘルパースレッドに停止フラグを通知
            stop_flag.store(true, Ordering::Relaxed);
            for h in handles {
                let _ = h.join();
            }

            if let Some(mv) = best_move {
                println!("bestmove {}", mv.to_usi());
            } else {
                println!("bestmove resign");
            }
        });

        self.search_handle = Some(handle);
    }

    fn start_mate_search(&mut self) {
        self.stop_search_if_running();

        let mut pos = self.pos.clone();
        let handle = thread::spawn(move || {
            let mut solver = crate::search::DfpnSolver::new(500_000);
            let (is_mate, mate_move) = solver.solve(&mut pos);
            if is_mate {
                if let Some(mv) = mate_move {
                    println!("checkmate {}", mv.to_usi());
                } else {
                    println!("checkmate notimplemented");
                }
            } else {
                println!("checkmate nomate");
            }
        });

        self.search_handle = Some(handle);
    }

    fn stop_search_if_running(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.search_handle.take() {
            let _ = handle.join();
        }
    }
}
