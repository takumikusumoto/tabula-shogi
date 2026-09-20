use super::config::SelfPlayConfig;
use super::dataset::{DatasetEntry, DatasetHandler};
use super::game::SimpleRng;
use super::manager::{SelfPlayManager, SelfPlayStats};
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// 分割型自己対局データセット生成の設定
#[derive(Debug, Clone)]
pub struct PartitionConfig {
    /// 全体の総生成対局数 (例: 1,000,000)
    pub total_games: usize,
    /// 1パーティションあたりの対局数 (例: 10,000)
    pub games_per_partition: usize,
    /// データセット出力先ディレクトリ (例: "data/titan")
    pub output_dir: String,
    /// 各パーティションで共通利用する自己対局基本設定
    pub base_config: SelfPlayConfig,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            total_games: 100,
            games_per_partition: 20,
            output_dir: "data/titan".to_string(),
            base_config: SelfPlayConfig {
                num_games: 20,
                threads: 4,
                depth: 2,
                random_opening_plies: 8,
                max_plies: 300,
                resign_threshold: -2500,
                csa_output: None,
                data_output: None,
                tt_size_mb: 16,
                seed: 0x9E3779B97F4A7C15,
                eval_mode: crate::eval::EvalMode::Hce,
                temperature_plies: 24,
                use_book: false,
                start_game_id: 0,
            },
        }
    }
}

/// 分割型自己対局セッション全体の集約統計
#[derive(Debug, Default, Clone)]
pub struct PartitionedSessionStats {
    pub total_partitions: usize,
    pub completed_partitions: usize,
    pub skipped_partitions: usize,
    pub total_games_completed: usize,
    pub total_io_errors: usize,
    pub total_duration_secs: f64,
}

impl PartitionedSessionStats {
    /// 全パーティションが I/O エラーなく 100% 正常完了したかを判定
    pub fn is_success(&self) -> bool {
        self.completed_partitions == self.total_partitions && self.total_io_errors == 0
    }
}

/// 大規模自己対局の分割生成マネージャー
pub struct PartitionedSelfPlayManager;

impl PartitionedSelfPlayManager {
    /// パーティションTSVファイルのパスを生成
    pub fn partition_tsv_path(dir: &str, part_idx: usize) -> PathBuf {
        Path::new(dir).join(format!("part_{part_idx:04}.tsv"))
    }

    /// パーティション一時生成ファイルのパスを生成 (.tmp.tsv)
    pub fn partition_tmp_tsv_path(dir: &str, part_idx: usize) -> PathBuf {
        Path::new(dir).join(format!("part_{part_idx:04}.tmp.tsv"))
    }

    /// パーティション完了マーカーファイルのパスを生成
    pub fn partition_done_path(dir: &str, part_idx: usize) -> PathBuf {
        Path::new(dir).join(format!("part_{part_idx:04}.done"))
    }

    /// 指定ディレクトリ内の全完了パーティションTSVファイル一覧を取得（連番ソート済み）
    pub fn discover_completed_partitions(dir: &str) -> io::Result<Vec<PathBuf>> {
        let dir_path = Path::new(dir);
        if !dir_path.exists() {
            return Ok(Vec::new());
        }

        let mut tsv_files = Vec::new();
        for entry in fs::read_dir(dir_path)? {
            let entry = entry?;
            let path = entry.path();
            if let Some(ext) = path.extension()
                && ext == "tsv"
            {
                let done_path = path.with_extension("done");
                if done_path.exists() {
                    tsv_files.push(path);
                }
            }
        }

        tsv_files.sort();
        Ok(tsv_files)
    }

    /// 分割自己対局セッションを実行（Resume対応・中断耐性）
    pub fn run(config: PartitionConfig) -> PartitionedSessionStats {
        let overall_start = Instant::now();
        fs::create_dir_all(&config.output_dir).expect("Failed to create output directory");

        let games_per_part = config.games_per_partition.max(1);
        let num_parts = config.total_games.div_ceil(games_per_part);

        println!("========================================================");
        println!("=== TabulaShogi Partitioned Self-Play Pipeline (Titan) ===");
        println!(
            "Total Games: {}, Games/Partition: {}, Partitions: {}",
            config.total_games, games_per_part, num_parts
        );
        println!("Output Directory: {}", config.output_dir);
        println!(
            "Threads: {}, Depth: {}, Base Seed: 0x{:X}",
            config.base_config.threads, config.base_config.depth, config.base_config.seed
        );
        println!("========================================================");

        let mut stats = PartitionedSessionStats {
            total_partitions: num_parts,
            ..Default::default()
        };

        for part_idx in 0..num_parts {
            let tsv_path = Self::partition_tsv_path(&config.output_dir, part_idx);
            let tmp_tsv_path = Self::partition_tmp_tsv_path(&config.output_dir, part_idx);
            let done_path = Self::partition_done_path(&config.output_dir, part_idx);

            let start_game_id = part_idx * games_per_part;
            let games_this_part = if part_idx == num_parts - 1 {
                let rem = config.total_games % games_per_part;
                if rem == 0 { games_per_part } else { rem }
            } else {
                games_per_part
            };

            let expected_signature = format!(
                "part={part_idx}\ngames={games_this_part}\nstart_id={start_game_id}\ndepth={}\nseed=0x{:X}\n",
                config.base_config.depth, config.base_config.seed
            );

            // 既に完了マーカーが存在し、かつ設定署名が完全一致するか照合（厳格な Resume 機能）
            if done_path.exists() && tsv_path.exists() {
                if let Ok(existing_done) = fs::read_to_string(&done_path)
                    && existing_done.starts_with(&expected_signature)
                {
                    println!(
                        "[Resume] Partition {:04}/{} already completed with matching config. Skipping.",
                        part_idx + 1,
                        num_parts
                    );
                    stats.skipped_partitions += 1;
                    stats.completed_partitions += 1;
                    stats.total_games_completed += games_this_part;
                    continue;
                } else {
                    println!(
                        "[Notice] Partition {:04} exists but configuration mismatched or marker corrupted. Invalidating and regenerating.",
                        part_idx + 1
                    );
                    let _ = fs::remove_file(&done_path);
                    let _ = fs::remove_file(&tsv_path);
                }
            }

            // 中断・不完全な一時ファイルが存在する場合は再生成のためにクリーンアップ
            if tmp_tsv_path.exists() {
                let _ = fs::remove_file(&tmp_tsv_path);
            }
            if tsv_path.exists() {
                let _ = fs::remove_file(&tsv_path);
            }
            if done_path.exists() {
                let _ = fs::remove_file(&done_path);
            }

            let mut part_sp_cfg = config.base_config.clone();
            part_sp_cfg.num_games = games_this_part;
            part_sp_cfg.start_game_id = start_game_id;
            // 直接本番パスではなく一時ファイル .tmp.tsv へ出力
            part_sp_cfg.data_output = Some(tmp_tsv_path.to_string_lossy().to_string());

            println!(
                "\n>>> Starting Partition {:04}/{} (Game IDs: {}..{}) >>>",
                part_idx + 1,
                num_parts,
                start_game_id,
                start_game_id + games_this_part - 1
            );

            let part_stats: SelfPlayStats = SelfPlayManager::run(part_sp_cfg);
            stats.total_io_errors += part_stats.io_errors;
            stats.total_games_completed += part_stats.completed_games;

            // 完了マーカーの作成 & アトミック公開
            if part_stats.io_errors == 0 && part_stats.completed_games == games_this_part {
                // 1. 一時 TSV を本番 TSV へアトミックリネーム
                if let Err(e) = fs::rename(&tmp_tsv_path, &tsv_path) {
                    eprintln!(
                        "[Error] Failed to rename tmp TSV {:?} to {:?}: {e}",
                        tmp_tsv_path, tsv_path
                    );
                    stats.total_io_errors += 1;
                    continue;
                }

                // 2. 厳格な設定署名を含む完了マーカーの作成
                let marker_content = format!(
                    "{expected_signature}plies={}\nio_errors={}\n",
                    part_stats.total_plies, part_stats.io_errors
                );
                if let Err(e) = fs::write(&done_path, marker_content) {
                    eprintln!("[Error] Failed to write done marker {:?}: {e}", done_path);
                    stats.total_io_errors += 1;
                } else {
                    stats.completed_partitions += 1;
                    println!(
                        "[Success] Partition {:04} completed, atomically published, and marked done.",
                        part_idx + 1
                    );
                }
            } else {
                eprintln!(
                    "[Warning] Partition {:04} finished with anomalies (completed {}/{}, io_errors={}). tmp TSV discarded.",
                    part_idx + 1,
                    part_stats.completed_games,
                    games_this_part,
                    part_stats.io_errors
                );
                let _ = fs::remove_file(&tmp_tsv_path);
            }
        }

        stats.total_duration_secs = overall_start.elapsed().as_secs_f64();

        println!("\n========================================================");
        println!("=== Partitioned Self-Play Pipeline Completed ===");
        println!("Total Duration: {:.2}s", stats.total_duration_secs);
        println!(
            "Partitions: {} total, {} completed ({} skipped by resume)",
            stats.total_partitions, stats.completed_partitions, stats.skipped_partitions
        );
        println!("Total Games Completed: {}", stats.total_games_completed);
        println!("Total I/O Errors: {}", stats.total_io_errors);
        println!("========================================================");

        stats
    }
}

/// 複数パーティションTSVファイルを透過的に跨いで省メモリにバッチ読み込みするストリーミングリーダー
pub struct MultiPartitionStreamingReader {
    file_paths: Vec<PathBuf>,
    current_file_idx: usize,
    current_reader: Option<BufReader<File>>,
    batch_size: usize,
    line_buf: String,
}

impl MultiPartitionStreamingReader {
    /// ファイルパス一覧とバッチサイズを指定して新規作成
    pub fn new(file_paths: Vec<PathBuf>, batch_size: usize) -> Self {
        Self {
            file_paths,
            current_file_idx: 0,
            current_reader: None,
            batch_size,
            line_buf: String::new(),
        }
    }

    /// ディレクトリ内の全完了パーティションを検索して新規作成
    pub fn from_directory(dir: &str, batch_size: usize) -> io::Result<Self> {
        let files = PartitionedSelfPlayManager::discover_completed_partitions(dir)?;
        Ok(Self::new(files, batch_size))
    }

    /// 読み込み対象のファイル数を取得
    pub fn file_count(&self) -> usize {
        self.file_paths.len()
    }

    /// 読み込み順序を Fisher-Yates でシャッフル（エポック間でのファイル順偏向を低減）
    pub fn shuffle_files(&mut self, seed: u64) {
        if self.file_paths.len() > 1 {
            let mut rng = SimpleRng::new(if seed == 0 { 0xdeadbeefcafe } else { seed });
            for i in (1..self.file_paths.len()).rev() {
                let j = (rng.next_u64() as usize) % (i + 1);
                self.file_paths.swap(i, j);
            }
        }
        // シャッフル後は先頭からリセット
        self.reset();
    }

    /// 読み込み位置を全ファイルの先頭に巻き戻す
    pub fn reset(&mut self) {
        self.current_file_idx = 0;
        self.current_reader = None;
        self.line_buf.clear();
    }

    /// 次のバッチを読み込む。全ファイルの末尾に達した場合は None を返す。
    pub fn next_batch(&mut self) -> io::Result<Option<Vec<DatasetEntry>>> {
        let mut batch = Vec::with_capacity(self.batch_size);

        while batch.len() < self.batch_size {
            // 現在のリーダーが存在しない、または EOF に達した場合は次のファイルを開く
            if self.current_reader.is_none() {
                if self.current_file_idx >= self.file_paths.len() {
                    break; // 全ファイル終了
                }
                let next_path = &self.file_paths[self.current_file_idx];
                let file = File::open(next_path)?;
                self.current_reader = Some(BufReader::new(file));
                self.current_file_idx += 1;
            }

            let reader = self.current_reader.as_mut().unwrap();
            self.line_buf.clear();
            let bytes_read = reader.read_line(&mut self.line_buf)?;

            if bytes_read == 0 {
                // 現在のファイルが EOF に達した
                self.current_reader = None;
                continue;
            }

            if let Some(entry) = DatasetHandler::parse_entry(&self.line_buf) {
                batch.push(entry);
            }
        }

        if batch.is_empty() {
            Ok(None)
        } else {
            Ok(Some(batch))
        }
    }
}
