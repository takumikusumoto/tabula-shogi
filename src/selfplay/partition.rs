use super::config::SelfPlayConfig;
use super::dataset::DatasetBatchReader;
use super::manager::SelfPlayManager;
use std::fs::{self, File};
use std::io::{self, ErrorKind, Write};
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
    pub total_duration_secs: f64,
}

impl DatasetBatchReader {
    /// ディレクトリ内の全完了パーティション TSV を検索して新規作成
    /// (`dataset -> partition` の循環依存を防ぐため、partition モジュール側で定義)
    pub fn from_partition_dir(dir: &str, batch_size: usize) -> io::Result<Self> {
        let files = PartitionedSelfPlayManager::discover_completed_partitions(dir)?;
        Self::from_files(files, batch_size)
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
        if !path_exists_fail_closed(dir_path)? {
            return Ok(Vec::new());
        }

        let mut tsv_files = Vec::new();
        for entry in fs::read_dir(dir_path)? {
            let entry = entry?;
            let path = entry.path();
            let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if file_name.ends_with(".tmp.tsv") {
                continue;
            }
            if let Some(ext) = path.extension()
                && ext == "tsv"
            {
                let done_path = path.with_extension("done");
                if path_exists_fail_closed(&done_path)? {
                    tsv_files.push(path);
                }
            }
        }

        tsv_files.sort();
        Ok(tsv_files)
    }

    /// 分割自己対局セッションを fail-closed に実行（Resume対応・中断耐性）
    pub fn run(config: PartitionConfig) -> Result<PartitionedSessionStats, String> {
        if config.output_dir.trim().is_empty() {
            return Err("Partition output_dir must not be empty".to_string());
        }
        let overall_start = Instant::now();
        fs::create_dir_all(&config.output_dir).map_err(|e| {
            format!(
                "Failed to create partition output directory '{}': {e}",
                config.output_dir
            )
        })?;

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
            if can_resume_partition(&done_path, &tsv_path, &expected_signature)? {
                println!(
                    "[Resume] Partition {:04}/{} already completed with matching config. Skipping.",
                    part_idx + 1,
                    num_parts
                );
                stats.skipped_partitions += 1;
                stats.completed_partitions += 1;
                stats.total_games_completed += games_this_part;
                continue;
            }

            // 中断・不完全・設定不一致な既存ファイルを再生成前にクリーンアップ（fail-closed）
            remove_if_exists(&tmp_tsv_path)?;
            remove_if_exists(&tsv_path)?;
            remove_if_exists(&done_path)?;

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

            let part_stats = match SelfPlayManager::run(part_sp_cfg) {
                Ok(s) => s,
                Err(e) => {
                    let _ = fs::remove_file(&tmp_tsv_path);
                    return Err(format!(
                        "Partition {:04}/{} self-play failed (tmp TSV discarded): {e}",
                        part_idx + 1,
                        num_parts
                    ));
                }
            };
            stats.total_games_completed += part_stats.completed_games;

            // 1. 一時 TSV を本番 TSV へアトミックリネーム
            fs::rename(&tmp_tsv_path, &tsv_path).map_err(|e| {
                let _ = fs::remove_file(&tmp_tsv_path);
                format!(
                    "Failed to publish partition {:04} TSV '{}' -> '{}': {e}",
                    part_idx + 1,
                    tmp_tsv_path.display(),
                    tsv_path.display()
                )
            })?;

            // 2. 厳格な設定署名を含む完了マーカーを原子的作成（途中障害で 0 byte .done を残さない）
            let marker_content = format!(
                "{expected_signature}plies={}\nio_errors={}\n",
                part_stats.total_plies, part_stats.io_errors
            );
            write_done_marker_atomic(&done_path, &marker_content).map_err(|e| {
                format!(
                    "Failed to write partition {:04} done marker '{}': {e}",
                    part_idx + 1,
                    done_path.display()
                )
            })?;

            stats.completed_partitions += 1;
            println!(
                "[Success] Partition {:04} completed, atomically published, and marked done.",
                part_idx + 1
            );
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
        println!("========================================================");

        Ok(stats)
    }
}

fn path_exists_fail_closed(path: &Path) -> io::Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

fn can_resume_partition(
    done_path: &Path,
    tsv_path: &Path,
    expected_signature: &str,
) -> Result<bool, String> {
    let has_done = path_exists_fail_closed(done_path).map_err(|e| {
        format!(
            "Failed to inspect done marker '{}': {e}",
            done_path.display()
        )
    })?;
    let has_tsv = path_exists_fail_closed(tsv_path).map_err(|e| {
        format!(
            "Failed to inspect partition TSV '{}': {e}",
            tsv_path.display()
        )
    })?;
    if !has_done || !has_tsv {
        return Ok(false);
    }

    let existing_done = fs::read_to_string(done_path)
        .map_err(|e| format!("Failed to read done marker '{}': {e}", done_path.display()))?;
    if existing_done.starts_with(expected_signature) {
        Ok(true)
    } else {
        println!(
            "[Notice] Partition '{}' exists but configuration mismatched or marker corrupted. Invalidating and regenerating.",
            tsv_path.display()
        );
        Ok(false)
    }
}

fn write_done_marker_atomic(done_path: &Path, content: &str) -> io::Result<()> {
    let mut tmp_os = done_path.as_os_str().to_os_string();
    tmp_os.push(".tmp");
    let tmp_path = PathBuf::from(tmp_os);

    let write_res = (|| -> io::Result<()> {
        let mut file = File::create(&tmp_path)?;
        file.write_all(content.as_bytes())?;
        file.sync_all()?;
        Ok(())
    })();
    if let Err(e) = write_res {
        let _ = fs::remove_file(&tmp_path);
        return Err(e);
    }
    if let Err(e) = fs::rename(&tmp_path, done_path) {
        let _ = fs::remove_file(&tmp_path);
        return Err(e);
    }
    Ok(())
}

fn remove_if_exists(path: &Path) -> Result<(), String> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!(
            "Failed to remove stale partition file '{}': {e}",
            path.display()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TempDirGuard(PathBuf);

    impl TempDirGuard {
        fn new(prefix: &str) -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "tabula_shogi_part_{prefix}_{}_{}",
                std::process::id(),
                id
            ));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn partitioned_manager_returns_err_on_file_as_output_dir_and_empty_dir() {
        let tmp = TempDirGuard::new("bad_dir");
        let file_path = tmp.0.join("not_a_dir");
        fs::write(&file_path, b"x").unwrap();

        let cfg_file = PartitionConfig {
            total_games: 1,
            games_per_partition: 1,
            output_dir: file_path.to_string_lossy().into_owned(),
            ..Default::default()
        };
        assert!(PartitionedSelfPlayManager::run(cfg_file).is_err());

        let cfg_empty = PartitionConfig {
            total_games: 1,
            games_per_partition: 1,
            output_dir: "  ".to_string(),
            ..Default::default()
        };
        assert!(PartitionedSelfPlayManager::run(cfg_empty).is_err());
    }
}
