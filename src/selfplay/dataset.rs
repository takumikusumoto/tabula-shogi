use super::game::{GameRecord, SimpleRng};
use crate::types::Color;
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, ErrorKind, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub struct DatasetEntry {
    pub sfen: String,
    pub score: i32,
    pub result: f32, // 1.0: 手番側勝利, 0.5: 引分, 0.0: 手番側敗北
    pub move_usi: String,
}

pub struct DatasetHandler;

impl DatasetHandler {
    /// 1対局の記録から学習用レコード群を抽出
    /// (序盤のランダム手はノイズを避けるため除外し、探索が行われた局面のみを抽出)
    pub fn extract_entries(game: &GameRecord, min_ply: usize) -> Vec<DatasetEntry> {
        Self::extract_entries_checked(game, min_ply).unwrap_or_default()
    }

    /// `GameRecord` の構造整合性を確認してから学習用レコード群を抽出する。
    ///
    /// `GameResult` は通常引分と最大手数引分を型で区別しており、どちらも 0.5 として
    /// 保持する。一方、破損した記録（手数とレコード数の不一致）は引分に読み替えず拒否する。
    pub fn extract_entries_checked(
        game: &GameRecord,
        min_ply: usize,
    ) -> Result<Vec<DatasetEntry>, String> {
        if game.total_plies != game.plies.len() {
            return Err(format!(
                "Invalid game record {}: total_plies={} but ply records={}",
                game.game_id,
                game.total_plies,
                game.plies.len()
            ));
        }

        if game.plies.windows(2).any(|pair| pair[0].ply >= pair[1].ply) {
            return Err(format!(
                "Invalid game record {}: ply numbers are not strictly increasing",
                game.game_id
            ));
        }

        let mut entries = Vec::with_capacity(game.plies.len());
        let game_score_black = game.result.score_black();

        for ply_rec in &game.plies {
            if ply_rec.ply < min_ply {
                continue;
            }

            // 手番視点での対局結果 (1.0 = 勝ち, 0.0 = 負け, 0.5 = 引分)
            let result_for_turn = match ply_rec.side_to_move {
                Color::Black => game_score_black,
                Color::White => 1.0 - game_score_black,
            };

            entries.push(DatasetEntry {
                sfen: ply_rec.sfen.clone(),
                score: ply_rec.score,
                result: result_for_turn,
                move_usi: ply_rec.mv.to_usi(),
            });
        }

        Ok(entries)
    }

    /// TSV形式の1行にシリアライズ
    pub fn format_entry(entry: &DatasetEntry) -> String {
        format!(
            "{}\t{}\t{:.1}\t{}\n",
            entry.sfen, entry.score, entry.result, entry.move_usi
        )
    }

    /// TSV形式の1行からパース
    pub fn parse_entry(line: &str) -> Option<DatasetEntry> {
        let parts: Vec<&str> = line.trim().split('\t').collect();
        if parts.len() < 4 {
            return None;
        }

        let sfen = parts[0].to_string();
        let score = parts[1].parse::<i32>().ok()?;
        let result = parts[2].parse::<f32>().ok()?;
        let move_usi = parts[3].to_string();

        Some(DatasetEntry {
            sfen,
            score,
            result,
            move_usi,
        })
    }

    /// データセットファイルへ追記保存
    pub fn append_to_file(path: &str, entries: &[DatasetEntry]) -> io::Result<()> {
        if let Some(parent) = Path::new(path).parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;

        for entry in entries {
            file.write_all(Self::format_entry(entry).as_bytes())?;
        }
        file.flush()?;
        Ok(())
    }

    /// データセットファイルから全件読み込み
    pub fn load_from_file(path: &str) -> io::Result<Vec<DatasetEntry>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut entries = Vec::new();

        for line in reader.lines() {
            let l = line?;
            if let Some(entry) = Self::parse_entry(&l) {
                entries.push(entry);
            }
        }

        Ok(entries)
    }

    /// データセットから最新世代および過去履歴をバランスよくサンプリングして読み込み
    /// - total_sample: 抽出する総局面数 (例: 100,000)
    /// - recent_ratio: 最新局面の比率 (例: 0.5 = 50% 最新, 50% 過去)
    /// - メモリ使用量は sample_size に制限され、ファイル全体の巨大アロケーションを防止
    pub fn load_sampled(
        path: &str,
        total_sample: usize,
        recent_ratio: f64,
        seed: u64,
    ) -> io::Result<Vec<DatasetEntry>> {
        let mut result = Self::load_sampled_unshuffled(path, total_sample, recent_ratio, seed)?;
        shuffle_slice(&mut result, seed);
        Ok(result)
    }

    /// 深読みプールを優先してサンプリングするデータローダー（既存ファイルの読込障害は fail-closed）
    /// - main_path: 通常の自己対局データ（loop_dataset.tsv）
    /// - deep_path: 永続化された深読み再評価データ（deep_dataset.tsv）
    /// - total_sample: 抽出する総局面数 (例: 100,000)
    /// - deep_ratio: 深読みデータの目標比率 (例: 0.5 = 最大5万件を深読みデータから優先抽出)
    /// - recent_ratio: 通常データ枠における最新局面比率 (例: 0.5)
    pub fn load_sampled_with_deep_pool(
        main_path: &str,
        deep_path: Option<&str>,
        total_sample: usize,
        deep_ratio: f64,
        recent_ratio: f64,
        seed: u64,
    ) -> io::Result<Vec<DatasetEntry>> {
        let target_deep = ((total_sample as f64) * deep_ratio.clamp(0.0, 1.0)).round() as usize;
        let deep_entries = match deep_path {
            Some(dp) if target_deep > 0 => match path_exists_fail_closed(dp)? {
                true => Self::load_sampled_unshuffled(dp, target_deep, 0.5, seed)?,
                false => Vec::new(),
            },
            _ => Vec::new(),
        };

        let remaining = total_sample.saturating_sub(deep_entries.len());
        let mut main_entries = if remaining > 0 && path_exists_fail_closed(main_path)? {
            Self::load_sampled_unshuffled(main_path, remaining, recent_ratio, seed.wrapping_add(1))?
        } else {
            Vec::new()
        };

        let mut combined = deep_entries;
        combined.append(&mut main_entries);

        // 結合後に 1 回だけインプレースシャッフル (Fisher-Yates)
        shuffle_slice(&mut combined, seed);

        Ok(combined)
    }

    fn load_sampled_unshuffled(
        path: &str,
        total_sample: usize,
        recent_ratio: f64,
        seed: u64,
    ) -> io::Result<Vec<DatasetEntry>> {
        let file = File::open(path)?;
        if total_sample == 0 {
            return Ok(Vec::new());
        }
        let reader = BufReader::new(file);

        let clamped_ratio = recent_ratio.clamp(0.0, 1.0);
        let recent_target = ((total_sample as f64) * clamped_ratio).round() as usize;
        let history_target = total_sample.saturating_sub(recent_target);

        let mut recent_queue = VecDeque::with_capacity(recent_target);
        let mut history_samples = Vec::with_capacity(history_target);
        let mut history_count = 0usize;
        let mut rng = SimpleRng::new(if seed == 0 { 0xdeadbeefcafe } else { seed });

        for line in reader.lines() {
            let l = line?;
            if let Some(entry) = Self::parse_entry(&l) {
                if recent_target == 0 {
                    // 全量を過去履歴リザーバサンプリングとして直接処理
                    history_count += 1;
                    if history_samples.len() < history_target {
                        history_samples.push(entry);
                    } else if history_target > 0 {
                        let j = rng.gen_range(history_count);
                        if j < history_target {
                            history_samples[j] = entry;
                        }
                    }
                } else if recent_queue.len() < recent_target {
                    recent_queue.push_back(entry);
                } else {
                    // 最新バッファからあふれた最古の要素が過去プールに流入
                    let displaced = recent_queue.pop_front().unwrap_or_else(|| entry.clone());
                    recent_queue.push_back(entry);

                    history_count += 1;
                    if history_samples.len() < history_target {
                        history_samples.push(displaced);
                    } else if history_target > 0 {
                        // リザーバサンプリング (一様確率で置換)
                        let j = rng.gen_range(history_count);
                        if j < history_target {
                            history_samples[j] = displaced;
                        }
                    }
                }
            }
        }

        let mut result = history_samples;
        result.extend(recent_queue);
        Ok(result)
    }
}

fn path_exists_fail_closed(path: &str) -> io::Result<bool> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

fn shuffle_slice<T>(items: &mut [T], seed: u64) {
    if items.len() <= 1 {
        return;
    }
    let mut rng = SimpleRng::new(if seed == 0 { 0xdeadbeefcafe } else { seed });
    for i in (1..items.len()).rev() {
        let j = rng.gen_range(i + 1);
        items.swap(i, j);
    }
}

fn validate_batch_size(batch_size: usize) -> io::Result<()> {
    if batch_size == 0 {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "batch_size must be greater than 0",
        ));
    }
    Ok(())
}

/// 単一 TSV ファイルおよび複数パーティション TSV ディレクトリを透過的にバッチ読込する単一ストリーミングリーダー
pub struct DatasetBatchReader {
    file_paths: Vec<PathBuf>,
    current_file_idx: usize,
    current_reader: Option<BufReader<File>>,
    batch_size: usize,
    line_buf: String,
}

impl DatasetBatchReader {
    /// ファイルパス一覧とバッチサイズを指定して新規作成
    pub fn from_files(file_paths: Vec<PathBuf>, batch_size: usize) -> io::Result<Self> {
        validate_batch_size(batch_size)?;
        Ok(Self {
            file_paths,
            current_file_idx: 0,
            current_reader: None,
            batch_size,
            line_buf: String::new(),
        })
    }

    /// 単一 TSV ファイルを即時オープン検証して新規作成
    pub fn from_file(path: impl Into<PathBuf>, batch_size: usize) -> io::Result<Self> {
        validate_batch_size(batch_size)?;
        let path_buf = path.into();
        let file = File::open(&path_buf)?;
        Ok(Self {
            file_paths: vec![path_buf],
            current_file_idx: 1,
            current_reader: Some(BufReader::new(file)),
            batch_size,
            line_buf: String::new(),
        })
    }

    /// 読み込み対象のファイル数を取得
    pub fn file_count(&self) -> usize {
        self.file_paths.len()
    }

    /// 読み込み順序を Fisher-Yates でシャッフルし、先頭へ巻き戻す（エポック間でのファイル順偏向を低減）
    pub fn shuffle_files(&mut self, seed: u64) {
        shuffle_slice(&mut self.file_paths, seed);
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
            let reader = match self.current_reader.as_mut() {
                Some(reader) => reader,
                None => {
                    if self.current_file_idx >= self.file_paths.len() {
                        break;
                    }
                    let next_path = &self.file_paths[self.current_file_idx];
                    let file = File::open(next_path)?;
                    self.current_file_idx += 1;
                    self.current_reader.insert(BufReader::new(file))
                }
            };

            self.line_buf.clear();
            let bytes_read = reader.read_line(&mut self.line_buf)?;

            if bytes_read == 0 {
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
                "tabula_shogi_ds_{prefix}_{}_{}",
                std::process::id(),
                id
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn path(&self, name: &str) -> String {
            self.0.join(name).to_string_lossy().into_owned()
        }
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn sample_entry(score: i32) -> DatasetEntry {
        DatasetEntry {
            sfen: "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1".to_string(),
            score,
            result: 1.0,
            move_usi: "7g7f".to_string(),
        }
    }

    #[test]
    fn dataset_batch_reader_supports_single_file_reset_and_rejects_invalid_inputs() {
        let tmp = TempDirGuard::new("batch_single");
        let path = tmp.path("single.tsv");
        let entries: Vec<DatasetEntry> = (0..7).map(sample_entry).collect();
        DatasetHandler::append_to_file(&path, &entries).unwrap();

        assert!(DatasetBatchReader::from_file(&path, 0).is_err());
        assert!(DatasetBatchReader::from_file(tmp.path("missing.tsv"), 4).is_err());

        let mut reader = DatasetBatchReader::from_file(&path, 3).unwrap();
        assert_eq!(reader.file_count(), 1);
        assert_eq!(reader.next_batch().unwrap().unwrap().len(), 3);
        assert_eq!(reader.next_batch().unwrap().unwrap().len(), 3);
        assert_eq!(reader.next_batch().unwrap().unwrap().len(), 1);
        assert!(reader.next_batch().unwrap().is_none());

        reader.reset();
        assert_eq!(reader.next_batch().unwrap().unwrap().len(), 3);
    }

    #[test]
    fn dataset_batch_reader_from_files_crosses_boundaries_and_shuffles() {
        let tmp = TempDirGuard::new("batch_multi");
        let p1 = tmp.path("a.tsv");
        let p2 = tmp.path("b.tsv");
        DatasetHandler::append_to_file(&p1, &[sample_entry(1), sample_entry(2), sample_entry(3)])
            .unwrap();
        DatasetHandler::append_to_file(&p2, &[sample_entry(4), sample_entry(5)]).unwrap();

        assert!(DatasetBatchReader::from_files(vec![PathBuf::from(&p1)], 0).is_err());

        let mut reader =
            DatasetBatchReader::from_files(vec![PathBuf::from(&p1), PathBuf::from(&p2)], 2)
                .unwrap();
        assert_eq!(reader.file_count(), 2);

        let mut total = 0;
        while let Some(batch) = reader.next_batch().unwrap() {
            assert!(batch.len() <= 2);
            total += batch.len();
        }
        assert_eq!(total, 5);

        reader.shuffle_files(7);
        let mut after_shuffle = 0;
        while let Some(batch) = reader.next_batch().unwrap() {
            after_shuffle += batch.len();
        }
        assert_eq!(after_shuffle, 5);
    }

    #[test]
    fn load_sampled_with_deep_pool_fails_closed_on_unreadable_deep_pool() {
        let tmp = TempDirGuard::new("deep_fail_closed");
        let main_path = tmp.path("main.tsv");
        let deep_dir = tmp.path("deep_as_dir");
        std::fs::create_dir_all(&deep_dir).unwrap();
        DatasetHandler::append_to_file(&main_path, &[sample_entry(10), sample_entry(20)]).unwrap();

        // Missing deep pool is allowed and falls back to main_path
        let missing_deep = tmp.path("absent_deep.tsv");
        let ok = DatasetHandler::load_sampled_with_deep_pool(
            &main_path,
            Some(&missing_deep),
            10,
            0.5,
            0.5,
            42,
        )
        .unwrap();
        assert_eq!(ok.len(), 2);

        // Existing but unreadable deep pool (directory) must fail closed
        let err = DatasetHandler::load_sampled_with_deep_pool(
            &main_path,
            Some(&deep_dir),
            10,
            0.5,
            0.5,
            42,
        );
        assert!(
            err.is_err(),
            "Unreadable existing deep_path must fail closed instead of silently degrading"
        );
    }
}
