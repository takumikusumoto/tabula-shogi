use super::game::{GameRecord, SimpleRng};
use crate::types::Color;
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};

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
        // 手数超過（打ち切り引き分け）は優劣が不明瞭なまま打ち切られるため、
        // 学習ラベルの汚染を防ぐためデータセットから除外
        if let super::game::GameResult::Draw(super::game::DrawReason::MaxPliesExceeded) =
            game.result
        {
            return Vec::new();
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

        entries
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
        if let Some(parent) = std::path::Path::new(path).parent() {
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
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let recent_target = ((total_sample as f64) * recent_ratio).round() as usize;
        let history_target = total_sample.saturating_sub(recent_target);

        let mut recent_queue = VecDeque::with_capacity(recent_target);
        let mut history_samples = Vec::with_capacity(history_target);
        let mut history_count = 0usize;
        let mut rng = SimpleRng::new(if seed == 0 { 0xdeadbeefcafe } else { seed });

        for line in reader.lines() {
            let l = line?;
            if let Some(entry) = Self::parse_entry(&l) {
                if recent_queue.len() < recent_target {
                    recent_queue.push_back(entry);
                } else {
                    // 最新バッファからあふれた最古の要素が過去プールに流入
                    let displaced = recent_queue.pop_front().unwrap();
                    recent_queue.push_back(entry);

                    history_count += 1;
                    if history_samples.len() < history_target {
                        history_samples.push(displaced);
                    } else {
                        // リザーバサンプリング (一様確率で置換)
                        let j = rng.gen_range(history_count);
                        if j < history_target {
                            history_samples[j] = displaced;
                        }
                    }
                }
            }
        }

        // 過去プールと最新バッファを統合
        let mut result = history_samples;
        result.extend(recent_queue);

        // ミニバッチ学習のバイアスを防ぐため、全体をインプレースでシャッフル (Fisher-Yates)
        if result.len() > 1 {
            for i in (1..result.len()).rev() {
                let j = rng.gen_range(i + 1);
                result.swap(i, j);
            }
        }

        Ok(result)
    }

    /// 深読みプールを優先してサンプリングするデータローダー
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
        let mut deep_entries = Vec::new();
        let target_deep = ((total_sample as f64) * deep_ratio).round() as usize;

        if let Some(dp) = deep_path
            && std::path::Path::new(dp).exists()
            && let Ok(loaded) = Self::load_sampled(dp, target_deep, 0.5, seed)
        {
            deep_entries = loaded;
        }

        let remaining = total_sample.saturating_sub(deep_entries.len());
        let mut main_entries = if remaining > 0 && std::path::Path::new(main_path).exists() {
            Self::load_sampled(main_path, remaining, recent_ratio, seed.wrapping_add(1))?
        } else {
            Vec::new()
        };

        let mut combined = deep_entries;
        combined.append(&mut main_entries);

        // 全体をインプレースでシャッフル (Fisher-Yates)
        if combined.len() > 1 {
            let mut rng = SimpleRng::new(if seed == 0 { 0xdeadbeefcafe } else { seed });
            for i in (1..combined.len()).rev() {
                let j = rng.gen_range(i + 1);
                combined.swap(i, j);
            }
        }

        Ok(combined)
    }

    /// サンプリングされたデータセットのうち指定件数をマルチスレッドで深い探索 (Depth 4等) により再評価 (IIZ / 知識蒸留)
    /// 戻り値: 深読み再評価が正常完了（または詰み証明）された高品質エントリのリスト（未完了・中断は含まれない）
    pub fn relabel_deep(
        entries: &mut [DatasetEntry],
        count: usize,
        depth: u8,
        threads: usize,
        eval_mode: &crate::eval::EvalMode,
    ) -> Vec<DatasetEntry> {
        // 深さ0の探索による無意味・危険な上書きを即座に拒否
        if depth == 0 {
            return Vec::new();
        }

        let target_len = count.min(entries.len());
        if target_len == 0 {
            return Vec::new();
        }

        let num_threads = threads.clamp(1, 64).min(target_len);
        let chunk_size = target_len.div_ceil(num_threads);

        let slice_to_relabel = &mut entries[..target_len];
        let min_required_depth = 2.min(depth);
        let successful_entries = std::sync::Mutex::new(Vec::with_capacity(target_len));

        std::thread::scope(|s| {
            for chunk in slice_to_relabel.chunks_mut(chunk_size) {
                s.spawn(|| {
                    let mut engine = crate::search::SearchEngine::new(4);
                    engine.eval_mode = eval_mode.clone();
                    engine.max_nodes = Some(30_000); // 1局面最大3万ノードで確実に打ち切り、ハング・長時間スタックを完全防止
                    engine.use_book = false; // 深読み再評価では定跡手をスキップし、純粋な深さNの探索評価値を算出

                    let mut local_successes = Vec::new();

                    for entry in chunk {
                        if let Ok(mut pos) = crate::board::Position::from_sfen(&entry.sfen) {
                            let outcome = engine.search_fixed_depth_outcome(&mut pos, depth);
                            // 定跡(Book)や中断(Aborted)による破壊的0上書き・不完全上書きを型レベルで完全排除
                            if let Some(reliable_score) =
                                outcome.reliable_score_for_relabel(min_required_depth)
                            {
                                entry.score = reliable_score;
                                local_successes.push(entry.clone());
                            }
                        }
                    }

                    if !local_successes.is_empty() {
                        let mut lock = successful_entries.lock().unwrap();
                        lock.extend(local_successes);
                    }
                });
            }
        });

        successful_entries.into_inner().unwrap()
    }
}

/// 巨大データセット（数百万〜1億局面）をメモリ 500MB 以内で省メモリにバッチ読み込みするストリーミングローダー
pub struct StreamingBatchReader {
    reader: BufReader<File>,
    batch_size: usize,
}

impl StreamingBatchReader {
    /// 新規ストリーミングリーダーを生成
    pub fn new(path: &str, batch_size: usize) -> io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Ok(Self { reader, batch_size })
    }

    /// 次のバッチを読み込む。ファイルの末尾 (EOF) に到達した場合は None を返す。
    pub fn next_batch(&mut self) -> io::Result<Option<Vec<DatasetEntry>>> {
        let mut batch = Vec::with_capacity(self.batch_size);
        let mut line = String::new();

        while batch.len() < self.batch_size {
            line.clear();
            let bytes_read = self.reader.read_line(&mut line)?;
            if bytes_read == 0 {
                break; // EOF
            }
            if let Some(entry) = DatasetHandler::parse_entry(&line) {
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
