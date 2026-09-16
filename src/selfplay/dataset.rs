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

    /// サンプリングされたデータセットのうち指定件数をマルチスレッドで深い探索 (Depth 4等) により再評価 (IIZ / 知識蒸留)
    /// - entries: 再評価対象のデータセット
    /// - count: 再評価する局面数 (先頭から count 件、例: 10,000)
    /// - depth: 探索深さ (例: 4)
    /// - threads: 並行スレッド数 (例: 4)
    pub fn relabel_deep(entries: &mut [DatasetEntry], count: usize, depth: u8, threads: usize) {
        let target_len = count.min(entries.len());
        if target_len == 0 {
            return;
        }

        let num_threads = threads.clamp(1, 64).min(target_len);
        let chunk_size = (target_len + num_threads - 1) / num_threads;

        let slice_to_relabel = &mut entries[..target_len];

        std::thread::scope(|s| {
            for chunk in slice_to_relabel.chunks_mut(chunk_size) {
                s.spawn(move || {
                    let mut engine = crate::search::SearchEngine::new(4);
                    engine.eval_mode = crate::eval::EvalMode::Hce;
                    engine.max_nodes = Some(30_000); // 1局面最大3万ノードで確実に打ち切り、ハング・長時間スタックを完全防止

                    for entry in chunk {
                        if let Ok(mut pos) = crate::board::Position::from_sfen(&entry.sfen) {
                            let res = engine.search_fixed_depth_detail(&mut pos, depth);
                            // 要求深さに達した（または深さ2以上の有意な探索が完了した）場合のみラベルを更新
                            if res.completed_depth >= 2.min(depth) {
                                entry.score = res.score;
                            }
                        }
                    }
                });
            }
        });
    }
}
