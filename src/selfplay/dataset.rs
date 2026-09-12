use super::game::GameRecord;
use crate::types::Color;
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
}
