use crate::board::Position;
use crate::eval::halfkp::HalfKPEvaluator;
use crate::eval::halfkp_trainer::{HalfKPTrainer, TrainStepLoss};
use crate::selfplay::partition::MultiPartitionStreamingReader;
use std::path::Path;
use std::time::Instant;

/// 大規模ストリーミング学習の設定
#[derive(Debug, Clone)]
pub struct StreamTrainConfig {
    /// パーティションデータセットが格納されたディレクトリ
    pub data_dir: String,
    /// ミニバッチサイズ (例: 1024)
    pub batch_size: usize,
    /// AdamW 学習率 (例: 0.001)
    pub lr: f32,
    /// シグモイド感度係数 (通常 600.0)
    pub k: f32,
    /// 総学習エポック数
    pub epochs: usize,
    /// チェックポイント保存間隔（バッチ数、0で無効化）
    pub checkpoint_interval_batches: usize,
    /// AdamW 状態チェックポイント保存先パス
    pub checkpoint_path: Option<String>,
    /// 有倍率固定小数点推論モデルバイナリ (TABU_HK2) 保存先パス
    pub model_output_path: String,
    /// 既存チェックポイントが存在する場合に自動再開するか
    pub resume_from_checkpoint: bool,
}

impl Default for StreamTrainConfig {
    fn default() -> Self {
        Self {
            data_dir: "data/titan".to_string(),
            batch_size: 1024,
            lr: 0.001,
            k: 600.0,
            epochs: 1,
            checkpoint_interval_batches: 2000,
            checkpoint_path: Some("models/halfkp_titan_ckpt.bin".to_string()),
            model_output_path: "models/halfkp_titan.bin".to_string(),
            resume_from_checkpoint: true,
        }
    }
}

/// ストリーミング学習の実行結果統計
#[derive(Debug, Default, Clone)]
pub struct StreamTrainSummary {
    pub epochs_completed: usize,
    pub total_positions_trained: usize,
    pub total_batches: usize,
    pub final_mse_loss: f32,
    pub final_range_loss: f32,
    pub final_total_loss: f32,
    pub total_duration_secs: f64,
    pub positions_per_sec: f64,
}

/// 巨大データセット（1億局面規模）をメモリ 500MB 以内でストリーミング学習するエンジン
pub struct HalfKPStreamTrainer;

impl HalfKPStreamTrainer {
    /// 複数パーティションTSVからストリーミング学習を実行
    pub fn train(config: StreamTrainConfig) -> Result<StreamTrainSummary, String> {
        let overall_start = Instant::now();

        // 0. 入力パラメータの厳格バリデーション (メモリ上限超過および不正値の即座拒否)
        if config.batch_size == 0 || config.batch_size > 16384 {
            return Err(format!(
                "Invalid batch_size: {}. batch_size must be between 1 and 16384 to strictly guarantee the 500MB memory limit.",
                config.batch_size
            ));
        }
        if config.epochs == 0 {
            return Err("Invalid epochs: 0. epochs must be at least 1.".to_string());
        }
        if config.lr <= 0.0 || !config.lr.is_finite() {
            return Err(format!(
                "Invalid learning rate: {}. lr must be positive and finite.",
                config.lr
            ));
        }
        if config.k <= 0.0 || !config.k.is_finite() {
            return Err(format!(
                "Invalid sigmoid sensitivity k: {}. k must be positive and finite.",
                config.k
            ));
        }

        // 1. ストリーミングリーダーの初期化
        let mut reader =
            MultiPartitionStreamingReader::from_directory(&config.data_dir, config.batch_size)
                .map_err(|e| {
                    format!(
                        "Failed to initialize MultiPartitionStreamingReader from '{}': {e}",
                        config.data_dir
                    )
                })?;

        if reader.file_count() == 0 {
            return Err(format!(
                "No completed partition files found in '{}'. Did you run self-play generation first?",
                config.data_dir
            ));
        }

        println!("========================================================");
        println!("=== TabulaShogi HalfKP Large-Scale Stream Trainer ===");
        println!("Dataset Directory: {}", config.data_dir);
        println!("Discovered Partitions: {}", reader.file_count());
        println!(
            "Epochs: {}, Batch Size: {}, LR: {}",
            config.epochs, config.batch_size, config.lr
        );
        if let Some(ref ckpt) = config.checkpoint_path {
            println!("Checkpoint Path: {ckpt}");
        }
        println!("Model Output: {}", config.model_output_path);
        println!("========================================================");

        // 2. トレーナーの初期化またはチェックポイントからのウォームスタート復元
        let mut trainer = if config.resume_from_checkpoint
            && let Some(ref ckpt_path) = config.checkpoint_path
            && Path::new(ckpt_path).exists()
        {
            println!(
                "[Resume] Warm-starting HalfKPTrainer weights and Adam momentum from '{ckpt_path}'..."
            );
            HalfKPTrainer::load_checkpoint(ckpt_path)?
        } else {
            println!("[Init] Initializing fresh HalfKPTrainer from initial evaluator weights...");
            HalfKPTrainer::new()
        };

        let mut summary = StreamTrainSummary::default();
        let mut global_batch_count = 0usize;
        let mut last_loss = TrainStepLoss::zero();

        // 出力先ディレクトリの確保
        if let Some(parent) = Path::new(&config.model_output_path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Some(ref ckpt_path) = config.checkpoint_path
            && let Some(parent) = Path::new(ckpt_path).parent()
        {
            let _ = std::fs::create_dir_all(parent);
        }

        // 3. エポックループ
        for epoch in 1..=config.epochs {
            let epoch_start = Instant::now();
            println!("\n>>> Starting Epoch {}/{} >>>", epoch, config.epochs);

            // ファイル順序のシャッフル
            reader.shuffle_files(0x123456789ABCDEF0u64.wrapping_add(epoch as u64));

            let mut epoch_positions = 0usize;
            let mut epoch_mse = 0.0f64;
            let mut epoch_range = 0.0f64;
            let mut epoch_batches = 0usize;

            while let Some(batch_entries) = reader
                .next_batch()
                .map_err(|e| format!("I/O error reading batch: {e}"))?
            {
                if batch_entries.is_empty() {
                    continue;
                }

                // SFEN から HalfKP 特徴量をオンザフライ抽出 (メモリ低フットプリント)
                let mut batch_samples: Vec<(Vec<usize>, Vec<usize>, f32)> =
                    Vec::with_capacity(batch_entries.len());
                for entry in &batch_entries {
                    if let Ok(pos) = Position::from_sfen(&entry.sfen) {
                        let mover_feats =
                            HalfKPEvaluator::extract_halfkp_features(&pos, pos.side_to_move);
                        let opp_feats = HalfKPEvaluator::extract_halfkp_features(
                            &pos,
                            pos.side_to_move.opposite(),
                        );
                        batch_samples.push((mover_feats, opp_feats, entry.result));
                    }
                }

                if batch_samples.is_empty() {
                    continue;
                }

                let batch_len = batch_samples.len();
                let loss = trainer.train_batch(&batch_samples, config.lr, config.k);
                last_loss = loss;

                epoch_mse += (loss.mse_loss as f64) * (batch_len as f64);
                epoch_range += (loss.range_loss as f64) * (batch_len as f64);
                epoch_positions += batch_len;
                epoch_batches += 1;
                global_batch_count += 1;

                // 定期進捗表示
                if global_batch_count.is_multiple_of(100) || global_batch_count == 1 {
                    let elapsed = epoch_start.elapsed().as_secs_f64().max(0.001);
                    let pos_per_sec = epoch_positions as f64 / elapsed;
                    print!(
                        "\r[Epoch {} | Batch {}] Pos: {} | MSE: {:.6} | Range: {:.6} | Total: {:.6} | {:.0} pos/s    ",
                        epoch,
                        epoch_batches,
                        epoch_positions,
                        loss.mse_loss,
                        loss.range_loss,
                        loss.total_loss,
                        pos_per_sec
                    );
                    let _ = std::io::Write::flush(&mut std::io::stdout());
                }

                // 定期チェックポイント保存
                if config.checkpoint_interval_batches > 0
                    && global_batch_count.is_multiple_of(config.checkpoint_interval_batches)
                    && let Some(ref ckpt_path) = config.checkpoint_path
                {
                    println!(
                        "\n[Checkpoint] Saving periodic Adam checkpoint to '{ckpt_path}' at batch {global_batch_count}..."
                    );
                    if let Err(e) = trainer.save_checkpoint(ckpt_path) {
                        eprintln!("[Warning] Failed to save checkpoint '{ckpt_path}': {e}");
                    }
                }
            }

            println!();
            let epoch_elapsed = epoch_start.elapsed().as_secs_f64().max(0.001);
            let avg_mse = if epoch_positions > 0 {
                epoch_mse / epoch_positions as f64
            } else {
                0.0
            };
            let avg_range = if epoch_positions > 0 {
                epoch_range / epoch_positions as f64
            } else {
                0.0
            };
            let pos_sec = epoch_positions as f64 / epoch_elapsed;

            println!(
                "=== Epoch {} Completed in {:.2}s: {} positions ({:.0} pos/s) | Avg MSE: {:.6} | Avg Range: {:.6} ===",
                epoch, epoch_elapsed, epoch_positions, pos_sec, avg_mse, avg_range
            );

            summary.epochs_completed += 1;
            summary.total_positions_trained += epoch_positions;
            summary.total_batches += epoch_batches;

            if epoch_positions == 0 {
                eprintln!(
                    "[Warning] Epoch {epoch} processed 0 valid positions. Skipping model/checkpoint persistence."
                );
                continue;
            }

            // 各エポック終了時に推論モデルおよびチェックポイントを永続化
            println!(
                "[Persist] Exporting inference model to '{}'...",
                config.model_output_path
            );
            let eval = trainer.to_evaluator();
            eval.save_to_file(&config.model_output_path).map_err(|e| {
                format!(
                    "Failed to save exported model to '{}': {e}",
                    config.model_output_path
                )
            })?;

            if let Some(ref ckpt_path) = config.checkpoint_path {
                println!("[Persist] Saving epoch checkpoint to '{ckpt_path}'...");
                trainer.save_checkpoint(ckpt_path).map_err(|e| {
                    format!("Failed to save epoch checkpoint to '{ckpt_path}': {e}")
                })?;
            }
        }

        if summary.total_positions_trained == 0 {
            return Err("Zero valid positions were processed across all epochs. Model and checkpoint were not updated.".to_string());
        }

        let overall_duration = overall_start.elapsed().as_secs_f64().max(0.001);
        summary.final_mse_loss = last_loss.mse_loss;
        summary.final_range_loss = last_loss.range_loss;
        summary.final_total_loss = last_loss.total_loss;
        summary.total_duration_secs = overall_duration;
        summary.positions_per_sec = summary.total_positions_trained as f64 / overall_duration;

        println!("\n========================================================");
        println!("=== HalfKP Large-Scale Stream Training Completed ===");
        println!("Total Epochs: {}", summary.epochs_completed);
        println!("Total Positions: {}", summary.total_positions_trained);
        println!(
            "Total Duration: {:.2}s ({:.0} positions/sec)",
            overall_duration, summary.positions_per_sec
        );
        println!(
            "Final MSE Loss: {:.6}, Range Loss: {:.6}, Total Loss: {:.6}",
            summary.final_mse_loss, summary.final_range_loss, summary.final_total_loss
        );
        println!("Final Model saved to: {}", config.model_output_path);
        println!("========================================================");

        Ok(summary)
    }
}
