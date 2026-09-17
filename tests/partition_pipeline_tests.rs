use std::fs;
use std::path::Path;
use tabula_shogi::eval::halfkp::HalfKPEvaluator;
use tabula_shogi::eval::halfkp_stream_trainer::{HalfKPStreamTrainer, StreamTrainConfig};
use tabula_shogi::eval::halfkp_trainer::HalfKPTrainer;
use tabula_shogi::selfplay::config::SelfPlayConfig;
use tabula_shogi::selfplay::dataset::{DatasetEntry, DatasetHandler};
use tabula_shogi::selfplay::partition::{
    MultiPartitionStreamingReader, PartitionConfig, PartitionedSelfPlayManager,
};

#[test]
fn test_partitioned_selfplay_generation_and_resume() {
    let test_dir = "target/test_partition_selfplay";
    let _ = fs::remove_dir_all(test_dir);

    let config = PartitionConfig {
        total_games: 4,
        games_per_partition: 2,
        output_dir: test_dir.to_string(),
        base_config: SelfPlayConfig {
            num_games: 2,
            threads: 2,
            depth: 1, // 高速実行
            random_opening_plies: 2,
            max_plies: 20,
            resign_threshold: -1000,
            csa_output: None,
            data_output: None,
            tt_size_mb: 2,
            seed: 0x12345678,
            eval_mode: tabula_shogi::eval::EvalMode::Hce,
            temperature_plies: 4,
            start_game_id: 0,
        },
    };

    // 1回目の実行: 2パーティション（計4対局）が新規生成される
    let stats1 = PartitionedSelfPlayManager::run(config.clone());
    assert_eq!(stats1.total_partitions, 2);
    assert_eq!(stats1.completed_partitions, 2);
    assert_eq!(stats1.skipped_partitions, 0);
    assert_eq!(stats1.total_games_completed, 4);
    assert_eq!(stats1.total_io_errors, 0);

    // 生成ファイルとマーカーの存在確認
    let part0_tsv = PartitionedSelfPlayManager::partition_tsv_path(test_dir, 0);
    let part0_done = PartitionedSelfPlayManager::partition_done_path(test_dir, 0);
    let part1_tsv = PartitionedSelfPlayManager::partition_tsv_path(test_dir, 1);
    let part1_done = PartitionedSelfPlayManager::partition_done_path(test_dir, 1);

    assert!(part0_tsv.exists(), "part_0000.tsv must exist");
    assert!(part0_done.exists(), "part_0000.done must exist");
    assert!(part1_tsv.exists(), "part_0001.tsv must exist");
    assert!(part1_done.exists(), "part_0001.done must exist");

    // 2回目の実行: 既に完了しているため2パーティションともスキップ（Resume 機能）
    let stats2 = PartitionedSelfPlayManager::run(config);
    assert_eq!(stats2.total_partitions, 2);
    assert_eq!(stats2.completed_partitions, 2);
    assert_eq!(
        stats2.skipped_partitions, 2,
        "Both partitions must be skipped on resume"
    );
    assert_eq!(stats2.total_games_completed, 4);

    let _ = fs::remove_dir_all(test_dir);
}

#[test]
fn test_multi_partition_streaming_reader_across_boundaries() {
    let test_dir = "target/test_multi_partition_reader";
    let _ = fs::remove_dir_all(test_dir);
    fs::create_dir_all(test_dir).unwrap();

    // 2つのパーティションファイルにそれぞれ 3 件ずつのダミーレコードを作成
    let entries_part0 = vec![
        DatasetEntry {
            sfen: "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1".to_string(),
            score: 10,
            result: 1.0,
            move_usi: "7g7f".to_string(),
        },
        DatasetEntry {
            sfen: "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2".to_string(),
            score: -10,
            result: 0.0,
            move_usi: "3c3d".to_string(),
        },
        DatasetEntry {
            sfen: "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL b - 3".to_string(),
            score: 15,
            result: 1.0,
            move_usi: "2g2f".to_string(),
        }
    ];

    let entries_part1 = vec![
        DatasetEntry {
            sfen: "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P4P1/PP1PPPP1P/1B5R1/LNSGKGSNL w - 4".to_string(),
            score: -15,
            result: 0.0,
            move_usi: "8c8d".to_string(),
        },
        DatasetEntry {
            sfen: "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P4P1/PP1PPPP1P/1B5R1/LNSGKGSNL b - 5".to_string(),
            score: 20,
            result: 1.0,
            move_usi: "2f2e".to_string(),
        },
        DatasetEntry {
            sfen: "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P4P1/PP1PPPP1P/1B5R1/LNSGKGSNL w - 6".to_string(),
            score: -20,
            result: 0.0,
            move_usi: "8d8e".to_string(),
        }
    ];

    let tsv0 = PartitionedSelfPlayManager::partition_tsv_path(test_dir, 0);
    let tsv1 = PartitionedSelfPlayManager::partition_tsv_path(test_dir, 1);
    let done0 = PartitionedSelfPlayManager::partition_done_path(test_dir, 0);
    let done1 = PartitionedSelfPlayManager::partition_done_path(test_dir, 1);

    DatasetHandler::append_to_file(tsv0.to_str().unwrap(), &entries_part0).unwrap();
    fs::write(done0, "done").unwrap();
    DatasetHandler::append_to_file(tsv1.to_str().unwrap(), &entries_part1).unwrap();
    fs::write(done1, "done").unwrap();

    // MultiPartitionStreamingReader を初期化（batch_size = 2）
    let mut reader = MultiPartitionStreamingReader::from_directory(test_dir, 2).unwrap();
    assert_eq!(reader.file_count(), 2);

    let mut total_read = 0usize;
    let mut batches_count = 0usize;

    while let Some(batch) = reader.next_batch().unwrap() {
        assert!(batch.len() <= 2);
        total_read += batch.len();
        batches_count += 1;
    }

    assert_eq!(
        total_read, 6,
        "Total read positions must equal 6 across both partitions"
    );
    assert_eq!(
        batches_count, 3,
        "6 positions with batch_size 2 must yield exactly 3 batches"
    );

    // EOF 後の next_batch は None
    assert!(reader.next_batch().unwrap().is_none());

    // リセット後の再読み込み検証
    reader.reset();
    let batch1 = reader.next_batch().unwrap();
    assert!(batch1.is_some());
    assert_eq!(batch1.unwrap().len(), 2);

    let _ = fs::remove_dir_all(test_dir);
}

#[test]
fn test_halfkp_stream_training_integration() {
    let test_dir = "target/test_stream_training_data";
    let _ = fs::remove_dir_all(test_dir);
    fs::create_dir_all(test_dir).unwrap();

    // 局面エントリの作成
    let entries = vec![
        DatasetEntry {
            sfen: "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1".to_string(),
            score: 50,
            result: 1.0,
            move_usi: "7g7f".to_string(),
        },
        DatasetEntry {
            sfen: "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2".to_string(),
            score: -50,
            result: 0.0,
            move_usi: "3c3d".to_string(),
        },
        DatasetEntry {
            sfen: "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL b - 3".to_string(),
            score: 80,
            result: 1.0,
            move_usi: "2g2f".to_string(),
        }
    ];

    let tsv_path = PartitionedSelfPlayManager::partition_tsv_path(test_dir, 0);
    let done_path = PartitionedSelfPlayManager::partition_done_path(test_dir, 0);
    DatasetHandler::append_to_file(tsv_path.to_str().unwrap(), &entries).unwrap();
    fs::write(done_path, "done").unwrap();

    let model_path = "target/test_stream_model.bin";
    let ckpt_path = "target/test_stream_ckpt.bin";
    let _ = fs::remove_file(model_path);
    let _ = fs::remove_file(ckpt_path);

    let config = StreamTrainConfig {
        data_dir: test_dir.to_string(),
        batch_size: 2,
        lr: 0.01,
        k: 600.0,
        epochs: 2,
        checkpoint_interval_batches: 1,
        checkpoint_path: Some(ckpt_path.to_string()),
        model_output_path: model_path.to_string(),
        resume_from_checkpoint: false,
    };

    let summary = HalfKPStreamTrainer::train(config).expect("Stream training must succeed");
    assert_eq!(summary.epochs_completed, 2);
    assert_eq!(summary.total_positions_trained, 6); // 3 entries * 2 epochs
    assert!(summary.final_total_loss > 0.0);

    // モデルファイルが正常に保存され、ロード可能であることを確認
    assert!(Path::new(model_path).exists());
    let loaded_eval =
        HalfKPEvaluator::load_from_file(model_path).expect("Failed to load trained model");
    assert_eq!(loaded_eval.output_weights.len(), 256);

    // チェックポイントファイルが正常に保存され、ロード可能であることを確認
    assert!(Path::new(ckpt_path).exists());
    let loaded_trainer =
        HalfKPTrainer::load_checkpoint(ckpt_path).expect("Failed to load checkpoint");
    assert_eq!(loaded_trainer.output_weights.len(), 256);

    // クリーンアップ
    let _ = fs::remove_dir_all(test_dir);
    let _ = fs::remove_file(model_path);
    let _ = fs::remove_file(ckpt_path);
}
