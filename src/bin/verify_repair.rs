use std::fs::File;
use std::io::{BufRead, BufReader};
use tabula_shogi::board::Position;
use tabula_shogi::eval::Evaluator;
use tabula_shogi::eval::nnue::NNUEEvaluator;
use tabula_shogi::eval::trainer::NNUETrainer;
use tabula_shogi::selfplay::dataset::DatasetHandler;
use tabula_shogi::types::Color;

fn main() {
    println!("============================================================");
    println!("   TABULA SHOGI ARCHITECTURE V5 EMPIRICAL REPAIR VERIFICATION");
    println!("============================================================");

    // 1. 本番データセットのロード (10,000局面)
    let dataset_path = "data/loop_dataset.tsv";
    let file = File::open(dataset_path).expect("Failed to open loop_dataset.tsv");
    let reader = BufReader::new(file);

    let mut dataset = Vec::with_capacity(10_000);
    for line in reader.lines().map_while(Result::ok) {
        if let Some(entry) = DatasetHandler::parse_entry(&line) {
            dataset.push(entry);
            if dataset.len() >= 10_000 {
                break;
            }
        }
    }
    println!(
        "[1] Loaded {} sample positions from production dataset.",
        dataset.len()
    );

    // 2. 新アーキテクチャ (TABU_NN5) による学習
    let mut trainer = NNUETrainer::new();
    println!("[2] Starting 5-epoch training with Range Penalty + AdamW + Mover-first...");
    let (eval, init_loss, final_loss) = trainer.train_dataset(&dataset, 5, 0.001, 64, 400.0);
    println!(
        "    Initial Loss: {:.6}, Final Loss: {:.6} (Loss Reduced: {:.2}%)",
        init_loss,
        final_loss,
        (init_loss - final_loss) / init_loss * 100.0
    );

    // 3. 隠れ層バイアスの検査 (死滅ニューロン自己修復の検証)
    let fb = &eval.feature_biases;
    let fb_min = fb.iter().copied().min().unwrap_or(0);
    let fb_max = fb.iter().copied().max().unwrap_or(0);
    let fb_avg = fb.iter().map(|&x| x as f64).sum::<f64>() / fb.len() as f64;
    let dead_neurons = fb.iter().filter(|&&x| x <= 0).count();
    println!("\n[3] Hidden Layer Biases (128 neurons, ClippedReLU linear range [0, 64]):");
    println!("    Min: {}, Max: {}, Avg: {:.2}", fb_min, fb_max, fb_avg);
    println!(
        "    Dead Neurons (bias <= 0): {} / 128 ({:.2}%)",
        dead_neurons,
        dead_neurons as f64 / 128.0 * 100.0
    );

    // 4. 重み量子化の健全性検査 (AdamW Weight Decay による肥大化防止)
    let mut saturated_weights = 0;
    let mut total_weights = 0;
    for row in &eval.feature_weights {
        for &w in row {
            total_weights += 1;
            if w == 127 || w == -127 {
                saturated_weights += 1;
            }
        }
    }
    println!("\n[4] Weight Quantization Saturation (Clamped at [-127, 127]):");
    println!(
        "    Saturated Weights: {} / {} ({:.2}%)",
        saturated_weights,
        total_weights,
        saturated_weights as f64 / total_weights as f64 * 100.0
    );

    // 5. 評価値残差の有界性検査 (Bounded Residual [-300, 300] cp)
    let mut residuals = Vec::new();
    let mut hce_diffs = Vec::new();
    for entry in dataset.iter().take(500) {
        if let Ok(pos) = Position::from_sfen(&entry.sfen) {
            let eval_score = eval.evaluate(&pos);
            let mat = NNUEEvaluator::material_stm(&pos);
            let res = eval_score - mat;
            residuals.push(res);

            let hce_score = Evaluator::evaluate(&pos);
            hce_diffs.push((eval_score - hce_score).abs() as f64);
        }
    }

    let res_min = residuals.iter().copied().min().unwrap_or(0);
    let res_max = residuals.iter().copied().max().unwrap_or(0);
    let res_avg = residuals.iter().map(|&x| x.abs() as f64).sum::<f64>() / residuals.len() as f64;
    let res_violating = residuals.iter().filter(|&&r| r.abs() > 300).count();
    println!("\n[5] Residual Distribution across 500 positions:");
    println!(
        "    Residual Min: {} cp, Max: {} cp, Avg Absolute: {:.1} cp",
        res_min, res_max, res_avg
    );
    println!(
        "    Violations (|residual| > 300 cp): {} (Strictly Bounded!)",
        res_violating
    );

    let mae = hce_diffs.iter().sum::<f64>() / hce_diffs.len() as f64;
    println!("\n[6] HCE vs TABU_NN5 Evaluation Alignment:");
    println!("    Mean Absolute Error (MAE): {:.1} cp", mae);

    // 7. 手番対称性検証
    println!("\n[7] Move-first Symmetry Check:");
    let pos_start = Position::startpos();
    let b_eval = eval.evaluate(&pos_start);
    let mut pos_white = pos_start.clone();
    pos_white.side_to_move = Color::White;
    let w_eval = eval.evaluate(&pos_white);
    println!(
        "    Startpos Black: {} cp, Startpos White: {} cp (Diff: {} cp)",
        b_eval,
        w_eval,
        (b_eval - w_eval).abs()
    );

    println!("\n============================================================");
    if dead_neurons == 0 && res_violating == 0 && (b_eval - w_eval).abs() == 0 {
        println!("  >>> ARCHITECTURAL VERIFICATION: 100% SUCCESS <<<");
    } else {
        println!("  >>> ARCHITECTURAL VERIFICATION: SOME CHECKS FAILED <<<");
    }
    println!("============================================================");
}
