use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::sync::Arc;
use tabula_shogi::board::Position;
use tabula_shogi::eval::nnue::{NNUE_HIDDEN_SIZE, NNUE_INPUT_SIZE, NNUEEvaluator};
use tabula_shogi::eval::{EvalMode, Evaluator};
use tabula_shogi::search::engine::SearchEngine;
use tabula_shogi::types::Color;

fn main() {
    println!("============================================================");
    println!("       TABULA SHOGI DEEP MATHEMATICAL & EMPIRICAL PROBE     ");
    println!("============================================================");

    // 1. candidate_nnue.bin のバイト列直接解析
    let nnue_path = "candidate_nnue.bin";
    let mut file = match File::open(nnue_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to open {}: {:?}", nnue_path, e);
            return;
        }
    };
    let mut raw_bytes = Vec::new();
    if let Err(e) = file.read_to_end(&mut raw_bytes) {
        eprintln!("Failed to read {}: {:?}", nnue_path, e);
        return;
    }
    println!("\n[1] Candidate NNUE Internal Weights Raw Inspection:");
    println!("  Total Binary Size: {} bytes", raw_bytes.len());

    let nnue = NNUEEvaluator::from_bytes(&raw_bytes).expect("Valid NNUE binary");

    // バイト列から重みを復元
    let mut cursor = 16; // magic(8) + in_dim(4) + hid_dim(4)

    // feature_weights: 2520 * 128 * 2
    let fw_count = NNUE_INPUT_SIZE * NNUE_HIDDEN_SIZE;
    let mut feature_weights = Vec::with_capacity(fw_count);
    for _ in 0..fw_count {
        let w = i16::from_le_bytes([raw_bytes[cursor], raw_bytes[cursor + 1]]);
        cursor += 2;
        feature_weights.push(w);
    }

    // feature_biases: 128 * 2
    let mut feature_biases = Vec::with_capacity(NNUE_HIDDEN_SIZE);
    for _ in 0..NNUE_HIDDEN_SIZE {
        let b = i16::from_le_bytes([raw_bytes[cursor], raw_bytes[cursor + 1]]);
        cursor += 2;
        feature_biases.push(b);
    }

    // output_weights: 256 * 2
    let mut output_weights = Vec::with_capacity(NNUE_HIDDEN_SIZE * 2);
    for _ in 0..(NNUE_HIDDEN_SIZE * 2) {
        let w = i16::from_le_bytes([raw_bytes[cursor], raw_bytes[cursor + 1]]);
        cursor += 2;
        output_weights.push(w);
    }

    // output_bias: 4 bytes
    let output_bias = i32::from_le_bytes([
        raw_bytes[cursor],
        raw_bytes[cursor + 1],
        raw_bytes[cursor + 2],
        raw_bytes[cursor + 3]
    ]);

    // feature_biases (初期値 32)
    let fb_min = feature_biases.iter().copied().min().unwrap_or(0);
    let fb_max = feature_biases.iter().copied().max().unwrap_or(0);
    let fb_sum: i64 = feature_biases.iter().map(|&x| x as i64).sum();
    let fb_avg = fb_sum as f64 / feature_biases.len() as f64;
    let fb_changed = feature_biases.iter().filter(|&&x| x != 32).count();
    println!(
        "  Hidden Biases (len={}, initial=32):",
        feature_biases.len()
    );
    println!(
        "    Range: [{}, {}], Avg: {:.3}, Changed from 32: {} / {} ({:.1}%)",
        fb_min,
        fb_max,
        fb_avg,
        fb_changed,
        feature_biases.len(),
        fb_changed as f64 / feature_biases.len() as f64 * 100.0
    );

    // output_weights (初期値 0)
    let ow_min = output_weights.iter().copied().min().unwrap_or(0);
    let ow_max = output_weights.iter().copied().max().unwrap_or(0);
    let ow_nonzero = output_weights.iter().filter(|&&x| x != 0).count();
    let ow_l1: i64 = output_weights.iter().map(|&x| (x as i64).abs()).sum();
    let ow_l2_sq: f64 = output_weights
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum();
    let ow_l2 = ow_l2_sq.sqrt();
    let ow_avg =
        output_weights.iter().map(|&x| x as f64).sum::<f64>() / output_weights.len() as f64;
    println!(
        "  Output Weights (len={}, initial=0):",
        output_weights.len()
    );
    println!(
        "    Range: [{}, {}], Non-zero: {} / {} ({:.1}%)",
        ow_min,
        ow_max,
        ow_nonzero,
        output_weights.len(),
        ow_nonzero as f64 / output_weights.len() as f64 * 100.0
    );
    println!(
        "    Avg: {:.4}, L1 Norm: {}, L2 Norm: {:.2}",
        ow_avg, ow_l1, ow_l2
    );
    println!("  Output Bias (initial=0): {}", output_bias);

    // feature_weights
    let fw_min = feature_weights.iter().copied().min().unwrap_or(0);
    let fw_max = feature_weights.iter().copied().max().unwrap_or(0);
    let fw_nonzero = feature_weights.iter().filter(|&&x| x != 0).count();
    let fw_l1: i64 = feature_weights.iter().map(|&x| (x as i64).abs()).sum();
    println!(
        "  Feature Weights (len={}, 2520x128):",
        feature_weights.len()
    );
    println!(
        "    Range: [{}, {}], Non-zero: {} / {} ({:.1}%), L1 Norm: {}",
        fw_min,
        fw_max,
        fw_nonzero,
        feature_weights.len(),
        fw_nonzero as f64 / feature_weights.len() as f64 * 100.0,
        fw_l1
    );

    // 2. データセットから局面サンプリング
    println!("\n[2] Sampling Positions from loop_dataset.tsv...");
    let dataset_path = "data/loop_dataset.tsv";
    let file = match File::open(dataset_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to open {}: {:?}", dataset_path, e);
            return;
        }
    };
    let reader = BufReader::new(file);

    let mut sampled_positions: Vec<(Position, i32, f32)> = Vec::new();
    let mut total_lines = 0;
    for l in reader.lines().map_while(Result::ok) {
        total_lines += 1;
        if total_lines % 480 == 0 && sampled_positions.len() < 1000 {
            let parts: Vec<&str> = l.split('\t').collect();
            if parts.len() >= 3
                && let Ok(pos) = Position::from_sfen(parts[0])
            {
                let score: i32 = parts[1].parse().unwrap_or(0);
                let result: f32 = parts[2].parse().unwrap_or(0.5);
                sampled_positions.push((pos, score, result));
            }
        }
    }
    println!(
        "  Sampled {} positions out of {} total recorded positions.",
        sampled_positions.len(),
        total_lines
    );

    // 3. 活性化と残差スケール分析
    println!("\n[3] Hidden Unit Activation & Residual Magnitude Analysis:");
    let mut dead_nodes_count = 0;
    let mut saturated_nodes_count = 0;
    let mut linear_nodes_count = 0;
    let mut total_node_evals = 0;

    let mut residual_magnitudes: Vec<f64> = Vec::new();
    let mut nnue_scores: Vec<i32> = Vec::new();
    let mut hce_scores: Vec<i32> = Vec::new();

    let mut node_dead_freq = vec![0usize; NNUE_HIDDEN_SIZE];
    let mut node_sat_freq = vec![0usize; NNUE_HIDDEN_SIZE];

    for (pos, _, _) in &sampled_positions {
        let b_feats = NNUEEvaluator::extract_features(pos, Color::Black);
        let mut b_acc = feature_biases.clone();
        for &feat_idx in &b_feats {
            let offset = feat_idx * NNUE_HIDDEN_SIZE;
            for i in 0..NNUE_HIDDEN_SIZE {
                b_acc[i] += feature_weights[offset + i];
            }
        }

        for i in 0..NNUE_HIDDEN_SIZE {
            total_node_evals += 1;
            if b_acc[i] <= 0 {
                dead_nodes_count += 1;
                node_dead_freq[i] += 1;
            } else if b_acc[i] >= 64 {
                saturated_nodes_count += 1;
                node_sat_freq[i] += 1;
            } else {
                linear_nodes_count += 1;
            }
        }

        let w_feats = NNUEEvaluator::extract_features(pos, Color::White);
        let mut w_acc = feature_biases.clone();
        for &feat_idx in &w_feats {
            let offset = feat_idx * NNUE_HIDDEN_SIZE;
            for i in 0..NNUE_HIDDEN_SIZE {
                w_acc[i] += feature_weights[offset + i];
            }
        }

        let mut out_accum = output_bias;
        for i in 0..NNUE_HIDDEN_SIZE {
            out_accum += (b_acc[i].clamp(0, 64) as i32) * (output_weights[i] as i32);
            out_accum +=
                (w_acc[i].clamp(0, 64) as i32) * (output_weights[NNUE_HIDDEN_SIZE + i] as i32);
        }
        let residual_cp = (out_accum / 16) as f64;
        let total_nnue = nnue.evaluate(pos);
        let total_hce = Evaluator::evaluate(pos);

        residual_magnitudes.push(residual_cp);
        nnue_scores.push(total_nnue);
        hce_scores.push(total_hce);
    }

    let dead_pct = dead_nodes_count as f64 / total_node_evals as f64 * 100.0;
    let sat_pct = saturated_nodes_count as f64 / total_node_evals as f64 * 100.0;
    let lin_pct = linear_nodes_count as f64 / total_node_evals as f64 * 100.0;
    println!(
        "  Hidden Unit States across {} evaluations (Black perspective):",
        total_node_evals
    );
    println!(
        "    Dead (<= 0):      {:.2}% ({} evals)",
        dead_pct, dead_nodes_count
    );
    println!(
        "    Saturated (>=64): {:.2}% ({} evals)",
        sat_pct, saturated_nodes_count
    );
    println!(
        "    Linear (1..63):   {:.2}% ({} evals)  <-- ACTIVE UNITS",
        lin_pct, linear_nodes_count
    );

    let dead_all_time = node_dead_freq
        .iter()
        .filter(|&&c| c == sampled_positions.len())
        .count();
    let sat_all_time = node_sat_freq
        .iter()
        .filter(|&&c| c == sampled_positions.len())
        .count();
    println!(
        "    Permanently Dead Units (0% active in all sampled positions): {} / 128",
        dead_all_time
    );
    println!(
        "    Permanently Saturated Units (100% active in all sampled positions): {} / 128",
        sat_all_time
    );

    let res_min = residual_magnitudes
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let res_max = residual_magnitudes
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let res_avg = residual_magnitudes.iter().sum::<f64>() / residual_magnitudes.len() as f64;
    let res_abs_avg = residual_magnitudes.iter().map(|&x| x.abs()).sum::<f64>()
        / residual_magnitudes.len() as f64;
    println!("\n  Residual Magnitude [output / 16] (cp):");
    println!(
        "    Min: {:.1} cp, Max: {:.1} cp, Avg: {:.2} cp, Mean Absolute: {:.2} cp",
        res_min, res_max, res_avg, res_abs_avg
    );

    // HCE と NNUE の評価値比較
    println!("\n[4] HCE vs NNUE Evaluation Discrepancy:");
    let diffs: Vec<f64> = nnue_scores
        .iter()
        .zip(hce_scores.iter())
        .map(|(&n, &h)| (n - h).abs() as f64)
        .collect();
    let mae = diffs.iter().sum::<f64>() / diffs.len() as f64;
    let diff_max = diffs.iter().copied().fold(0.0, f64::max);
    println!(
        "    Mean Absolute Error (MAE): {:.1} cp, Max Diff: {:.1} cp",
        mae, diff_max
    );

    // HCE の内訳（PSTや玉安全性がどれくらいのスコアを持っているか）
    println!("\n[5] HCE Positional Component Magnitude (PST + King Safety + Mobility):");
    let mut psts = Vec::new();
    for (pos, _, _) in sampled_positions.iter().take(200) {
        let hce_eval = Evaluator::evaluate(pos);
        let mat = NNUEEvaluator::material_stm(pos);
        let positional = hce_eval - mat;
        psts.push(positional.abs());
    }
    let pst_avg = psts.iter().sum::<i32>() as f64 / psts.len() as f64;
    let pst_max = psts.iter().copied().max().unwrap_or(0);
    println!(
        "    Average Absolute Positional Component: {:.1} cp",
        pst_avg
    );
    println!("    Max Absolute Positional Component:     {} cp", pst_max);

    // 6. 探索深さ Depth 2 における指し手一致率 (Move Agreement)
    println!("\n[6] Move Agreement Test (Depth 2, 40 random positions):");
    let mut agree_count = 0;
    let mut total_moves_tested = 0;
    let mut engine_hce = SearchEngine::new(16);
    engine_hce.eval_mode = EvalMode::Hce;

    let mut engine_nnue = SearchEngine::new(16);
    engine_nnue.eval_mode = EvalMode::Nnue(Arc::new(nnue));

    for (pos, _, _) in sampled_positions.iter().take(40) {
        total_moves_tested += 1;
        let mut p1 = pos.clone();
        let (m_hce, _) = engine_hce.search_fixed_depth(&mut p1, 2);
        let mut p2 = pos.clone();
        let (m_nnue, _) = engine_nnue.search_fixed_depth(&mut p2, 2);

        if m_hce == m_nnue {
            agree_count += 1;
        }
    }
    println!(
        "    Move Agreement (Depth 2, HCE vs Candidate NNUE): {} / {} ({:.1}%)",
        agree_count,
        total_moves_tested,
        agree_count as f64 / total_moves_tested as f64 * 100.0
    );

    println!("\n============================================================");
    println!("                      PROBE COMPLETE                        ");
    println!("============================================================");
}
