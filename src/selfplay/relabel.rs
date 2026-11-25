use super::dataset::DatasetEntry;
use crate::board::Position;
use crate::eval::EvalMode;
use crate::search::SearchEngine;

/// サンプリングされたデータセットのうち指定件数をマルチスレッドで深い探索により再評価する (Deep Relabeling)
/// 戻り値: 深読み再評価が正常完了（または詰み証明）された高品質エントリのリスト（未完了・中断は含まれない）
pub fn relabel_deep(
    entries: &mut [DatasetEntry],
    count: usize,
    depth: u8,
    threads: usize,
    eval_mode: &EvalMode,
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

    let mut all_successes = Vec::with_capacity(target_len);
    std::thread::scope(|s| {
        let handles: Vec<_> = slice_to_relabel
            .chunks_mut(chunk_size)
            .map(|chunk| {
                s.spawn(move || {
                    let mut engine = SearchEngine::new(4);
                    engine.eval_mode = eval_mode.clone();
                    engine.max_nodes = Some(30_000); // 1局面最大3万ノードで確実に打ち切り、ハング・長時間スタックを完全防止
                    engine.use_book = false; // 深読み再評価では定跡手をスキップし、純粋な深さNの探索評価値を算出

                    let mut local_successes = Vec::new();
                    for entry in chunk {
                        if let Ok(mut pos) = Position::from_sfen(&entry.sfen) {
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
                    local_successes
                })
            })
            .collect();

        for handle in handles {
            if let Ok(local_successes) = handle.join() {
                all_successes.extend(local_successes);
            }
        }
    });

    all_successes
}
