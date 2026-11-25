use super::match_runner::MatchResult;
use super::sprt::PromotionVerdict;
use crate::eval::EvalMode;
use crate::eval::halfkp::HalfKPEvaluator;
use crate::selfplay::dataset::{DatasetEntry, DatasetHandler};
use std::io::ErrorKind;
use std::path::Path;
use std::sync::Arc;

/// データセット・モデル永続化パス設定
#[derive(Clone, Debug)]
pub struct LoopStoragePaths {
    pub state_path: String,
    pub data_path: String,
    pub deep_data_path: String,
    pub best_model_path: String,
    pub candidate_model_path: String,
    pub candidate_ckpt_path: String,
    pub summary_path: String,
}

impl Default for LoopStoragePaths {
    fn default() -> Self {
        Self {
            state_path: "data/loop_state.txt".to_string(),
            data_path: "data/loop_dataset.tsv".to_string(),
            deep_data_path: "data/deep_dataset.tsv".to_string(),
            best_model_path: "models/best_halfkp.bin".to_string(),
            candidate_model_path: "models/candidate_halfkp.bin".to_string(),
            candidate_ckpt_path: "models/candidate_halfkp_ckpt.bin".to_string(),
            summary_path: "data/loop_summary.csv".to_string(),
        }
    }
}

/// 世代ごとの学習・評価メトリクスサマリ
#[derive(Clone, Copy, Debug)]
pub struct GenerationMetrics {
    pub dataset_len: usize,
    pub relabelled_count: usize,
    pub init_mse: f32,
    pub final_mse: f32,
    pub loss_reduction: f32,
}

/// サマリ CSV 追記および世代状態永続化のための構造化レコード
#[derive(Clone, Debug)]
pub struct GenerationSummaryRow<'a> {
    pub generation: usize,
    pub champion_mode: &'a str,
    pub duration_secs: f64,
    pub metrics: &'a GenerationMetrics,
    pub match_result: &'a MatchResult,
    pub promoted: bool,
}

/// 自律改善ループの永続ストレージ・パス衝突防護・昇格ライフサイクルを所有する深モジュール
#[derive(Clone, Debug)]
pub struct LoopWorkspace {
    paths: LoopStoragePaths,
}

impl LoopWorkspace {
    /// 全パスの検証・親ディレクトリ作成・全ストレージパスの衝突チェックを実行して開く
    pub fn open(paths: LoopStoragePaths) -> Result<Self, String> {
        let labeled_paths = [
            ("state_path", &paths.state_path),
            ("data_path", &paths.data_path),
            ("deep_data_path", &paths.deep_data_path),
            ("best_model_path", &paths.best_model_path),
            ("candidate_model_path", &paths.candidate_model_path),
            ("candidate_ckpt_path", &paths.candidate_ckpt_path),
            ("summary_path", &paths.summary_path),
        ];

        for (label, path_str) in labeled_paths {
            if path_str.trim().is_empty() {
                return Err(format!("{label} must not be empty"));
            }
            ensure_parent_dir(path_str, label)?;
        }

        // 全 7 ストレージパスのペア衝突チェック (未作成ファイルも祖先正規化で検出)
        for i in 0..labeled_paths.len() {
            for j in (i + 1)..labeled_paths.len() {
                let (label_a, path_a) = labeled_paths[i];
                let (label_b, path_b) = labeled_paths[j];
                crate::eval::ensure_distinct_paths(path_a, path_b, label_a, label_b).map_err(
                    |err| {
                        format!(
                            "{err}! Aborting to protect champion model, datasets, and checkpoints."
                        )
                    },
                )?;
            }
        }

        Ok(Self { paths })
    }

    pub fn paths(&self) -> &LoopStoragePaths {
        &self.paths
    }

    /// 現行 Champion モデルを fail-closed にロード（未存在時は初期 HCE）
    pub fn load_champion(&self) -> Result<EvalMode, String> {
        match EvalMode::load_if_exists_halfkp(&self.paths.best_model_path) {
            Ok(Some(mode)) => {
                println!(
                    "Loaded initial best HalfKP model from '{}'",
                    self.paths.best_model_path
                );
                Ok(mode)
            }
            Ok(None) => {
                println!("No existing best HalfKP model found. Starting with initial HCE.");
                Ok(EvalMode::Hce)
            }
            Err(e) => Err(format!(
                "Failed to load existing best HalfKP model '{}': {e}. Aborting to prevent silent HCE fallback.",
                self.paths.best_model_path
            )),
        }
    }

    /// 探査自己対局用の既存 Candidate モデルを fail-closed にロード
    pub fn load_candidate_if_exists(&self) -> Result<Option<EvalMode>, String> {
        EvalMode::load_if_exists_halfkp(&self.paths.candidate_model_path).map_err(|e| {
            format!(
                "Failed to load existing candidate HalfKP model '{}' (fail-closed): {e}",
                self.paths.candidate_model_path
            )
        })
    }

    /// 開始世代番号を解決（未存在時は 1、破損 state ファイルは fail-closed）
    pub fn resolve_start_generation(&self, explicit: Option<usize>) -> Result<usize, String> {
        if let Some(start) = explicit {
            return Ok(start);
        }

        match std::fs::symlink_metadata(&self.paths.state_path) {
            Ok(_) => {
                let raw = std::fs::read_to_string(&self.paths.state_path).map_err(|e| {
                    format!(
                        "Failed to read generation state file '{}': {e}",
                        self.paths.state_path
                    )
                })?;
                let last = raw.trim().parse::<usize>().map_err(|e| {
                    format!(
                        "Invalid generation state in '{}': {:?} ({e})",
                        self.paths.state_path,
                        raw.trim()
                    )
                })?;
                last.checked_add(1).ok_or_else(|| {
                    format!(
                        "Generation counter overflow in '{}': {last}",
                        self.paths.state_path
                    )
                })
            }
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(1),
            Err(e) => Err(format!(
                "Failed to inspect generation state file '{}': {e}",
                self.paths.state_path
            )),
        }
    }

    /// 深読み再評価済み局面プールへの fail-closed 追記
    pub fn append_deep_pool(&self, entries: &[DatasetEntry]) -> Result<(), String> {
        if entries.is_empty() {
            return Ok(());
        }
        DatasetHandler::append_to_file(&self.paths.deep_data_path, entries).map_err(|e| {
            format!(
                "Failed to persist deep relabeled pool to '{}': {e}",
                self.paths.deep_data_path
            )
        })
    }

    /// SPRT 昇格判定と成果物遷移（昇格時は王者原子的保存 + Candidate チェックポイント整理、非昇格時は維持）
    pub fn apply_promotion(
        &self,
        generation: usize,
        match_res: &MatchResult,
        candidate_eval: &Arc<HalfKPEvaluator>,
        current_best_eval: &mut EvalMode,
        min_promotion_games: usize,
    ) -> Result<PromotionVerdict, String> {
        let verdict = match_res.promotion_verdict(min_promotion_games);
        match verdict {
            PromotionVerdict::Promote => {
                let reason = format!(
                    "SPRT Pass (Statistically Significant Superiority, {} games >= {} required)",
                    match_res.total_games, min_promotion_games
                );
                println!(
                    "\n>>> [PROMOTION] Gen {} Candidate won ({}) with {:.1}% win rate ({:+.1} Elo). Promoting to Best Model! <<<",
                    generation,
                    reason,
                    match_res.win_rate_a * 100.0,
                    match_res.elo_diff_a
                );
                ensure_parent_dir(&self.paths.best_model_path, "best_model_path")?;
                candidate_eval
                    .save_to_file(&self.paths.best_model_path)
                    .map_err(|e| {
                        format!(
                            "Failed to write promoted best model to '{}': {e}",
                            self.paths.best_model_path
                        )
                    })?;

                // 昇格時は古い Candidate チェックポイントを整理し、次代は新王者から Warm-Start する
                // すべてのディスク更新が完了してからメモリ上の Champion 参照を切り替える
                remove_file_if_exists(&self.paths.candidate_ckpt_path).map_err(|e| {
                    format!(
                        "Promoted best model was written to '{}', but cleanup of stale checkpoint failed: {e}",
                        self.paths.best_model_path
                    )
                })?;
                let bak_path = format!("{}.bak", self.paths.candidate_ckpt_path);
                remove_file_if_exists(&bak_path).map_err(|e| {
                    format!(
                        "Promoted best model was written to '{}', but cleanup of stale checkpoint backup failed: {e}",
                        self.paths.best_model_path
                    )
                })?;

                *current_best_eval = EvalMode::HalfKP(Arc::clone(candidate_eval));
                println!(
                    "Successfully promoted and updated '{}'!",
                    self.paths.best_model_path
                );
                println!(
                    "[Progression] Gen {generation} Candidate successfully promoted! Fresh cycle will warm-start from new champion."
                );
            }
            PromotionVerdict::DeferredInsufficientGames { played, required } => {
                println!(
                    "\n>>> [GATE REJECTED] Gen {generation} Candidate passed SPRT early but had only {played} games (< {required} required). Promotion deferred. <<<"
                );
                println!(
                    "[Progression] Promotion deferred in Gen {generation} due to insufficient games. Retaining trained weights & Adam momentum for Gen {}.",
                    generation + 1
                );
            }
            PromotionVerdict::Reject => {
                println!(
                    "\n>>> [REJECTED] Gen {} Candidate did not surpass Best Model ({:.1}% win rate, Elo {:+.1}). Keeping existing best. <<<",
                    generation,
                    match_res.win_rate_a * 100.0,
                    match_res.elo_diff_a
                );
                println!(
                    "[Progression] Candidate did not beat champion in Gen {generation}. Retaining trained weights & Adam momentum for Gen {}.",
                    generation + 1
                );
            }
        }
        Ok(verdict)
    }

    /// 世代サマリ CSV 追記と世代番号の原子的永続化（fail-closed）
    pub fn record_generation(&self, row: &GenerationSummaryRow<'_>) -> Result<(), String> {
        self.append_summary_csv(row)?;
        ensure_parent_dir(&self.paths.state_path, "state_path")?;
        write_text_atomic(&self.paths.state_path, &row.generation.to_string()).map_err(|e| {
            format!(
                "Failed to persist generation state to '{}': {e}",
                self.paths.state_path
            )
        })?;
        Ok(())
    }

    fn append_summary_csv(&self, row: &GenerationSummaryRow<'_>) -> Result<(), String> {
        ensure_parent_dir(&self.paths.summary_path, "summary_path")?;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.paths.summary_path)
            .map_err(|e| {
                format!(
                    "Failed to open summary CSV '{}': {e}",
                    self.paths.summary_path
                )
            })?;

        let needs_header = file
            .metadata()
            .map_err(|e| {
                format!(
                    "Failed to inspect summary CSV '{}': {e}",
                    self.paths.summary_path
                )
            })?
            .len()
            == 0;

        use std::io::Write;
        if needs_header {
            writeln!(
                file,
                "generation,timestamp,champion_mode,duration_secs,dataset_size,relabel_count,init_mse,final_mse,reduction_pct,match_games,win_rate,elo_diff,sprt_status,promoted"
            )
            .map_err(|e| {
                format!(
                    "Failed to write header to summary CSV '{}': {e}",
                    self.paths.summary_path
                )
            })?;
        }

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let sprt_str = row
            .match_result
            .sprt
            .as_ref()
            .map(|s| format!("{:?}", s.status))
            .unwrap_or_else(|| "None".to_string());

        writeln!(
            file,
            "{},{},{},{:.2},{},{},{:.6},{:.6},{:.2},{},{:.4},{:+.1},{},{}",
            row.generation,
            timestamp,
            row.champion_mode,
            row.duration_secs,
            row.metrics.dataset_len,
            row.metrics.relabelled_count,
            row.metrics.init_mse,
            row.metrics.final_mse,
            row.metrics.loss_reduction,
            row.match_result.total_games,
            row.match_result.win_rate_a,
            row.match_result.elo_diff_a,
            sprt_str,
            row.promoted
        )
        .map_err(|e| {
            format!(
                "Failed to write row to summary CSV '{}': {e}",
                self.paths.summary_path
            )
        })?;

        Ok(())
    }
}

fn ensure_parent_dir(path_str: &str, label: &str) -> Result<(), String> {
    if let Some(parent) = Path::new(path_str).parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| {
            format!("Failed to create parent directory for {label} '{path_str}': {e}")
        })?;
    }
    Ok(())
}

fn remove_file_if_exists(path_str: &str) -> Result<(), String> {
    match std::fs::remove_file(path_str) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!("Failed to remove stale file '{path_str}': {e}")),
    }
}

fn write_text_atomic(path_str: &str, content: &str) -> std::io::Result<()> {
    use std::io::Write;
    let tmp_path = format!("{path_str}.tmp");
    let write_res = (|| -> std::io::Result<()> {
        let mut file = std::fs::File::create(&tmp_path)?;
        file.write_all(content.as_bytes())?;
        file.sync_all()?;
        Ok(())
    })();
    if let Err(e) = write_res {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e);
    }
    if let Err(e) = std::fs::rename(&tmp_path, path_str) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::sprt::{Sprt, SprtConfig};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TempDirGuard(PathBuf);

    impl TempDirGuard {
        fn new(prefix: &str) -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "tabula_shogi_ws_{prefix}_{}_{}",
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

        fn storage_paths(&self) -> LoopStoragePaths {
            LoopStoragePaths {
                state_path: self.path("data/state.txt"),
                data_path: self.path("data/dataset.tsv"),
                deep_data_path: self.path("data/deep.tsv"),
                best_model_path: self.path("models/best.bin"),
                candidate_model_path: self.path("models/cand.bin"),
                candidate_ckpt_path: self.path("models/cand_ckpt.bin"),
                summary_path: self.path("data/summary.csv"),
            }
        }
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn make_passing_match(games: usize) -> MatchResult {
        let mut sprt = Sprt::new(SprtConfig {
            elo0: 0.0,
            elo1: 50.0,
            alpha: 0.05,
            beta: 0.05,
            min_games: 0,
        });
        sprt.record_batch(games, 0, 0);
        MatchResult {
            wins_a: games,
            wins_b: 0,
            draws: 0,
            total_games: games,
            win_rate_a: 1.0,
            elo_diff_a: 1000.0,
            sprt: Some(sprt),
        }
    }

    #[test]
    fn workspace_open_rejects_collisions_and_empty_paths() {
        let tmp = TempDirGuard::new("open");
        let mut paths = tmp.storage_paths();
        paths.candidate_ckpt_path = tmp.path("models/./best.bin");
        assert!(LoopWorkspace::open(paths).is_err());

        let mut data_collision = tmp.storage_paths();
        data_collision.deep_data_path = data_collision.data_path.clone();
        assert!(LoopWorkspace::open(data_collision).is_err());

        let mut empty_paths = tmp.storage_paths();
        empty_paths.summary_path = "   ".to_string();
        assert!(LoopWorkspace::open(empty_paths).is_err());
    }

    #[test]
    fn workspace_resolve_start_generation_reads_and_fails_closed_on_corrupt_state() {
        let tmp = TempDirGuard::new("state");
        let ws = LoopWorkspace::open(tmp.storage_paths()).unwrap();

        assert_eq!(ws.resolve_start_generation(None).unwrap(), 1);
        assert_eq!(ws.resolve_start_generation(Some(7)).unwrap(), 7);

        std::fs::write(&ws.paths().state_path, "12\n").unwrap();
        assert_eq!(ws.resolve_start_generation(None).unwrap(), 13);

        std::fs::write(&ws.paths().state_path, "not-a-number").unwrap();
        assert!(ws.resolve_start_generation(None).is_err());
    }

    #[test]
    fn workspace_append_deep_pool_fails_closed_on_unwritable_path() {
        let tmp = TempDirGuard::new("deep_fail");
        let mut paths = tmp.storage_paths();
        let dir_as_file = tmp.path("deep_dir");
        std::fs::create_dir_all(&dir_as_file).unwrap();
        paths.deep_data_path = dir_as_file;

        let ws = LoopWorkspace::open(paths).unwrap();
        let entry = DatasetEntry {
            sfen: "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1".to_string(),
            score: 42,
            result: 1.0,
            move_usi: "7g7f".to_string(),
        };
        assert!(
            ws.append_deep_pool(&[entry]).is_err(),
            "Appending deep pool to a directory path must fail closed"
        );
    }

    #[test]
    fn workspace_apply_promotion_cleans_up_ckpt_on_promote_and_retains_on_reject() {
        let tmp = TempDirGuard::new("promo");
        let ws = LoopWorkspace::open(tmp.storage_paths()).unwrap();
        let ckpt_bak = format!("{}.bak", ws.paths().candidate_ckpt_path);

        std::fs::write(&ws.paths().candidate_ckpt_path, b"ckpt").unwrap();
        std::fs::write(&ckpt_bak, b"bak").unwrap();

        let candidate = Arc::new(HalfKPEvaluator::new());
        let mut champion = EvalMode::Hce;

        // 1. SPRT Fail => Reject & checkpoint + .bak retained
        let mut fail_sprt = Sprt::new(SprtConfig::default());
        fail_sprt.record_batch(0, 30, 0);
        let fail_res = MatchResult {
            wins_a: 0,
            wins_b: 30,
            draws: 0,
            total_games: 30,
            win_rate_a: 0.0,
            elo_diff_a: -1000.0,
            sprt: Some(fail_sprt),
        };
        let verdict = ws
            .apply_promotion(1, &fail_res, &candidate, &mut champion, 20)
            .unwrap();
        assert_eq!(verdict, PromotionVerdict::Reject);
        assert!(matches!(champion, EvalMode::Hce));
        assert!(!Path::new(&ws.paths().best_model_path).exists());
        assert!(Path::new(&ws.paths().candidate_ckpt_path).exists());
        assert!(Path::new(&ckpt_bak).exists());

        // 2. SPRT Pass but total_games < min_promotion_games => deferred & checkpoint retained
        let early_res = make_passing_match(6);
        let verdict = ws
            .apply_promotion(2, &early_res, &candidate, &mut champion, 20)
            .unwrap();
        assert_eq!(
            verdict,
            PromotionVerdict::DeferredInsufficientGames {
                played: 6,
                required: 20,
            }
        );
        assert!(matches!(champion, EvalMode::Hce));
        assert!(!Path::new(&ws.paths().best_model_path).exists());
        assert!(Path::new(&ws.paths().candidate_ckpt_path).exists());
        assert!(Path::new(&ckpt_bak).exists());

        // 3. SPRT Pass and total_games >= min_promotion_games => promoted & checkpoint + .bak removed
        let pass_res = make_passing_match(24);
        let verdict = ws
            .apply_promotion(3, &pass_res, &candidate, &mut champion, 20)
            .unwrap();
        assert_eq!(verdict, PromotionVerdict::Promote);
        assert!(matches!(champion, EvalMode::HalfKP(_)));
        assert!(Path::new(&ws.paths().best_model_path).exists());
        assert!(!Path::new(&ws.paths().candidate_ckpt_path).exists());
        assert!(!Path::new(&ckpt_bak).exists());
    }

    #[test]
    fn workspace_record_generation_writes_csv_header_once_and_persists_state() {
        let tmp = TempDirGuard::new("record");
        let ws = LoopWorkspace::open(tmp.storage_paths()).unwrap();
        let metrics = GenerationMetrics {
            dataset_len: 100,
            relabelled_count: 10,
            init_mse: 0.25,
            final_mse: 0.20,
            loss_reduction: 20.0,
        };
        let match_res = make_passing_match(20);

        for generation in 1..=2 {
            ws.record_generation(&GenerationSummaryRow {
                generation,
                champion_mode: "HalfKP",
                duration_secs: 1.5,
                metrics: &metrics,
                match_result: &match_res,
                promoted: generation == 2,
            })
            .unwrap();
        }

        let csv = std::fs::read_to_string(&ws.paths().summary_path).unwrap();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 3, "Expected header + 2 rows");
        assert!(lines[0].starts_with("generation,timestamp,champion_mode"));
        assert!(lines[1].starts_with("1,"));
        assert!(lines[2].starts_with("2,"));
        assert_eq!(
            std::fs::read_to_string(&ws.paths().state_path)
                .unwrap()
                .trim(),
            "2"
        );
    }
}
