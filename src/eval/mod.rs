mod atomic_io;
pub mod evaluator;
pub mod halfkp;
pub mod halfkp_trainer;
pub mod nnue;
pub mod trainer;

pub(crate) use atomic_io::ensure_distinct_paths;
pub use evaluator::{EvalBreakdown, Evaluator};
pub use halfkp::{
    HALFKP_HIDDEN_SIZE, HALFKP_INPUT_SIZE, HALFKP_MAGIC, HALFKP_PIECE_SIZE, HalfKPAccumulator,
    HalfKPEvaluator,
};
pub use halfkp_trainer::{
    HalfKPCheckpointSummary, HalfKPExportReport, HalfKPInitMode, HalfKPInitPolicy,
    HalfKPQuantizationStats, HalfKPTargetPolicy, HalfKPTrainProgress, HalfKPTrainSessionConfig,
    HalfKPTrainSessionOutcome, HalfKPTrainSource, MAX_BATCH_SIZE, TrainStepLoss,
    export_halfkp_checkpoint, inspect_halfkp_checkpoint, train_halfkp_session,
    train_halfkp_session_with_progress,
};
pub use nnue::{MAX_EVAL_CP, NNUEEvaluator, RESIDUAL_BOUND_CP};
pub use trainer::NNUETrainer;

use crate::board::Position;
use std::io::Read;
use std::sync::Arc;

/// 評価関数の動作モード（fail-closed モデルロード・種別解決・推論の単一モジュール）
#[derive(Clone, Debug, Default)]
pub enum EvalMode {
    /// 手動評価関数 (Hand-Crafted Evaluation, デフォルト)
    #[default]
    Hce,
    /// スクラッチ NNUE 評価ネットワーク (1駒特徴量)
    Nnue(Arc<NNUEEvaluator>),
    /// 本格 HalfKP 評価ネットワーク (自玉81マス×全駒2,520)
    HalfKP(Arc<HalfKPEvaluator>),
}

impl EvalMode {
    /// Built-in default-initialized HalfKP evaluator.
    pub fn builtin_halfkp() -> Self {
        EvalMode::HalfKP(Arc::new(HalfKPEvaluator::new()))
    }

    /// Built-in default-initialized NNUE evaluator.
    pub fn builtin_nnue() -> Self {
        EvalMode::Nnue(Arc::new(NNUEEvaluator::new()))
    }

    /// Canonical mode label (`"HCE"`, `"NNUE"`, `"HalfKP"`).
    pub fn label(&self) -> &'static str {
        match self {
            EvalMode::Hce => "HCE",
            EvalMode::Nnue(_) => "NNUE",
            EvalMode::HalfKP(_) => "HalfKP",
        }
    }

    /// Fail-closed load of a `TABU_HK2` HalfKP Evaluation Model file.
    pub fn load_halfkp(path: &str) -> Result<Self, String> {
        HalfKPEvaluator::load_from_file(path).map(|e| EvalMode::HalfKP(Arc::new(e)))
    }

    /// Fail-closed load of a `TABU_NN5` NNUE Evaluation Model file.
    pub fn load_nnue(path: &str) -> Result<Self, String> {
        NNUEEvaluator::load_from_file(path).map(|e| EvalMode::Nnue(Arc::new(e)))
    }

    /// Loads a HalfKP Evaluation Model if `path` exists (fail-closed if corrupt, unreadable,
    /// or a broken symlink), or returns `Ok(None)` only when the path is truly absent.
    pub fn load_if_exists_halfkp(path: &str) -> Result<Option<Self>, String> {
        Self::load_if_exists(path, "HalfKP", Self::load_halfkp)
    }

    /// Loads an NNUE Evaluation Model if `path` exists (fail-closed if corrupt, unreadable,
    /// or a broken symlink), or returns `Ok(None)` only when the path is truly absent.
    pub fn load_if_exists_nnue(path: &str) -> Result<Option<Self>, String> {
        Self::load_if_exists(path, "NNUE", Self::load_nnue)
    }

    fn load_if_exists(
        path: &str,
        kind: &str,
        loader: fn(&str) -> Result<Self, String>,
    ) -> Result<Option<Self>, String> {
        match std::fs::symlink_metadata(path) {
            Ok(_) => loader(path).map(Some),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(format!("Failed to inspect {kind} model path '{path}': {e}")),
        }
    }

    /// Resolves a CLI/user evaluation model specification fail-closed without exiting the process.
    ///
    /// Supported forms:
    /// - Built-in keywords (case-insensitive): `"hce"`, `"nnue"`, `"halfkp"`
    /// - Explicit typed prefixes (case-insensitive): `"halfkp:<path>"`, `"nnue:<path>"`
    /// - Bare file path: inspects the 8-byte magic header first to dispatch `TABU_HK2` HalfKP or
    ///   `TABU_NN5` NNUE without reading unrelated large files into RAM, while rejecting legacy
    ///   `TABU_HKP` explicitly.
    pub fn resolve_spec(spec: &str) -> Result<(String, Self), String> {
        if spec.eq_ignore_ascii_case("hce") {
            return Ok(("HCE".to_string(), EvalMode::Hce));
        }
        if spec.eq_ignore_ascii_case("nnue") {
            return Ok(("NNUE(built-in)".to_string(), Self::builtin_nnue()));
        }
        if spec.eq_ignore_ascii_case("halfkp") {
            return Ok(("HalfKP(built-in)".to_string(), Self::builtin_halfkp()));
        }

        if let Some((prefix, path)) = spec.split_once(':') {
            if prefix.eq_ignore_ascii_case("halfkp") {
                if path.trim().is_empty() {
                    return Err("Empty path after 'halfkp:' prefix (fail-closed).".to_string());
                }
                let mode = Self::load_halfkp(path)?;
                return Ok((path.to_string(), mode));
            }
            if prefix.eq_ignore_ascii_case("nnue") {
                if path.trim().is_empty() {
                    return Err("Empty path after 'nnue:' prefix (fail-closed).".to_string());
                }
                let mode = Self::load_nnue(path)?;
                return Ok((path.to_string(), mode));
            }
        }

        if let Err(e) = std::fs::symlink_metadata(spec) {
            if e.kind() == std::io::ErrorKind::NotFound {
                return Err(format!(
                    "Evaluation model file '{spec}' does not exist (fail-closed)."
                ));
            }
            return Err(format!(
                "Failed to inspect evaluation model path '{spec}': {e}"
            ));
        }

        // Peek only the first 8 bytes so accidental non-model files (e.g. 300 MiB checkpoints)
        // never get loaded into memory via NNUE's fs::read fallback.
        let mut magic = [0u8; 8];
        {
            let mut file = std::fs::File::open(spec)
                .map_err(|e| format!("Failed to open evaluation model '{spec}': {e}"))?;
            file.read_exact(&mut magic).map_err(|e| {
                format!(
                    "Failed to read 8-byte magic header from evaluation model '{spec}': {e} (fail-closed)"
                )
            })?;
        }

        if &magic == halfkp::HALFKP_LEGACY_MAGIC {
            return Err(
                "Legacy HalfKP format TABU_HKP is not supported by the scaled evaluator; re-export the original float checkpoint with `tabula-shogi export-halfkp`"
                    .to_string(),
            );
        }
        if &magic == HALFKP_MAGIC {
            let mode = Self::load_halfkp(spec)?;
            return Ok((spec.to_string(), mode));
        }
        if &magic == b"TABU_NN5" {
            let mode = Self::load_nnue(spec)?;
            return Ok((spec.to_string(), mode));
        }

        Err(format!(
            "Unrecognized evaluation model header {:?} in '{spec}' (expected TABU_HK2 or TABU_NN5; fail-closed).",
            String::from_utf8_lossy(&magic)
        ))
    }

    #[inline(always)]
    pub fn evaluate(&self, pos: &Position) -> i32 {
        match self {
            EvalMode::Hce => Evaluator::evaluate(pos),
            EvalMode::Nnue(nnue) => nnue.evaluate(pos),
            EvalMode::HalfKP(halfkp) => halfkp.evaluate(pos),
        }
    }
}

/// 評価値 (centipawns) を勝率 [0.0, 1.0] に変換するシグモイド関数 (Elo 式: 1 / (1 + 10^(-score/k)))
#[inline(always)]
pub fn sigmoid_win_rate(score: f32, k: f32) -> f32 {
    1.0 / (1.0 + 10.0f32.powf(-score / k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(prefix: &str) -> Self {
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "tabula_eval_mode_{prefix}_{}_{}",
                std::process::id(),
                id
            ));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn path(&self, name: &str) -> String {
            self.0.join(name).to_string_lossy().into_owned()
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn resolve_spec_supports_builtin_keywords_and_labels() {
        let (n1, m1) = EvalMode::resolve_spec("hce").unwrap();
        assert_eq!(n1, "HCE");
        assert_eq!(m1.label(), "HCE");

        let (n2, m2) = EvalMode::resolve_spec("NNUE").unwrap();
        assert_eq!(n2, "NNUE(built-in)");
        assert_eq!(m2.label(), "NNUE");

        let (n3, m3) = EvalMode::resolve_spec("HalfKP").unwrap();
        assert_eq!(n3, "HalfKP(built-in)");
        assert_eq!(m3.label(), "HalfKP");
    }

    #[test]
    fn resolve_spec_loads_halfkp_and_nnue_polymorphically_and_by_prefix() {
        let dir = TempDir::new("poly");
        let hkp_path = dir.path("model_hkp.bin");
        let nnue_path = dir.path("model_nnue.bin");

        HalfKPEvaluator::new().save_to_file(&hkp_path).unwrap();
        NNUEEvaluator::new().save_to_file(&nnue_path).unwrap();

        let (_, m_hkp) = EvalMode::resolve_spec(&hkp_path).unwrap();
        assert_eq!(m_hkp.label(), "HalfKP");

        let (_, m_nnue) = EvalMode::resolve_spec(&nnue_path).unwrap();
        assert_eq!(m_nnue.label(), "NNUE");

        let (_, m_pref_hkp) = EvalMode::resolve_spec(&format!("halfkp:{hkp_path}")).unwrap();
        assert_eq!(m_pref_hkp.label(), "HalfKP");

        let (_, m_pref_nnue) = EvalMode::resolve_spec(&format!("NNUE:{nnue_path}")).unwrap();
        assert_eq!(m_pref_nnue.label(), "NNUE");
    }

    #[test]
    fn resolve_spec_rejects_empty_prefix_missing_corrupt_checkpoint_and_legacy_halfkp() {
        let dir = TempDir::new("fail_closed");
        assert!(EvalMode::resolve_spec("halfkp:").is_err());
        assert!(EvalMode::resolve_spec("nnue:   ").is_err());

        let missing = dir.path("does_not_exist.bin");
        assert!(EvalMode::resolve_spec(&missing).is_err());

        let corrupt = dir.path("corrupt.bin");
        fs::write(&corrupt, b"NOT_A_VALID_MODEL").unwrap();
        assert!(EvalMode::resolve_spec(&corrupt).is_err());
        assert!(EvalMode::load_if_exists_halfkp(&corrupt).is_err());
        assert!(EvalMode::load_if_exists_nnue(&corrupt).is_err());
        assert!(EvalMode::load_if_exists_halfkp(&missing).unwrap().is_none());
        assert!(EvalMode::load_if_exists_nnue(&missing).unwrap().is_none());

        // Accidentally passing a checkpoint header must fail fast on the 8-byte peek
        let ckpt_like = dir.path("ckpt_header.bin");
        fs::write(&ckpt_like, b"TB_HKPCK12345678").unwrap();
        let ckpt_err = EvalMode::resolve_spec(&ckpt_like).unwrap_err();
        assert!(ckpt_err.contains("Unrecognized evaluation model header"));

        let legacy = dir.path("legacy.bin");
        fs::write(&legacy, b"TABU_HKP").unwrap();
        let err = EvalMode::resolve_spec(&legacy).expect_err("Legacy TABU_HKP must be rejected");
        assert!(err.contains("Legacy HalfKP format TABU_HKP"));
        assert!(err.contains("export-halfkp"));
    }

    #[cfg(unix)]
    #[test]
    fn load_if_exists_fails_closed_on_broken_symlink() {
        let dir = TempDir::new("symlink");
        let dangling_target = dir.path("missing_target.bin");
        let link = dir.path("broken_link.bin");
        std::os::unix::fs::symlink(&dangling_target, &link).unwrap();

        assert!(
            EvalMode::load_if_exists_halfkp(&link).is_err(),
            "broken symlink must fail closed rather than silently returning Ok(None)"
        );
    }
}
