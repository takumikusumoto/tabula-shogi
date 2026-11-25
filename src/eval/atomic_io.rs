use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Component, Path, PathBuf};

/// Removes a temporary file on drop unless explicitly disarmed after a successful rename.
struct TempFileGuard {
    path: PathBuf,
    armed: bool,
}

impl TempFileGuard {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn validate_file_path(path: &Path) -> io::Result<()> {
    if path.as_os_str().is_empty() || path.file_name().is_none() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("Invalid target file path '{}'", path.display()),
        ));
    }
    Ok(())
}

fn ensure_parent_dir(path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn normalize_lexical(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if matches!(
                    normalized.components().next_back(),
                    Some(Component::Normal(_))
                ) {
                    normalized.pop();
                } else {
                    normalized.push(component.as_os_str());
                }
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

/// Resolves a path to a canonical comparable path even when the target file does not yet exist
/// by canonicalizing its nearest existing ancestor and re-appending the remaining relative tail.
pub(super) fn resolve_comparable_path(path: &Path) -> io::Result<PathBuf> {
    validate_file_path(path)?;
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let lexical = normalize_lexical(&abs);

    if let Ok(canon) = fs::canonicalize(&lexical) {
        return Ok(canon);
    }

    let mut tail = Vec::new();
    let mut cursor = lexical.as_path();
    while let Some(name) = cursor.file_name() {
        tail.push(name.to_os_string());
        match cursor.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => {
                if let Ok(mut base) = fs::canonicalize(parent) {
                    for seg in tail.into_iter().rev() {
                        base.push(seg);
                    }
                    return Ok(base);
                }
                cursor = parent;
            }
            _ => break,
        }
    }

    Ok(lexical)
}

/// Ensures two file paths do not resolve to the same destination, even if neither file exists yet.
pub(crate) fn ensure_distinct_paths(
    path_a: &str,
    path_b: &str,
    label_a: &str,
    label_b: &str,
) -> Result<(), String> {
    let resolved_a = resolve_comparable_path(Path::new(path_a))
        .map_err(|e| format!("Invalid {label_a} path '{path_a}': {e}"))?;
    let resolved_b = resolve_comparable_path(Path::new(path_b))
        .map_err(|e| format!("Invalid {label_b} path '{path_b}': {e}"))?;
    if paths_collide(&resolved_a, &resolved_b) {
        return Err(format!(
            "{label_a} and {label_b} paths must be different; both resolve to '{}'",
            resolved_a.display()
        ));
    }
    Ok(())
}

fn paths_collide(a: &Path, b: &Path) -> bool {
    if a == b {
        return true;
    }
    #[cfg(windows)]
    {
        if a.to_string_lossy()
            .eq_ignore_ascii_case(&b.to_string_lossy())
        {
            return true;
        }
    }
    false
}

fn replace_file_atomically(src: &Path, dst: &Path, backup: Option<&Path>) -> io::Result<()> {
    // Retry atomic rename briefly before any destructive fallback (transient locks / AV scanners).
    let mut last_err = None;
    for attempt in 0..3 {
        match fs::rename(src, dst) {
            Ok(()) => return Ok(()),
            Err(err) => {
                last_err = Some(err);
                if attempt < 2 {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
            }
        }
    }

    // Only attempt delete-then-rename when the destination could actually be removed.
    match fs::remove_file(dst) {
        Ok(()) => match fs::rename(src, dst) {
            Ok(()) => Ok(()),
            Err(retry_err) => {
                if let Some(bak) = backup
                    && bak.exists()
                {
                    let _ = fs::copy(bak, dst);
                }
                Err(retry_err)
            }
        },
        Err(_) => Err(last_err.expect("rename failed at least once")),
    }
}

/// Atomically writes a binary file via `{path}.tmp` with `sync_all`, fail-closed atomic `.bak`
/// preservation of the previous file (via `{path}.bak.tmp` -> `{path}.bak`), and RAII cleanup of
/// temporary files on any failure.
pub(super) fn write_atomic_with_backup<F>(path: &str, write_fn: F) -> io::Result<()>
where
    F: FnOnce(&mut BufWriter<File>) -> io::Result<()>,
{
    let target_path = Path::new(path);
    validate_file_path(target_path)?;
    ensure_parent_dir(target_path)?;

    let tmp_path = PathBuf::from(format!("{path}.tmp"));
    let bak_path = PathBuf::from(format!("{path}.bak"));

    let mut tmp_guard = TempFileGuard::new(tmp_path.clone());
    {
        let file = File::create(&tmp_path)?;
        let mut writer = BufWriter::new(file);
        write_fn(&mut writer)?;
        writer.flush()?;
        let file = writer
            .into_inner()
            .map_err(io::IntoInnerError::into_error)?;
        file.sync_all()?;
    }

    if target_path.try_exists()? {
        let bak_tmp_path = PathBuf::from(format!("{path}.bak.tmp"));
        let mut bak_tmp_guard = TempFileGuard::new(bak_tmp_path.clone());
        {
            let mut src_file = File::open(target_path)?;
            let mut bak_file = File::create(&bak_tmp_path)?;
            io::copy(&mut src_file, &mut bak_file)?;
            bak_file.sync_all()?;
        }
        replace_file_atomically(&bak_tmp_path, &bak_path, None)?;
        bak_tmp_guard.disarm();
    }

    replace_file_atomically(&tmp_path, target_path, Some(&bak_path))?;
    tmp_guard.disarm();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(prefix: &str) -> Self {
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "tabula_atomic_io_{prefix}_{}_{}",
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

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn write_atomic_creates_backup_and_cleans_temp_files() {
        let dir = TestDir::new("backup");
        let target = dir.path("sub/artifact.bin");
        let tmp = format!("{target}.tmp");
        let bak = format!("{target}.bak");
        let bak_tmp = format!("{target}.bak.tmp");

        write_atomic_with_backup(&target, |w| w.write_all(b"gen-1")).unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"gen-1");
        assert!(!Path::new(&tmp).exists());
        assert!(!Path::new(&bak).exists());

        write_atomic_with_backup(&target, |w| w.write_all(b"gen-2")).unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"gen-2");
        assert_eq!(fs::read(&bak).unwrap(), b"gen-1");
        assert!(!Path::new(&tmp).exists());
        assert!(!Path::new(&bak_tmp).exists());
    }

    #[test]
    fn write_atomic_cleans_up_tmp_and_preserves_target_on_write_error() {
        let dir = TestDir::new("err");
        let target = dir.path("artifact.bin");
        let tmp = format!("{target}.tmp");

        write_atomic_with_backup(&target, |w| w.write_all(b"stable")).unwrap();
        let res = write_atomic_with_backup(&target, |w| {
            w.write_all(b"partial")?;
            Err(io::Error::other("simulated write failure"))
        });
        assert!(res.is_err());
        assert_eq!(fs::read(&target).unwrap(), b"stable");
        assert!(
            !Path::new(&tmp).exists(),
            "temporary file must be removed on failure"
        );
    }

    #[test]
    fn write_atomic_rejects_empty_and_directory_paths() {
        assert!(write_atomic_with_backup("", |_| Ok(())).is_err());
        assert!(write_atomic_with_backup(".", |_| Ok(())).is_err());
    }

    #[test]
    fn ensure_distinct_paths_catches_nonexistent_collisions() {
        let dir = TestDir::new("distinct");
        let a = dir.path("nested/model.bin");
        let b = dir.path("nested/./other/../model.bin");
        let c = dir.path("nested/ckpt.bin");

        assert!(ensure_distinct_paths(&a, &b, "model", "checkpoint").is_err());
        assert!(ensure_distinct_paths(&a, &c, "model", "checkpoint").is_ok());
    }
}
