use std::fs;
use std::path::{Path, PathBuf};

/// RAII ガード付きテスト専用一時作業ディレクトリ
/// スコープを抜けた際（テストパニック時含む）に自動でディレクトリごと完全削除される。
#[allow(dead_code)]
pub struct TestTempDir {
    dir: PathBuf,
}

impl TestTempDir {
    #[allow(dead_code)]
    pub fn new(prefix: &str) -> Self {
        let unique_name = format!(
            "{}_{}_{}",
            prefix,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );
        let dir = std::env::temp_dir().join(unique_name);
        fs::create_dir_all(&dir).expect("Failed to create test temp directory");
        Self { dir }
    }

    #[allow(dead_code)]
    pub fn path(&self) -> &Path {
        &self.dir
    }

    #[allow(dead_code)]
    pub fn file_path(&self, filename: &str) -> String {
        self.dir
            .join(filename)
            .to_str()
            .expect("Valid UTF-8 path")
            .to_string()
    }
}

impl Drop for TestTempDir {
    fn drop(&mut self) {
        if self.dir.exists() {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }
}
