//! The embedding model memory runs on this machine: EmbeddingGemma-300m, 4-bit (onnx-community/embeddinggemma-300m-ONNX
//! on Hugging Face, one pinned revision), fetched on first use into the cache, each file checked against its pinned
//! size and SHA-256. 218 MB. The model is Google's, under the Gemma Terms of Use (ai.google.dev/gemma/terms);
//! memtrails does not ship it.

use std::fs;
use std::path::{Path, PathBuf};

use crate::download::{download_verified, with_lock, DownloadOptions, LockOptions, Pin};
use crate::error::{Error, Result};

/// The model's repository on Hugging Face.
pub const MODEL_REPO: &str = "onnx-community/embeddinggemma-300m-ONNX";
/// The pinned revision.
pub const MODEL_REVISION: &str = "5090578d9565bb06545b4552f76e6bc2c93e4a66";
/// Each file: its path in the repository, size and SHA-256.
pub const MODEL_FILES: &[(&str, u64, &str)] = &[
    ("onnx/model_q4.onnx", 519_322, "ad1dfee81a70f7944b9b9d1cc6e48075b832881cf33fab2f2b248be78f3f0043"),
    ("onnx/model_q4.onnx_data", 196_725_760, "599962c3143b040de2dd05e5975be3e9091dd067cacc6a8f7186e3203bab9e02"),
    ("tokenizer.json", 20_323_312, "4dda02faaf32bc91031dc8c88457ac272b00c1016cc679757d1c441b248b9c47"),
    ("tokenizer_config.json", 1_156_830, "3ca953eea6c3c9fcda9cf3df22949ff18b216f7c74bd6459230f3f1013953f3a"),
];

/// Where the model is fetched to in a cache folder.
pub fn default_model_dir(cache: &Path) -> PathBuf {
    cache.join(format!("embeddinggemma-300m-{}", &MODEL_REVISION[..8]))
}

fn stamp() -> String {
    MODEL_FILES.iter().map(|(rel, size, sha)| format!("{rel}:{size}:{sha}")).collect::<Vec<_>>().join("\n")
}

/// Every file in place at its pinned size, and the marker written after the last of them was verified.
pub fn model_ready(dir: &Path) -> bool {
    fs::read_to_string(dir.join(".verified")).is_ok_and(|t| t == stamp())
        && MODEL_FILES.iter().all(|(rel, size, _)| fs::metadata(dir.join(rel)).is_ok_and(|m| m.len() == *size))
}

/// The model in `dir` (default: in the cache), fetched and checked unless it is already there; the folder, ready to
/// load. Each file is fetched to a temp of its own, checked on disk and renamed into place; one run at a time per
/// folder, and the marker is taken away first, so a set left half-replaced is never taken as ready.
pub async fn ensure_model(o: &DownloadOptions, dir: Option<PathBuf>) -> Result<PathBuf> {
    let dir = dir.unwrap_or_else(|| default_model_dir(&o.cache_dir()));
    if model_ready(&dir) {
        return Ok(dir);
    }
    fs::create_dir_all(&dir).map_err(|e| Error::io(&dir, e))?;
    with_lock(&dir, LockOptions::default(), || async {
        if model_ready(&dir) {
            return Ok(dir.clone()); // another run finished it while this one waited
        }
        o.say(&format!("[memtrails] fetching {MODEL_REPO} (218 MB, Gemma Terms of Use)"));
        let marker = dir.join(".verified");
        let _ = fs::remove_file(&marker);
        let base = o
            .base_url
            .clone()
            .unwrap_or_else(|| format!("https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}"));
        for (rel, size, sha) in MODEL_FILES {
            let path = dir.join(rel);
            let parent = path.parent().unwrap_or(&dir).to_path_buf();
            let pin = Pin { size: *size, sha256: (*sha).to_string() };
            let (file, tmp) =
                download_verified(&format!("{}/{rel}", base.trim_end_matches('/')), &parent, &pin, o).await?;
            let placed = fs::rename(&file, &path).map_err(|e| Error::io(&path, e));
            let _ = fs::remove_dir_all(&tmp);
            placed?;
        }
        fs::write(&marker, stamp()).map_err(|e| Error::io(&marker, e))?;
        o.say("[memtrails] embedding model ready");
        Ok(dir.clone())
    })
    .await
}
