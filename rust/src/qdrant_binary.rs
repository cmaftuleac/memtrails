//! The Qdrant binary memory runs when you give it no Qdrant server of your own: one pinned release, fetched on first
//! use from Qdrant's GitHub release, verified against the size and SHA-256 pinned below, unpacked next to it (the
//! cache folder), and trusted after that through a `.verified` marker while the folder is this user's alone.
//!
//! Qdrant publishes no checksums, so the hashes are pinned here from its release. Upgrading means a new version and new
//! hashes; Qdrant reads storage only from the minor version before it, so after a version change the index is written
//! again from the logs rather than migrated: it is derived data.

use std::fs;
use std::path::{Component, Path, PathBuf};

use crate::download::{download_verified, private_cache, with_lock, DownloadOptions, LockOptions, Pin, TempFolder};
use crate::error::{Error, Result};
use crate::store::blocking;
use crate::sys;

/// The Qdrant version memtrails runs.
pub const QDRANT_VERSION: &str = "1.20.0";

/// One platform's release archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QdrantAsset {
    /// The archive's file name in the release.
    pub asset: String,
    /// Its size.
    pub size: u64,
    /// Its SHA-256.
    pub sha256: String,
    /// The executable's name inside the archive.
    pub exe: String,
}

/// The pinned archives, per `platform-arch` as Node.js names them. Linux takes the static musl build: it runs on any
/// distribution.
pub const QDRANT_ASSETS: &[(&str, &str, u64, &str, &str)] = &[
    (
        "darwin-arm64",
        "qdrant-aarch64-apple-darwin.tar.gz",
        29_070_255,
        "531904e822050790bb8c1da6705f801ee48b91eb4c229d0d605ff06f6331b20a",
        "qdrant",
    ),
    (
        "darwin-x64",
        "qdrant-x86_64-apple-darwin.tar.gz",
        31_917_921,
        "94a2e6f6b8ef3a3b5756a11172fd66554ac753e0a71c1d8d8b5349356c084f4d",
        "qdrant",
    ),
    (
        "linux-arm64",
        "qdrant-aarch64-unknown-linux-musl.tar.gz",
        31_475_234,
        "bd8a458b8ef8006ce2c603c5bd6d1d973e39d787f794008dd959112e747c3bb7",
        "qdrant",
    ),
    (
        "linux-x64",
        "qdrant-x86_64-unknown-linux-musl.tar.gz",
        33_773_335,
        "b03a64616a1f41ddab78b480a318466541765f19835318113b1821cbea8df52c",
        "qdrant",
    ),
    (
        "win32-x64",
        "qdrant-x86_64-pc-windows-msvc.zip",
        30_801_664,
        "6d13553584fd0411f00637519e293331ff0aa1fb79bd9e30db45839bb75de2be",
        "qdrant.exe",
    ),
];

/// This machine as Node.js names it: `darwin-arm64`, `linux-x64`, `win32-x64`, …
pub fn platform() -> String {
    let os = match std::env::consts::OS {
        "macos" => "darwin",
        "windows" => "win32",
        other => other,
    };
    let arch = match std::env::consts::ARCH {
        "aarch64" => "arm64",
        "x86_64" => "x64",
        other => other,
    };
    format!("{os}-{arch}")
}

/// The archive for a platform, `None` where Qdrant ships none.
pub fn qdrant_asset(platform: &str) -> Option<QdrantAsset> {
    QDRANT_ASSETS.iter().find(|a| a.0 == platform).map(|a| QdrantAsset {
        asset: a.1.into(),
        size: a.2,
        sha256: a.3.into(),
        exe: a.4.into(),
    })
}

/// Where the binary is unpacked: `<cache>/qdrant-<version>`.
pub fn qdrant_dir(cache: &Path) -> PathBuf {
    cache.join(format!("qdrant-{QDRANT_VERSION}"))
}

/// Qdrant's release downloads.
pub fn qdrant_release_base() -> String {
    format!("https://github.com/qdrant/qdrant/releases/download/v{QDRANT_VERSION}")
}

/// The binary is in place, its archive was verified when it was unpacked, and no one else could have changed it since.
pub async fn qdrant_ready(cache: &Path, a: &QdrantAsset) -> bool {
    let (cache, a) = (cache.to_path_buf(), a.clone());
    blocking(move || ready_now(&cache, &a)).await.unwrap_or(false)
}

fn ready_now(cache: &Path, a: &QdrantAsset) -> bool {
    let dir = qdrant_dir(cache);
    let bin = dir.join(&a.exe);
    let marker = dir.join(".verified");
    fs::read_to_string(&marker).is_ok_and(|t| t.trim() == format!("{}:{}", a.size, a.sha256))
        && fs::metadata(&bin).is_ok_and(|m| m.len() > 0)
        && [cache, &dir, &bin, &marker].iter().all(|p| sys::owned_safely(p))
}

/// An entry's path inside the folder it is unpacked into, or `None` when it would lead out of it (absolute, `..`).
fn inside(into: &Path, name: &Path) -> Option<PathBuf> {
    let mut out = into.to_path_buf();
    for c in name.components() {
        match c {
            Component::Normal(p) => out.push(p),
            Component::CurDir => {}
            _ => return None,
        }
    }
    (out != into).then_some(out)
}

/// Unpack an archive's files and folders (never a link, never a path out of `into`).
fn unpack(archive: &Path, into: &Path) -> Result<()> {
    let name = archive.to_string_lossy().to_string();
    if name.ends_with(".zip") {
        return unpack_zip(archive, into);
    }
    let file = fs::File::open(archive).map_err(|e| Error::io(archive, e))?;
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(file));
    for entry in tar.entries().map_err(|e| Error::io(archive, e))? {
        let mut entry = entry.map_err(|e| Error::io(archive, e))?;
        let path = entry.path().map_err(|e| Error::io(archive, e))?.into_owned();
        let Some(dest) = inside(into, &path) else {
            return Err(Error::Download(format!("{name}: an entry leads out of the archive ({})", path.display())));
        };
        let kind = entry.header().entry_type();
        if kind.is_dir() {
            fs::create_dir_all(&dest).map_err(|e| Error::io(&dest, e))?;
        } else if kind.is_file() {
            if let Some(dir) = dest.parent() {
                fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
            }
            let mut out = fs::File::create(&dest).map_err(|e| Error::io(&dest, e))?;
            std::io::copy(&mut entry, &mut out).map_err(|e| Error::io(&dest, e))?;
        }
    }
    Ok(())
}

#[cfg(windows)]
fn unpack_zip(archive: &Path, into: &Path) -> Result<()> {
    let file = fs::File::open(archive).map_err(|e| Error::io(archive, e))?;
    let mut zip = zip::ZipArchive::new(file).map_err(|e| Error::Download(format!("{}: {e}", archive.display())))?;
    for i in 0..zip.len() {
        let mut f = zip.by_index(i).map_err(|e| Error::Download(format!("{}: {e}", archive.display())))?;
        let Some(name) = f.enclosed_name() else {
            return Err(Error::Download(format!("{}: an entry leads out of the archive", archive.display())));
        };
        let Some(dest) = inside(into, &name) else {
            return Err(Error::Download(format!("{}: an entry leads out of the archive", archive.display())));
        };
        if f.is_dir() {
            fs::create_dir_all(&dest).map_err(|e| Error::io(&dest, e))?;
        } else if f.is_file() {
            if let Some(dir) = dest.parent() {
                fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
            }
            let mut out = fs::File::create(&dest).map_err(|e| Error::io(&dest, e))?;
            std::io::copy(&mut f, &mut out).map_err(|e| Error::io(&dest, e))?;
        }
    }
    Ok(())
}

#[cfg(not(windows))]
fn unpack_zip(archive: &Path, _into: &Path) -> Result<()> {
    Err(Error::Download(format!("{}: zip archives are unpacked on Windows only", archive.display())))
}

/// Fetch, verify and unpack the binary unless it is already there; its path. Fails on a network failure or stall, a
/// size or hash mismatch (the download is deleted), a platform Qdrant does not ship for, or a folder others could
/// change; the caller retries with backoff. `asset` is for tests: another archive with its own pins.
pub async fn ensure_qdrant(o: &DownloadOptions, asset: Option<QdrantAsset>) -> Result<PathBuf> {
    let a = match asset.or_else(|| qdrant_asset(&platform())) {
        Some(a) => a,
        None => return Err(Error::Download(format!("no Qdrant build for {}", platform()))),
    };
    let cache = o.cache_dir();
    let dir = qdrant_dir(&cache);
    let bin = dir.join(&a.exe);
    if qdrant_ready(&cache, &a).await {
        return Ok(bin);
    }
    private_cache(&cache).await?;
    with_lock(&dir, LockOptions::default(), || async {
        if qdrant_ready(&cache, &a).await {
            return Ok(bin.clone()); // another run finished it while this one waited
        }
        o.say(&format!("[memtrails] fetching Qdrant {QDRANT_VERSION} ({} MB)", (a.size as f64 / 1e6).round()));
        let base = o.base_url.clone().unwrap_or_else(qdrant_release_base);
        let pin = Pin { size: a.size, sha256: a.sha256.clone() };
        let (file, tmp) =
            download_verified(&format!("{}/{}", base.trim_end_matches('/'), a.asset), &cache, &pin, o).await?;
        let tmp = TempFolder::new(tmp); // removed however this ends, a cancelled run too
        let (a2, dir2, tmp2) = (a.clone(), dir.clone(), tmp.path().to_path_buf());
        // Unpacking and moving folders: file work, on a blocking thread.
        let placed = blocking(move || -> Result<()> {
            let (a, dir, tmp) = (a2, dir2, tmp2);
            // Unpacked into a fresh folder and moved into place whole: never a half-unpacked binary under the real name.
            let archive = tmp.join(&a.asset);
            fs::rename(&file, &archive).map_err(|e| Error::io(&archive, e))?;
            let unpacked = tmp.join("unpacked");
            fs::create_dir(&unpacked).map_err(|e| Error::io(&unpacked, e))?;
            unpack(&archive, &unpacked)?;
            let exe = unpacked.join(&a.exe);
            let ok = fs::symlink_metadata(&exe).is_ok_and(|m| m.file_type().is_file() && m.len() > 0);
            if !ok {
                return Err(Error::Download(format!("{} did not contain {}", a.asset, a.exe)));
            }
            sys::chmod(&exe, 0o755).map_err(|e| Error::io(&exe, e))?;
            let marker = unpacked.join(".verified");
            fs::write(&marker, format!("{}:{}\n", a.size, a.sha256)).map_err(|e| Error::io(&marker, e))?;
            sys::chmod(&marker, 0o644).map_err(|e| Error::io(&marker, e))?;
            sys::chmod(&unpacked, 0o755).map_err(|e| Error::io(&unpacked, e))?;
            if dir.exists() {
                // Unverified or not this user's alone; removed with tmp.
                fs::rename(&dir, tmp.join("old")).map_err(|e| Error::io(&dir, e))?;
            }
            fs::rename(&unpacked, &dir).map_err(|e| Error::io(&dir, e))
        })
        .await;
        tmp.remove().await;
        placed??;
        if !qdrant_ready(&cache, &a).await {
            return Err(Error::Download(format!(
                "{} is not this user's alone (its owner or permissions): not running a binary others could change",
                dir.display()
            )));
        }
        o.say(&format!("[memtrails] Qdrant {QDRANT_VERSION} ready"));
        Ok(bin.clone())
    })
    .await
}
