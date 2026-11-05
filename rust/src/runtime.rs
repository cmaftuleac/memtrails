//! The ONNX Runtime the local model runs on: Microsoft's own release, the version the reference's `onnxruntime-node`
//! carries (1.23.2), fetched on first use from its GitHub release, checked against the size and SHA-256 pinned below,
//! and loaded at run time. ONNX Runtime builds differ in their 4-bit matrix kernels: with another build the vectors
//! drift by up to ~0.007 a number (cosine ~0.999) from the reference's; with this one they are the same on the same processor family, so an index
//! written by either library is searched alike by the other.
//!
//! `MEMTRAILS_ONNXRUNTIME` names a library of your own instead (nothing is fetched then).

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::download::{download_verified, with_lock, DownloadOptions, LockOptions, Pin};
use crate::error::{Error, Result};
use crate::qdrant_binary::platform;
use crate::store::blocking;
use crate::sys;

/// The ONNX Runtime version the reference runs the model on.
pub const ONNXRUNTIME_VERSION: &str = "1.23.2";

/// Per platform: the release archive, its size and SHA-256, and the library inside it.
pub const ONNXRUNTIME_ASSETS: &[(&str, &str, u64, &str, &str)] = &[
    (
        "darwin-arm64",
        "onnxruntime-osx-arm64-1.23.2.tgz",
        9_999_931,
        "b4d513ab2b26f088c66891dbbc1408166708773d7cc4163de7bdca0e9bbb7856",
        "lib/libonnxruntime.1.23.2.dylib",
    ),
    (
        "darwin-x64",
        "onnxruntime-osx-x86_64-1.23.2.tgz",
        11_676_322,
        "d10359e16347b57d9959f7e80a225a5b4a66ed7d7e007274a15cae86836485a6",
        "lib/libonnxruntime.1.23.2.dylib",
    ),
    (
        "linux-x64",
        "onnxruntime-linux-x64-1.23.2.tgz",
        8_309_231,
        "1fa4dcaef22f6f7d5cd81b28c2800414350c10116f5fdd46a2160082551c5f9b",
        "lib/libonnxruntime.so.1.23.2",
    ),
    (
        "linux-arm64",
        "onnxruntime-linux-aarch64-1.23.2.tgz",
        7_254_068,
        "7c63c73560ed76b1fac6cff8204ffe34fe180e70d6582b5332ec094810241e5c",
        "lib/libonnxruntime.so.1.23.2",
    ),
    (
        "win32-x64",
        "onnxruntime-win-x64-1.23.2.zip",
        78_127_794,
        "0b38df9af21834e41e73d602d90db5cb06dbd1ca618948b8f1d66d607ac9f3cd",
        "lib/onnxruntime.dll",
    ),
];

fn asset(platform: &str) -> Option<(&'static str, u64, &'static str, &'static str)> {
    ONNXRUNTIME_ASSETS.iter().find(|a| a.0 == platform).map(|a| (a.1, a.2, a.3, a.4))
}

/// Where the library is kept: `<cache>/onnxruntime-<version>/<its file name>`.
fn runtime_dir(cache: &Path) -> PathBuf {
    cache.join(format!("onnxruntime-{ONNXRUNTIME_VERSION}"))
}

fn file_name(inner: &str) -> &str {
    inner.rsplit('/').next().unwrap_or(inner)
}

/// The library is in place, its archive was verified when it was taken out, and no one else could have changed it.
fn ready(cache: &Path, size: u64, sha: &str, inner: &str) -> bool {
    let dir = runtime_dir(cache);
    let lib = dir.join(file_name(inner));
    let marker = dir.join(".verified");
    fs::read_to_string(&marker).is_ok_and(|t| t.trim() == format!("{size}:{sha}"))
        && fs::metadata(&lib).is_ok_and(|m| m.len() > 0)
        && [&dir, &lib, &marker].iter().all(|p| sys::owned_safely(p))
}

/// Only the library out of the archive, under its own name (nothing else of it is written).
fn take_out(archive: &Path, inner: &str, to: &Path) -> Result<()> {
    let name = archive.display().to_string();
    let wanted = |path: &str| path.trim_start_matches("./").split_once('/').is_some_and(|(_, rest)| rest == inner);
    let mut out = fs::File::create(to).map_err(|e| Error::io(to, e))?;
    if name.ends_with(".zip") {
        #[cfg(windows)]
        {
            let file = fs::File::open(archive).map_err(|e| Error::io(archive, e))?;
            let mut zip = zip::ZipArchive::new(file).map_err(|e| Error::Download(format!("{name}: {e}")))?;
            for i in 0..zip.len() {
                let mut f = zip.by_index(i).map_err(|e| Error::Download(format!("{name}: {e}")))?;
                if f.is_file() && wanted(f.name()) {
                    std::io::copy(&mut f, &mut out).map_err(|e| Error::io(to, e))?;
                    return Ok(());
                }
            }
        }
        return Err(Error::Download(format!("{name} did not contain {inner}")));
    }
    let file = fs::File::open(archive).map_err(|e| Error::io(archive, e))?;
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(file));
    for entry in tar.entries().map_err(|e| Error::io(archive, e))? {
        let mut entry = entry.map_err(|e| Error::io(archive, e))?;
        let path = entry.path().map_err(|e| Error::io(archive, e))?.to_string_lossy().to_string();
        if entry.header().entry_type().is_file() && wanted(&path) {
            let mut buf = Vec::new();
            entry.read_to_end(&mut buf).map_err(|e| Error::io(archive, e))?;
            std::io::Write::write_all(&mut out, &buf).map_err(|e| Error::io(to, e))?;
            return Ok(());
        }
    }
    Err(Error::Download(format!("{name} did not contain {inner}")))
}

/// The library, fetched, verified and taken out of its archive unless it is already there; or the one
/// `MEMTRAILS_ONNXRUNTIME` names.
pub async fn ensure_runtime(o: &DownloadOptions) -> Result<PathBuf> {
    if let Some(own) = std::env::var_os("MEMTRAILS_ONNXRUNTIME").filter(|v| !v.is_empty()) {
        return Ok(PathBuf::from(own));
    }
    let p = platform();
    let Some((archive, size, sha, inner)) = asset(&p) else {
        return Err(Error::Download(format!("no ONNX Runtime build for {p}")));
    };
    let cache = o.cache_dir();
    let dir = runtime_dir(&cache);
    let lib = dir.join(file_name(inner));
    let (c, s) = (cache.clone(), sha.to_string());
    if blocking(move || ready(&c, size, &s, inner)).await? {
        return Ok(lib);
    }
    let made = cache.clone();
    blocking(move || sys::make_dir(&made, 0o700)).await?.map_err(|e| Error::io(&cache, e))?;
    with_lock(&dir, LockOptions::default(), || async {
        let (c, s) = (cache.clone(), sha.to_string());
        if blocking(move || ready(&c, size, &s, inner)).await? {
            return Ok(lib.clone()); // another run finished it while this one waited
        }
        o.say(&format!("[memtrails] fetching ONNX Runtime {ONNXRUNTIME_VERSION} ({} MB)", (size as f64 / 1e6).round()));
        let base = o.base_url.clone().unwrap_or_else(|| {
            format!("https://github.com/microsoft/onnxruntime/releases/download/v{ONNXRUNTIME_VERSION}")
        });
        let pin = Pin { size, sha256: sha.to_string() };
        let (file, tmp) =
            download_verified(&format!("{}/{archive}", base.trim_end_matches('/')), &cache, &pin, o).await?;
        let (d, t, s) = (dir.clone(), tmp.clone(), sha.to_string());
        // Taken out into a fresh folder and moved into place whole: never a half-written library under the real name.
        let placed = blocking(move || -> Result<()> {
            let unpacked = t.join("unpacked");
            fs::create_dir(&unpacked).map_err(|e| Error::io(&unpacked, e))?;
            take_out(&file, inner, &unpacked.join(file_name(inner)))?;
            let marker = unpacked.join(".verified");
            fs::write(&marker, format!("{size}:{s}\n")).map_err(|e| Error::io(&marker, e))?;
            sys::chmod(&marker, 0o644).map_err(|e| Error::io(&marker, e))?;
            sys::chmod(&unpacked, 0o755).map_err(|e| Error::io(&unpacked, e))?;
            if d.exists() {
                fs::rename(&d, t.join("old")).map_err(|e| Error::io(&d, e))?;
            }
            fs::rename(&unpacked, &d).map_err(|e| Error::io(&d, e))
        })
        .await;
        let _ = tokio::fs::remove_dir_all(&tmp).await;
        placed??;
        let (c, s) = (cache.clone(), sha.to_string());
        if !blocking(move || ready(&c, size, &s, inner)).await? {
            return Err(Error::Download(format!(
                "{} is not this user's alone (its owner or permissions): not loading a library others could change",
                dir.display()
            )));
        }
        o.say(&format!("[memtrails] ONNX Runtime {ONNXRUNTIME_VERSION} ready"));
        Ok(lib.clone())
    })
    .await
}

static LOADED: OnceLock<std::result::Result<PathBuf, String>> = OnceLock::new();

/// Load the library into this process, once: every model after it runs on it. A second library asked for later is
/// refused (a process holds one ONNX Runtime).
pub(crate) fn load(lib: &Path) -> Result<()> {
    let loaded = LOADED.get_or_init(|| match ort::init_from(lib) {
        Ok(builder) => {
            let _ = builder.commit();
            Ok(lib.to_path_buf())
        }
        Err(e) => Err(format!("ONNX Runtime could not be loaded from {}: {e}", lib.display())),
    });
    match loaded {
        Ok(at) if at == lib => Ok(()),
        Ok(at) => Err(Error::Model(format!("ONNX Runtime is already loaded from {}", at.display()))),
        Err(e) => Err(Error::Model(e.clone())),
    }
}
