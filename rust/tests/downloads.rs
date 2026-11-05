//! What memtrails downloads: every platform pinned, a download that does not match never unpacked or kept, a stalled
//! one given up, two at once never sharing a file, one run at a time under a lock — over a fake server.

mod common;

use std::fs;
use std::io::Write;

use common::{serve, Response};
use memtrails::download::{download_verified, with_lock, DownloadOptions, LockOptions, Pin};
use memtrails::model::{model_ready, MODEL_FILES};
use memtrails::qdrant_binary::{ensure_qdrant, qdrant_asset, qdrant_dir, qdrant_ready, QdrantAsset, QDRANT_ASSETS};
use sha2::{Digest, Sha256};

fn pin_of(bytes: &[u8]) -> Pin {
    Pin { size: bytes.len() as u64, sha256: hex::encode(Sha256::digest(bytes)) }
}

#[test]
fn every_supported_platform_has_a_pinned_build() {
    for p in ["darwin-arm64", "darwin-x64", "linux-x64", "linux-arm64", "win32-x64"] {
        let a = qdrant_asset(p).unwrap_or_else(|| panic!("{p}"));
        assert_eq!(a.sha256.len(), 64);
        assert!(a.size > 10_000_000);
    }
    assert_eq!(qdrant_asset("win32-x64").unwrap().exe, "qdrant.exe");
    assert!(qdrant_asset("freebsd-x64").is_none());
    assert_eq!(QDRANT_ASSETS.len(), 5);
    for p in ["darwin-arm64", "darwin-x64", "linux-x64", "linux-arm64", "win32-x64"] {
        let a = memtrails::runtime::ONNXRUNTIME_ASSETS.iter().find(|a| a.0 == p).unwrap_or_else(|| panic!("{p}"));
        assert!(a.1.contains(memtrails::runtime::ONNXRUNTIME_VERSION) && a.3.len() == 64 && a.2 > 5_000_000);
    }
    assert_eq!(MODEL_FILES.len(), 4);
}

#[tokio::test]
async fn refuses_a_download_whose_size_or_hash_is_not_pinned_and_keeps_nothing() {
    let s = serve(|_| Response::new(200, vec![1u8, 2, 3])).await;
    let cache = tempfile::tempdir().unwrap();
    let o = DownloadOptions { cache: Some(cache.path().into()), base_url: Some(s.url.clone()), ..Default::default() };
    let err = ensure_qdrant(&o, None).await.unwrap_err().to_string();
    assert!(err.contains("expected"), "{err}");
    let a = qdrant_asset(&memtrails::qdrant_binary::platform()).unwrap();
    assert!(!qdrant_ready(cache.path(), &a).await);
    let left: Vec<_> = fs::read_dir(cache.path()).unwrap().map(|e| e.unwrap().file_name()).collect();
    assert!(left.iter().all(|n| !n.to_string_lossy().starts_with(".download-")), "{left:?}");
}

#[tokio::test]
async fn says_why_when_the_download_fails() {
    let s = serve(|_| Response::new(404, "nope")).await;
    let cache = tempfile::tempdir().unwrap();
    let o = DownloadOptions { cache: Some(cache.path().into()), base_url: Some(s.url.clone()), ..Default::default() };
    assert!(ensure_qdrant(&o, None).await.unwrap_err().to_string().contains("HTTP 404"));
}

#[tokio::test]
async fn two_downloads_at_once_never_take_each_others_file() {
    let bytes: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
    let pin = pin_of(&bytes);
    let s = serve(move |_| Response::new(200, bytes.clone())).await;
    let dir = tempfile::tempdir().unwrap();
    let o = DownloadOptions::default();
    let url = format!("{}/f.bin", s.url);
    let (a, b) =
        tokio::join!(download_verified(&url, dir.path(), &pin, &o), download_verified(&url, dir.path(), &pin, &o));
    let (a, b) = (a.unwrap(), b.unwrap());
    assert_ne!(a.0, b.0);
    assert!(a.0.exists() && b.0.exists());
}

#[tokio::test]
async fn gives_up_on_a_download_that_stops_sending_and_leaves_nothing() {
    let s = serve(|_| Response { status: 200, chunks: vec![(0, vec![1]), (5000, vec![2])] }).await;
    let dir = tempfile::tempdir().unwrap();
    let o = DownloadOptions { stall_ms: Some(100), ..Default::default() };
    let pin = Pin { size: 10, sha256: "0".repeat(64) };
    let err = download_verified(&format!("{}/f.bin", s.url), dir.path(), &pin, &o).await.unwrap_err().to_string();
    assert!(err.contains("no data"), "{err}");
    assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
}

fn tar_gz(files: &[(&str, &[u8])]) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let enc = flate2::write::GzEncoder::new(&mut out, flate2::Compression::default());
        let mut tar = tar::Builder::new(enc);
        for (name, data) in files {
            let mut h = tar::Header::new_gnu();
            h.set_size(data.len() as u64);
            h.set_mode(0o644);
            h.set_entry_type(tar::EntryType::Regular);
            // Written by hand so a hostile name (`../x`) goes into the archive as it is.
            let raw = h.as_old_mut();
            raw.name[..name.len()].copy_from_slice(name.as_bytes());
            h.set_cksum();
            tar.append(&h, *data).unwrap();
        }
        tar.into_inner().unwrap().finish().unwrap();
    }
    out
}

#[tokio::test]
async fn unpacks_a_verified_archive_into_place_and_trusts_it_after() {
    let archive = tar_gz(&[("qdrant", b"#!/bin/sh\necho fake\n")]);
    let pin = pin_of(&archive);
    let s = serve(move |_| Response::new(200, archive.clone())).await;
    let cache = tempfile::tempdir().unwrap();
    let asset = QdrantAsset {
        asset: "qdrant-test.tar.gz".into(),
        size: pin.size,
        sha256: pin.sha256.clone(),
        exe: "qdrant".into(),
    };
    let o = DownloadOptions { cache: Some(cache.path().into()), base_url: Some(s.url.clone()), ..Default::default() };
    let bin = ensure_qdrant(&o, Some(asset.clone())).await.unwrap();
    assert_eq!(bin, qdrant_dir(cache.path()).join("qdrant"));
    assert!(qdrant_ready(cache.path(), &asset).await);
    // Ready: not fetched again.
    ensure_qdrant(&o, Some(asset)).await.unwrap();
    assert_eq!(s.requests().len(), 1);
}

#[tokio::test]
async fn refuses_an_archive_with_a_path_out_of_it() {
    let archive = tar_gz(&[("../escaped", b"x"), ("qdrant", b"bin")]);
    let pin = pin_of(&archive);
    let s = serve(move |_| Response::new(200, archive.clone())).await;
    let cache = tempfile::tempdir().unwrap();
    let asset = QdrantAsset { asset: "q.tar.gz".into(), size: pin.size, sha256: pin.sha256, exe: "qdrant".into() };
    let o = DownloadOptions { cache: Some(cache.path().into()), base_url: Some(s.url.clone()), ..Default::default() };
    let err = ensure_qdrant(&o, Some(asset)).await.unwrap_err().to_string();
    assert!(err.contains("leads out"), "{err}");
    assert!(!cache.path().join("escaped").exists());
}

#[tokio::test]
async fn one_holder_of_a_lock_at_a_time_and_a_dead_ones_lock_taken_over() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x");
    let order = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let o = LockOptions { poll_ms: 5, ..Default::default() };
    let run = |n: &'static str| {
        let order = order.clone();
        let path = path.clone();
        async move {
            with_lock(&path, o, || async {
                order.lock().unwrap().push(format!("{n}+"));
                tokio::time::sleep(std::time::Duration::from_millis(30)).await;
                order.lock().unwrap().push(format!("{n}-"));
                Ok(())
            })
            .await
        }
    };
    let (a, b) = tokio::join!(run("a"), run("b"));
    a.unwrap();
    b.unwrap();
    let got = order.lock().unwrap().join(" ");
    assert!(got == "a+ a- b+ b-" || got == "b+ b- a+ a-", "{got}");
    let host = String::from_utf8(std::process::Command::new("hostname").output().unwrap().stdout).unwrap();
    let mut f = fs::File::create(dir.path().join("x.lock")).unwrap();
    write!(f, "999999999 {} dead", host.trim()).unwrap();
    drop(f);
    let got = with_lock(&path, o, || async { Ok("taken") }).await.unwrap();
    assert_eq!(got, "taken");
    assert!(!dir.path().join("x.lock").exists());
}

#[tokio::test]
async fn a_model_folder_without_its_marker_is_not_ready() {
    let dir = tempfile::tempdir().unwrap();
    assert!(!model_ready(dir.path()).await);
}
