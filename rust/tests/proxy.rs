//! The proxy variables are not followed: the reference's own fetch ignores them, and following them sent the local
//! Qdrant's key to whoever serves the proxy (and a Qdrant on loopback could not be reached behind one). A test binary
//! of its own: the variables are set before the shared client is first made, and nothing else in it reads them.

mod common;

use common::fake_qdrant;
use common::{serve, Response};
use memtrails::download::{download_verified, DownloadOptions, Pin};
use memtrails::qdrant_client::{Endpoint, QdrantClient};
use memtrails::qdrant_process::QdrantEndpoint;
use sha2::{Digest, Sha256};

#[tokio::test]
async fn requests_never_go_through_a_proxy_the_environment_names() {
    let proxy = serve(|_| Response::new(502, "proxied")).await;
    for v in ["HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"] {
        std::env::set_var(v, &proxy.url);
    }
    for v in ["NO_PROXY", "no_proxy"] {
        std::env::remove_var(v);
    }
    let fake = fake_qdrant::start().await;
    let client = QdrantClient::new(
        Endpoint::Fixed(QdrantEndpoint { url: fake.server.url.clone(), api_key: "local-secret".into() }),
        None,
    );
    client.ensure_collection().await.unwrap();
    assert_eq!(client.count(None).await.unwrap(), 0);
    let bytes = b"a file".to_vec();
    let pin = Pin { size: bytes.len() as u64, sha256: hex::encode(Sha256::digest(&bytes)) };
    let files = serve(move |_| Response::new(200, bytes.clone())).await;
    let dir = tempfile::tempdir().unwrap();
    let (file, _) = download_verified(&format!("{}/f.bin", files.url), dir.path(), &pin, &DownloadOptions::default())
        .await
        .unwrap();
    assert_eq!(std::fs::read(file).unwrap(), b"a file");
    assert!(proxy.requests().is_empty(), "the proxy was sent {:?}", proxy.requests());
    assert!(fake.server.requests().iter().all(|r| r.header("api-key") == Some("local-secret")));
}
