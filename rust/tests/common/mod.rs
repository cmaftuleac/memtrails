//! A small HTTP server on loopback for the tests: every request is recorded and answered by a function of it — no
//! network, no dependency.
#![allow(dead_code)]

pub mod fake_qdrant;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// A request as it came.
#[derive(Debug, Clone)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }

    /// The body, gunzipped when it was sent gzipped, as JSON.
    pub fn json(&self) -> serde_json::Value {
        let raw = if self.header("content-encoding") == Some("gzip") {
            use std::io::Read;
            let mut out = Vec::new();
            flate2::read::GzDecoder::new(&self.body[..]).read_to_end(&mut out).unwrap();
            out
        } else {
            self.body.clone()
        };
        serde_json::from_slice(&raw).unwrap()
    }
}

/// An answer: a status and the body in pieces, each after a pause (a stalled download is a long pause).
pub struct Response {
    pub status: u16,
    pub chunks: Vec<(u64, Vec<u8>)>,
}

impl Response {
    pub fn new(status: u16, body: impl Into<Vec<u8>>) -> Self {
        Response { status, chunks: vec![(0, body.into())] }
    }

    pub fn json(status: u16, v: &serde_json::Value) -> Self {
        Response::new(status, serde_json::to_vec(v).unwrap())
    }
}

type Handler = Arc<dyn Fn(&Request) -> Response + Send + Sync>;

pub struct Server {
    pub url: String,
    pub seen: Arc<Mutex<Vec<Request>>>,
}

impl Server {
    pub fn requests(&self) -> Vec<Request> {
        self.seen.lock().unwrap().clone()
    }
}

async fn read_request(sock: &mut tokio::net::TcpStream) -> Option<Request> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    let head_end = loop {
        let n = sock.read(&mut tmp).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i;
        }
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let mut lines = head.split("\r\n");
    let first = lines.next()?;
    let mut parts = first.split(' ');
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();
    let headers: Vec<(String, String)> =
        lines.filter_map(|l| l.split_once(':').map(|(k, v)| (k.trim().to_lowercase(), v.trim().to_string()))).collect();
    let len: usize = headers.iter().find(|(k, _)| k == "content-length").and_then(|(_, v)| v.parse().ok()).unwrap_or(0);
    let mut body = buf[head_end + 4..].to_vec();
    while body.len() < len {
        let n = sock.read(&mut tmp).await.ok()?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&tmp[..n]);
    }
    Some(Request { method, path, headers, body })
}

/// A server answering every request with `f(request)`; one connection a request (`Connection: close`).
pub async fn serve(f: impl Fn(&Request) -> Response + Send + Sync + 'static) -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let handler: Handler = Arc::new(f);
    let log = seen.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else { break };
            let handler = handler.clone();
            let log = log.clone();
            tokio::spawn(async move {
                let Some(req) = read_request(&mut sock).await else { return };
                log.lock().unwrap().push(req.clone());
                let res = handler(&req);
                let head =
                    format!("HTTP/1.1 {} X\r\nconnection: close\r\ncontent-type: application/json\r\n\r\n", res.status);
                if sock.write_all(head.as_bytes()).await.is_err() {
                    return;
                }
                for (pause, bytes) in res.chunks {
                    if pause > 0 {
                        tokio::time::sleep(Duration::from_millis(pause)).await;
                    }
                    if sock.write_all(&bytes).await.is_err() {
                        return;
                    }
                    let _ = sock.flush().await;
                }
                let _ = sock.shutdown().await;
            });
        }
    });
    Server { url, seen }
}
