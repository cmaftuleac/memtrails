//! The few Qdrant REST calls the memory index makes — plain HTTP, no client library, so the payloads are the
//! reference's own. One collection, `memory`, two tiers (payload `tier`):
//!
//!   - `stm` — short-term memory: one point per record (its start as the key) and its chunks;
//!   - `ltm` — long-term memory: per item a KEY point (`part: key` — its content title + ~150-word summary as the
//!     dense vector, a BM25 sparse vector over its title, summary, body and files, the item itself as payload) and
//!     CHILD points (`part: child`, ~96-word windows of its body, dense only) and CUE points pointing at it.

use sha1::{Digest, Sha1};

/// A point's id from its item and its place (0 = the key, n = the n-th child, -(k+1) = the k-th cue): the same item
/// writes the same ids. SHA-1 of `item#n`, laid out as a UUID version 5.
pub fn point_id(item: &str, n: i64) -> String {
    let h = hex::encode(Sha1::digest(format!("{item}#{n}").as_bytes()));
    let variant = (u8::from_str_radix(&h[16..17], 16).unwrap_or(0) & 0x3) | 0x8;
    format!("{}-{}-5{}-{:x}{}-{}", &h[0..8], &h[8..12], &h[13..16], variant, &h[17..20], &h[20..32])
}
