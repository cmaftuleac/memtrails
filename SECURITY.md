# Security

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
[security advisories](https://github.com/cmaftuleac/memtrails/security/advisories/new). You will get an answer
within a week. Please do not open a public issue for a vulnerability.

## What memtrails does with your data

- Records are stored in the memory's `home` directory, readable by its owner only (files are written with mode 0600).
- Nothing is sent over the network unless you configure a language model, an embedding server or a rerank server.
  Then the texts they need are sent to them: for a language model, windows of records and candidates (consolidation,
  sleep, gathering); for servers, the texts to embed or rerank.
- Before anything is sent to a language model, and in everything consolidation keeps, secrets are replaced: provider
  keys, tokens, credentials in URLs, private keys, JWTs, `key=value` secrets, e-mail local parts; home directories become
  `~`. Your records themselves are kept as you wrote them.
- The Qdrant process memtrails starts listens on loopback only, behind an API key made for that run, with telemetry off.
- Downloads (the embedding model and the Qdrant binary) are verified against pinned sizes and SHA-256 hashes before
  they are used.

## Known advisories in dependencies

**adm-zip, through onnxruntime-node (npm).** `npm audit` reports advisories for adm-zip 0.5.x (crafted ZIP files:
memory exhaustion, symlink and SUID handling, duplicate entries). memtrails never loads adm-zip. onnxruntime-node uses
it only in its install script, and only on Linux x64, to unpack the CUDA libraries it downloads from NuGet. memtrails
runs on the CPU and never uses those libraries. To reach the flaws, that download itself would have to be malicious,
and a malicious download could already plant code in those libraries, so adm-zip adds no risk of its own.

To leave the download, and adm-zip, out of the install entirely (it also saves about 350 MB and most of the install
time on Linux x64), install with the installer's own switch:

```bash
ONNXRUNTIME_NODE_INSTALL=skip npm install memtrails
```

memtrails works the same with or without it; its CI installs this way. onnxruntime-node stays at 1.23.2 because the
Python and Rust versions run the same ONNX Runtime, so that all three give the same vectors, and because 1.23.2 is the
last release that runs on every platform memtrails supports: every later ONNX Runtime (1.24 to 1.31) ships no build
for Intel Macs, in its npm package, its Python wheels and its own releases alike, and no Python 3.10 wheels. A later
one (1.31.1 still downloads the CUDA libraries, with adm-zip 0.6, which has the fixes) would mean dropping those.

## Supported versions

The latest minor release receives security fixes.
