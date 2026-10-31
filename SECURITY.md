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

## Supported versions

The latest minor release receives security fixes.
