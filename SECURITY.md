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

**adm-zip, through onnxruntime-node (npm).** `npm audit` and other scanners report advisories for adm-zip 0.5.x
(crafted ZIP files: memory exhaustion, symlink and SUID handling, duplicate entries). memtrails never loads adm-zip.
onnxruntime-node uses it only in its install script, and only on Linux x64, to unpack the CUDA libraries it downloads
from NuGet. memtrails runs on the CPU and never uses those libraries. What that means for each advisory:

- **SUID/SGID bits (GHSA-j5f4-cc29-5x44)** and the other crafted-archive flaws: the archive would itself have to be
  malicious, and a malicious download could already plant code in those libraries. The installer also does not ask
  adm-zip to keep file permissions, which the SUID flaw needs.
- **Destination symlinks (GHSA-vwc7-r8mq-g2x9, no fixed adm-zip yet):** this one is about the folder, not the
  archive. The installer unpacks into a folder under the system temp directory named after the current time
  (`onnxruntime-node-pkgs_<milliseconds>`, the same in onnxruntime-node 1.31). On a Linux x64 machine shared with
  other users, one of them could prepare that folder with a symlink in advance and have the install overwrite a file
  of yours, or swap the unpacked library before it is copied. A machine only you use is not exposed.

To leave the download, and adm-zip, out of the install entirely (it also saves about 350 MB and most of the install
time on Linux x64), install with the installer's own switch. On a shared Linux x64 machine, always install this way:

```bash
ONNXRUNTIME_NODE_INSTALL=skip npm install memtrails
```

memtrails works the same with or without it; its CI installs this way. onnxruntime-node stays at 1.23.2 because the
Python and Rust versions run the same ONNX Runtime, so that all three give the same vectors, and because 1.23.2 is the
last release that runs on every platform memtrails supports: every later ONNX Runtime (1.24 to 1.31) ships no build
for Intel Macs, in its npm package, its Python wheels and its own releases alike, and no Python 3.10 wheels. A later
one would mean dropping those, and would not remove the risk: 1.31.1 still downloads the CUDA libraries into the same
kind of temp folder, only with a newer adm-zip.

**Other scanner notes.** Package scanners such as Socket also list behaviour that is expected here: memtrails starts
programs (it unpacks and runs the Qdrant binary), onnxruntime-node runs an install script that uses the network and
the shell (the download above), and that script's proxy helper, global-agent, pulls in old or deprecated packages
(`boolean`). langsmith, flagged for building a git command from a remote name, arrives only with `@langchain/core`,
an optional peer that you install yourself for the LangChain retriever; memtrails never calls it.

## Supported versions

The latest minor release receives security fixes.
