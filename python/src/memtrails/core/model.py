"""
The embedding model memory runs on this machine: EmbeddingGemma-300m, 4-bit (onnx-community/embeddinggemma-300m-ONNX on
Hugging Face, one pinned revision), fetched on first use into the cache (paths.py), each file checked against its pinned
size and SHA-256. 218 MB. The model is Google's, under the Gemma Terms of Use (ai.google.dev/gemma/terms); memtrails
does not ship it. The same files, the same folder and the same ``.verified`` marker as the reference library.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from collections.abc import Callable

import httpx

from .paths import cache_dir
from .qdrant_binary import download_verified, with_lock

MODEL_REPO = "onnx-community/embeddinggemma-300m-ONNX"
MODEL_REVISION = "5090578d9565bb06545b4552f76e6bc2c93e4a66"
#: (path in the repository, size, SHA-256).
MODEL_FILES: list[tuple[str, int, str]] = [
    ("onnx/model_q4.onnx", 519_322, "ad1dfee81a70f7944b9b9d1cc6e48075b832881cf33fab2f2b248be78f3f0043"),
    ("onnx/model_q4.onnx_data", 196_725_760, "599962c3143b040de2dd05e5975be3e9091dd067cacc6a8f7186e3203bab9e02"),
    ("tokenizer.json", 20_323_312, "4dda02faaf32bc91031dc8c88457ac272b00c1016cc679757d1c441b248b9c47"),
    ("tokenizer_config.json", 1_156_830, "3ca953eea6c3c9fcda9cf3df22949ff18b216f7c74bd6459230f3f1013953f3a"),
]


def default_model_dir() -> str:
    return os.path.join(cache_dir(), f"embeddinggemma-300m-{MODEL_REVISION[:8]}")


def _marker(directory: str) -> str:
    return os.path.join(directory, ".verified")


_STAMP = "\n".join(f"{rel}:{size}:{sha}" for rel, size, sha in MODEL_FILES)


def model_ready(directory: str | None = None) -> bool:
    """Every file in place at its pinned size, and the marker written after the last of them was verified."""
    d = directory or default_model_dir()
    try:
        with open(_marker(d), encoding="utf-8") as f:
            if f.read() != _STAMP:
                return False
        return all(os.path.exists(os.path.join(d, rel)) and os.stat(os.path.join(d, rel)).st_size == size for rel, size, _ in MODEL_FILES)
    except OSError:
        return False


async def ensure_model_async(
    directory: str | None = None, log: Callable[[str], None] | None = None, transport: httpx.AsyncBaseTransport | None = None
) -> str:
    """The model in ``directory`` (fetched and checked unless it is already there); the directory, ready for
    ``model_dir``. Each file is fetched to a temp of its own, checked on disk and renamed into place; one run at a time
    per directory (with_lock), and the marker is taken away first, so a set left half-replaced is never taken as
    ready."""
    d = directory or default_model_dir()
    if model_ready(d):
        return d
    os.makedirs(d, exist_ok=True)

    async def fetch() -> str:
        if model_ready(d):
            return d  # another run finished it while this one waited
        if log:
            log(f"[memtrails] fetching {MODEL_REPO} (218 MB, Gemma Terms of Use)")
        try:
            os.remove(_marker(d))
        except FileNotFoundError:
            pass
        for rel, size, sha in MODEL_FILES:
            path = os.path.join(d, rel)
            file, tmp = await download_verified(
                f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{rel}", os.path.dirname(path), size, sha, transport=transport
            )
            try:
                os.replace(file, path)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        with open(_marker(d), "w", encoding="utf-8") as f:
            f.write(_STAMP)
        if log:
            log("[memtrails] embedding model ready")
        return d

    return await with_lock(d, fetch)


def ensure_model(directory: str | None = None, log: Callable[[str], None] | None = None) -> str:
    """The model, fetched and checked unless it is already there; its directory."""
    return asyncio.run(ensure_model_async(directory, log))
