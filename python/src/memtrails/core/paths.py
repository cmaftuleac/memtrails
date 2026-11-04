"""Where memtrails keeps what it downloads (the Qdrant binary, the embedding model): $MEMTRAILS_CACHE, else
$XDG_CACHE_HOME/memtrails, else ~/.cache/memtrails — the reference's folder, so both libraries share one download.
Shared by every memory on the machine; safe to delete."""

from __future__ import annotations

import os


def cache_dir() -> str:
    own = (os.environ.get("MEMTRAILS_CACHE") or "").strip()
    if own:
        return own
    xdg = (os.environ.get("XDG_CACHE_HOME") or "").strip()
    return os.path.join(xdg or os.path.join(os.path.expanduser("~"), ".cache"), "memtrails")
