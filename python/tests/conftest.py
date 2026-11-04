"""Every test run gets its own memtrails cache, so nothing on this machine is read or left behind (and a real Qdrant or
model already downloaded here does not change what a test sees) — unless the integration tests ask for the real one."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).resolve().parents[2] / "conformance" / "fixtures"

if os.environ.get("MEMTRAILS_INTEGRATION") != "1":
    os.environ["MEMTRAILS_CACHE"] = tempfile.mkdtemp(prefix="memtrails-test-")


def fixture(name: str) -> Any:
    """One of the conformance fixtures the TypeScript library wrote (conformance/generate*.ts)."""
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return tmp_path / "memory"
