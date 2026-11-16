#!/usr/bin/env python3
"""Write a sorted SHA-256 manifest for the public repository package."""
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "ARTIFACTS.sha256"
EXCLUDED_PARTS = {".git", "__pycache__"}
EXCLUDED_NAMES = {".DS_Store", "ARTIFACTS.sha256"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    lines = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name in EXCLUDED_NAMES:
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        lines.append(f"{digest(path)}  {relative.as_posix()}")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} records to {OUTPUT.name}")


if __name__ == "__main__":
    main()
