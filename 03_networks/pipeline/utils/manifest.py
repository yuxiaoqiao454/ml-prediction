#!/usr/bin/env python3
"""
Tiny helpers for writing/reading per-stage manifests.
"""
from pathlib import Path
import json

def write_manifest(path: str | Path, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def read_manifest(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))
