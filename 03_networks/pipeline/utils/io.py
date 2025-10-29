#!/usr/bin/env python3
"""
Unified I/O helpers:
- YAML config loader
- CSV save/load
- ensure_dir
All scripts should import from here to avoid drift.
"""

from pathlib import Path
import pandas as pd
import yaml

# --- dirs ---
def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)

# --- csv ---
def save_csv(df: pd.DataFrame, path: str | Path, **kwargs):
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8", **kwargs)
    print(f"✓ Saved CSV → {path}")

def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, **kwargs)

# --- yaml config ---
def load_config(path: str | Path) -> dict:
    """
    Load a YAML config and return a dict. Use this in ALL scripts.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML structure in {path}")
    return cfg
