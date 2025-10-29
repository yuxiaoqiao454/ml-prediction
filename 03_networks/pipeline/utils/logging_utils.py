#!/usr/bin/env python3
"""
Simple CSV logging + console echo, shared by all scripts.
"""
from pathlib import Path
from datetime import datetime
import csv

def log_message(log_path: str | Path, message: str):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([ts, message])
    print(f"[{ts}] {message}")

def log_summary_csv(path: str | Path, row: dict, header: list[str]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new_file:
            w.writeheader()
        w.writerow(row)
