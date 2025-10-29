#!/usr/bin/env python3
"""
Window utility helpers.
NOTE: Main window discovery lives in build_window_index.py.
These helpers may be used by multiple scripts.
"""
from datetime import timedelta
import pandas as pd

def window_bounds(window_end: pd.Timestamp, win_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.to_datetime(window_end).normalize()
    start = end - timedelta(days=win_days)
    return start, end

def is_datestr_yyyymmdd(name: str) -> bool:
    return len(name) == 10 and name[4] == "-" and name[7] == "-"
