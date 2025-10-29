#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build bipartite influencer–audience graphs for all rolling windows.

Stage: s1_bipartite

Inputs (from data/hashtag folder)
--------------------------------
exposures.parquet          # ['hashtag','A','B','exposure_epoch',...]
user_hashtag_first_post.parquet  # ['user','hashtag','first_post_epoch']

Config fields (from YAML)
-------------------------
base_dir: /path/to/03_networks/data
win_days: 90
step_days: 30
min_edges: 10

Outputs (per window)
--------------------
03_networks/data/windows_<hashtag>/<YYYY-MM-DD>/s1_bipartite/
  ├─ edges.csv
  ├─ nodes.csv
  ├─ _manifest.json
  └─ .ok
"""

from __future__ import annotations
from pathlib import Path
import argparse, yaml, json
import pandas as pd
import numpy as np
from datetime import timedelta

# === 1️⃣ Utilities reused from project_rolling_projections.py ===
def safe_epoch_to_day(series: pd.Series) -> pd.Series:
    """Convert epoch (s or ms) to daily timestamps (naive UTC)."""
    s = pd.to_datetime(series, unit="s", utc=True, errors="coerce")
    if s.dropna().median() < pd.Timestamp("1980-01-01", tz="UTC"):
        s = pd.to_datetime(series, unit="ms", utc=True, errors="coerce")
    return s.dt.tz_convert(None).dt.floor("D")


def load_tables(exposures_parq: Path, adoptions_parq: Path, tag: str):
    """Load exposures & adoptions, restrict to this hashtag."""
    e = pd.read_parquet(exposures_parq)
    a = pd.read_parquet(adoptions_parq)
    tag = tag.lower()

    # Clean exposures
    e["hashtag"] = e["hashtag"].astype(str).str.lower().str.lstrip("#")
    e = e[e["hashtag"].str.contains(tag, na=False)].copy()
    if e.empty:
        raise SystemExit(f"[!] No exposures matched {tag}")

    e["A"] = e["A"].astype(str).str.lower()
    e["B"] = e["B"].astype(str).str.lower()
    e["date"] = safe_epoch_to_day(e["exposure_epoch"])

    # Clean adoptions
    a["hashtag"] = a["hashtag"].astype(str).str.lower().str.lstrip("#")
    a = a[a["hashtag"].str.contains(tag, na=False)].copy()
    a["user"] = a["user"].astype(str).str.lower()

    infl = set(a["user"].unique())
    e = e[e["A"].isin(infl)].copy()

    infl_first = a[["user", "first_post_epoch"]].rename(columns={"user": "node", "first_post_epoch": "node_ts_infl"})
    aud_first = e.groupby("B", as_index=False)["exposure_epoch"].min().rename(columns={"B": "node", "exposure_epoch": "node_ts_aud"})
    return e, infl, infl_first, aud_first


def build_bipartite_tables(e_win: pd.DataFrame, infl_first: pd.DataFrame, aud_first: pd.DataFrame):
    """Aggregate influencer–audience interactions within one window."""
    if e_win.empty:
        return pd.DataFrame(), pd.DataFrame(), 0, 0, 0

    agg = (e_win.groupby(["A", "B"], as_index=False)
                 .agg(weight=("hashtag", "size"),
                      first_exposure_epoch=("exposure_epoch", "min")))
    agg = agg[agg["A"] != agg["B"]]  # remove self-loops

    A_nodes = pd.DataFrame({"id": pd.unique(agg["A"]), "partition": "influencer"})
    B_nodes = pd.DataFrame({"id": pd.unique(agg["B"]), "partition": "audience"})
    nodes = pd.concat([A_nodes, B_nodes], ignore_index=True)

    out_deg = agg.groupby("A")["B"].nunique().rename("out_degree").reset_index().rename(columns={"A": "id"})
    in_deg  = agg.groupby("B")["A"].nunique().rename("in_degree").reset_index().rename(columns={"B": "id"})
    nodes = nodes.merge(out_deg, on="id", how="left").merge(in_deg, on="id", how="left").fillna(0)
    nodes["out_degree"] = nodes["out_degree"].astype(int)
    nodes["in_degree"]  = nodes["in_degree"].astype(int)

    nodes = nodes.merge(infl_first, left_on="id", right_on="node", how="left").drop(columns=["node"])
    nodes = nodes.merge(aud_first,  left_on="id", right_on="node", how="left").drop(columns=["node"])
    nodes["ts"] = nodes.apply(
        lambda r: r["node_ts_infl"] if r["partition"] == "influencer" else r["node_ts_aud"],
        axis=1
    ).fillna(0).astype(int)

    nA = (nodes["partition"] == "influencer").sum()
    nB = (nodes["partition"] == "audience").sum()
    E = len(agg)

    return agg.rename(columns={"A": "source", "B": "target"}), nodes, nA, nB, E


def write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# === 2️⃣ Main ===
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config file")
    ap.add_argument("--hashtag", required=True, help="Hashtag (no #)")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    tag = args.hashtag.lower()
    base_dir = Path(cfg["base_dir"])
    data_dir = base_dir / f"windows_{tag}"

    tag_dir = base_dir / "parquets"/ f"networks_{tag}"
    exposures_path = tag_dir / "exposures.parquet"
    adoptions_path = tag_dir / "user_hashtag_first_post.parquet"

    win_days = int(cfg.get("win_days", 90))
    step_days = int(cfg.get("step_days", 30))
    min_edges = int(cfg.get("min_edges", 10))

    e, infl, infl_first, aud_first = load_tables(exposures_path, adoptions_path, tag)

    start = e["date"].min()
    end = e["date"].max()
    win = timedelta(days=win_days)
    step = timedelta(days=step_days)

    t_end = start + win
    while t_end <= end:
        t_start = t_end - win
        e_win = e[(e["date"] > t_start) & (e["date"] <= t_end)].copy()

        edges, nodes, nA, nB, E = build_bipartite_tables(e_win, infl_first, aud_first)
        if E < min_edges:
            print(f"[-] Skip {t_end.date()} (E={E})")
            t_end += step
            continue

        out_dir = data_dir / t_end.strftime("%Y-%m-%d") / "s1_bipartite"
        write_csv(edges, out_dir / "edges.csv")
        write_csv(nodes, out_dir / "nodes.csv")

        manifest = {
            "stage": "s1_bipartite",
            "hashtag": tag,
            "window_start": str(t_start.date()),
            "window_end": str(t_end.date()),
            "n_edges": int(E),
            "n_influencers": int(nA),
            "n_audience": int(nB),
            "win_days": win_days,
            "step_days": step_days
        }
        (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2))
        (out_dir / ".ok").touch()

        print(f"[✓] {t_end.date()} → {E} edges ({nA} infl / {nB} aud)")
        t_end += step


if __name__ == "__main__":
    main()
