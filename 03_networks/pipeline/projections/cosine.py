#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cosine-similarity projection for one window.

Input  (within one time-window folder):
  s1_bipartite/edges.csv   # columns: A,B,weight
  s1_bipartite/nodes.csv

Output (creates under s2_proj/cosine_A_imported and cosine_B_imported):
  edges.csv, nodes.csv, _manifest.json, .ok
"""

from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import json

def run_cosine_projection(window_dir: Path,
                          min_sim: float = 0.0,
                          min_shared: int = 1,
                          topk: int | None = None,
                          weighted: bool = False,
                          overwrite: bool = False):
    """Perform cosine projection for a single window."""

    bip_path = window_dir / "s1_bipartite" / "edges.csv"
    if not bip_path.exists():
        print(f"[skip] No bipartite edges at {bip_path}")
        return

    df = pd.read_csv(bip_path)
    if "weight" not in df.columns:
        df["weight"] = 1.0
    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)

    # ---- Projection helper ----
    def _cosine(side: str):
        """Project to influencers (A) or audiences (B)."""
        if side == "A":
            L, R = "source", "target"
        else:
            L, R = "target", "source"

        # Precompute ||v_i||^2
        if weighted:
            w2 = df.groupby(L)["weight"].apply(lambda x: np.sum(x**2))
        else:
            w2 = df.groupby(L)[R].nunique().astype(float)
        norms = np.sqrt(w2 + 1e-12)

        # Accumulate dot products for pairs sharing R
        dot = defaultdict(float)
        for _, grp in df.groupby(R):
            if weighted:
                sub = grp.groupby(L)["weight"].sum().reset_index()
                arrL, arrW = sub[L].to_numpy(), sub["weight"].to_numpy()
                for i in range(len(arrL)-1):
                    for j in range(i+1, len(arrL)):
                        dot[(arrL[i], arrL[j])] += arrW[i] * arrW[j]
            else:
                lefts = grp[L].drop_duplicates().to_numpy()
                for i in range(len(lefts)-1):
                    for j in range(i+1, len(lefts)):
                        dot[(lefts[i], lefts[j])] += 1.0

        if not dot:
            return pd.DataFrame(), pd.DataFrame()

        u, v, val = zip(*((a, b, s) for (a, b), s in dot.items()))
        edges = pd.DataFrame({"source": u, "target": v, "dot": val})
        edges["sim"] = edges["dot"] / (edges["source"].map(norms) * edges["target"].map(norms) + 1e-12)
        edges = edges[edges["sim"] >= min_sim]
        if topk:
            keep = set()
            for col in ["source", "target"]:
                for node, g in edges.groupby(col):
                    idx = g["sim"].nlargest(topk).index
                    keep.update(idx)
            edges = edges.loc[sorted(keep)]

        nodes = pd.concat([edges["source"], edges["target"]]).value_counts()
        nodes = nodes.rename_axis("id").reset_index(name="degree")

        return edges[["source", "target", "sim"]], nodes

    # ---- Save ----
    out_root = window_dir / "s2_proj"
    out_root.mkdir(exist_ok=True)

    for side in ["A", "B"]:
        edges, nodes = _cosine(side)
        if edges.empty:
            print(f"[warn] Empty projection for {side}")
            continue
        folder = out_root / f"cosine_{side}_imported"
        folder.mkdir(parents=True, exist_ok=True)
        edges.to_csv(folder / "edges.csv", index=False)
        nodes.to_csv(folder / "nodes.csv", index=False)
        (folder / ".ok").touch()
        with open(folder / "_manifest.json", "w") as f:
            json.dump({
                "method": "cosine",
                "side": side,
                "min_sim": min_sim,
                "topk": topk,
                "weighted": weighted
            }, f, indent=2)

    print(f"[✓] {window_dir.name}: cosine projections done.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Cosine projection for one window folder")
    ap.add_argument("--window-dir", required=True)
    ap.add_argument("--min-sim", type=float, default=0.0)
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--weighted", action="store_true")
    args = ap.parse_args()
    run_cosine_projection(Path(args.window-dir),
                          min_sim=args.min_sim,
                          topk=args.topk or None,
                          weighted=args.weighted)
