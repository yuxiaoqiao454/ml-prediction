#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count-based projection for one window.

Input  (within one time-window folder):
  s1_bipartite/edges.csv   # columns: source,target,weight
  s1_bipartite/nodes.csv

Output (creates under s2_proj/count_A_imported and count_B_imported):
  edges.csv, nodes.csv, _manifest.json, .ok

Key difference from cosine:
  - Weight = raw count of shared neighbors (no normalization)
  - Preserves influence of high-degree nodes (better for trend prediction)
  - No similarity threshold - uses min_shared parameter instead
"""

from __future__ import annotations
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import json
from itertools import combinations

def run_count_projection(window_dir: Path,
                         min_sim: float = 0.0,  # ignored, kept for API compatibility
                         topk: int | None = None,
                         weighted: bool = False,
                         min_shared: int = 1,  # use this instead of min_sim
                         overwrite: bool = False):
    """
    Perform count-based projection for a single window.
    
    Parameters:
    -----------
    window_dir : Path
        Directory containing s1_bipartite/ folder
    min_sim : float
        Ignored (kept for compatibility with cosine API)
    topk : int
        Keep only top-k edges per node by weight
    weighted : bool
        If True, use edge weights from bipartite; if False, treat all edges as weight=1
    min_shared : int
        Minimum number of shared neighbors to create projection edge (default=1)
    overwrite : bool
        If True, regenerate even if outputs exist
    """
    
    bip_path = window_dir / "s1_bipartite" / "edges.csv"
    if not bip_path.exists():
        print(f"[skip] No bipartite edges at {bip_path}")
        return

    # Check if already done (unless overwrite)
    out_root = window_dir / "s2_proj"
    if not overwrite:
        if (out_root / "count_A_imported" / ".ok").exists() and \
           (out_root / "count_B_imported" / ".ok").exists():
            print(f"[skip] Count projection already exists for {window_dir.name}")
            return

    df = pd.read_csv(bip_path)
    if "weight" not in df.columns:
        df["weight"] = 1.0
    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)

    # ---- Projection helper ----
    def _count_projection(side: str):
        """
        Project to influencers (A) or audiences (B).
        Weight = number of shared neighbors (no normalization).
        """
        if side == "A":
            L, R = "source", "target"  # project to sources (influencers)
        else:
            L, R = "target", "source"  # project to targets (audience)

        # Build projection: count shared R neighbors for each pair of L nodes
        pair_weight = Counter()
        pair_ts_min = defaultdict(lambda: np.inf)
        
        # Group by right side (the side we're projecting across)
        for _, grp in df.groupby(R):
            if weighted:
                # Weighted version: each shared R contributes product of weights
                sub = grp.groupby(L)["weight"].sum().reset_index()
                arrL, arrW = sub[L].to_numpy(), sub["weight"].to_numpy()
                if len(arrL) < 2:
                    continue
                for i in range(len(arrL)-1):
                    for j in range(i+1, len(arrL)):
                        u, v = (arrL[i], arrL[j]) if arrL[i] < arrL[j] else (arrL[j], arrL[i])
                        pair_weight[(u, v)] += arrW[i] * arrW[j]
            else:
                # Unweighted version: each shared R contributes +1
                # lefts = grp[L].drop_duplicates().to_numpy()
                # if len(lefts) < 2:
                #     continue
                # for i in range(len(lefts)-1):
                #     for j in range(i+1, len(lefts)):
                #         u, v = (lefts[i], lefts[j]) if lefts[i] < lefts[j] else (lefts[j], lefts[i])
                #         pair_weight[(u, v)] += 1
                lefts = sorted(grp[L].drop_duplicates())
                if len(lefts) < 2:
                    continue
                for u, v in combinations(lefts, 2):  # much faster!
                    pair_weight[(u, v)] += 1

            # # Track timestamp: earliest co-exposure time
            # if "first_exposure_epoch" in grp.columns:
            #     grp_sorted = grp.sort_values("first_exposure_epoch")
            #     L_list = grp_sorted[L].to_numpy()
            #     T_list = grp_sorted["first_exposure_epoch"].to_numpy()
            #     if len(L_list) >= 2:
            #         for i in range(len(L_list)-1):
            #             li, ti = L_list[i], T_list[i]
            #             for j in range(i+1, len(L_list)):
            #                 lj, tj = L_list[j], T_list[j]
            #                 u, v = (li, lj) if li < lj else (lj, li)
            #                 coexp_time = max(ti, tj)
            #                 if coexp_time < pair_ts_min[(u,v)]:
            #                     pair_ts_min[(u,v)] = coexp_time

            # Track timestamp: earliest co-exposure time
            if "first_exposure_epoch" in grp.columns:
                grp_sorted = grp.sort_values("first_exposure_epoch")
                L_list = grp_sorted[L].to_numpy()
                T_list = grp_sorted["first_exposure_epoch"].to_numpy()
                if len(L_list) >= 2:
                    # NEW: use combinations here too
                    for (i, li), (j, lj) in combinations(enumerate(L_list), 2):
                        u, v = (li, lj) if li < lj else (lj, li)
                        coexp_time = max(T_list[i], T_list[j])
                        if coexp_time < pair_ts_min[(u,v)]:
                            pair_ts_min[(u,v)] = coexp_time

        if not pair_weight:
            return pd.DataFrame(), pd.DataFrame()

        # Filter by min_shared
        filtered = [(u, v, w, pair_ts_min.get((u,v), 0)) 
                   for (u, v), w in pair_weight.items() 
                   if w >= min_shared]
        
        if not filtered:
            return pd.DataFrame(), pd.DataFrame()

        u, v, w, ts = zip(*filtered)
        edges = pd.DataFrame({
            "source": u, 
            "target": v, 
            "weight": w,
            "ts": [int(t) for t in ts]
        })

        # Apply topk filtering
        if topk:
            keep = set()
            for col in ["source", "target"]:
                for node, g in edges.groupby(col):
                    idx = g["weight"].nlargest(topk).index
                    keep.update(idx)
            edges = edges.loc[sorted(keep)]

        # Build node table
        nodes = pd.concat([edges["source"], edges["target"]]).value_counts()
        nodes = nodes.rename_axis("id").reset_index(name="degree")

        return edges, nodes

    # ---- Save ----
    out_root.mkdir(exist_ok=True)

    for side in ["A", "B"]:
        edges, nodes = _count_projection(side)
        if edges.empty:
            print(f"[warn] Empty count projection for {side} in {window_dir.name}")
            continue
        
        folder = out_root / f"count_{side}_imported"
        folder.mkdir(parents=True, exist_ok=True)
        
        edges.to_csv(folder / "edges.csv", index=False)
        nodes.to_csv(folder / "nodes.csv", index=False)
        (folder / ".ok").touch()
        
        with open(folder / "_manifest.json", "w") as f:
            json.dump({
                "method": "count",
                "side": side,
                "min_shared": min_shared,
                "topk": topk,
                "weighted": weighted
            }, f, indent=2)

    print(f"[✓] {window_dir.name}: count projections done.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Count-based projection for one window folder")
    ap.add_argument("--window-dir", required=True)
    ap.add_argument("--min-shared", type=int, default=1)
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--weighted", action="store_true")
    args = ap.parse_args()
    run_count_projection(
        Path(args.window_dir),
        min_shared=args.min_shared,
        topk=args.topk or None,
        weighted=args.weighted
    )