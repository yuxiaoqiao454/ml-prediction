#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run clustering across all time windows and analyze temporal trends.

Steps:
1. Loop through each window folder (YYYY-MM-DD).
2. Load the projection edges + nodes (count_shared_A_imported by default).
3. Run clustering and save results (labels.csv + metrics.json).
4. After all windows are done, compute:
   - Modularity trend
   - Number of communities trend
   - NMI between consecutive windows
5. Save summary CSV.
"""

from pathlib import Path
import pandas as pd
import json
from sklearn.metrics import normalized_mutual_info_score

import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
networks_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(networks_dir))

from pipeline.clustering import CLUSTERING_REGISTRY
import pipeline.clustering.infomap
import pipeline.clustering.louvain



# # import your Louvain clustering function
# from pipeline.clustering.infomap import cluster

# === 1️⃣ Define paths and parameters ===
import argparse, yaml

# --- Configurable parameters via command line ---
ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True, help="Path to YAML config file")
ap.add_argument("--hashtag", required=True, help="Hashtag (no #)")
args = ap.parse_args()

# --- Load YAML config ---
with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

# --- Extract parameters ---
TAG = args.hashtag
# PROJ_SIDE = cfg.get("projection_side", "B")  # "A" or "B"
# PROJ_METHOD = f"{cfg.get('projection_method', 'cosine')}_{PROJ_SIDE}_imported"
PROJ_SIDES = ["A", "B"]  # A = influencers, B = audience
PROJ_BASE_METHOD = cfg.get('projection_method', 'cosine')
CLUSTER_METHOD = cfg.get("clustering_method", "infomap")
RESOLUTION = cfg.get("resolution", 1.0)

BASE_DIR = Path(cfg["base_dir"]) / f"windows_{TAG}"
OUT_SUMMARY = BASE_DIR / f"summary_{CLUSTER_METHOD}_{PROJ_BASE_METHOD}_{PROJ_SIDES}.csv"




cluster = CLUSTERING_REGISTRY[CLUSTER_METHOD]


# === 2️⃣ Helper: compute NMI between two clustering label files ===
def compute_nmi(labels_prev: pd.DataFrame, labels_curr: pd.DataFrame) -> float:
    merged = labels_prev.merge(labels_curr, on="id", how="inner", suffixes=("_prev", "_curr"))
    if len(merged) == 0:
        return float("nan")
    return normalized_mutual_info_score(merged["cluster_prev"], merged["cluster_curr"])


# # === 3️⃣ Main loop through all window folders ===
# def main():
#     results = []  # store per-window stats for trend analysis

#     # sort ensures chronological order
#     for win_dir in sorted(BASE_DIR.iterdir()):
#         if not win_dir.is_dir():
#             continue
#         if len(win_dir.name) != 10 or win_dir.name[4] != "-" or win_dir.name[7] != "-":
#             continue  # skip non-date folders

#         proj_dir = win_dir / "s2_proj" / PROJ_METHOD
#         edges_path = proj_dir / "edges.csv"
#         nodes_path = proj_dir / "nodes.csv"

#         if not edges_path.exists() or not nodes_path.exists():
#             print(f"[-] Skip {win_dir.name}: projection missing.")
#             continue

#         # --- clustering output directory ---
#         cluster_out = win_dir / "s3_cluster" / PROJ_METHOD / CLUSTER_METHOD
#         cluster_out.mkdir(parents=True, exist_ok=True)

#         # --- skip if already done ---
#         if (cluster_out / "metrics.json").exists():
#             print(f"[=] Louvain already done for {win_dir.name}")
#             with open(cluster_out / "metrics.json", "r") as f:
#                 metrics = json.load(f)
#         else:
#             # --- load projection ---
#             edges = pd.read_csv(edges_path)
#             nodes = pd.read_csv(nodes_path)

#             # --- run Louvain ---
#             print(f"[+] Running {CLUSTER_METHOD.capitalize()} on {win_dir.name} ({len(nodes)} nodes, {len(edges)} edges)")
#             labels, metrics = cluster(edges, nodes, {}, cluster_out)

#         # store modularity, number of communities
#         results.append({
#             "date": win_dir.name,
#             "map_equation": metrics["map_equation"],
#             "n_communities": metrics["n_communities"]
#         })

#     # === 4️⃣ Compute NMI between consecutive windows ===
#     print("\n[→] Computing NMI between consecutive clusterings...")
#     summary_df = pd.DataFrame(results).sort_values("date").reset_index(drop=True)
#     nmis = []

#     prev_labels = None
#     for i, row in summary_df.iterrows():
#         date = row["date"]
#         label_path = BASE_DIR / date / "s3_cluster" / PROJ_METHOD / CLUSTER_METHOD / "labels.csv"

#         if not label_path.exists():
#             nmis.append(float("nan"))
#             continue

#         labels_curr = pd.read_csv(label_path)
#         if prev_labels is None:
#             nmis.append(float("nan"))  # no previous window to compare
#         else:
#             nmis.append(compute_nmi(prev_labels, labels_curr))
#         prev_labels = labels_curr

#     summary_df["NMI_to_prev"] = nmis

#     # === 5️⃣ Save final summary CSV ===
#     summary_df.to_csv(OUT_SUMMARY, index=False)
#     print(f"\n[✓] Summary written → {OUT_SUMMARY}")
#     print(summary_df.tail())


def main():
    # Loop over both projection sides
    for PROJ_SIDE in PROJ_SIDES:
        PROJ_METHOD = f"{PROJ_BASE_METHOD}_{PROJ_SIDE}_imported"
        OUT_SUMMARY = BASE_DIR / f"summary_{CLUSTER_METHOD}_{PROJ_METHOD}.csv"
        
        print(f"\n{'='*80}")
        print(f"Processing side {PROJ_SIDE} ({'influencers' if PROJ_SIDE == 'A' else 'audience'})")
        print(f"{'='*80}\n")
        
        results = []  # store per-window stats for trend analysis

        # sort ensures chronological order
        for win_dir in sorted(BASE_DIR.iterdir()):
            if not win_dir.is_dir():
                continue
            if len(win_dir.name) != 10 or win_dir.name[4] != "-" or win_dir.name[7] != "-":
                continue  # skip non-date folders

            proj_dir = win_dir / "s2_proj" / PROJ_METHOD
            edges_path = proj_dir / "edges.csv"
            nodes_path = proj_dir / "nodes.csv"

            if not edges_path.exists() or not nodes_path.exists():
                print(f"[-] Skip {win_dir.name}: projection missing for side {PROJ_SIDE}.")
                continue

            # --- clustering output directory ---
            cluster_out = win_dir / "s3_cluster" / PROJ_METHOD / CLUSTER_METHOD
            cluster_out.mkdir(parents=True, exist_ok=True)

            # --- skip if already done ---
            if (cluster_out / "metrics.json").exists():
                print(f"[=] {CLUSTER_METHOD.capitalize()} already done for {win_dir.name} (side {PROJ_SIDE})")
                with open(cluster_out / "metrics.json", "r") as f:
                    metrics = json.load(f)
            else:
                # --- load projection ---
                edges = pd.read_csv(edges_path)
                nodes = pd.read_csv(nodes_path)

                # --- run clustering ---
                print(f"[+] Running {CLUSTER_METHOD.capitalize()} on {win_dir.name} side {PROJ_SIDE} ({len(nodes)} nodes, {len(edges)} edges)")
                labels, metrics = cluster(edges, nodes, {}, cluster_out)

            # store metrics
            results.append({
                "date": win_dir.name,
                "map_equation": metrics["map_equation"],
                "n_communities": metrics["n_communities"]
            })

        # === Compute NMI between consecutive windows ===
        print(f"\n[→] Computing NMI for side {PROJ_SIDE} between consecutive clusterings...")
        # summary_df = pd.DataFrame(results).sort_values("date").reset_index(drop=True)
        if not results:
            print(f"\n[!] No windows were processed for side {PROJ_SIDE}. Skipping summary.")
            continue  # Skip to next side
            
        summary_df = pd.DataFrame(results).sort_values("date").reset_index(drop=True)
        nmis = []

        prev_labels = None
        for i, row in summary_df.iterrows():
            date = row["date"]
            label_path = BASE_DIR / date / "s3_cluster" / PROJ_METHOD / CLUSTER_METHOD / "labels.csv"

            if not label_path.exists():
                nmis.append(float("nan"))
                continue

            labels_curr = pd.read_csv(label_path)
            if prev_labels is None:
                nmis.append(float("nan"))  # no previous window to compare
            else:
                nmis.append(compute_nmi(prev_labels, labels_curr))
            prev_labels = labels_curr

        summary_df["NMI_to_prev"] = nmis

        # === Save final summary CSV ===
        summary_df.to_csv(OUT_SUMMARY, index=False)
        print(f"\n[✓] Summary for side {PROJ_SIDE} written → {OUT_SUMMARY}")
        print(summary_df.tail())


if __name__ == "__main__":
    main()
