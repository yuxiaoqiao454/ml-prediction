#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run graph projection (cosine, count_shared, etc.) across all rolling windows.

Steps:
1. Loop through each window folder (YYYY-MM-DD).
2. Load its s1_bipartite/edges.csv.
3. Run the selected projection method (registered under pipeline/projections/__init__.py).
4. Save results under s2_proj/<method>_A_imported and s2_proj/<method>_B_imported.
"""

from pathlib import Path
import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
networks_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(networks_dir))

from pipeline.projections import PROJECTION_REGISTRY


# === 1️⃣ Parameters ===
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
BASE_DIR = Path(cfg["base_dir"]) / f"windows_{TAG}"
PROJ_METHOD = cfg.get("projection_method", "cosine")
MIN_SIM = cfg.get("min_sim", 0.15)
MIN_SHARED = cfg.get("min_shared", 1)
TOPK = cfg.get("topk", 50)
WEIGHTED = cfg.get("weighted", False)
OVERWRITE = cfg.get("overwrite", False)



# === 2️⃣ Load projection function from registry ===
if PROJ_METHOD not in PROJECTION_REGISTRY:
    raise ValueError(f"Projection method '{PROJ_METHOD}' not registered.")
project = PROJECTION_REGISTRY[PROJ_METHOD]


# === 3️⃣ Main ===
def main():
    print(f"\n[→] Running '{PROJ_METHOD}' projection across all windows under {BASE_DIR}\n")
    for win_dir in sorted(BASE_DIR.iterdir()):
        if not win_dir.is_dir():
            continue
        if len(win_dir.name) != 10 or win_dir.name[4] != "-" or win_dir.name[7] != "-":
            continue  # only YYYY-MM-DD folders

        bip_path = win_dir / "s1_bipartite" / "edges.csv"
        if not bip_path.exists():
            print(f"[-] Skip {win_dir.name}: no bipartite edges found.")
            continue

        print(f"[+] Projecting {win_dir.name} using {PROJ_METHOD} ...")
        project(win_dir,
                min_sim=MIN_SIM,
                min_shared=MIN_SHARED,
                topk=TOPK,
                weighted=WEIGHTED,
                overwrite=OVERWRITE)

    print(f"\n[✓] Projection complete for all windows ({PROJ_METHOD}).")


if __name__ == "__main__":
    main()
