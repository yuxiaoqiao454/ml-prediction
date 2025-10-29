"""
Migrate legacy per window files (bipartite + projected) into the new layout.

Old (per window):
  bipartite_edges.csv
  bipartite_nodes.csv
  proj_influencer_edges.csv
  proj_audience_edges.csv
  triplet.html

New (per window):
  s1_bipartite/{edges.csv, nodes.csv, _manifest.json, .ok}
  s2_proj/count_shared_A_imported/{edges.csv, nodes.csv, _manifest.json, .ok}
  s2_proj/count_shared_B_imported/{edges.csv, nodes.csv, _manifest.json, .ok}
  triplet.html (left as-is)
"""

from pathlib import Path
import json
import shutil
import pandas as pd

WINDOWS_ROOT = Path("/Users/xiaoqiao/Documents/GitHub/yu-4yp/Ins_dataset/03-networks/networks_athleisure/unipartite/windows_athleisure")

def write_manifest(path:Path, payload:dict):
    path.write_text(json.dumps(payload, indent=2))

def ensure_node_degrees(edges_csv: Path, partition_label: str) -> pd.DataFrame:
    """
    Build a nodes.csv from the edges.csv:
        id, degree (times a node appears across edges), partition
    """
    if not edges_csv.exists():
        return pd.DataFrame(columns=["id", "degress", "partition"])
    
    df = pd.read_csv(edges_csv)
    if {"source", "target"}.issubset(df.columns):
        u = df["source"]; v = df["target"]
    else:
        raise SystemExit(f"Edges File {edges_csv} missing source/targert columns")
    
    deg = pd.concat([u,v]).value_counts().rename_axis("id").reset_index(name="degree") # The concatenation -> series of ids -> value counts -> ids as index axis and counts as series values
    deg["partition"] = partition_label
    return deg

def migrate_one_window(win_dir: Path):
    # Paths to old files
    bip_edges_old = win_dir / "bipartite_edges.csv"
    bip_nodes_old = win_dir / "bipartite_nodes.csv"
    projA_old     = win_dir / "proj_influencer_edges.csv"
    projB_old     = win_dir / "proj_audience_edges.csv"
    triplet_old   = win_dir / "triplet.html"  

    if not (bip_edges_old.exists() and bip_nodes_old.exists()):
        print(f"[-]Skip {win_dir.name}: missing bipartite csvs")
        return
    
    # Stage 1: s1_bipartite
    s1_dir = win_dir / "s1_bipartite"
    s1_dir.mkdir(parents=True, exist_ok=True)

    s1_edges_new = s1_dir / "edges.csv"
    s1_nodes_new = s1_dir / "nodes.csv"
    if not s1_edges_new.exists():
        shutil.copy(bip_edges_old, s1_edges_new)
    if not s1_nodes_new.exists():
        shutil.copy(bip_nodes_old, s1_nodes_new)

    write_manifest(s1_dir / "_manifest.json", {
        "stage": "s1_bipartite",
        "status": "imported",
        "source_files": ["bipartite_edges.csv", "bipartite_nodes.csv"],
        "note": "Migrated from legacy outputs."
    })
    (s1_dir / ".ok").touch()

    # Stage 2: projections (influencer side = A)
    if projA_old.exists():
        projA_dir = win_dir / "s2_proj" / "count_shared_A_imported"
        projA_dir.mkdir(parents=True, exist_ok=True)
        projA_edges_new = projA_dir / "edges.csv"
        if not projA_edges_new.exists():
            shutil.copy2(projA_old, projA_edges_new)

        projA_nodes_new = projA_dir / "nodes.csv"
        if not projA_nodes_new.exists():
            degA = ensure_node_degrees(projA_edges_new, partition_label="influencer")
            degA.to_csv(projA_nodes_new, index=False)

        write_manifest(projA_dir / "_manifest.json", {
            "stage": "s2_proj",
            "method": "count_shared",
            "side": "A",
            "status": "imported",
            "source_files": ["proj_influencer_edges.csv"]
        })

        (projA_dir / ".ok").touch()

    # projections (audience side = B)
    if projB_old.exists():
        projB_dir = win_dir / "s2_proj" / "count_shared_B_imported"
        projB_dir.mkdir(parents=True, exist_ok=True)
        projB_edges_new = projB_dir / "edges.csv"
        if not projB_edges_new.exists():
            shutil.copy2(projB_old, projB_edges_new)

        projB_nodes_new = projB_dir / "nodes.csv"
        if not projB_nodes_new.exists():
            degB = ensure_node_degrees(projB_edges_new, partition_label="audience")
            degB.to_csv(projB_nodes_new, index=False)

        write_manifest(projB_dir / "_manifest.json", {
            "stage": "s2_proj",
            "method": "count_shared",
            "side": "B",
            "status": "imported",
            "source_files": ["proj_influencer_edges.csv"]
        })

        (projB_dir / ".ok").touch()

    
    print(f"[✓] Migrated {win_dir.name}")

def main():
    # A "for loop" over all subfolders that look like dates.
    # A for loop in Python means: take each item in a collection, do the body once per item.
    for win_dir in sorted(WINDOWS_ROOT.iterdir()):
        if not win_dir.is_dir():
            continue
        if len(win_dir.name) == 10 and win_dir.name[4] == "-" and win_dir.name[7] == "-":
            migrate_one_window(win_dir)

if __name__ == "__main__":
    main()

    





