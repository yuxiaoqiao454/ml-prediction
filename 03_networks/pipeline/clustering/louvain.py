"""
Louvain clustering wrapper
input: projected graph (nodes + edges)
output: cluster labels and metrics

"""

import json
from pathlib import Path
import pandas as pd
import networkx as nx
from community import community_louvain
from typing import Optional

# register the method so pipeline can read it (Registry Pattern)
from . import register_clustering

@register_clustering("louvain")
def cluster(proj_edges: pd.DataFrame,
            proj_nodes: pd.DataFrame,
            params: dict,
            out_dir: Optional[Path]=None):
    """
    Run Louvain clustering on a weighted undirected graph.

    Parameters
    ----------
    proj_edges : DataFrame
        Must have columns source, target, weight_shared_audience
    proj_nodes : DataFrame
        Must have column id
    params : dict
        e.g. {"resolution": 1.0, "random_state": 42}
    out_dir : Path, optional
        If given, save outputs here (labels.csv, metrics.json)

    """
    # 1. Build a networkx graph
    G = nx.Graph()
    for node_id in proj_nodes["id"]:
        G.add_node(node_id)
    G.add_weighted_edges_from(proj_edges[['source', 'target', 'sim']].values)


    if G.number_of_edges() == 0 or G.number_of_nodes() == 0:
        print("[!] Empty graph — skipping Louvain")
        return pd.DataFrame(columns=["id","cluster"]), {
            "modularity": float("nan"),
            "n_communities": 0,
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
        }


    # 2. Run Louvain 
    partition = community_louvain.best_partition(
        G,
        weight = "sim",
        resolution = float(params.get("resolution", 1.0)),
        random_state = params.get("random_state", 42)
    )
        


    # 3. Convert result (dict) to DataFrame
    labels = pd.DataFrame({
        "id": list(partition.keys()),
        "cluster": list(partition.values())
    })

    # 4. compute metrics
    Q = community_louvain.modularity(partition, G, weight="weight")
    metrics = {
        "modularity": Q,
        "n_communities": labels["cluster"].nunique(),
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
    }

    # 5. Save results to out_dir
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        labels.to_csv(out_dir / "labels.csv", index=False)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"[✓] Louvain done → {out_dir}")

    return labels, metrics