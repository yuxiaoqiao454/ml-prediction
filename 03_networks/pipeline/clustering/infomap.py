import json
import pandas as pd
from pathlib import Path
import networkx as nx
from infomap import Infomap
from typing import Optional

from . import register_clustering

@register_clustering("infomap")
def cluster(proj_edges:pd.DataFrame,
            proj_nodes: pd.DataFrame,
            params: dict,
            out_dir: Optional[Path]=None):
    """
    Run Infomap on weighted undirected graph.

    Parameters
    ----------
    proj_edges : DataFrame with columns u, v, weight
    proj_nodes : DataFrame with column id
    params : dict, may include {"trials": 10, "two_level": True}
    out_dir : where to save results
    """

    # 1. Build a graph
    G = nx.Graph()
    for node_id in proj_nodes["id"]:
        G.add_node(node_id)
    G.add_weighted_edges_from(proj_edges[["source", "target", "weight"]].values)

    if G.number_of_edges() == 0 or G.number_of_nodes() == 0:
        print("[!] Empty graph — skipping Infomap")
        return pd.DataFrame(columns=["id","cluster"]), {
            "map_equation": float("nan"),
            "n_communities": 0,
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
        }

    # 2. Build the infomap
    two_level = params.get("two_level", True)
    im = Infomap(two_level=two_level, silent=True)

    # 3. Add edges
    id_map = {node: i for i, node in enumerate(G.nodes())}

    for u, v, data in G.edges(data=True):
        iu, iv = id_map[u], id_map[v]
        im.addLink(iu, iv, float(data.get("weight", 1.0)))

    # 4. Run Infomap algo
    im.run()
    modules_int = im.getModules()
    modules = {node: modules_int[id_map[node]] for node in G.nodes()}
    code_length = im.codelength

    # 5. convert to dataframe
    labels = pd.DataFrame(list(modules.items()), columns=["id", "cluster"])

    metrics = {
        "map_equation": code_length,
        "n_communities": labels["cluster"].nunique(),
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges()
    }

    # 6. Save results
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        labels.to_csv(out_dir / "labels.csv", index=False)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"[✓] Infomap done → {out_dir}")

    return labels, metrics