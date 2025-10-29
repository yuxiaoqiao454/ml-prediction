#!/usr/bin/env python3
"""
Network Feature Registry

Modular feature extraction from network graphs (bipartite, projections, clustered).
"""

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

# ============================================================================
# Registry
# ============================================================================

NETWORK_FEATURE_REGISTRY = {}

def register_network_feature(name):
    """Decorator to register a network feature extraction function."""
    def decorator(fn):
        NETWORK_FEATURE_REGISTRY[name] = fn
        return fn
    return decorator


# ============================================================================
# Helper Functions
# ============================================================================

def load_graph_from_edges(edges_path, directed=False):
    """
    Load graph from edges.csv file.
    
    Returns NetworkX graph or None if file doesn't exist.
    """
    if not Path(edges_path).exists():
        return None
    
    try:
        edges_df = pd.read_csv(edges_path)
        if len(edges_df) == 0:
            return None
        
        # Create graph
        if directed:
            G = nx.from_pandas_edgelist(
                edges_df, 'source', 'target', 
                edge_attr='weight' if 'weight' in edges_df.columns else None,
                create_using=nx.DiGraph()
            )
        else:
            G = nx.from_pandas_edgelist(
                edges_df, 'source', 'target',
                edge_attr='weight' if 'weight' in edges_df.columns else None,
                create_using=nx.Graph()
            )
        
        return G if len(G.nodes()) > 0 else None
        
    except Exception as e:
        return None


def compute_gini(values):
    """
    Compute Gini coefficient for inequality measurement.
    
    Returns value between 0 (perfect equality) and 1 (perfect inequality).
    """
    if len(values) == 0:
        return 0.0
    
    sorted_values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(sorted_values)
    
    return (2 * np.sum((np.arange(1, n+1)) * sorted_values)) / (n * cumsum[-1]) - (n + 1) / n


def safe_divide(a, b, default=0.0):
    """Safe division."""
    return float(a / b) if b != 0 else default


# ============================================================================
# Feature Families
# ============================================================================

@register_network_feature("basic_stats")
def extract_basic_stats(G, config, graph_name):
    """
    Basic graph statistics: nodes, edges, density.
    """
    if G is None:
        return {}
    
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    # Density
    if isinstance(G, nx.Graph) and not G.is_directed():
        max_edges = n_nodes * (n_nodes - 1) / 2
    elif isinstance(G, nx.DiGraph):
        max_edges = n_nodes * (n_nodes - 1)
    else:
        max_edges = n_nodes * n_nodes
    
    density = safe_divide(n_edges, max_edges)
    
    return {
        f'net_{graph_name}_n_nodes': int(n_nodes),
        f'net_{graph_name}_n_edges': int(n_edges),
        f'net_{graph_name}_density': float(density),
    }


@register_network_feature("degree_stats")
def extract_degree_stats(G, config, graph_name):
    """
    Degree distribution statistics.
    """
    if G is None or G.number_of_nodes() == 0:
        return {}
    
    degrees = [d for n, d in G.degree()]
    
    if len(degrees) == 0:
        return {}
    
    return {
        f'net_{graph_name}_avg_degree': float(np.mean(degrees)),
        f'net_{graph_name}_max_degree': int(np.max(degrees)),
        f'net_{graph_name}_std_degree': float(np.std(degrees)),
        f'net_{graph_name}_degree_gini': float(compute_gini(degrees)),
    }


@register_network_feature("clustering")
def extract_clustering(G, config, graph_name):
    """
    Clustering coefficient and transitivity.
    """
    if G is None or G.number_of_nodes() < 3:
        return {}
    
    try:
        # Sample for speed if graph is large
        sample_size = config.get('sample_size', 1000)
        if G.number_of_nodes() > sample_size:
            sampled_nodes = np.random.choice(list(G.nodes()), sample_size, replace=False)
            avg_clustering = np.mean([nx.clustering(G, n) for n in sampled_nodes])
        else:
            avg_clustering = nx.average_clustering(G)
        
        # Transitivity (global clustering coefficient)
        transitivity = nx.transitivity(G)
        
        return {
            f'net_{graph_name}_avg_clustering': float(avg_clustering),
            f'net_{graph_name}_transitivity': float(transitivity),
        }
    except Exception:
        return {}


@register_network_feature("community_basic")
def extract_community_basic(G, labels_path, config, graph_name):
    """
    Community structure metrics from clustering results.
    
    Requires labels.csv from clustering (infomap/louvain).
    """
    if G is None or not Path(labels_path).exists():
        return {}
    
    try:
        # Load community labels
        labels_df = pd.read_csv(labels_path)
        
        if len(labels_df) == 0:
            return {}
        
        # Get community assignments
        communities = labels_df['cluster'].values
        unique_communities = np.unique(communities)
        n_communities = len(unique_communities)
        
        # Community sizes
        community_sizes = [np.sum(communities == c) for c in unique_communities]
        total_nodes = len(communities)
        
        # Size distribution entropy
        size_probs = np.array(community_sizes) / total_nodes
        size_entropy = -np.sum(size_probs * np.log(size_probs + 1e-10))
        
        # Largest community percentage
        largest_community_pct = max(community_sizes) / total_nodes
        
        # Modularity (requires graph structure + labels)
        # Convert labels to dict
        node_to_community = dict(zip(labels_df['id'], labels_df['cluster']))
        
        # Create partition (list of sets, one per community)
        partition_dict = {}
        for node, comm in node_to_community.items():
            if node in G.nodes():
                partition_dict[node] = comm
        
        # Compute modularity if possible
        try:
            from networkx.algorithms.community import modularity
            # Convert partition_dict to list of sets
            communities_sets = {}
            for node, comm in partition_dict.items():
                if comm not in communities_sets:
                    communities_sets[comm] = set()
                communities_sets[comm].add(node)
            
            mod = modularity(G, communities_sets.values())
        except Exception:
            mod = 0.0
        
        return {
            f'net_{graph_name}_n_communities': int(n_communities),
            f'net_{graph_name}_community_size_entropy': float(size_entropy),
            f'net_{graph_name}_largest_community_pct': float(largest_community_pct),
            f'net_{graph_name}_modularity': float(mod),
        }
        
    except Exception:
        return {}


@register_network_feature("temporal_delta")
def extract_temporal_delta(curr_features, prev_features, config, graph_name):
    """
    Compute changes between current and previous window.
    
    Requires features from both windows.
    """
    if prev_features is None:
        return {}
    
    delta_features = {}
    
    # Define metrics to compute deltas for
    delta_metrics = config.get('metrics', ['edge_turnover', 'delta_modularity', 'delta_avg_degree', 'delta_density'])
    
    # Edge turnover (if n_edges available)
    curr_edges = curr_features.get(f'net_{graph_name}_n_edges')
    prev_edges = prev_features.get(f'net_{graph_name}_n_edges')
    
    if curr_edges is not None and prev_edges is not None and prev_edges > 0:
        turnover = abs(curr_edges - prev_edges) / prev_edges
        delta_features[f'net_{graph_name}_edge_turnover'] = float(turnover)
    
    # Modularity change
    curr_mod = curr_features.get(f'net_{graph_name}_modularity')
    prev_mod = prev_features.get(f'net_{graph_name}_modularity')
    
    if curr_mod is not None and prev_mod is not None:
        delta_features[f'net_{graph_name}_delta_modularity'] = float(curr_mod - prev_mod)
    
    # Avg degree change
    curr_deg = curr_features.get(f'net_{graph_name}_avg_degree')
    prev_deg = prev_features.get(f'net_{graph_name}_avg_degree')
    
    if curr_deg is not None and prev_deg is not None:
        delta_features[f'net_{graph_name}_delta_avg_degree'] = float(curr_deg - prev_deg)
    
    # Density change
    curr_dens = curr_features.get(f'net_{graph_name}_density')
    prev_dens = prev_features.get(f'net_{graph_name}_density')
    
    if curr_dens is not None and prev_dens is not None:
        delta_features[f'net_{graph_name}_delta_density'] = float(curr_dens - prev_dens)
    
    return delta_features