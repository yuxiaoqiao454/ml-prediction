#!/usr/bin/env python3
"""
FAST Graph Embedding Extraction - No Node2Vec Precomputation!

Uses gensim Word2Vec directly on random walks to avoid 60-second init overhead.

Expected: ~3-5 seconds per window instead of 60+ seconds.
"""

import argparse
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import networkx as nx
import random
import gc
from scipy import stats
from scipy.linalg import svd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from gensim.models import Word2Vec


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_hashtag_list(labels_path):
    labels = pd.read_parquet(labels_path)
    return sorted(labels['hashtag'].unique())


def find_network_windows(hashtag, base_path):
    windows_dir = Path(base_path.format(hashtag=hashtag))
    if not windows_dir.exists():
        return []
    
    windows = []
    for window_folder in sorted(windows_dir.iterdir()):
        if window_folder.is_dir() and len(window_folder.name) == 10:
            try:
                window_date = pd.to_datetime(window_folder.name)
                windows.append((hashtag, window_date.strftime('%Y-%m-%d')))
            except:
                continue
    return windows


def load_graph_edges(hashtag, window_date, base_path):
    windows_dir = Path(base_path.format(hashtag=hashtag))
    edges_path = windows_dir / window_date / 's2_proj' / 'count_B_imported' / 'edges.csv'
    
    if not edges_path.exists():
        return None
    
    try:
        return pd.read_csv(edges_path)
    except:
        return None


def build_graph(edges):
    return nx.from_pandas_edgelist(edges, 'source', 'target')


def sample_large_graph(G, target_nodes=7000):
    """Forest fire sampling for large graphs."""
    if len(G) <= target_nodes:
        return G, False
    
    sampled_nodes = set()
    seed = random.choice(list(G.nodes()))
    sampled_nodes.add(seed)
    frontier = {seed}
    p_forward = 0.7
    
    while len(sampled_nodes) < target_nodes and frontier:
        current = frontier.pop()
        neighbors = list(G.neighbors(current))
        
        n_to_burn = int(np.random.geometric(1 - p_forward) * len(neighbors))
        n_to_burn = min(n_to_burn, len(neighbors), target_nodes - len(sampled_nodes))
        
        if n_to_burn > 0:
            burned = set(random.sample(neighbors, n_to_burn))
            new_nodes = burned - sampled_nodes
            sampled_nodes.update(new_nodes)
            frontier.update(new_nodes)
        
        if not frontier and len(sampled_nodes) < target_nodes:
            remaining = set(G.nodes()) - sampled_nodes
            if remaining:
                seed = random.choice(list(remaining))
                sampled_nodes.add(seed)
                frontier.add(seed)
    
    return G.subgraph(list(sampled_nodes)[:target_nodes]).copy(), True


def analyze_graph_structure(G, was_sampled=False, original_size=None):
    if len(G) == 0:
        return {
            'n_nodes': 0, 'n_edges': 0, 'n_components': 0,
            'largest_component_pct': 0.0, 'was_sampled': False, 'original_n_nodes': 0,
        }
    
    components = list(nx.connected_components(G))
    largest = max(components, key=len) if components else set()
    
    return {
        'n_nodes': len(G.nodes()),
        'n_edges': len(G.edges()),
        'n_components': len(components),
        'largest_component_size': len(largest),
        'largest_component_pct': len(largest) / len(G.nodes()),
        'was_sampled': was_sampled,
        'original_n_nodes': original_size if was_sampled else len(G.nodes()),
    }


def generate_embeddings_gensim(G, config):
    """
    Generate embeddings using gensim Word2Vec directly.
    
    NO node2vec precomputation overhead - goes straight to random walks!
    """
    if len(G) == 0:
        return {}
    
    try:
        # Get config
        dim = config['embeddings']['node2vec']['dimensions']
        walk_length = config['embeddings']['node2vec']['walk_length']
        num_walks = config['embeddings']['node2vec']['num_walks']
        
        # Generate random walks (DeepWalk-style: uniform random)
        walks = []
        nodes = list(G.nodes())
        
        for _ in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = [node]
                for _ in range(walk_length - 1):
                    neighbors = list(G.neighbors(walk[-1]))
                    if not neighbors:
                        break
                    walk.append(random.choice(neighbors))
                walks.append([str(n) for n in walk])
        
        # Train Word2Vec
        model = Word2Vec(
            walks,
            vector_size=dim,
            window=5,
            min_count=1,
            workers=1,
            epochs=1,
            sg=1  # Skip-gram
        )
        
        # Extract embeddings
        embeddings = {}
        for node in G.nodes():
            try:
                embeddings[node] = model.wv[str(node)]
            except KeyError:
                pass
        
        del model, walks
        gc.collect()
        
        return embeddings
    
    except Exception as e:
        return {}


def pool_embeddings(embeddings, graph_info):
    if len(embeddings) == 0:
        return {}
    
    embedding_matrix = np.array(list(embeddings.values()))
    n_nodes, n_dims = embedding_matrix.shape
    
    features = {
        'graph_n_nodes': graph_info['n_nodes'],
        'graph_n_edges': graph_info['n_edges'],
        'graph_n_components': graph_info['n_components'],
        'graph_was_sampled': 1 if graph_info.get('was_sampled', False) else 0,
        'graph_original_n_nodes': graph_info.get('original_n_nodes', graph_info['n_nodes']),
    }
    
    # Per-dimension stats
    mean_vals = np.mean(embedding_matrix, axis=0)
    std_vals = np.std(embedding_matrix, axis=0)
    
    for i in range(n_dims):
        features[f'mean_dim{i}'] = mean_vals[i]
        features[f'std_dim{i}'] = std_vals[i]
    
    # Global stats
    features['mean_global'] = np.mean(embedding_matrix)
    features['std_global'] = np.std(embedding_matrix)
    features['skew_global'] = stats.skew(embedding_matrix.flatten())
    features['kurt_global'] = stats.kurtosis(embedding_matrix.flatten())
    
    # Size normalization
    if n_nodes > 1:
        log_n = np.log(n_nodes)
        features['mean_norm'] = features['mean_global'] / log_n
        features['std_norm'] = features['std_global'] / log_n
    
    # SVD
    try:
        _, S, _ = svd(embedding_matrix, full_matrices=False)
        features['sv1'] = S[0]
        features['sv2'] = S[1] if len(S) > 1 else 0
        features['sv3'] = S[2] if len(S) > 2 else 0
    except:
        pass
    
    return features


def process_hashtag(hashtag, config, sample_threshold=7000):
    base_path = config['input']['network_windows']
    windows = find_network_windows(hashtag, base_path)
    
    if not windows:
        return []
    
    hashtag_features = []
    prev_features = None
    
    for _, window_date in sorted(windows):
        edges = load_graph_edges(hashtag, window_date, base_path)
        if edges is None or len(edges) == 0:
            continue
        
        G = build_graph(edges)
        original_size = len(G.nodes())
        was_sampled = False
        
        if original_size > sample_threshold:
            G, was_sampled = sample_large_graph(G, target_nodes=sample_threshold)
        
        graph_info = analyze_graph_structure(G, was_sampled=was_sampled, original_size=original_size)
        
        # Largest component
        if graph_info['n_components'] > 1:
            components = list(nx.connected_components(G))
            largest = max(components, key=len)
            G = G.subgraph(largest).copy()
        
        if len(G.nodes()) < 5:
            continue
        
        # Generate embeddings (FAST!)
        embeddings = generate_embeddings_gensim(G, config)
        
        if len(embeddings) == 0:
            continue
        
        features = pool_embeddings(embeddings, graph_info)
        
        row = {'hashtag': hashtag, 'window_end': window_date}
        
        for key, val in features.items():
            row[f'emb_{key}'] = val
        
        # Deltas
        if prev_features and config['embeddings']['pooling'].get('compute_delta', False):
            for key, val in features.items():
                if key in prev_features and not key.startswith('graph_'):
                    row[f'emb_{key}_delta'] = val - prev_features[key]
        
        hashtag_features.append(row)
        prev_features = features
        
        del embeddings, G
        gc.collect()
    
    return hashtag_features


def load_checkpoint(output_path):
    if Path(output_path).exists():
        df = pd.read_parquet(output_path)
        return df, set(df['hashtag'].unique())
    return None, set()


def save_checkpoint(all_features, output_path):
    pd.DataFrame(all_features).to_parquet(output_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--labels', default='04_ml_prediction/02_labels/labels_h550_cpinside.parquet')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--sample-threshold', type=int, default=7000)
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    print("="*80)
    print("FAST Embedding Extraction (gensim, no precomputation)")
    print("="*80)
    
    config = load_config(args.config)
    output_path = Path(config['output']['features_output'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.resume:
        existing_df, processed = load_checkpoint(output_path)
        if existing_df is not None:
            print(f"\n✓ Resuming: {len(processed)} hashtags done, {len(existing_df)} windows\n")
            all_features = existing_df.to_dict('records')
        else:
            processed = set()
            all_features = []
    else:
        processed = set()
        all_features = []
    
    hashtags = load_hashtag_list(args.labels)
    if args.limit:
        hashtags = hashtags[:args.limit]
    
    hashtags_to_process = [h for h in hashtags if h not in processed]
    
    print(f"Config: dim={config['embeddings']['node2vec']['dimensions']}, "
          f"walks={config['embeddings']['node2vec']['num_walks']}, "
          f"len={config['embeddings']['node2vec']['walk_length']}")
    print(f"Hashtags: {len(hashtags_to_process)} to process\n")
    
    for i, hashtag in enumerate(tqdm(hashtags_to_process, desc="Hashtags")):
        try:
            features = process_hashtag(hashtag, config, sample_threshold=args.sample_threshold)
            
            if features:
                all_features.extend(features)
                if args.verbose:
                    print(f"  [{hashtag}] {len(features)} windows")
            
            if (i + 1) % 10 == 0:
                save_checkpoint(all_features, output_path)
                if args.verbose:
                    print(f"  ✓ Checkpoint: {len(all_features)} windows")
        
        except Exception as e:
            if args.verbose:
                print(f"  [{hashtag}] ERROR: {e}")
            continue
        
        gc.collect()
    
    save_checkpoint(all_features, output_path)
    
    print(f"\n{'='*80}")
    print(f"✓ Done! {len(all_features)} windows, {len(all_features[0])-2 if all_features else 0} features")
    print(f"  Output: {output_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()