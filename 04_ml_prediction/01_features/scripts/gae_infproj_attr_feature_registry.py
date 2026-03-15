#!/usr/bin/env python3
"""
Feature registry for GAE embeddings on influencer projection with node attributes.

Graph: count_A_imported (influencer-to-influencer projection)
Node features: [followers, followees, posts, audience_size, first_post_time, has_profile]
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.decomposition import PCA
from scipy.stats import entropy, skew, kurtosis

# Import GAE training function
from gae_model import train_gae

# ============================================================================
# Feature Registry
# ============================================================================

GAE_INFPROJ_ATTR_FEATURES = {}

def register_feature(name: str):
    """Decorator to register feature extraction functions."""
    def decorator(func):
        GAE_INFPROJ_ATTR_FEATURES[name] = func
        return func
    return decorator


# ============================================================================
# Helper: Build Projection Graph with Attributes
# ============================================================================

def build_projection_with_attributes(
    projection_edges_path: Path,
    bipartite_edges_path: Path,
    influencer_attrs_path: Path,
    verbose: bool = False
) -> Tuple[Optional[sp.csr_matrix], Optional[np.ndarray], Optional[Dict], Optional[int]]:
    """
    Build influencer projection graph with node attributes.
    
    Returns:
    --------
    adj : sparse matrix (N x N) - adjacency matrix
    features : np.ndarray (N x 6) - [followers, followees, posts, audience_size, first_post_time, has_profile]
    node_to_idx : dict - mapping from username to index
    n_nodes : int
    """
    try:
        # Load projection edges
        proj_df = pd.read_csv(projection_edges_path)
        if len(proj_df) == 0:
            return None, None, None, None
        
        # Load bipartite edges for audience_size and timestamps
        bip_df = pd.read_csv(bipartite_edges_path)
        if len(bip_df) == 0:
            return None, None, None, None
        
        # Load influencer attributes
        attr_df = pd.read_parquet(influencer_attrs_path)
        
        # Get unique influencers from projection
        influencers = sorted(set(proj_df['source'].unique()) | set(proj_df['target'].unique()))
        node_to_idx = {node: i for i, node in enumerate(influencers)}
        n_nodes = len(influencers)
        
        if n_nodes < 3:
            return None, None, None, None
        
        # ============================================================================
        # Build Adjacency Matrix
        # ============================================================================
        row = proj_df['source'].map(node_to_idx).values
        col = proj_df['target'].map(node_to_idx).values
        
        # Use weight if available, else 1
        if 'weight' in proj_df.columns:
            data = proj_df['weight'].values
        else:
            data = np.ones(len(proj_df))
        
        # Create symmetric adjacency (undirected)
        adj = sp.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
        adj = adj + adj.T  # Symmetrize
        adj = adj.tocsr()
        
        # ============================================================================
        # Build Feature Matrix (N x 6)
        # ============================================================================
        features = np.zeros((n_nodes, 6), dtype=np.float32)
        
        # Compute audience_size (out-degree in bipartite)
        # In bipartite edges.csv: source=influencer, target=audience, weight=count
        audience_counts = bip_df.groupby('source')['target'].nunique()
        
        # Compute first_post_time (normalized within window)
        if 'first_exposure_epoch' in bip_df.columns:
            timestamps = bip_df.groupby('source')['first_exposure_epoch'].min()
            t_min, t_max = timestamps.min(), timestamps.max()
            if t_max > t_min:
                timestamps_norm = (timestamps - t_min) / (t_max - t_min)
            else:
                timestamps_norm = pd.Series(0.5, index=timestamps.index)
        else:
            timestamps_norm = pd.Series(0.5, index=audience_counts.index)
        
        # Load profile attributes (indexed by username)
        attr_lookup = attr_df.set_index('username')[['followers', 'followees', 'posts']]
        
        # Fill feature matrix
        for i, username in enumerate(influencers):
            # Attribute 0-2: Profile data (followers, followees, posts)
            if username in attr_lookup.index:
                profile = attr_lookup.loc[username]
                features[i, 0] = profile['followers'] if pd.notna(profile['followers']) else 0.0
                features[i, 1] = profile['followees'] if pd.notna(profile['followees']) else 0.0
                features[i, 2] = profile['posts'] if pd.notna(profile['posts']) else 0.0
                features[i, 5] = 1.0  # has_profile_data flag
            else:
                features[i, 0:3] = 0.0
                features[i, 5] = 0.0  # missing profile data
            
            # Attribute 3: Audience size
            features[i, 3] = audience_counts.get(username, 0.0)
            
            # Attribute 4: First post time (normalized)
            features[i, 4] = timestamps_norm.get(username, 0.5)
        
        # ============================================================================
        # Normalize Features to [0, 1] (per column)
        # ============================================================================
        for j in range(5):  # Don't normalize the flag (column 5)
            col_min = features[:, j].min()
            col_max = features[:, j].max()
            if col_max > col_min:
                features[:, j] = (features[:, j] - col_min) / (col_max - col_min)
            else:
                features[:, j] = 0.5  # Constant column -> midpoint
        
        return adj, features, node_to_idx, n_nodes
    
    except Exception as e:
        if verbose:
            print(f"[ERROR] Failed to build projection graph: {e}")
        return None, None, None, None


# ============================================================================
# Feature Extraction Functions
# ============================================================================

@register_feature("tier1_core")
def extract_tier1_core(embeddings: np.ndarray) -> Dict[str, float]:
    """
    Tier 1 core features: mean/std per dimension, norms, energy.
    
    Returns 39 features:
    - gae_mean_dim0...15 (16)
    - gae_std_dim0...15 (16)
    - gae_mean_norm, gae_std_norm, gae_energy (3)
    - gae_pca1, gae_pca2, gae_pca3 (3)
    - gae_norm_entropy (1)
    """
    N, d = embeddings.shape
    features = {}
    
    # Mean per dimension
    for i in range(d):
        features[f'gae_mean_dim{i}'] = float(embeddings[:, i].mean())
    
    # Std per dimension
    for i in range(d):
        features[f'gae_std_dim{i}'] = float(embeddings[:, i].std())
    
    # Norms
    norms = np.linalg.norm(embeddings, axis=1)
    features['gae_mean_norm'] = float(norms.mean())
    features['gae_std_norm'] = float(norms.std())
    features['gae_energy'] = float((norms ** 2).sum())
    
    return features


@register_feature("tier1_pca")
def extract_tier1_pca(embeddings: np.ndarray) -> Dict[str, float]:
    """Top 3 PCA eigenvalues of covariance matrix."""
    try:
        pca = PCA(n_components=min(3, embeddings.shape[1]))
        pca.fit(embeddings)
        
        features = {}
        for i in range(len(pca.explained_variance_)):
            features[f'gae_pca{i+1}'] = float(pca.explained_variance_[i])
        
        # Pad with zeros if < 3 components
        for i in range(len(pca.explained_variance_), 3):
            features[f'gae_pca{i+1}'] = 0.0
        
        return features
    except:
        return {f'gae_pca{i+1}': 0.0 for i in range(3)}


@register_feature("tier1_entropy")
def extract_tier1_entropy(embeddings: np.ndarray) -> Dict[str, float]:
    """Entropy of norm distribution."""
    norms = np.linalg.norm(embeddings, axis=1)
    
    # Discretize into 20 bins
    hist, _ = np.histogram(norms, bins=20)
    hist = hist + 1e-10  # Avoid log(0)
    prob = hist / hist.sum()
    
    return {'gae_norm_entropy': float(entropy(prob))}


@register_feature("tier2_extended")
def extract_tier2_extended(embeddings: np.ndarray) -> Dict[str, float]:
    """
    Tier 2 extended features.
    
    Returns 8 features:
    - Norm percentiles: max, median, p25, p75
    - Global stats: mean, std, skew, kurtosis
    """
    norms = np.linalg.norm(embeddings, axis=1)
    global_vec = embeddings.flatten()
    
    return {
        'gae_max_norm': float(norms.max()),
        'gae_median_norm': float(np.median(norms)),
        'gae_norm_p25': float(np.percentile(norms, 25)),
        'gae_norm_p75': float(np.percentile(norms, 75)),
        'gae_global_mean': float(global_vec.mean()),
        'gae_global_std': float(global_vec.std()),
        'gae_global_skew': float(skew(global_vec)),
        'gae_global_kurt': float(kurtosis(global_vec))
    }


@register_feature("tier3_advanced")
def extract_tier3_advanced(embeddings: np.ndarray) -> Dict[str, float]:
    """
    Tier 3 advanced features.
    
    Returns 4 features:
    - L2 norms of mean/std vectors
    - Frobenius norm of covariance
    - Effective rank
    """
    mean_vec = embeddings.mean(axis=0)
    std_vec = embeddings.std(axis=0)
    cov = np.cov(embeddings.T)
    
    # Effective rank
    try:
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        if len(eigenvalues) > 0:
            probs = eigenvalues / eigenvalues.sum()
            eff_rank = np.exp(entropy(probs + 1e-10))
        else:
            eff_rank = 0.0
    except:
        eff_rank = 0.0
    
    return {
        'gae_l2_mean': float(np.linalg.norm(mean_vec)),
        'gae_l2_std': float(np.linalg.norm(std_vec)),
        'gae_cov_frob': float(np.linalg.norm(cov, 'fro')),
        'gae_effective_rank': float(eff_rank)
    }


@register_feature("context")
def extract_context(n_nodes: int, n_edges: int) -> Dict[str, float]:
    """Context features: graph size."""
    return {
        'gae_n_nodes': float(n_nodes),
        'gae_n_edges': float(n_edges)
    }


# ============================================================================
# Main Feature Extraction
# ============================================================================

def extract_gae_infproj_attr_features(
    projection_edges_path: Path,
    bipartite_edges_path: Path,
    influencer_attrs_path: Path,
    config: dict,
    verbose: bool = False
) -> Tuple[Dict[str, float], Dict[str, any]]:
    """
    Extract GAE features from influencer projection with attributes.
    
    Returns:
    --------
    features : dict - extracted features
    metadata : dict - diagnostic info
    """
    # Build graph with attributes
    adj, node_features, node_to_idx, n_nodes = build_projection_with_attributes(
        projection_edges_path, bipartite_edges_path, influencer_attrs_path, verbose
    )
    
    # Check if graph is valid
    if adj is None or n_nodes is None:
        return {}, {
            'gae_too_small': 1,
            'gae_train_failed': 0,
            'success': False,
            'n_nodes': 0,
            'n_edges': 0
        }
    
    n_edges = adj.nnz // 2  # Undirected
    min_size = config.get('min_graph_size', 10)
    
    if n_nodes < min_size:
        if verbose:
            print(f"      [SKIP] Graph too small (n={n_nodes} < {min_size})")
        return {}, {
            'gae_too_small': 1,
            'gae_train_failed': 0,
            'success': False,
            'n_nodes': n_nodes,
            'n_edges': n_edges
        }
    
    # Train GAE
    embeddings = train_gae(
        adj=adj,
        features=node_features,
        hidden_dim=config.get('hidden_dim', 32),
        embedding_dim=config.get('embedding_dim', 16),
        epochs=config.get('epochs', 200),
        learning_rate=config.get('learning_rate', 0.01),
        dropout=config.get('dropout', 0.0),
        verbose=verbose
    )
    
    if embeddings is None:
        if verbose:
            print(f"      [FAIL] GAE training failed")
        return {}, {
            'gae_too_small': 0,
            'gae_train_failed': 1,
            'success': False,
            'n_nodes': n_nodes,
            'n_edges': n_edges
        }
    
    # Extract features from embeddings
    features = {}
    
    # Tier 1
    if config.get('tier1_core', True):
        features.update(extract_tier1_core(embeddings))
    if config.get('tier1_pca', True):
        features.update(extract_tier1_pca(embeddings))
    if config.get('tier1_entropy', True):
        features.update(extract_tier1_entropy(embeddings))
    
    # Tier 2
    if config.get('tier2_extended', True):
        features.update(extract_tier2_extended(embeddings))
    
    # Tier 3
    if config.get('tier3_advanced', True):
        features.update(extract_tier3_advanced(embeddings))
    
    # Context
    if config.get('context', True):
        features.update(extract_context(n_nodes, n_edges))
    
    # Metadata
    metadata = {
        'gae_too_small': 0,
        'gae_train_failed': 0,
        'success': True,
        'n_nodes': n_nodes,
        'n_edges': n_edges
    }
    
    if verbose:
        print(f"      ✓ n={n_nodes}, e={n_edges}")
    
    return features, metadata


# ============================================================================
# Delta Features (Temporal Changes)
# ============================================================================

def compute_delta_features(
    curr_features: Dict[str, float],
    prev_features: Optional[Dict[str, float]],
    prefix: str = "gae"
) -> Dict[str, float]:
    """
    Compute temporal delta features.
    
    Only compute deltas for scalar aggregates, not per-dimension features.
    """
    if prev_features is None:
        return {}
    
    # List of keys to compute deltas for
    delta_keys = [
        'gae_mean_norm', 'gae_std_norm', 'gae_energy',
        'gae_pca1', 'gae_pca2', 'gae_pca3',
        'gae_norm_entropy',
        'gae_max_norm', 'gae_median_norm',
        'gae_norm_p25', 'gae_norm_p75',
        'gae_global_mean', 'gae_global_std',
        'gae_global_skew', 'gae_global_kurt',
        'gae_l2_mean', 'gae_l2_std',
        'gae_cov_frob', 'gae_effective_rank',
        'gae_n_nodes', 'gae_n_edges'
    ]
    
    deltas = {}
    for key in delta_keys:
        if key in curr_features and key in prev_features:
            curr_val = curr_features[key]
            prev_val = prev_features[key]
            
            if prev_val != 0:
                delta = (curr_val - prev_val) / abs(prev_val)
            else:
                delta = 0.0 if curr_val == 0 else 1.0
            
            deltas[f'{key}_delta'] = delta
    
    return deltas