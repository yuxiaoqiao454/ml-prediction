#!/usr/bin/env python3
"""
GAE Bipartite Structural Feature Registry

Extracts graph-level features from GAE embeddings on bipartite networks.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from typing import Dict, Optional
from pathlib import Path

# Import GAE model
from gae_model import train_gae


# ============================================================================
# FEATURE REGISTRY
# ============================================================================

GAE_BIP_STRUCT_REGISTRY = {}

def register_feature(name):
    """Decorator to register feature extraction functions."""
    def decorator(fn):
        GAE_BIP_STRUCT_REGISTRY[name] = fn
        return fn
    return decorator


# ============================================================================
# HELPER: BUILD BIPARTITE GRAPH + TYPE FEATURES
# ============================================================================

def compute_edge_weights(
    edges_df: pd.DataFrame,
    exposures_df: pd.DataFrame,
    window_start_epoch: int,
    window_end_epoch: int,
    weight_mode: str = 'decay_sum',
    tau_days: float = 14.0,
    transform: str = 'log1p',  # Can be str or list
    rescale: bool = True
) -> dict:
    """
    Compute recency-based edge weights from event timestamps.
    
    Parameters:
    -----------
    edges_df : pd.DataFrame
        Bipartite edges (source=influencer, target=audience)
    exposures_df : pd.DataFrame
        Exposure data with timestamps (columns: A, B, exposure_epoch)
    window_start_epoch : int
        Window start timestamp
    window_end_epoch : int
        Window end timestamp
    weight_mode : str
        "decay_sum" or "first"
    tau_days : float
        Decay constant in days
    transform : str or list
        Single transform ("none", "log1p", "clip99") or list of transforms to apply in sequence
    rescale : bool
        Whether to rescale median weight to 1.0
    
    Returns:
    --------
    edge_weights : dict
        Mapping {(source, target): weight}
    """
    # Filter exposures to window
    window_exposures = exposures_df[
        (exposures_df['exposure_epoch'] >= window_start_epoch) &
        (exposures_df['exposure_epoch'] <= window_end_epoch)
    ].copy()
    
    if len(window_exposures) == 0:
        return {}
    
    # Convert tau to seconds
    tau_seconds = tau_days * 24 * 3600
    
    # Group by edge (A, B)
    edge_weights = {}
    
    if weight_mode == 'first':
        # Weight by first event only
        first_events = window_exposures.groupby(['A', 'B'])['exposure_epoch'].min()
        
        for (a, b), t_first in first_events.items():
            # Recency: more recent = higher weight
            time_since = window_end_epoch - t_first
            weight = np.exp(-time_since / tau_seconds)
            edge_weights[(a, b)] = weight
    
    else:  # decay_sum
        # Sum of decayed weights for all events
        for (a, b), group in window_exposures.groupby(['A', 'B']):
            timestamps = group['exposure_epoch'].values
            
            # Compute decayed sum
            time_since = window_end_epoch - timestamps
            weights = np.exp(-time_since / tau_seconds)
            total_weight = weights.sum()
            
            edge_weights[(a, b)] = total_weight
    
    if len(edge_weights) == 0:
        return {}
    
    # Apply transform(s)
    weights_array = np.array(list(edge_weights.values()))
    
    # Handle both single transform and list of transforms
    transforms = transform if isinstance(transform, list) else [transform]
    
    for trans in transforms:
        if trans == 'log1p':
            weights_array = np.log1p(weights_array)
        elif trans == 'clip99':
            p99 = np.percentile(weights_array, 99)
            weights_array = np.clip(weights_array, 0, p99)
        elif trans == 'none':
            pass  # No transform
        else:
            print(f"[WARN] Unknown transform '{trans}', skipping")
    
    # Rescale to median=1
    if rescale and len(weights_array) > 0:
        median_weight = np.median(weights_array[weights_array > 0])
        if median_weight > 0:
            weights_array = weights_array / median_weight
    
    # Update dictionary with transformed weights
    edge_weights = {edge: weight for edge, weight in zip(edge_weights.keys(), weights_array)}
    
    return edge_weights



def build_bipartite_graph_with_type_features(
    edges_df: pd.DataFrame,
    exposures_df: pd.DataFrame,
    window_start_epoch: int,
    window_end_epoch: int,
    type_encoding: str = 'onehot',
    temporal_encoding: str = 'norm_flag',
    compute_type: bool = True,
    compute_temporal: bool = True,
    adj_mode: str = 'binary',  # ← NEW!
    recency_config: dict = None  # ← NEW!
) -> tuple:
    """
    Build bipartite adjacency matrix with optional type and temporal features.
    
    Parameters:
    -----------
    ... (previous parameters)
    adj_mode : str
        "binary" or "recency_weighted"
    recency_config : dict
        Config for recency weighting (only used if adj_mode='recency_weighted')
        Keys: weight_mode, tau_days, transform, rescale
    
    Returns:
    --------
    adj : scipy.sparse.csr_matrix (N x N)
    node_to_idx : dict
    n_inf : int
    n_aud : int
    type_features : np.ndarray (N x type_dim) or None
    temporal_features : np.ndarray (N x temporal_dim) or None
    """
    # Get unique nodes
    influencers = edges_df['source'].unique()
    audience = edges_df['target'].unique()
    
    n_inf = len(influencers)
    n_aud = len(audience)
    n_total = n_inf + n_aud
    
    # Create node-to-index mapping
    node_to_idx = {}
    for i, node in enumerate(influencers):
        node_to_idx[node] = i
    for i, node in enumerate(audience):
        node_to_idx[node] = n_inf + i
    
    # ===== Compute edge weights (if needed) =====
    edge_weights_map = {}
    if adj_mode == 'recency_weighted' and exposures_df is not None:
        if recency_config is None:
            recency_config = {
                'weight_mode': 'decay_sum',
                'tau_days': 14.0,
                'transform': 'log1p',
                'rescale': True
            }
        
        edge_weights_map = compute_edge_weights(
            edges_df=edges_df,
            exposures_df=exposures_df,
            window_start_epoch=window_start_epoch,
            window_end_epoch=window_end_epoch,
            **recency_config
        )
    
    # ===== Build adjacency matrix =====
    row_indices = []
    col_indices = []
    data = []

    # Create index-based weight map for train_gae() to use
    edge_weights_by_idx = {}

    for _, edge in edges_df.iterrows():
        src = edge['source']
        tgt = edge['target']
        src_idx = node_to_idx[src]
        tgt_idx = node_to_idx[tgt]
        
        # Determine weight
        if adj_mode == 'binary':
            weight = 1.0
        else:  # recency_weighted
            # Look up weight (default to 1.0 if not found)
            weight = edge_weights_map.get((src, tgt), 1.0)
            
            # Store by matrix indices for train_gae()
            edge_weights_by_idx[(src_idx, tgt_idx)] = weight
            edge_weights_by_idx[(tgt_idx, src_idx)] = weight  # Symmetric
        
        # Add bidirectional edges
        row_indices.extend([src_idx, tgt_idx])
        col_indices.extend([tgt_idx, src_idx])
        data.extend([weight, weight])
    
    adj = sp.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_total, n_total)
    )
    
    # ===== Build type features ===== (unchanged)
    type_features = None
    if compute_type:
        if type_encoding == 'onehot':
            type_features = np.zeros((n_total, 2), dtype=np.float32)
            type_features[:n_inf, 0] = 1.0
            type_features[n_inf:, 1] = 1.0
        elif type_encoding == 'flag':
            type_features = np.zeros((n_total, 1), dtype=np.float32)
            type_features[:n_inf, 0] = 1.0
    
    # ===== Build temporal features ===== (unchanged)
    temporal_features = None
    if compute_temporal:
        window_exposures = exposures_df[
            (exposures_df['exposure_epoch'] >= window_start_epoch) &
            (exposures_df['exposure_epoch'] <= window_end_epoch)
        ]
        
        inf_first_appearance = window_exposures.groupby('A')['post_epoch'].min()
        aud_first_appearance = window_exposures.groupby('B')['exposure_epoch'].min()
        
        window_duration = window_end_epoch - window_start_epoch
        
        if temporal_encoding == 'norm_flag':
            temporal_features = np.zeros((n_total, 2), dtype=np.float32)
            temporal_features[:, 1] = 1.0
        else:
            temporal_features = np.zeros((n_total, 1), dtype=np.float32)
        
        for i, node in enumerate(influencers):
            if node in inf_first_appearance:
                first_time = inf_first_appearance[node]
                t_norm = (first_time - window_start_epoch) / window_duration
                t_norm = np.clip(t_norm, 0.0, 1.0)
                temporal_features[i, 0] = t_norm
                if temporal_encoding == 'norm_flag':
                    temporal_features[i, 1] = 0.0
        
        for i, node in enumerate(audience):
            if node in aud_first_appearance:
                first_time = aud_first_appearance[node]
                t_norm = (first_time - window_start_epoch) / window_duration
                t_norm = np.clip(t_norm, 0.0, 1.0)
                temporal_features[n_inf + i, 0] = t_norm
                if temporal_encoding == 'norm_flag':
                    temporal_features[n_inf + i, 1] = 0.0
    
    # At end of function
    return adj, node_to_idx, n_inf, n_aud, type_features, temporal_features, edge_weights_by_idx


def get_gae_feature_template(prefix: str, embedding_dim: int) -> dict:
    """
    Get template of all GAE features with NaN values.
    Used when GAE is unavailable to ensure consistent schema.
    """
    features = {}
    
    # Availability flags (these are always set)
    features[f'{prefix}_available'] = 0
    features[f'{prefix}_skip_reason'] = 'unknown'
    
    # Per-dimension features (inf and aud)
    for node_type in ['inf', 'aud']:
        for dim in range(embedding_dim):
            features[f'{prefix}_{node_type}_mean_dim{dim}'] = np.nan
            features[f'{prefix}_{node_type}_std_dim{dim}'] = np.nan
    
    # Scalar features (inf and aud)
    for node_type in ['inf', 'aud']:
        scalar_features = [
            'mean_norm', 'std_norm', 'energy',
            'pca1', 'pca2', 'pca3',
            'norm_entropy',
            'max_norm', 'median_norm', 'norm_p25', 'norm_p75',
            'global_mean', 'global_std', 'global_skew', 'global_kurt',
            'l2_mean', 'l2_std', 'cov_frob', 'effective_rank'
        ]
        for feat in scalar_features:
            features[f'{prefix}_{node_type}_{feat}'] = np.nan
    
    # Context features
    features[f'{prefix}_n_nodes'] = np.nan
    features[f'{prefix}_n_edges'] = np.nan
    features[f'{prefix}_n_influencers'] = np.nan
    features[f'{prefix}_n_audience'] = np.nan
    
    # Test metrics
    features[f'{prefix}_test_auc'] = np.nan
    features[f'{prefix}_test_ap'] = np.nan
    features[f'{prefix}_best_epoch'] = np.nan
    
    return features

# ============================================================================
# TIER 1 FEATURES: Core Statistics
# ============================================================================

@register_feature("tier1_core")
def extract_tier1_core_features(embeddings: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Tier 1: Mean & std per dimension, mean/std of norms, energy.
    
    Parameters:
    -----------
    embeddings : np.ndarray (N x d)
        Node embeddings
    prefix : str
        Feature name prefix
    
    Returns:
    --------
    features : dict
    """
    features = {}
    N, d = embeddings.shape
    
    # Per-dimension mean and std
    mean_per_dim = np.mean(embeddings, axis=0)
    std_per_dim = np.std(embeddings, axis=0)
    
    for k in range(d):
        features[f'{prefix}_mean_dim{k}'] = mean_per_dim[k]
        features[f'{prefix}_std_dim{k}'] = std_per_dim[k]
    
    # L2 norms of node embeddings
    norms = np.linalg.norm(embeddings, axis=1)
    
    features[f'{prefix}_mean_norm'] = np.mean(norms)
    features[f'{prefix}_std_norm'] = np.std(norms)
    
    # Energy (Frobenius norm squared)
    features[f'{prefix}_energy'] = np.sum(embeddings ** 2)
    
    return features


@register_feature("tier1_pca")
def extract_tier1_pca_features(embeddings: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Tier 1: PCA eigenvalues of covariance matrix.
    
    Parameters:
    -----------
    embeddings : np.ndarray (N x d)
        Node embeddings
    prefix : str
        Feature name prefix
    
    Returns:
    --------
    features : dict
    """
    features = {}
    N, d = embeddings.shape
    
    try:
        # Covariance matrix: C = Z^T Z / N
        cov = (embeddings.T @ embeddings) / N
        
        # Compute eigenvalues
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]  # Descending order
        
        # Top 3 eigenvalues
        features[f'{prefix}_pca1'] = eigvals[0] if len(eigvals) > 0 else np.nan
        features[f'{prefix}_pca2'] = eigvals[1] if len(eigvals) > 1 else np.nan
        features[f'{prefix}_pca3'] = eigvals[2] if len(eigvals) > 2 else np.nan
        
    except Exception as e:
        features[f'{prefix}_pca1'] = np.nan
        features[f'{prefix}_pca2'] = np.nan
        features[f'{prefix}_pca3'] = np.nan
    
    return features


@register_feature("tier1_entropy")
def extract_tier1_entropy_features(embeddings: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Tier 1: Entropy of norm distribution.
    
    Parameters:
    -----------
    embeddings : np.ndarray (N x d)
        Node embeddings
    prefix : str
        Feature name prefix
    
    Returns:
    --------
    features : dict
    """
    features = {}
    
    # L2 norms
    norms = np.linalg.norm(embeddings, axis=1)
    
    # Normalize to probabilities
    norm_sum = np.sum(norms)
    if norm_sum > 0:
        probs = norms / norm_sum
        # Remove zeros to avoid log(0)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))
        features[f'{prefix}_norm_entropy'] = entropy
    else:
        features[f'{prefix}_norm_entropy'] = np.nan
    
    return features


# ============================================================================
# TIER 2 FEATURES: Extended Statistics
# ============================================================================

@register_feature("tier2_extended")
def extract_tier2_extended_features(embeddings: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Tier 2: Max, median, percentiles, global stats, skew, kurtosis.
    """
    features = {}
    N, d = embeddings.shape
    
    # L2 norms
    norms = np.linalg.norm(embeddings, axis=1)
    
    features[f'{prefix}_max_norm'] = np.max(norms)
    features[f'{prefix}_median_norm'] = np.median(norms)
    features[f'{prefix}_norm_p25'] = np.percentile(norms, 25)
    features[f'{prefix}_norm_p75'] = np.percentile(norms, 75)
    
    # Global statistics (all embedding entries)
    flat_embeddings = embeddings.flatten()
    features[f'{prefix}_global_mean'] = np.mean(flat_embeddings)
    features[f'{prefix}_global_std'] = np.std(flat_embeddings)
    features[f'{prefix}_global_skew'] = stats.skew(flat_embeddings)
    features[f'{prefix}_global_kurt'] = stats.kurtosis(flat_embeddings)
    
    return features


# ============================================================================
# TIER 3 FEATURES: Advanced
# ============================================================================

@register_feature("tier3_advanced")
def extract_tier3_advanced_features(embeddings: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Tier 3: L2 of mean/std vectors, covariance Frobenius, effective rank.
    """
    features = {}
    N, d = embeddings.shape
    
    # L2 norm of mean embedding vector
    mean_vec = np.mean(embeddings, axis=0)
    features[f'{prefix}_l2_mean'] = np.linalg.norm(mean_vec)
    
    # L2 norm of std vector
    std_vec = np.std(embeddings, axis=0)
    features[f'{prefix}_l2_std'] = np.linalg.norm(std_vec)
    
    try:
        # Covariance matrix
        cov = (embeddings.T @ embeddings) / N
        
        # Frobenius norm of covariance
        features[f'{prefix}_cov_frob'] = np.linalg.norm(cov, 'fro')
        
        # Effective rank
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = eigvals[eigvals > 0]
        if len(eigvals) > 0:
            # Normalize eigenvalues
            eigvals_norm = eigvals / np.sum(eigvals)
            # Entropy
            entropy = -np.sum(eigvals_norm * np.log(eigvals_norm))
            # Effective rank = exp(entropy)
            features[f'{prefix}_effective_rank'] = np.exp(entropy)
        else:
            features[f'{prefix}_effective_rank'] = np.nan
            
    except Exception as e:
        features[f'{prefix}_cov_frob'] = np.nan
        features[f'{prefix}_effective_rank'] = np.nan
    
    return features


# ============================================================================
# CONTEXT FEATURES
# ============================================================================

@register_feature("context")
def extract_context_features(adj: sp.spmatrix, n_inf: int, n_aud: int, prefix: str) -> Dict[str, float]:
    """
    Context: Node/edge counts.
    """
    features = {}
    
    features[f'{prefix}_n_nodes'] = adj.shape[0]
    features[f'{prefix}_n_edges'] = adj.nnz // 2  # Divide by 2 for undirected
    features[f'{prefix}_n_influencers'] = n_inf
    features[f'{prefix}_n_audience'] = n_aud
    
    return features


# ============================================================================
# MAIN EXTRACTION FUNCTION
# ============================================================================



def extract_gae_bip_struct_features(edges_df: pd.DataFrame, prefix: str, 
                                     comp_config: dict, feature_config: dict,
                                     window_dir: Path,
                                     verbose: bool = False) -> tuple:
    """
    Extract all GAE bipartite structural features for one window.
    
    Parameters:
    -----------
    edges_df : pd.DataFrame
        Bipartite edges (source=influencer, target=audience)
    prefix : str
        Feature name prefix (e.g., 'gae_bip_struct')
    comp_config : dict
        Computation config
    feature_config : dict
        Feature extraction config
    window_dir : Path
        Path to window directory (to get window date and load exposures)
    verbose : bool
    
    Returns:
    --------
    features : dict
    metadata : dict
    """
    features = {}
    metadata = {
        'success': False,
        'n_nodes': 0,
        'n_edges': 0,
        'too_small': False,
        'train_failed': False,
        'skip_reason': None  # ← ADD THIS
    }
    
    # Extract window date from directory name
    window_end_str = window_dir.name  # e.g., "2019-03-31"
    try:
        from datetime import datetime, timedelta
        window_end_dt = datetime.strptime(window_end_str, "%Y-%m-%d")
        window_start_dt = window_end_dt - timedelta(days=90)
        
        window_end_epoch = int(window_end_dt.timestamp())
        window_start_epoch = int(window_start_dt.timestamp())
    except Exception as e:
        if verbose:
            print(f"[ERROR] Could not parse window date: {e}")
        features[f'{prefix}_too_small'] = 1
        features[f'{prefix}_train_failed'] = 0
        return features, metadata
    
    # Load exposures.parquet from parent directory
    # window_dir structure: .../windows_<hashtag>/2019-03-31/
    # exposures at: .../data/parquets/networks_<hashtag>/exposures.parquet
    hashtag_dir = window_dir.parent
    hashtag_name = hashtag_dir.name.replace('windows_', '')
    
    exposures_path = hashtag_dir.parent / 'parquets' / f'networks_{hashtag_name}' / 'exposures.parquet'
    
    if not exposures_path.exists():
        if verbose:
            print(f"[WARN] Exposures file not found: {exposures_path}")
        # Continue without temporal features
        exposures_df = None
    else:
        try:
            exposures_df = pd.read_parquet(exposures_path)
        except Exception as e:
            if verbose:
                print(f"[WARN] Could not load exposures: {e}")
            exposures_df = None
    
    # Determine what features to compute
    feature_mode = comp_config.get('feature_mode', 'type')
    compute_type = 'type' in feature_mode
    compute_temporal = 'time' in feature_mode and exposures_df is not None
    
    # Build graph with features
    try:
        adj, node_to_idx, n_inf, n_aud, type_features, temporal_features, edge_weights_map = \
            build_bipartite_graph_with_type_features(
                edges_df=edges_df,
                exposures_df=exposures_df,
                window_start_epoch=window_start_epoch,
                window_end_epoch=window_end_epoch,
                type_encoding=comp_config.get('type_encoding', 'onehot'),
                temporal_encoding=comp_config.get('temporal_encoding', 'norm_flag'),
                compute_type=compute_type,
                compute_temporal=compute_temporal,
                adj_mode=comp_config.get('adj_mode', 'binary'),  # ← ADD THIS
                recency_config=comp_config.get('recency')  # ← ADD THIS
            )
        
        metadata['n_nodes'] = adj.shape[0]
        metadata['n_edges'] = adj.nnz // 2
        
    except Exception as e:
        if verbose:
            print(f"[ERROR] Graph building failed: {e}")
        features[f'{prefix}_too_small'] = 1
        features[f'{prefix}_train_failed'] = 0
        return features, metadata
    
    # # ADD THIS DEBUG PRINT
    # print(f"  [DEBUG] adj_mode={comp_config.get('adj_mode')}")
    # print(f"  [DEBUG] edge_weights_map type: {type(edge_weights_map)}")
    # print(f"  [DEBUG] edge_weights_map size: {len(edge_weights_map) if edge_weights_map else 0}")
    # if edge_weights_map and len(edge_weights_map) > 0:
    #     sample_weights = list(edge_weights_map.values())[:5]
    #     print(f"  [DEBUG] Sample weights: {sample_weights}")

    # Check minimum size
    min_nodes = comp_config.get('min_graph_size', 10)
    if adj.shape[0] < min_nodes or adj.nnz == 0:
        metadata['too_small'] = True
        metadata['skip_reason'] = 'too_few_nodes'  # ← ADD THIS
        features[f'{prefix}_available'] = 0  # ← ADD THIS
        features[f'{prefix}_skip_reason'] = 'too_few_nodes'  # ← ADD THIS
        # features[f'{prefix}_too_small'] = 1
        # features[f'{prefix}_train_failed'] = 0
        return features, metadata
    
    # Train GAE
    try:
        embeddings, training_curve, train_metrics = train_gae(
            adj=adj,
            n_inf=n_inf,
            n_aud=n_aud,
            type_features=type_features,
            temporal_features=temporal_features,
            edge_weights_map=edge_weights_map if comp_config.get('adj_mode') == 'recency_weighted' else None,  # ← ADD THIS
            feature_mode=comp_config.get('feature_mode', 'type'),
            id_dim=comp_config.get('id_dim', 64),
            hidden_dim=comp_config.get('hidden_dim', 32),
            embedding_dim=comp_config.get('embedding_dim', 16),
            max_epochs=comp_config.get('max_epochs', 500),
            learning_rate=comp_config.get('learning_rate', 0.01),
            dropout=comp_config.get('dropout', 0.0),
            train_ratio=comp_config.get('train_ratio', 0.85),
            val_ratio=comp_config.get('val_ratio', 0.05),
            test_ratio=comp_config.get('test_ratio', 0.10),
            patience=comp_config.get('patience', 20),
            check_every=comp_config.get('check_every', 5),
            neg_sample_ratio=comp_config.get('neg_sample_ratio', 1.0),
            min_train_edges=comp_config.get('min_train_edges', 50),
            min_val_edges=comp_config.get('min_val_edges', 10),
            min_test_edges=comp_config.get('min_test_edges', 10),
            use_weighted_bce=comp_config.get('use_weighted_bce', False),  # ← ADD
            pos_weight_scale=comp_config.get('pos_weight_scale', 1.0),    # ← ADD
            random_seed=comp_config.get('random_seed', 42),
            verbose=verbose
        )
        
        if embeddings is None:
            metadata['train_failed'] = True
            # Determine skip reason from train_metrics
            if 'skip_reason' in train_metrics:
                metadata['skip_reason'] = train_metrics['skip_reason']
            else:
                metadata['skip_reason'] = 'training_failed'
            
            features[f'{prefix}_available'] = 0  # ← ADD THIS
            features[f'{prefix}_skip_reason'] = metadata['skip_reason']  # ← ADD THIS
            return features, metadata
        
        metadata.update(train_metrics)
        metadata['training_curve'] = training_curve
        
    except Exception as e:
        metadata['train_failed'] = True
        
        # Determine skip reason from exception
        error_msg = str(e)
        if 'Insufficient train edges' in error_msg:
            skip_reason = 'insufficient_train_edges'
        elif 'Insufficient val edges' in error_msg:
            skip_reason = 'insufficient_val_edges'
        elif 'Insufficient test edges' in error_msg:
            skip_reason = 'insufficient_test_edges'
        else:
            skip_reason = 'training_error'
        
        metadata['skip_reason'] = skip_reason
        features[f'{prefix}_available'] = 0  # ← ADD THIS
        features[f'{prefix}_skip_reason'] = skip_reason  # ← ADD THIS
        return features, metadata
    
    # After train_gae() returns (around line 370):
    if embeddings is not None and comp_config.get('save_embeddings', False):
        # Save embeddings for later analysis
        emb_dir = Path('04_ml_prediction/01_features/outputs/embeddings')
        emb_dir.mkdir(parents=True, exist_ok=True)
        
        hashtag_name = window_dir.parent.name.replace('windows_', '')
        window_name = window_dir.name
        adj_mode = comp_config.get('adj_mode', 'binary')
        feature_mode = comp_config.get('feature_mode', 'type')
        
        emb_file = emb_dir / f"{hashtag_name}_{window_name}_{adj_mode}_{feature_mode}.npz"
        
        np.savez(
            emb_file,
            embeddings=embeddings,
            n_inf=n_inf,
            n_aud=n_aud,
            train_metrics=train_metrics
        )

    # Split embeddings and extract features
    embeddings_inf = embeddings[:n_inf]
    embeddings_aud = embeddings[n_inf:]
    
    if verbose:
        print(f"  Split embeddings: {n_inf} influencers, {n_aud} audience")
        print(f"  Inf embedding shape: {embeddings_inf.shape}")
        print(f"  Aud embedding shape: {embeddings_aud.shape}")
    
    # Extract features separately for each node type
    if feature_config.get('tier1_core', True):
        tier1_inf = GAE_BIP_STRUCT_REGISTRY['tier1_core'](embeddings_inf, f'{prefix}_inf')
        tier1_aud = GAE_BIP_STRUCT_REGISTRY['tier1_core'](embeddings_aud, f'{prefix}_aud')
        features.update(tier1_inf)
        features.update(tier1_aud)
    
    if feature_config.get('tier1_pca', True):
        pca_inf = GAE_BIP_STRUCT_REGISTRY['tier1_pca'](embeddings_inf, f'{prefix}_inf')
        pca_aud = GAE_BIP_STRUCT_REGISTRY['tier1_pca'](embeddings_aud, f'{prefix}_aud')
        features.update(pca_inf)
        features.update(pca_aud)
    
    if feature_config.get('tier1_entropy', True):
        ent_inf = GAE_BIP_STRUCT_REGISTRY['tier1_entropy'](embeddings_inf, f'{prefix}_inf')
        ent_aud = GAE_BIP_STRUCT_REGISTRY['tier1_entropy'](embeddings_aud, f'{prefix}_aud')
        features.update(ent_inf)
        features.update(ent_aud)
    
    if feature_config.get('tier2_extended', True):
        tier2_inf = GAE_BIP_STRUCT_REGISTRY['tier2_extended'](embeddings_inf, f'{prefix}_inf')
        tier2_aud = GAE_BIP_STRUCT_REGISTRY['tier2_extended'](embeddings_aud, f'{prefix}_aud')
        features.update(tier2_inf)
        features.update(tier2_aud)
    
    if feature_config.get('tier3_advanced', True):
        tier3_inf = GAE_BIP_STRUCT_REGISTRY['tier3_advanced'](embeddings_inf, f'{prefix}_inf')
        tier3_aud = GAE_BIP_STRUCT_REGISTRY['tier3_advanced'](embeddings_aud, f'{prefix}_aud')
        features.update(tier3_inf)
        features.update(tier3_aud)
    
    if feature_config.get('context', True):
        context = GAE_BIP_STRUCT_REGISTRY['context'](adj, n_inf, n_aud, prefix)
        features.update(context)
    
    # Success flags
    features[f'{prefix}_too_small'] = 0
    features[f'{prefix}_train_failed'] = 0
    # Success - mark as available
    features[f'{prefix}_available'] = 1  # ← ADD THIS
    features[f'{prefix}_skip_reason'] = ''  # ← ADD THIS (empty for success)
    

    metadata['success'] = True
    
    # Add test metrics as features
    if 'test_auc' in metadata:
        features[f'{prefix}_test_auc'] = metadata['test_auc']
        features[f'{prefix}_test_ap'] = metadata['test_ap']
        features[f'{prefix}_best_epoch'] = metadata['best_epoch']
    
    return features, metadata








# def build_bipartite_graph_with_type_features(
#     edges_df: pd.DataFrame,
#     exposures_df: pd.DataFrame,
#     window_start_epoch: int,
#     window_end_epoch: int,
#     type_encoding: str = 'onehot',
#     temporal_encoding: str = 'norm_flag',
#     compute_type: bool = True,
#     compute_temporal: bool = True
# ) -> tuple:
#     """
#     Build bipartite adjacency matrix with optional type and temporal features.
    
#     Parameters:
#     -----------
#     edges_df : pd.DataFrame
#         Bipartite edges (source=influencer, target=audience)
#     exposures_df : pd.DataFrame
#         Exposure data with timestamps (columns: A, B, post_epoch, exposure_epoch)
#     window_start_epoch : int
#         Window start timestamp
#     window_end_epoch : int
#         Window end timestamp
#     type_encoding : str
#         "flag" (N×1) or "onehot" (N×2)
#     temporal_encoding : str
#         "norm_flag" (N×2) or "norm_only" (N×1)
#     compute_type : bool
#         Whether to compute type features
#     compute_temporal : bool
#         Whether to compute temporal features
    
#     Returns:
#     --------
#     adj : scipy.sparse.csr_matrix (N x N)
#     node_to_idx : dict
#     n_inf : int
#     n_aud : int
#     type_features : np.ndarray (N x type_dim) or None
#     temporal_features : np.ndarray (N x temporal_dim) or None
#     """
#     # Get unique nodes
#     influencers = edges_df['source'].unique()
#     audience = edges_df['target'].unique()
    
#     n_inf = len(influencers)
#     n_aud = len(audience)
#     n_total = n_inf + n_aud
    
#     # Create node-to-index mapping
#     node_to_idx = {}
#     for i, node in enumerate(influencers):
#         node_to_idx[node] = i
#     for i, node in enumerate(audience):
#         node_to_idx[node] = n_inf + i
    
#     # Build adjacency matrix
#     row_indices = []
#     col_indices = []
#     data = []
    
#     for _, edge in edges_df.iterrows():
#         src_idx = node_to_idx[edge['source']]
#         tgt_idx = node_to_idx[edge['target']]
        
#         row_indices.extend([src_idx, tgt_idx])
#         col_indices.extend([tgt_idx, src_idx])
        
#         weight = edge.get('weight', 1.0)
#         data.extend([weight, weight])
    
#     adj = sp.csr_matrix(
#         (data, (row_indices, col_indices)),
#         shape=(n_total, n_total)
#     )
    
#     # ===== Build type features =====
#     type_features = None
#     if compute_type:
#         if type_encoding == 'onehot':
#             type_features = np.zeros((n_total, 2), dtype=np.float32)
#             type_features[:n_inf, 0] = 1.0   # Influencers = [1, 0]
#             type_features[n_inf:, 1] = 1.0   # Audience = [0, 1]
#         elif type_encoding == 'flag':
#             type_features = np.zeros((n_total, 1), dtype=np.float32)
#             type_features[:n_inf, 0] = 1.0   # Influencers = [1]
#             # Audience = [0]
    
#     # ===== Build temporal features =====
#     temporal_features = None
#     if compute_temporal:
#         # Filter exposures to window
#         window_exposures = exposures_df[
#             (exposures_df['exposure_epoch'] >= window_start_epoch) &
#             (exposures_df['exposure_epoch'] <= window_end_epoch)
#         ]
        
#         # Compute first appearance times
#         # Influencers: first POST time
#         inf_first_appearance = window_exposures.groupby('A')['post_epoch'].min()
        
#         # Audience: first EXPOSURE time (comment time)
#         aud_first_appearance = window_exposures.groupby('B')['exposure_epoch'].min()
        
#         # Initialize temporal features
#         window_duration = window_end_epoch - window_start_epoch
        
#         # [t_norm, t_missing] or [t_norm] depending on encoding
#         if temporal_encoding == 'norm_flag':
#             temporal_features = np.zeros((n_total, 2), dtype=np.float32)
#             temporal_features[:, 1] = 1.0  # Initialize missing flag to 1
#         else:  # norm_only
#             temporal_features = np.zeros((n_total, 1), dtype=np.float32)
        
#         # Fill in influencer timestamps
#         for i, node in enumerate(influencers):
#             if node in inf_first_appearance:
#                 first_time = inf_first_appearance[node]
#                 t_norm = (first_time - window_start_epoch) / window_duration
#                 t_norm = np.clip(t_norm, 0.0, 1.0)
#                 temporal_features[i, 0] = t_norm
#                 if temporal_encoding == 'norm_flag':
#                     temporal_features[i, 1] = 0.0  # Not missing
        
#         # Fill in audience timestamps
#         for i, node in enumerate(audience):
#             if node in aud_first_appearance:
#                 first_time = aud_first_appearance[node]
#                 t_norm = (first_time - window_start_epoch) / window_duration
#                 t_norm = np.clip(t_norm, 0.0, 1.0)
#                 temporal_features[n_inf + i, 0] = t_norm
#                 if temporal_encoding == 'norm_flag':
#                     temporal_features[n_inf + i, 1] = 0.0  # Not missing
    
#     return adj, node_to_idx, n_inf, n_aud, type_features, temporal_features





# def extract_gae_bip_struct_features(edges_df: pd.DataFrame, prefix: str, 
#                                      comp_config: dict, feature_config: dict,
#                                      verbose: bool = False) -> tuple:
#     """
#     Extract all GAE bipartite structural features for one window.
    
#     Parameters:
#     -----------
#     edges_df : pd.DataFrame
#         Bipartite edges (source=influencer, target=audience)
#     prefix : str
#         Feature name prefix (e.g., 'gae_bip_struct')
#     config : dict
#         Config parameters (hidden_dim, embedding_dim, epochs, etc.)
#     verbose : bool
#         Print progress
    
#     Returns:
#     --------
#     features : dict
#         All extracted features
#     metadata : dict
#         Computation metadata
#     """
#     features = {}
#     metadata = {
#         'success': False,
#         'n_nodes': 0,
#         'n_edges': 0,
#         'too_small': False,
#         'train_failed': False
#     }
    
#     # Build graph
#     try:
#         adj, node_to_idx, n_inf, n_aud = \
#             build_bipartite_graph_with_type_features(edges_df)
        
#         metadata['n_nodes'] = adj.shape[0]
#         metadata['n_edges'] = adj.nnz // 2
        
#     except Exception as e:
#         if verbose:
#             print(f"[ERROR] Graph building failed: {e}")
#         features[f'{prefix}_too_small'] = 1
#         features[f'{prefix}_train_failed'] = 0
#         return features, metadata
    
#     # Check minimum size
#     min_nodes = comp_config.get('min_graph_size', 10)
#     if adj.shape[0] < min_nodes or adj.nnz == 0:
#         metadata['too_small'] = True
#         features[f'{prefix}_too_small'] = 1
#         features[f'{prefix}_train_failed'] = 0
#         return features, metadata
    
#     # Train GAE
#     try:
#         embeddings, training_curve, train_metrics = train_gae(
#             adj=adj,
#             n_inf=n_inf,
#             n_aud=n_aud,
#             feature_mode=comp_config.get('feature_mode', 'type_only'),
#             id_dim=comp_config.get('id_dim', 64),
#             type_encoding=comp_config.get('type_encoding', 'flag'),
#             hidden_dim=comp_config.get('hidden_dim', 32),
#             embedding_dim=comp_config.get('embedding_dim', 16),
#             max_epochs=comp_config.get('max_epochs', 500),
#             learning_rate=comp_config.get('learning_rate', 0.01),
#             dropout=comp_config.get('dropout', 0.0),
#             train_ratio=comp_config.get('train_ratio', 0.85),
#             val_ratio=comp_config.get('val_ratio', 0.05),
#             test_ratio=comp_config.get('test_ratio', 0.10),
#             patience=comp_config.get('patience', 20),
#             check_every=comp_config.get('check_every', 5),
#             neg_sample_ratio=comp_config.get('neg_sample_ratio', 1.0),
#             random_seed=comp_config.get('random_seed', 42),
#             verbose=verbose
#         )
        
#         if embeddings is None:
#             metadata['train_failed'] = True
#             features[f'{prefix}_too_small'] = 0
#             features[f'{prefix}_train_failed'] = 1
#             return features, metadata
        
#         # Store training metrics in metadata
#         metadata.update(train_metrics)
#         metadata['training_curve'] = training_curve  # For saving later
        
#     except Exception as e:
#         if verbose:
#             print(f"[ERROR] GAE training failed: {e}")
#         metadata['train_failed'] = True
#         features[f'{prefix}_too_small'] = 0
#         features[f'{prefix}_train_failed'] = 1
#         return features, metadata
    
#     # Split embeddings by node type
#     embeddings_inf = embeddings[:n_inf]    # Influencer embeddings
#     embeddings_aud = embeddings[n_inf:]    # Audience embeddings
        
#     if verbose:
#         print(f"  Split embeddings: {n_inf} influencers, {n_aud} audience")
#         print(f"  Inf embedding shape: {embeddings_inf.shape}")
#         print(f"  Aud embedding shape: {embeddings_aud.shape}")
        
#     # Extract features separately for each node type
        
#     # Tier 1 Core
#     if feature_config.get('tier1_core', True):
#         tier1_inf = GAE_BIP_STRUCT_REGISTRY['tier1_core'](embeddings_inf, f'{prefix}_inf')
#         tier1_aud = GAE_BIP_STRUCT_REGISTRY['tier1_core'](embeddings_aud, f'{prefix}_aud')
#         features.update(tier1_inf)
#         features.update(tier1_aud)

#     # Tier 1 PCA
#     if feature_config.get('tier1_pca', True):
#         pca_inf = GAE_BIP_STRUCT_REGISTRY['tier1_pca'](embeddings_inf, f'{prefix}_inf')
#         pca_aud = GAE_BIP_STRUCT_REGISTRY['tier1_pca'](embeddings_aud, f'{prefix}_aud')
#         features.update(pca_inf)
#         features.update(pca_aud)

#     # Tier 1 Entropy
#     if feature_config.get('tier1_entropy', True):
#         ent_inf = GAE_BIP_STRUCT_REGISTRY['tier1_entropy'](embeddings_inf, f'{prefix}_inf')
#         ent_aud = GAE_BIP_STRUCT_REGISTRY['tier1_entropy'](embeddings_aud, f'{prefix}_aud')
#         features.update(ent_inf)
#         features.update(ent_aud)

#     # Tier 2 Extended
#     if feature_config.get('tier2_extended', True):
#         tier2_inf = GAE_BIP_STRUCT_REGISTRY['tier2_extended'](embeddings_inf, f'{prefix}_inf')
#         tier2_aud = GAE_BIP_STRUCT_REGISTRY['tier2_extended'](embeddings_aud, f'{prefix}_aud')
#         features.update(tier2_inf)
#         features.update(tier2_aud)

#     # Tier 3 Advanced
#     if feature_config.get('tier3_advanced', True):
#         tier3_inf = GAE_BIP_STRUCT_REGISTRY['tier3_advanced'](embeddings_inf, f'{prefix}_inf')
#         tier3_aud = GAE_BIP_STRUCT_REGISTRY['tier3_advanced'](embeddings_aud, f'{prefix}_aud')
#         features.update(tier3_inf)
#         features.update(tier3_aud)

#     # Context (already role-aware, don't duplicate)
#     if feature_config.get('context', True):
#         context = GAE_BIP_STRUCT_REGISTRY['context'](adj, n_inf, n_aud, prefix)
#         features.update(context)
        
#     # Success flags
#     features[f'{prefix}_too_small'] = 0
#     features[f'{prefix}_train_failed'] = 0
    
#     metadata['success'] = True
    
#     # Add test metrics as features
#     if 'test_auc' in metadata:
#         features[f'{prefix}_test_auc'] = metadata['test_auc']
#         features[f'{prefix}_test_ap'] = metadata['test_ap']
#         features[f'{prefix}_best_epoch'] = metadata['best_epoch']
    
#     return features, metadata