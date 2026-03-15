#!/usr/bin/env python3
"""
Spectral Feature Registry

Modular extraction of spectral graph features for burst prediction.
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, lobpcg 
from scipy.stats import entropy as scipy_entropy
from scipy.sparse import csr_matrix, diags, eye
import scipy.sparse.linalg as spla
import warnings
import signal 


# ============================================================================
# Registry
# ============================================================================

SPECTRAL_FEATURE_REGISTRY = {}

def register_spectral_family(name):
    """Decorator to register a spectral feature extraction function."""
    def decorator(fn):
        SPECTRAL_FEATURE_REGISTRY[name] = fn
        return fn
    return decorator


# ============================================================================
# Graph Preprocessing
# ============================================================================

def build_graph_from_edges(edges_df, use_weights=True, remove_isolated=True):
    """Build NetworkX graph from edges DataFrame."""
    G = nx.Graph()
    
    for _, row in edges_df.iterrows():
        if use_weights:
            G.add_edge(row['source'], row['target'], weight=row['weight'])
        else:
            G.add_edge(row['source'], row['target'], weight=1.0)
    
    metadata = {
        'n_nodes_original': G.number_of_nodes(),
        'n_edges_original': G.number_of_edges()
    }
    
    if remove_isolated:
        isolated = list(nx.isolates(G))
        G.remove_nodes_from(isolated)
        metadata['n_isolated_removed'] = len(isolated)
    
    metadata['n_nodes_final'] = G.number_of_nodes()
    metadata['n_edges_final'] = G.number_of_edges()
    
    return G, metadata


def extract_largest_cc(G):
    """Extract largest connected component."""
    import time
    t0 = time.time()

    if G.number_of_nodes() == 0:
        return G, []
    
    components = sorted(nx.connected_components(G), key=len, reverse=True)

    t_cc = time.time() - t0
    print(f"[DEBUG] nx.connected_components: {t_cc:.3f}s for {len(components)} CCs")
    
    if len(components) == 0:
        return G, []
    
    G_lcc = G.subgraph(components[0]).copy()


    
    return G_lcc, components


def compute_normalized_laplacian(G, epsilon=1e-10):
    """
    Compute normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
    
    ✅ FIXED: Fully sparse implementation to avoid memory blowup.
    """
    if G.number_of_nodes() == 0:
        return None, []
    
    node_list = list(G.nodes())
    A = nx.adjacency_matrix(G, nodelist=node_list, weight='weight')
    
    degrees = np.array(A.sum(axis=1)).flatten()
    degrees = np.maximum(degrees, epsilon)
    
    D_inv_sqrt = np.power(degrees, -0.5)
    
    # ✅ FIXED: Sparse diagonal (not dense np.diag)
    D_inv_sqrt_mat = diags(D_inv_sqrt)
    
    # ✅ FIXED: Sparse identity (not dense np.eye)
    I = eye(len(node_list), format="csr")
    
    L = I - D_inv_sqrt_mat @ A @ D_inv_sqrt_mat
    
    return L, node_list

class TimeoutException(Exception):
    """Raised when computation exceeds time limit."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Computation timed out")



# ============================================================================
# Eigenvalue Computation
# ============================================================================

def compute_eigenvalues_arpack(L, k_small, k_tail, max_dense_size, timeout_seconds=300):
    """
    Compute eigenvalues using ARPACK (current method).
    
    Returns:
    --------
    result : dict or None
    success : bool
    method : str ('arpack', 'arpack_dense', or 'failed')
    """
    n = L.shape[0]
    
    result = {
        'small': None,
        'large': None,
        'all': None,
        'eigenvectors_fiedler': None,
        'eigenvectors_max': None
    }
    
    try:
        # Set timeout alarm (Unix only)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            
            # Small eigenvalues (shift-invert)
            vals_small, vecs_small = eigsh(
                L, 
                k=min(k_small + 1, n - 1),
                which='SM',
                sigma=0,
                return_eigenvectors=True
            )
            
            idx_small = np.argsort(vals_small)
            vals_small = vals_small[idx_small]
            vecs_small = vecs_small[:, idx_small]
            
            result['small'] = vals_small[1:]  # Skip λ₁ ≈ 0
            result['eigenvectors_fiedler'] = vecs_small[:, 1]
            
            # Large eigenvalues
            if k_tail > 0 and n > k_small + k_tail + 1:
                vals_large, vecs_large = eigsh(
                    L,
                    k=min(k_tail, n - k_small - 1),
                    which='LM',
                    return_eigenvectors=True
                )
                
                idx_large = np.argsort(vals_large)[::-1]
                vals_large = vals_large[idx_large]
                vecs_large = vecs_large[:, idx_large]
                
                result['large'] = vals_large
                result['eigenvectors_max'] = vecs_large[:, 0]
        
        # Cancel alarm
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        
        return result, True, 'arpack'
        
    except TimeoutException:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        print(f"[TIMEOUT] ARPACK exceeded {timeout_seconds}s (n={n})")
        return None, False, 'timeout'
        
    except Exception as e:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        
        # Fallback to dense for small graphs
        if n < max_dense_size:
            try:
                L_dense = L.toarray()
                vals_all = np.linalg.eigvalsh(L_dense)
                vals_all = np.sort(vals_all)
                
                result['small'] = vals_all[1:k_small+1]
                result['large'] = vals_all[-k_tail:] if k_tail > 0 else None
                result['all'] = vals_all
                
                return result, True, 'arpack_dense'
            except:
                pass
        
        return None, False, 'failed'


def compute_eigenvalues_lobpcg(L, k_small, tol=1e-3, maxiter=200):
    """
    Compute small eigenvalues using LOBPCG.
    
    For large graphs where ARPACK struggles.
    
    Returns:
    --------
    result : dict or None
    success : bool
    method : str ('lobpcg' or 'failed')
    """
    n = L.shape[0]
    
    result = {
        'small': None,
        'large': None,  # LOBPCG doesn't compute large eigenvalues
        'all': None,
        'eigenvectors_fiedler': None,
        'eigenvectors_max': None
    }
    
    try:
        # Initial guess: random vectors
        k_target = min(k_small + 1, n - 1)
        X = np.random.randn(n, k_target)
        
        # Orthogonalize
        X, _ = np.linalg.qr(X)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            
            # LOBPCG: smallest eigenvalues
            vals, vecs = lobpcg(
                L, 
                X, 
                largest=False,
                tol=tol,
                maxiter=maxiter
            )
        
        # Sort ascending
        idx = np.argsort(vals)
        vals = vals[idx]
        vecs = vecs[:, idx]
        
        # Skip λ₁ ≈ 0, keep λ₂...λ_(k+1)
        result['small'] = vals[1:k_target]
        result['eigenvectors_fiedler'] = vecs[:, 1]
        
        # Note: large eigenvalues not computed with LOBPCG
        
        return result, True, 'lobpcg'
        
    except Exception as e:
        print(f"[ERROR] LOBPCG failed: {e}")
        return None, False, 'failed'


def compute_eigenvalues(L, k_small=6, k_tail=3, max_dense_size=1500,
                        arpack_timeout=300, arpack_size_limit=15000,
                        lobpcg_tol=1e-3, lobpcg_maxiter=200):
    """
    Compute eigenvalues with hybrid strategy.
    
    Strategy:
    ---------
    1. For n ≤ arpack_size_limit: Try ARPACK with timeout
    2. If ARPACK times out or n > limit: Try LOBPCG
    3. If both fail: Return None with failed flag
    
    Parameters:
    -----------
    L : sparse matrix
        Normalized Laplacian
    k_small : int
        Number of small eigenvalues (excluding λ₁≈0)
    k_tail : int
        Number of large eigenvalues
    max_dense_size : int
        Max size for dense fallback
    arpack_timeout : int
        Timeout in seconds for ARPACK
    arpack_size_limit : int
        Max n for attempting ARPACK
    lobpcg_tol : float
        Tolerance for LOBPCG
    lobpcg_maxiter : int
        Max iterations for LOBPCG
    
    Returns:
    --------
    result : dict or None
        Eigenvalue results
    success : bool
        Whether computation succeeded
    method : str
        Which method succeeded ('arpack', 'lobpcg', 'arpack_dense', 'failed')
    """
    n = L.shape[0]
    
    if n < k_small + 1:
        return None, False, 'too_small'
    
    # Strategy 1: Try ARPACK for reasonable-sized graphs
    if n <= arpack_size_limit:
        result, success, method = compute_eigenvalues_arpack(
            L, k_small, k_tail, max_dense_size, arpack_timeout
        )
        
        if success:
            return result, True, method
        
        # If ARPACK timed out or failed, try LOBPCG
        if method == 'timeout' or n > 5000:
            print(f"[FALLBACK] Trying LOBPCG after ARPACK {method} (n={n})")
            result, success, method = compute_eigenvalues_lobpcg(
                L, k_small, lobpcg_tol, lobpcg_maxiter
            )
            return result, success, method
        else:
            return None, False, method
    
    # Strategy 2: Large graphs - use LOBPCG directly
    else:
        print(f"[LOBPCG] Graph large (n={n} > {arpack_size_limit}), using LOBPCG")
        result, success, method = compute_eigenvalues_lobpcg(
            L, k_small, lobpcg_tol, lobpcg_maxiter
        )
        return result, success, method




# def compute_eigenvalues(L, k_small=6, k_tail=3, max_dense_size=1500):
#     """Compute small and large eigenvalues of Laplacian."""
#     n = L.shape[0]
    
#     if n < k_small + 1:
#         return None, False
    
#     result = {
#         'small': None,
#         'large': None,
#         'all': None,
#         'eigenvectors_fiedler': None,
#         'eigenvectors_max': None
#     }
    
#     try:
#         with warnings.catch_warnings():
#             warnings.filterwarnings('ignore')
            
#             vals_small, vecs_small = eigsh(
#                 L, 
#                 k=min(k_small + 1, n - 1),
#                 which='SM',
#                 sigma=0,
#                 return_eigenvectors=True
#             )
        
#         idx = np.argsort(vals_small)
#         vals_small = vals_small[idx]
#         vecs_small = vecs_small[:, idx]
        
#         result['small'] = vals_small[1:k_small+1]
#         result['eigenvectors_fiedler'] = vecs_small[:, 1]
        
#         vals_large, vecs_large = eigsh(
#             L,
#             k=min(k_tail, n - k_small - 1),
#             which='LM',
#             return_eigenvectors=True
#         )
        
#         idx = np.argsort(vals_large)[::-1]
#         vals_large = vals_large[idx]
#         vecs_large = vecs_large[:, idx]
        
#         result['large'] = vals_large[:k_tail]
#         result['eigenvectors_max'] = vecs_large[:, 0]
        
#         return result, True
        
#     except Exception as e:
#         if n < max_dense_size:
#             try:
#                 L_dense = L.toarray()
#                 eigenvalues_all, eigenvectors_all = np.linalg.eigh(L_dense)
                
#                 idx = np.argsort(eigenvalues_all)
#                 eigenvalues_all = eigenvalues_all[idx]
#                 eigenvectors_all = eigenvectors_all[:, idx]
                
#                 result['small'] = eigenvalues_all[1:k_small+1]
#                 result['large'] = eigenvalues_all[-k_tail:][::-1]
#                 result['all'] = eigenvalues_all
#                 result['eigenvectors_fiedler'] = eigenvectors_all[:, 1]
#                 result['eigenvectors_max'] = eigenvectors_all[:, -1]
                
#                 return result, True
                
#             except Exception as e2:
#                 return None, False
        
#         return None, False


# ============================================================================
# Feature Families
# ============================================================================

@register_spectral_family("lcc_spectral")
def extract_lcc_spectral_features(eigenvalues, prefix, config):
    """Extract spectral features from largest connected component."""
    if eigenvalues is None:
        return {}
    
    features = {}
    epsilon = config.get('epsilon', 1e-10)
    
    small_eigs = eigenvalues['small']
    for i, val in enumerate(small_eigs, start=2):
        features[f'{prefix}_lcc_lambda{i}'] = float(val)
    
    large_eigs = eigenvalues['large']
    features[f'{prefix}_lcc_lambda_max'] = float(large_eigs[0])
    if len(large_eigs) > 1:
        features[f'{prefix}_lcc_lambda_max2'] = float(large_eigs[1])
    if len(large_eigs) > 2:
        features[f'{prefix}_lcc_lambda_max3'] = float(large_eigs[2])
    
    all_computed = np.concatenate([small_eigs, large_eigs])
    
    eig_sum = np.sum(all_computed) + epsilon
    p = all_computed / eig_sum
    H = scipy_entropy(p)
    features[f'{prefix}_lcc_entropy'] = float(H)
    
    features[f'{prefix}_lcc_effective_rank'] = float(np.exp(H))
    features[f'{prefix}_lcc_frob_norm'] = float(np.linalg.norm(all_computed, 2))
    
    fiedler_vec = eigenvalues['eigenvectors_fiedler']
    max_vec = eigenvalues['eigenvectors_max']
    
    if fiedler_vec is not None:
        ipr_fiedler = np.sum(fiedler_vec ** 4)
        features[f'{prefix}_lcc_ipr_fiedler'] = float(ipr_fiedler)
    
    if max_vec is not None:
        ipr_max = np.sum(max_vec ** 4)
        features[f'{prefix}_lcc_ipr_max'] = float(ipr_max)
    
    return features


@register_spectral_family("cc_summary")
def extract_cc_summary_features(components, G_full, prefix, computation_config):
    """Summarize connected component structure."""
    n_total = G_full.number_of_nodes()
    n_components = len(components)
    
    component_sizes = sorted([len(c) for c in components], reverse=True)
    
    features = {}
    
    features[f'{prefix}_cc_summary_num_components'] = n_components
    features[f'{prefix}_cc_summary_size1'] = component_sizes[0] if len(component_sizes) >= 1 else 0
    features[f'{prefix}_cc_summary_size2'] = component_sizes[1] if len(component_sizes) >= 2 else 0
    features[f'{prefix}_cc_summary_size3'] = component_sizes[2] if len(component_sizes) >= 3 else 0
    features[f'{prefix}_cc_summary_size4'] = component_sizes[3] if len(component_sizes) >= 4 else 0
    features[f'{prefix}_cc_summary_size5'] = component_sizes[4] if len(component_sizes) >= 5 else 0
    
    if n_total > 0:
        size_probs = [s / n_total for s in component_sizes if s > 0]
        entropy = -sum(p * np.log(p + 1e-12) for p in size_probs)
        features[f'{prefix}_cc_summary_entropy'] = entropy
    else:
        features[f'{prefix}_cc_summary_entropy'] = 0.0
    
    lcc_size = component_sizes[0] if len(component_sizes) > 0 else 0
    features[f'{prefix}_cc_summary_frac_outside_lcc'] = (n_total - lcc_size) / n_total if n_total > 0 else 0.0
    
    mid_components = [s for s in component_sizes if 10 <= s < lcc_size]
    features[f'{prefix}_cc_summary_num_mid_components'] = len(mid_components)
    
    return features


@register_spectral_family("cc2_features")
def extract_cc2_features(components, G, prefix, config):
    """Extract features from 2nd largest component."""
    if len(components) < 2:
        return {}
    
    features = {}
    min_size = config.get('cc2_min_size', 10)
    
    cc2_nodes = components[1]
    cc2_size = len(cc2_nodes)
    
    features[f'{prefix}_cc2_size'] = float(cc2_size)
    
    if cc2_size < min_size:
        # ✅ FIXED: Add validity flag
        features[f'{prefix}_cc2_valid'] = 0
        features[f'{prefix}_cc2_density'] = np.nan
        features[f'{prefix}_cc2_mean_degree'] = np.nan
        features[f'{prefix}_cc2_clustering'] = np.nan
        features[f'{prefix}_cc2_lambda2'] = np.nan
        return features
    
    features[f'{prefix}_cc2_valid'] = 1
    
    G_cc2 = G.subgraph(cc2_nodes).copy()
    
    density = nx.density(G_cc2)
    features[f'{prefix}_cc2_density'] = float(density)
    
    degrees = dict(G_cc2.degree())
    mean_degree = np.mean(list(degrees.values()))
    features[f'{prefix}_cc2_mean_degree'] = float(mean_degree)
    
    clustering = nx.average_clustering(G_cc2, weight='weight')
    features[f'{prefix}_cc2_clustering'] = float(clustering)
    
    # ✅ FIXED: Use config parameters
    try:
        L_cc2, _ = compute_normalized_laplacian(G_cc2, epsilon=config.get('epsilon', 1e-10))
        
        if L_cc2 is not None and L_cc2.shape[0] >= 3:
            vals, _ = eigsh(L_cc2, k=2, which='SM', sigma=0, return_eigenvectors=False)
            vals = np.sort(vals)
            features[f'{prefix}_cc2_lambda2'] = float(vals[1])
        else:
            features[f'{prefix}_cc2_lambda2'] = np.nan
    except:
        features[f'{prefix}_cc2_lambda2'] = np.nan
    
    return features


@register_spectral_family("global_indicators")
def extract_global_indicators(G, components, prefix, config):
    """Extract global spectral indicators."""
    features = {}
    
    features[f'{prefix}_global_num_zero_eigs'] = len(components)
    
    if len(components) == 1:
        # ✅ FIXED: Use NaN as placeholder, will be filled later
        features[f'{prefix}_global_lambda2_full'] = np.nan
    else:
        features[f'{prefix}_global_lambda2_full'] = 0.0
    
    return features


# ============================================================================
# Delta Features
# ============================================================================

def compute_delta_features(current_features, previous_features, prefix, feature_config):
    """
    Compute temporal delta features.
    
    ✅ FIXED: Only compute deltas for enabled families.
    """
    if previous_features is None:
        return {}
    
    delta_features = {}
    
    # ✅ FIXED: Respect config toggles
    lcc_keys = []
    if feature_config.get('lcc_spectral', True):
        lcc_keys = [
            'lambda2', 'lambda3', 'lambda4', 'lambda5', 'lambda6',
            'entropy', 'effective_rank', 'frob_norm',
            'ipr_fiedler', 'ipr_max'
        ]
    
    cc_keys = []
    if feature_config.get('cc_summary', True):
        cc_keys = [
            'cc_summary_entropy',
            'cc_summary_frac_outside_lcc',
            'cc_summary_num_mid_components'
        ]
    
    cc2_keys = []
    if feature_config.get('cc2_features', True):
        cc2_keys = [
            'cc2_size',
            'cc2_density',
            'cc2_mean_degree',
            'cc2_clustering'
        ]
    
    # Compute deltas for LCC spectral features
    for key in lcc_keys:
        curr_key = f'{prefix}_lcc_{key}'
        
        if curr_key in current_features and curr_key in previous_features:
            curr_val = current_features[curr_key]
            prev_val = previous_features[curr_key]
            
            if not (np.isnan(curr_val) or np.isnan(prev_val)):
                delta_features[f'{prefix}_delta_{key}'] = float(curr_val - prev_val)
    
    # Compute deltas for component summary features
    for key in cc_keys:
        curr_key = f'{prefix}_{key}'
        
        if curr_key in current_features and curr_key in previous_features:
            curr_val = current_features[curr_key]
            prev_val = previous_features[curr_key]
            
            if not (np.isnan(curr_val) or np.isnan(prev_val)):
                delta_features[f'{prefix}_delta_{key}'] = float(curr_val - prev_val)
    
    # Compute deltas for CC2 features (only if valid in both windows)
    for key in cc2_keys:
        curr_key = f'{prefix}_{key}'
        
        # ✅ FIXED: Check validity flag
        curr_valid = current_features.get(f'{prefix}_cc2_valid', 0)
        prev_valid = previous_features.get(f'{prefix}_cc2_valid', 0)
        
        if curr_valid and prev_valid:
            if curr_key in current_features and curr_key in previous_features:
                curr_val = current_features[curr_key]
                prev_val = previous_features[curr_key]
                
                if not (np.isnan(curr_val) or np.isnan(prev_val)):
                    delta_features[f'{prefix}_delta_{key}'] = float(curr_val - prev_val)
    
    return delta_features


# ============================================================================
# Flags
# ============================================================================

def create_flags(G, G_lcc, components, eigenvalues_success, has_prev, prefix):
    """Create diagnostic flags for this window."""
    flags = {}
    
    flags[f'{prefix}_graph_too_small'] = int(G.number_of_nodes() < 10)
    flags[f'{prefix}_eig_failed'] = int(not eigenvalues_success)
    flags[f'{prefix}_g_was_disconnected'] = int(len(components) > 1)
    flags[f'{prefix}_has_prev_window'] = int(has_prev)
    
    return flags