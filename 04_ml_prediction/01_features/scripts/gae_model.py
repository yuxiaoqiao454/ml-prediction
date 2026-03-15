#!/usr/bin/env python3
"""
Graph Autoencoder (GAE) Model - PyTorch Implementation

Adapted from Kipf & Welling (2016): "Variational Graph Auto-Encoders"
Original TF1 implementation: https://github.com/tkipf/gae

This is a clean PyTorch re-implementation for our feature extraction pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional

# DEBUG PRINTS
def print_weight_diagnostics(edge_weights_map: dict, train_edges: np.ndarray, 
                             adj_binary: sp.spmatrix, adj_weighted: sp.spmatrix,
                             n_inf: int, verbose: bool = False):
    """
    Print diagnostic statistics for weighted adjacency.
    
    Parameters:
    -----------
    edge_weights_map : dict
        {(u,v): weight} mapping
    train_edges : np.ndarray
        Training edges
    adj_binary : scipy.sparse
        Binary adjacency (for comparison)
    adj_weighted : scipy.sparse
        Weighted adjacency
    n_inf : int
        Number of influencers
    verbose : bool
    """
    if not verbose:
        return
    
    print(f"\n  {'='*60}")
    print(f"  WEIGHTED ADJACENCY DIAGNOSTICS")
    print(f"  {'='*60}")
    
    # A1: Weight distributions
    weights = np.array([edge_weights_map.get((u,v), 1.0) for u,v in train_edges])
    
    print(f"\n  A1) Weight Distribution:")
    print(f"    n_edges: {len(weights)}")
    print(f"    min:     {weights.min():.4f}")
    print(f"    p10:     {np.percentile(weights, 10):.4f}")
    print(f"    median:  {np.median(weights):.4f}")
    print(f"    p90:     {np.percentile(weights, 90):.4f}")
    print(f"    p99:     {np.percentile(weights, 99):.4f}")
    print(f"    max:     {weights.max():.4f}")
    print(f"    mean:    {weights.mean():.4f}")
    print(f"    std:     {weights.std():.4f}")
    print(f"    CV:      {weights.std()/weights.mean():.4f}")
    
    pct_at_one = (np.abs(weights - 1.0) < 0.01).mean() * 100
    print(f"    % at 1.0: {pct_at_one:.1f}%")
    
    # A3: Weight concentration
    sorted_weights = np.sort(weights)[::-1]
    top1pct_idx = max(1, int(len(weights) * 0.01))
    top1pct_mass = sorted_weights[:top1pct_idx].sum() / sorted_weights.sum()
    
    print(f"\n  A3) Weight Concentration:")
    print(f"    Top 1% mass fraction: {top1pct_mass:.3f}")
    print(f"    p99/median ratio:     {np.percentile(weights, 99)/np.median(weights):.2f}")
    print(f"    max/median ratio:     {weights.max()/np.median(weights):.2f}")
    
    # B1: Degree statistics
    deg_binary = np.array(adj_binary.sum(axis=1)).flatten()
    deg_weighted = np.array(adj_weighted.sum(axis=1)).flatten()
    
    deg_binary_inf = deg_binary[:n_inf]
    deg_weighted_inf = deg_weighted[:n_inf]
    deg_binary_aud = deg_binary[n_inf:]
    deg_weighted_aud = deg_weighted[n_inf:]
    
    print(f"\n  B1) Degree Statistics:")
    print(f"    Influencers (binary):  median={np.median(deg_binary_inf):.1f}, p90={np.percentile(deg_binary_inf, 90):.1f}, max={deg_binary_inf.max():.1f}")
    print(f"    Influencers (weighted): median={np.median(deg_weighted_inf):.1f}, p90={np.percentile(deg_weighted_inf, 90):.1f}, max={deg_weighted_inf.max():.1f}")
    print(f"    Audience (binary):     median={np.median(deg_binary_aud):.1f}, p90={np.percentile(deg_binary_aud, 90):.1f}, max={deg_binary_aud.max():.1f}")
    print(f"    Audience (weighted):    median={np.median(deg_weighted_aud):.1f}, p90={np.percentile(deg_weighted_aud, 90):.1f}, max={deg_weighted_aud.max():.1f}")
    
    # Correlations
    from scipy.stats import pearsonr
    corr_inf = pearsonr(deg_binary_inf, deg_weighted_inf)[0]
    corr_aud = pearsonr(deg_binary_aud, deg_weighted_aud)[0]
    
    print(f"    Degree correlation (influencers): {corr_inf:.4f}")
    print(f"    Degree correlation (audience):    {corr_aud:.4f}")
    
    print(f"  {'='*60}\n")

# ============================================================================
# GRAPH CONVOLUTION LAYER (PyTorch)
# ============================================================================

class GraphConvolution(nn.Module):
    """
    Graph Convolution Layer: H' = σ(AHW)
    
    Where:
    - A: Normalized adjacency matrix (sparse)
    - H: Node features (dense)
    - W: Learnable weight matrix
    - σ: Activation function
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 dropout: float = 0.0, activation: bool = True):
        super(GraphConvolution, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.use_activation = activation
        
        # Learnable weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        # Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x: torch.Tensor, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor (N x in_features)
            Node features
        adj : torch.sparse.FloatTensor (N x N)
            Normalized adjacency matrix
        
        Returns:
        --------
        output : torch.Tensor (N x out_features)
        """
        # Dropout on input features
        x = F.dropout(x, self.dropout, training=self.training)
        
        # H' = HW
        support = torch.mm(x, self.weight)
        
        # H'' = AH'
        output = torch.spmm(adj, support)
        
        # Activation
        if self.use_activation:
            output = F.relu(output)
        
        return output


# ============================================================================
# INNER PRODUCT DECODER (PyTorch)
# ============================================================================

class InnerProductDecoder(nn.Module):
    """
    Decoder for link prediction: A_hat = σ(ZZ^T)
    
    Where:
    - Z: Node embeddings (N x d)
    - σ: Sigmoid activation
    """
    
    def __init__(self, dropout: float = 0.0):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct adjacency matrix.
        
        Parameters:
        -----------
        z : torch.Tensor (N x d)
            Node embeddings
        
        Returns:
        --------
        adj_reconstructed : torch.Tensor (N x N)
            Reconstructed adjacency (logits, not probabilities)
        """
        z = F.dropout(z, self.dropout, training=self.training)
        
        # A_hat = ZZ^T
        adj = torch.mm(z, z.t())
        
        return adj


# ============================================================================
# GAE MODEL (2-layer GCN encoder + inner product decoder)
# ============================================================================

class GAE(nn.Module):
    """
    Graph Autoencoder (non-variational version).
    
    Architecture:
    - GCN Layer 1: input_dim → hidden_dim (ReLU)
    - GCN Layer 2: hidden_dim → embedding_dim (linear)
    - Decoder: Inner product (ZZ^T)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 32, 
                 embedding_dim: int = 16, dropout: float = 0.0):
        super(GAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Encoder: 2-layer GCN
        self.gc1 = GraphConvolution(input_dim, hidden_dim, 
                                     dropout=dropout, activation=True)
        self.gc2 = GraphConvolution(hidden_dim, embedding_dim, 
                                     dropout=dropout, activation=False)
        
        # Decoder: Inner product
        self.decoder = InnerProductDecoder(dropout=dropout)
    
    def encode(self, x: torch.Tensor, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        """
        Encode nodes to embeddings.
        
        Parameters:
        -----------
        x : torch.Tensor (N x input_dim)
            Node features
        adj : torch.sparse.FloatTensor (N x N)
            Normalized adjacency matrix
        
        Returns:
        --------
        z : torch.Tensor (N x embedding_dim)
            Node embeddings
        """
        h = self.gc1(x, adj)
        z = self.gc2(h, adj)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode embeddings to adjacency matrix."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor, adj: torch.sparse.FloatTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode + decode.
        
        Returns:
        --------
        z : Node embeddings
        adj_recon : Reconstructed adjacency
        """
        z = self.encode(x, adj)
        adj_recon = self.decode(z)
        return z, adj_recon


# ============================================================================
# GAE WITH FLEXIBLE INPUT FEATURES
# ============================================================================

# class GAEWithFeatures(nn.Module):
#     """
#     GAE wrapper that handles different input feature modes.
    
#     Modes:
#     ------
#     - type_only: Use only type flags (N×1 or N×2)
#     - id: Use learned node ID embeddings (N×id_dim)
#     - id+type: Concatenate ID embeddings + type flags (N×(id_dim+type_dim))
#     """
    
#     def __init__(self, feature_mode: str, n_nodes: int, 
#                  id_dim: int = 64, type_dim: int = 1,
#                  hidden_dim: int = 32, embedding_dim: int = 16, 
#                  dropout: float = 0.0):
#         super(GAEWithFeatures, self).__init__()
        
#         self.feature_mode = feature_mode
#         self.n_nodes = n_nodes
#         self.id_dim = id_dim
#         self.type_dim = type_dim
        
#         # Create ID embedding table if needed
#         if feature_mode in ['id', 'id+type']:
#             self.id_embed = nn.Embedding(n_nodes, id_dim)
#             # Xavier initialization
#             nn.init.xavier_uniform_(self.id_embed.weight)
#         else:
#             self.id_embed = None
        
#         # Determine input dimension for GCN
#         if feature_mode == 'type_only':
#             input_dim = type_dim
#         elif feature_mode == 'id':
#             input_dim = id_dim
#         else:  # id+type
#             input_dim = id_dim + type_dim
        
#         # Create the underlying GAE model
#         self.gae = GAE(input_dim, hidden_dim, embedding_dim, dropout)
    
#     def build_features(self, type_features: torch.Tensor) -> torch.Tensor:
#         """
#         Build node feature matrix based on feature_mode.
        
#         Parameters:
#         -----------
#         type_features : torch.Tensor (N, type_dim)
#             Type flags (0/1 for flag encoding, one-hot for onehot encoding)
        
#         Returns:
#         --------
#         X : torch.Tensor (N, input_dim)
#             Node feature matrix
#         """
#         n_nodes = type_features.shape[0]
        
#         if self.feature_mode == 'type_only':
#             return type_features
        
#         # Create node indices
#         node_idx = torch.arange(n_nodes, device=type_features.device)
        
#         # Get ID embeddings
#         id_feats = self.id_embed(node_idx)  # (N, id_dim)
        
#         if self.feature_mode == 'id':
#             return id_feats
#         else:  # id+type
#             return torch.cat([id_feats, type_features], dim=1)
    
#     def encode(self, type_features: torch.Tensor, 
#                adj: torch.sparse.FloatTensor) -> torch.Tensor:
#         """
#         Encode nodes to embeddings.
        
#         Parameters:
#         -----------
#         type_features : torch.Tensor (N, type_dim)
#             Type flags
#         adj : torch.sparse.FloatTensor (N, N)
#             Normalized adjacency matrix
        
#         Returns:
#         --------
#         z : torch.Tensor (N, embedding_dim)
#             Node embeddings
#         """
#         X = self.build_features(type_features)
#         return self.gae.encode(X, adj)
    
#     def decode(self, z: torch.Tensor) -> torch.Tensor:
#         """Decode embeddings to adjacency matrix."""
#         return self.gae.decode(z)
    
#     def forward(self, type_features: torch.Tensor, 
#                 adj: torch.sparse.FloatTensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Full forward pass: encode + decode.
        
#         Returns:
#         --------
#         z : Node embeddings
#         adj_recon : Reconstructed adjacency
#         """
#         z = self.encode(type_features, adj)
#         adj_recon = self.decode(z)
#         return z, adj_recon
class GAEWithFeatures(nn.Module):
    """GAE wrapper handling different input feature modes."""
    
    def __init__(self, feature_mode, n_nodes, id_dim, type_dim, temporal_dim,
                 hidden_dim, embedding_dim, dropout):
        super().__init__()
        self.feature_mode = feature_mode
        self.n_nodes = n_nodes
        
        # Parse feature_mode to determine which components to use
        self.use_type = 'type' in feature_mode
        self.use_id = 'id' in feature_mode
        self.use_time = 'time' in feature_mode
        
        # Create ID embedding if needed
        self.id_embed = None
        if self.use_id:
            self.id_embed = nn.Embedding(n_nodes, id_dim)
            nn.init.xavier_uniform_(self.id_embed.weight)
        
        # Determine total input dimension
        input_dim = 0
        if self.use_type:
            input_dim += type_dim
        if self.use_id:
            input_dim += id_dim
        if self.use_time:
            input_dim += temporal_dim
        
        if input_dim == 0:
            raise ValueError(f"Invalid feature_mode '{feature_mode}': must include at least one of type/id/time")
        
        self.gae = GAE(input_dim, hidden_dim, embedding_dim, dropout)
    
    def build_features(self, type_features, temporal_features):
        """
        Build input features by concatenating components based on mode.
        
        Parameters:
        -----------
        type_features : torch.Tensor (N x type_dim) or None
        temporal_features : torch.Tensor (N x temporal_dim) or None
        
        Returns:
        --------
        X : torch.Tensor (N x input_dim)
        """
        features = []
        
        # Add type features
        if self.use_type:
            if type_features is None:
                raise ValueError("feature_mode requires 'type' but type_features is None")
            features.append(type_features)
        
        # Add ID embeddings
        if self.use_id:
            device = type_features.device if type_features is not None else temporal_features.device
            node_idx = torch.arange(self.n_nodes, device=device)
            id_feats = self.id_embed(node_idx)
            features.append(id_feats)
        
        # Add temporal features
        if self.use_time:
            if temporal_features is None:
                raise ValueError("feature_mode requires 'time' but temporal_features is None")
            features.append(temporal_features)
        
        return torch.cat(features, dim=1)
    
    def encode(self, type_features, temporal_features, adj):
        """Encode with feature building."""
        X = self.build_features(type_features, temporal_features)
        return self.gae.encode(X, adj)

# ============================================================================
# PREPROCESSING UTILITIES
# ============================================================================

def normalize_adjacency(adj: sp.spmatrix, self_loops: bool = True) -> sp.spmatrix:
    """
    Normalize adjacency matrix: A_norm = D^(-1/2) A D^(-1/2)
    
    Parameters:
    -----------
    adj : scipy.sparse matrix (N x N)
        Adjacency matrix
    self_loops : bool
        Whether to add self-loops (I) before normalizing
    
    Returns:
    --------
    adj_norm : scipy.sparse matrix (N x N)
        Normalized adjacency
    """
    if self_loops:
        # Add self-loops: A_tilde = A + I
        adj = adj + sp.eye(adj.shape[0])
    
    # Compute degree matrix
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    
    # D^(-1/2)
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    # A_norm = D^(-1/2) A D^(-1/2)
    adj_norm = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    
    return adj_norm.tocoo()


def sparse_scipy_to_torch(sparse_mx: sp.spmatrix) -> torch.sparse.FloatTensor:
    """
    Convert scipy sparse matrix to PyTorch sparse tensor.
    
    Parameters:
    -----------
    sparse_mx : scipy.sparse matrix
    
    Returns:
    --------
    torch_sparse : torch.sparse.FloatTensor
    """
    sparse_mx = sparse_mx.tocoo()
    
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    
    return torch.sparse.FloatTensor(indices, values, shape)


def dense_to_torch(dense_mx: np.ndarray) -> torch.FloatTensor:
    """Convert numpy array to PyTorch tensor."""
    return torch.FloatTensor(dense_mx)


# ============================================================================
# LOSS FUNCTION
# ============================================================================

def gae_loss(adj_recon: torch.Tensor, adj_label: torch.Tensor, 
             pos_weight: float = 1.0) -> torch.Tensor:
    """
    Reconstruction loss with weighted binary cross-entropy.
    
    Parameters:
    -----------
    adj_recon : torch.Tensor (N*N,)
        Reconstructed adjacency (logits, flattened)
    adj_label : torch.Tensor (N*N,)
        True adjacency (binary, flattened)
    pos_weight : float
        Weight for positive examples (to handle class imbalance)
    
    Returns:
    --------
    loss : torch.Tensor (scalar)
    """
    # Weighted binary cross-entropy
    loss = F.binary_cross_entropy_with_logits(
        adj_recon, 
        adj_label,
        pos_weight=torch.tensor(pos_weight)
    )
    
    return loss


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_gae(adj: sp.spmatrix, n_inf: int, n_aud: int,
              type_features: Optional[np.ndarray], 
              temporal_features: Optional[np.ndarray],
              edge_weights_map: Optional[dict] = None,  # ← ADD THIS
              feature_mode: str = 'type+time',
              id_dim: int = 64,
              hidden_dim: int = 32, embedding_dim: int = 16,
              max_epochs: int = 500, learning_rate: float = 0.01,
              dropout: float = 0.0, 
              train_ratio: float = 0.85, val_ratio: float = 0.05, test_ratio: float = 0.10,
              patience: int = 20, check_every: int = 5,
              neg_sample_ratio: float = 1.0,
              min_train_edges: int = 50,
              min_val_edges: int = 10,
              min_test_edges: int = 10,
              use_weighted_bce: bool = False,  # ← ADD THIS
              pos_weight_scale: float = 1.0,   # ← ADD THIS
              random_seed: int = 42,
              verbose: bool = False) -> tuple:
    """
    Train a Graph Autoencoder with flexible input features.
    
    Parameters:
    -----------
    adj : scipy.sparse matrix (N x N)
    n_inf : int
        Number of influencer nodes
    n_aud : int
        Number of audience nodes
    type_features : np.ndarray (N x type_dim) or None
        Type features (if feature_mode includes 'type')
    temporal_features : np.ndarray (N x temporal_dim) or None
        Temporal features (if feature_mode includes 'time')
    feature_mode : str
        Feature combination: "type", "id", "time", "type+time", "id+time", "type+id", "type+id+time"
    id_dim : int
        Dimension of ID embeddings (if feature_mode includes 'id')
    ... (rest same as before)
    
    Returns:
    --------
    embeddings : np.ndarray (N x embedding_dim) or None
    training_curve : list of dict
    metrics : dict
    """
    n_nodes = adj.shape[0]
    
    # Check for degenerate cases
    if n_nodes < 3 or adj.nnz == 0:
        if verbose:
            print(f"[SKIP] Graph too small (n={n_nodes}, edges={adj.nnz})")
        return None, [], {}
    
    try:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Determine feature dimensions
        type_dim = type_features.shape[1] if type_features is not None else 0
        temporal_dim = temporal_features.shape[1] if temporal_features is not None else 0
        
        if verbose:
            print(f"  Feature mode: {feature_mode}")
            if 'type' in feature_mode:
                print(f"  Type features: {type_dim}D")
            if 'id' in feature_mode:
                print(f"  ID embeddings: {id_dim}D")
            if 'time' in feature_mode:
                print(f"  Temporal features: {temporal_dim}D")
        
        # ===== STEP 1: Split edges into train/val/test =====
        adj_triu = sp.triu(adj, k=1)
        edges = np.array(adj_triu.nonzero()).T
        n_edges = len(edges)
        
        if n_edges < 10:
            if verbose:
                print(f"[SKIP] Too few edges ({n_edges})")
            return None, [], {}
        
        # Shuffle edges
        perm = np.random.permutation(n_edges)
        edges = edges[perm]
        
        # Split
        n_val = int(n_edges * val_ratio)
        n_test = int(n_edges * test_ratio)
        n_train = n_edges - n_val - n_test

        train_edges = edges[:n_train]
        val_edges = edges[n_train:n_train+n_val]
        test_edges = edges[n_train+n_val:]

        if verbose:
            print(f"  Edges: {n_train} train / {n_val} val / {n_test} test")

        # Check minimum edges per split
        min_train_edges = 50   # Need enough to train
        min_val_edges = 10    # Need enough to evaluate
        min_test_edges = 10    # Need enough to test

        # if n_train < min_train_edges or n_val < min_val_edges or n_test < min_test_edges:
        #     if verbose:
        #         print(f"  [SKIP] Too few edges in splits (need train≥{min_train_edges}, val≥{min_val_edges}, test≥{min_test_edges})")
        #     return None, [], {}
        # Check minimum edges
        if n_train < min_train_edges:
            if verbose:
                print(f"  [skip] Insufficient train edges: {n_train} < {min_train_edges}")
            return None, [], {'skip_reason': 'insufficient_train_edges'}  # ← ADD skip_reason

        if n_val < min_val_edges:
            if verbose:
                print(f"  [skip] Insufficient val edges: {n_val} < {min_val_edges}")
            return None, [], {'skip_reason': 'insufficient_val_edges'}  # ← ADD skip_reason

        if n_test < min_test_edges:
            if verbose:
                print(f"  [skip] Insufficient test edges: {n_test} < {min_test_edges}")
            return None, [], {'skip_reason': 'insufficient_test_edges'}  # ← ADD skip_reason
                
        
        # ===== STEP 2: Sample negative edges =====
        # def sample_negative_edges(pos_edges, n_neg, n_inf, n_nodes):
        #     pos_set = set(map(tuple, pos_edges))
        #     neg_set = set()  # ← Track what we've already sampled
        #     neg_edges = []
            
        #     while len(neg_edges) < n_neg:
        #         # Sample influencer from [0, n_inf)
        #         i = np.random.randint(0, n_inf)
        #         # Sample audience from [n_inf, n_nodes)
        #         j = np.random.randint(n_inf, n_nodes)
                
        #         if (i, j) not in pos_set and (j, i) not in pos_set and (i, j) not in neg_set:
        #             neg_set.add((i, j))
        #             neg_edges.append([i, j])
            
        #     return np.array(neg_edges)

        def sample_negative_edges(edges, n_neg, n_inf, n_nodes):
            """
            Sample negative edges that don't exist in the graph.
            Uses rejection sampling with timeout fallback.
            """
            neg_edges = []
            existing = set(zip(edges[:, 0], edges[:, 1]))
            
            # Calculate density
            n_aud = n_nodes - n_inf
            max_possible = n_inf * n_aud
            density = len(existing) / max_possible if max_possible > 0 else 0
            
            # If graph is very dense (>80%), use alternative method
            if density > 0.8:
                # Generate all possible edges and sample from non-existing ones
                all_possible = set((i, j) for i in range(n_inf) for j in range(n_inf, n_nodes))
                non_existing = list(all_possible - existing)
                
                if len(non_existing) < n_neg:
                    # Not enough negatives available - return what we have
                    return np.array(non_existing) if non_existing else np.zeros((0, 2), dtype=int)
                
                # Random sample from non-existing edges
                sampled_idx = np.random.choice(len(non_existing), size=n_neg, replace=False)
                return np.array([non_existing[i] for i in sampled_idx])
            
            # Standard rejection sampling with MAX_ATTEMPTS limit
            MAX_ATTEMPTS = n_neg * 100  # Reasonable limit
            attempts = 0
            
            while len(neg_edges) < n_neg and attempts < MAX_ATTEMPTS:
                i = np.random.randint(0, n_inf)
                j = np.random.randint(n_inf, n_nodes)
                
                if (i, j) not in existing:
                    neg_edges.append([i, j])
                
                attempts += 1
            
            if len(neg_edges) < n_neg:
                # Couldn't sample enough negatives - graph too dense
                # Return what we got (train_gae will check min_edges threshold)
                pass
            
            return np.array(neg_edges) if neg_edges else np.zeros((0, 2), dtype=int)
        
        
        n_neg_train = int(n_train * neg_sample_ratio)
        n_neg_val = int(n_val * neg_sample_ratio)
        n_neg_test = int(n_test * neg_sample_ratio)
        
        train_neg = sample_negative_edges(edges, n_neg_train, n_inf, n_nodes)
        val_neg = sample_negative_edges(edges, n_neg_val, n_inf, n_nodes)
        test_neg = sample_negative_edges(edges, n_neg_test, n_inf, n_nodes)

        # # After sampling all negatives, add this:
        # if verbose:
        #     print(f"\n  === Negative Sampling Diagnostics ===")
            
        #     # Check false negatives
        #     all_pos_set = set(map(tuple, edges))  # Full edge list
            
        #     train_fn_rate = sum((i,j) in all_pos_set or (j,i) in all_pos_set 
        #                         for i,j in train_neg) / len(train_neg)
        #     val_fn_rate = sum((i,j) in all_pos_set or (j,i) in all_pos_set 
        #                     for i,j in val_neg) / len(val_neg)
        #     test_fn_rate = sum((i,j) in all_pos_set or (j,i) in all_pos_set 
        #                     for i,j in test_neg) / len(test_neg)
            
        #     print(f"  Train neg false-negative rate: {train_fn_rate:.4f} (should be 0.0)")
        #     print(f"  Val neg false-negative rate: {val_fn_rate:.4f} (should be 0.0)")
        #     print(f"  Test neg false-negative rate: {test_fn_rate:.4f} (should be 0.0)")
            
        #     # Check duplicates (already prevented by code above, but verify)
        #     train_dup_rate = 1 - len(set(map(tuple, train_neg))) / len(train_neg)
        #     val_dup_rate = 1 - len(set(map(tuple, val_neg))) / len(val_neg)
        #     test_dup_rate = 1 - len(set(map(tuple, test_neg))) / len(test_neg)
            
        #     print(f"  Train neg duplicate rate: {train_dup_rate:.4f} (should be 0.0)")
        #     print(f"  Val neg duplicate rate: {val_dup_rate:.4f} (should be 0.0)")
        #     print(f"  Test neg duplicate rate: {test_dup_rate:.4f} (should be 0.0)")
        
        # ===== STEP 3: Build training adjacency =====
        # train_adj = sp.csr_matrix(
        #     (np.ones(n_train), (train_edges[:, 0], train_edges[:, 1])),
        #     shape=(n_nodes, n_nodes)
        # )
        # train_adj = train_adj + train_adj.T
        
        # Build BOTH binary and weighted versions for diagnostics
        train_adj_binary = sp.csr_matrix(
            (np.ones(n_train), (train_edges[:, 0], train_edges[:, 1])),
            shape=(n_nodes, n_nodes)
        )
        train_adj_binary = train_adj_binary + train_adj_binary.T

        # For actual training, use weighted if available
        if edge_weights_map:
            # Use weighted adjacency
            weights = np.array([edge_weights_map.get(tuple(e), 1.0) for e in train_edges])

            # ADD DEBUG
            if verbose:
                print(f"\n  [DEBUG] Train edge weight stats:")
                print(f"    Lookups succeeded: {sum(tuple(e) in edge_weights_map for e in train_edges)}/{len(train_edges)}")
                print(f"    Sample train edges: {train_edges[:3].tolist()}")
                if len(train_edges) > 0:
                    sample_edge = tuple(train_edges[0])
                    print(f"    Sample lookup: edge {sample_edge} → weight {edge_weights_map.get(sample_edge, 'NOT FOUND')}")
                print(f"    Weight range: [{weights.min():.4f}, {weights.max():.4f}]")


            train_adj = sp.csr_matrix(
                (weights, (train_edges[:, 0], train_edges[:, 1])),
                shape=(n_nodes, n_nodes)
            )
            train_adj = train_adj + train_adj.T  # Symmetrize
            
            # Print diagnostics
            print_weight_diagnostics(edge_weights_map, train_edges, 
                                    train_adj_binary, train_adj, n_inf, verbose)
        else:
            # Use binary adjacency
            train_adj = train_adj_binary
        
        # Normalize for GCN
        adj_norm = normalize_adjacency(train_adj, self_loops=True)
        adj_norm_torch = sparse_scipy_to_torch(adj_norm).float()
        #type_features_torch = torch.FloatTensor(type_features)

        type_features_torch = torch.FloatTensor(type_features) if type_features is not None else None
        temporal_features_torch = torch.FloatTensor(temporal_features) if temporal_features is not None else None
        
        # ===== STEP 4: Initialize model =====
        model = GAEWithFeatures(
            feature_mode=feature_mode,
            n_nodes=n_nodes,
            id_dim=id_dim,
            type_dim=type_dim,
            temporal_dim=temporal_dim,  # Make sure this is here
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Convert edges to torch
        train_pos = torch.LongTensor(train_edges)
        train_neg = torch.LongTensor(train_neg)
        val_pos = torch.LongTensor(val_edges)
        val_neg = torch.LongTensor(val_neg)
        test_pos = torch.LongTensor(test_edges)
        test_neg = torch.LongTensor(test_neg)
        
        # Debug prints
        if verbose:
            with torch.no_grad():
                X_sample = model.build_features(type_features_torch, temporal_features_torch)  # ← FIXED!
                print(f"  Input features shape: {X_sample.shape}")
                print(f"  Input features stats: mean={X_sample.mean().item():.4f}, std={X_sample.std().item():.4f}")
                if model.id_embed is not None:
                    print(f"  ID embeddings trainable: {model.id_embed.weight.requires_grad}")
        
        # ===== STEP 5: Training loop with early stopping =====
        best_val_auc = 0
        best_epoch = 0
        patience_counter = 0
        training_curve = []
        best_embeddings = None

        # Prepare positive edge weights for BCE (if using weighted loss)
        train_pos_weights = None
        if use_weighted_bce and edge_weights_map:
            train_pos_weights = torch.FloatTensor(
                [edge_weights_map.get(tuple(e), 1.0) for e in train_edges]
            ) * pos_weight_scale

        model.train()
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            
            # Encode
            z = model.encode(type_features_torch, temporal_features_torch, adj_norm_torch)
            
            # Edge-restricted reconstruction
            pos_scores = (z[train_pos[:, 0]] * z[train_pos[:, 1]]).sum(dim=1)
            neg_scores = (z[train_neg[:, 0]] * z[train_neg[:, 1]]).sum(dim=1)
            
            # Loss with optional positive weighting
            if use_weighted_bce and train_pos_weights is not None:
                # Weighted BCE: each positive example weighted by edge weight
                pos_loss = (F.binary_cross_entropy_with_logits(
                    pos_scores, 
                    torch.ones(len(pos_scores)), 
                    reduction='none'
                ) * train_pos_weights).mean()
            else:
                # Standard BCE
                pos_loss = F.binary_cross_entropy_with_logits(
                    pos_scores, 
                    torch.ones(len(pos_scores))
                )
            
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_scores, 
                torch.zeros(len(neg_scores))
            )
            loss = pos_loss + neg_loss
            
            loss.backward()
            optimizer.step()
            
            # Validation check
            if (epoch + 1) % check_every == 0:
                model.eval()
                with torch.no_grad():
                    z_eval = model.encode(type_features_torch, temporal_features_torch, adj_norm_torch)  # ← FIXED!
                    
                    val_pos_scores = (z_eval[val_pos[:, 0]] * z_eval[val_pos[:, 1]]).sum(dim=1)
                    val_neg_scores = (z_eval[val_neg[:, 0]] * z_eval[val_neg[:, 1]]).sum(dim=1)
                    
                    from sklearn.metrics import roc_auc_score, average_precision_score
                    
                    all_scores = torch.cat([val_pos_scores, val_neg_scores]).cpu().numpy()
                    all_labels = np.concatenate([np.ones(len(val_pos_scores)), 
                                                 np.zeros(len(val_neg_scores))])
                    
                    val_auc = roc_auc_score(all_labels, all_scores)
                    val_ap = average_precision_score(all_labels, all_scores)
                
                model.train()
                
                training_curve.append({
                    'epoch': epoch + 1,
                    'train_loss': loss.item(),
                    'val_auc': val_auc,
                    'val_ap': val_ap
                })
                
                if verbose and (epoch + 1) % (check_every * 10) == 0:
                    print(f"  Epoch {epoch+1}/{max_epochs} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")
                
                # Early stopping
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_epoch = epoch + 1
                    patience_counter = 0
                    best_embeddings = z_eval.cpu().numpy()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch+1} (best: {best_epoch})")
                    break
        
        # ===== STEP 6: Final test evaluation =====
        if best_embeddings is None:
            best_embeddings = z.detach().cpu().numpy()

        z_test = torch.FloatTensor(best_embeddings)
        test_pos_scores = (z_test[test_pos[:, 0]] * z_test[test_pos[:, 1]]).sum(dim=1)
        test_neg_scores = (z_test[test_neg[:, 0]] * z_test[test_neg[:, 1]]).sum(dim=1)

        all_scores_test = torch.cat([test_pos_scores, test_neg_scores]).cpu().numpy()
        all_labels_test = np.concatenate([np.ones(len(test_pos_scores)), 
                                        np.zeros(len(test_neg_scores))])

        test_auc = roc_auc_score(all_labels_test, all_scores_test)
        test_ap = average_precision_score(all_labels_test, all_scores_test)

        # ===== DEBUG METRICS ===== 
        debug_metrics = {}

        # D1: Spearman correlation between score and weight on positive test edges
        if edge_weights_map and len(test_pos) > 0:
            test_pos_weights = np.array([edge_weights_map.get(tuple(e.tolist()), 1.0) for e in test_pos])
            
            from scipy.stats import spearmanr
            score_weight_corr = spearmanr(test_pos_scores.cpu().numpy(), test_pos_weights)[0]
            debug_metrics['score_weight_corr'] = score_weight_corr
            
            if verbose:
                print(f"\n  [DEBUG D1] Spearman corr(score, weight) on positive test edges: {score_weight_corr:.4f}")
        else:
            debug_metrics['score_weight_corr'] = np.nan

        # D2: AUC for "top 20% weighted edges vs negatives"
        if edge_weights_map and len(test_pos) > 0:
            test_pos_weights = np.array([edge_weights_map.get(tuple(e.tolist()), 1.0) for e in test_pos])
            
            # Get top 20% by weight
            top20_threshold = np.percentile(test_pos_weights, 80)
            top20_mask = test_pos_weights >= top20_threshold
            
            if top20_mask.sum() > 0:
                # Scores for top 20% positive edges
                top20_pos_scores = test_pos_scores[top20_mask].cpu().numpy()
                
                # Combine with all negative scores
                top20_all_scores = np.concatenate([top20_pos_scores, test_neg_scores.cpu().numpy()])
                top20_all_labels = np.concatenate([np.ones(len(top20_pos_scores)), 
                                                np.zeros(len(test_neg_scores))])
                
                top20_auc = roc_auc_score(top20_all_labels, top20_all_scores)
                debug_metrics['top20_auc'] = top20_auc
                debug_metrics['top20_n_edges'] = top20_mask.sum()
                
                if verbose:
                    print(f"  [DEBUG D2] AUC for top 20% weighted edges ({top20_mask.sum()} edges) vs negatives: {top20_auc:.4f}")
            else:
                debug_metrics['top20_auc'] = np.nan
                debug_metrics['top20_n_edges'] = 0
        else:
            debug_metrics['top20_auc'] = np.nan
            debug_metrics['top20_n_edges'] = 0

        metrics = {
            'test_auc': test_auc,
            'test_ap': test_ap,
            'best_epoch': best_epoch,
            'final_train_loss': loss.item(),
            'n_train_edges': n_train,
            'n_val_edges': n_val,
            'n_test_edges': n_test,
            **debug_metrics  # ← Add debug metrics
        }

        if verbose:
            print(f"  ✓ Test AUC: {test_auc:.4f} | Test AP: {test_ap:.4f}")

        return best_embeddings, training_curve, metrics
    
    except Exception as e:
        if verbose:
            print(f"[ERROR] GAE training failed: {e}")
            import traceback
            traceback.print_exc()
        return None, [], {}