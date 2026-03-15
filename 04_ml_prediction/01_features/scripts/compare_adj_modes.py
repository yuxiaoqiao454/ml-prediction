#!/usr/bin/env python3
"""
Compare Binary vs Weighted Adjacency Embeddings

Analyzes saved embeddings to measure the impact of recency weighting.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
import argparse

def load_embeddings(hashtag, window, adj_mode, feature_mode):
    """Load saved embeddings."""
    emb_dir = Path('04_ml_prediction/01_features/outputs/embeddings')
    emb_file = emb_dir / f"{hashtag}_{window}_{adj_mode}_{feature_mode}.npz"
    
    if not emb_file.exists():
        return None
    
    data = np.load(emb_file, allow_pickle=True)
    return data

def compare_embeddings(emb_binary, emb_weighted):
    """
    Comprehensive comparison between binary and weighted embeddings.
    
    Returns dict of metrics.
    """
    z_bin = emb_binary['embeddings']
    z_wgt = emb_weighted['embeddings']
    n_inf = emb_binary['n_inf']
    
    assert z_bin.shape == z_wgt.shape, "Embedding shapes must match!"
    
    results = {}
    
    # C1: Cosine similarity between corresponding nodes
    cosines = []
    for i in range(len(z_bin)):
        cos = np.dot(z_bin[i], z_wgt[i]) / (np.linalg.norm(z_bin[i]) * np.linalg.norm(z_wgt[i]) + 1e-10)
        cosines.append(cos)
    cosines = np.array(cosines)
    
    results['cosine_median'] = np.median(cosines)
    results['cosine_p10'] = np.percentile(cosines, 10)
    results['cosine_p90'] = np.percentile(cosines, 90)
    
    # C1: Distance matrix correlation (sample 500 nodes or all if fewer)
    n_sample = min(500, len(z_bin))
    idx_sample = np.random.choice(len(z_bin), n_sample, replace=False)
    
    dist_bin = pdist(z_bin[idx_sample], metric='cosine')
    dist_wgt = pdist(z_wgt[idx_sample], metric='cosine')
    
    results['dist_matrix_corr'] = pearsonr(dist_bin, dist_wgt)[0]
    
    # C2: Embedding geometry (influencers vs audience)
    for node_type, idx_range in [('inf', slice(0, n_inf)), ('aud', slice(n_inf, None))]:
        z_bin_subset = z_bin[idx_range]
        z_wgt_subset = z_wgt[idx_range]
        
        # Norms
        norms_bin = np.linalg.norm(z_bin_subset, axis=1)
        norms_wgt = np.linalg.norm(z_wgt_subset, axis=1)
        
        results[f'{node_type}_norm_median_bin'] = np.median(norms_bin)
        results[f'{node_type}_norm_median_wgt'] = np.median(norms_wgt)
        results[f'{node_type}_norm_p90_bin'] = np.percentile(norms_bin, 90)
        results[f'{node_type}_norm_p90_wgt'] = np.percentile(norms_wgt, 90)
        
        # PCA explained variance
        pca_bin = PCA(n_components=min(5, z_bin_subset.shape[1]))
        pca_wgt = PCA(n_components=min(5, z_wgt_subset.shape[1]))
        
        pca_bin.fit(z_bin_subset)
        pca_wgt.fit(z_wgt_subset)
        
        results[f'{node_type}_pca_top5_bin'] = pca_bin.explained_variance_[:5].tolist()
        results[f'{node_type}_pca_top5_wgt'] = pca_wgt.explained_variance_[:5].tolist()
        
        # Effective rank
        s_bin = np.linalg.svd(z_bin_subset, compute_uv=False)
        s_wgt = np.linalg.svd(z_wgt_subset, compute_uv=False)
        
        eff_rank_bin = np.exp(-np.sum(s_bin * np.log(s_bin + 1e-10)))
        eff_rank_wgt = np.exp(-np.sum(s_wgt * np.log(s_wgt + 1e-10)))
        
        results[f'{node_type}_eff_rank_bin'] = eff_rank_bin
        results[f'{node_type}_eff_rank_wgt'] = eff_rank_wgt
    
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hashtag', required=True)
    ap.add_argument('--window', required=True)
    ap.add_argument('--feature-mode', default='type+time')
    args = ap.parse_args()
    
    # Load embeddings
    print(f"Loading embeddings for {args.hashtag}/{args.window}...")
    emb_binary = load_embeddings(args.hashtag, args.window, 'binary', args.feature_mode)
    emb_weighted = load_embeddings(args.hashtag, args.window, 'recency_weighted', args.feature_mode)
    
    if emb_binary is None or emb_weighted is None:
        print("Error: Could not load embeddings for both modes!")
        return
    
    # Compare
    print("\nComparing embeddings...")
    results = compare_embeddings(emb_binary, emb_weighted)
    
    # Print results
    print(f"\n{'='*70}")
    print(f"EMBEDDING COMPARISON: {args.hashtag}/{args.window}")
    print(f"{'='*70}")
    
    print(f"\nC1) Alignment:")
    print(f"  Cosine similarity:      median={results['cosine_median']:.4f}, p10={results['cosine_p10']:.4f}, p90={results['cosine_p90']:.4f}")
    print(f"  Distance matrix corr:   {results['dist_matrix_corr']:.4f}")
    
    print(f"\nC2) Geometry (Influencers):")
    print(f"  Norm median:   binary={results['inf_norm_median_bin']:.3f}, weighted={results['inf_norm_median_wgt']:.3f}")
    print(f"  Norm p90:      binary={results['inf_norm_p90_bin']:.3f}, weighted={results['inf_norm_p90_wgt']:.3f}")
    print(f"  Effective rank: binary={results['inf_eff_rank_bin']:.1f}, weighted={results['inf_eff_rank_wgt']:.1f}")
    
    print(f"\nC2) Geometry (Audience):")
    print(f"  Norm median:   binary={results['aud_norm_median_bin']:.3f}, weighted={results['aud_norm_median_wgt']:.3f}")
    print(f"  Norm p90:      binary={results['aud_norm_p90_bin']:.3f}, weighted={results['aud_norm_p90_wgt']:.3f}")
    print(f"  Effective rank: binary={results['aud_eff_rank_bin']:.1f}, weighted={results['aud_eff_rank_wgt']:.1f}")
    
    print(f"\n{'='*70}")
    
    # Save results
    results_df = pd.DataFrame([results])
    results_df['hashtag'] = args.hashtag
    results_df['window'] = args.window
    results_df['feature_mode'] = args.feature_mode
    
    output_file = Path('04_ml_prediction/01_features/outputs/adj_mode_comparison.csv')
    
    if output_file.exists():
        existing = pd.read_csv(output_file)
        results_df = pd.concat([existing, results_df], ignore_index=True)
    
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved to {output_file}")

if __name__ == '__main__':
    main()