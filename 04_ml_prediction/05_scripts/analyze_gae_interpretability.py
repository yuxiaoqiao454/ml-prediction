#!/usr/bin/env python3
"""
analyze_gae_interpretability.py

Fast GAE embedding interpretability analysis:
- PCA on embeddings
- Correlation with degree, weighted degree, first-seen time
- Scatter plots for report

Analyzes 8-10 pre-selected windows for both:
- recency_weighted_type
- recency_weighted_type+time
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

sns.set_style("whitegrid")


def load_window_embeddings(hashtag, window_end, adj_mode, feature_mode, emb_dir):
    """Load saved embeddings for one window."""
    
    emb_file = emb_dir / f"{hashtag}_{window_end}_{adj_mode}_{feature_mode}.npz"
    
    if not emb_file.exists():
        raise FileNotFoundError(f"Embeddings not found: {emb_file}")
    
    data = np.load(emb_file)
    
    embeddings = data['embeddings']  # (N x d)
    n_inf = int(data['n_inf'])
    n_aud = int(data['n_aud'])
    
    return embeddings, n_inf, n_aud


def load_window_graph_data(hashtag, window_end):
    """Load bipartite edges and exposures for one window."""
    
    # Load bipartite edges
    edges_path = Path(f'03_networks/data/windows_{hashtag}/{window_end}/s1_bipartite/edges.csv')
    if not edges_path.exists():
        raise FileNotFoundError(f"Edges not found: {edges_path}")
    
    edges = pd.read_csv(edges_path)
    
    # Load exposures
    exposures_path = Path(f'03_networks/data/parquets/networks_{hashtag}/exposures.parquet')
    if not exposures_path.exists():
        raise FileNotFoundError(f"Exposures not found: {exposures_path}")
    
    exposures = pd.read_parquet(exposures_path)
    
    return edges, exposures


def compute_node_statistics(edges, exposures, n_inf, n_aud, window_end, tau_days=14):
    """
    Compute per-node statistics:
    - degree (binary adjacency)
    - weighted_degree (recency-weighted)
    - first_seen_time (normalized to [0, 1])
    """
    
    # Parse window dates
    window_end_dt = datetime.strptime(window_end, "%Y-%m-%d")
    window_start_dt = window_end_dt - timedelta(days=90)
    
    window_end_epoch = int(window_end_dt.timestamp())
    window_start_epoch = int(window_start_dt.timestamp())
    window_duration = window_end_epoch - window_start_epoch
    
    # Get unique nodes
    influencers = sorted(edges['source'].unique())
    audience = sorted(edges['target'].unique())
    n_total = len(influencers) + len(audience)
    
    assert n_total == n_inf + n_aud, f"Node count mismatch: {n_total} != {n_inf} + {n_aud}"
    
    # Map nodes to indices (must match GAE node ordering)
    node_to_idx = {}
    for i, node in enumerate(influencers):
        node_to_idx[node] = i  # Influencers first
    for i, node in enumerate(audience):
        node_to_idx[node] = n_inf + i  # Audience after
    
    # Initialize arrays
    degree = np.zeros(n_total)
    weighted_degree = np.zeros(n_total)
    first_seen = np.full(n_total, np.nan)
    
    # === Compute degree (binary) ===
    for _, edge in edges.iterrows():
        src = edge['source']
        tgt = edge['target']
        
        if src in node_to_idx and tgt in node_to_idx:
            i = node_to_idx[src]
            j = node_to_idx[tgt]
            degree[i] += 1
            degree[j] += 1
    
    # === Compute weighted degree (recency-weighted) ===
    # Filter exposures to window
    window_exp = exposures[
        (exposures['exposure_epoch'] >= window_start_epoch) &
        (exposures['exposure_epoch'] <= window_end_epoch)
    ].copy()
    
    # Group by edge and compute weight
    tau_sec = tau_days * 86400
    
    edge_weights = {}
    for (a, b), group in window_exp.groupby(['A', 'B']):
        if a not in node_to_idx or b not in node_to_idx:
            continue
        
        # Sum of exponential decay weights
        weights = np.exp(-(window_end_epoch - group['exposure_epoch'].values) / tau_sec)
        total_weight = weights.sum()
        
        edge_weights[(a, b)] = total_weight
    
    # Add to node weighted degrees
    for (a, b), weight in edge_weights.items():
        i = node_to_idx[a]
        j = node_to_idx[b]
        weighted_degree[i] += weight
        weighted_degree[j] += weight
    
    # === Compute first-seen time (normalized) ===
    # Influencers: first post time
    inf_first = window_exp.groupby('A')['post_epoch'].min()
    for node, t in inf_first.items():
        if node in node_to_idx:
            idx = node_to_idx[node]
            first_seen[idx] = (t - window_start_epoch) / window_duration
    
    # Audience: first exposure time
    aud_first = window_exp.groupby('B')['exposure_epoch'].min()
    for node, t in aud_first.items():
        if node in node_to_idx:
            idx = node_to_idx[node]
            first_seen[idx] = (t - window_start_epoch) / window_duration
    
    # Clip to [0, 1]
    first_seen = np.clip(first_seen, 0, 1)
    
    return degree, weighted_degree, first_seen


def analyze_one_window(hashtag, window_end, adj_mode, feature_mode, emb_dir):
    """
    Run PCA + correlation analysis for one window.
    
    Returns dict with correlations and data for plotting.
    """
    
    print(f"  Analyzing {hashtag} / {window_end} / {feature_mode}...")
    
    try:
        # Load embeddings
        embeddings, n_inf, n_aud = load_window_embeddings(
            hashtag, window_end, adj_mode, feature_mode, emb_dir
        )
        
        # Load graph data
        edges, exposures = load_window_graph_data(hashtag, window_end)
        
        # Compute node statistics
        degree, weighted_degree, first_seen = compute_node_statistics(
            edges, exposures, n_inf, n_aud, window_end, tau_days=14
        )
        
        # === PCA on embeddings ===
        pca = PCA(n_components=3)
        pc_scores = pca.fit_transform(embeddings)  # (N x 3)
        
        explained_var = pca.explained_variance_ratio_
        
        # === Compute correlations (Spearman) ===
        correlations = {}
        
        for k in range(3):
            pc = pc_scores[:, k]
            
            # PC vs degree
            correlations[f'PC{k+1}_deg'] = spearmanr(pc, degree)[0]
            
            # PC vs weighted degree
            correlations[f'PC{k+1}_wdeg'] = spearmanr(pc, weighted_degree)[0]
            
            # PC vs first-seen time (only for nodes with valid timestamps)
            valid_mask = ~np.isnan(first_seen)
            if valid_mask.sum() > 10:
                correlations[f'PC{k+1}_time'] = spearmanr(pc[valid_mask], first_seen[valid_mask])[0]
            else:
                correlations[f'PC{k+1}_time'] = np.nan
        
        return {
            'hashtag': hashtag,
            'window_end': window_end,
            'adj_mode': adj_mode,
            'feature_mode': feature_mode,
            'n_nodes': len(embeddings),
            'n_inf': n_inf,
            'n_aud': n_aud,
            'explained_var_pc1': explained_var[0],
            'explained_var_pc2': explained_var[1],
            'explained_var_pc3': explained_var[2],
            **correlations,
            # Save for plotting
            'pc_scores': pc_scores,
            'degree': degree,
            'weighted_degree': weighted_degree,
            'first_seen': first_seen
        }
    
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return None


def make_scatter_plots(results, output_dir):
    """Create scatter plots for top 3 windows (each feature mode)."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Group by feature_mode
    for feature_mode in ['type', 'type+time']:
        mode_results = [r for r in results if r['feature_mode'] == feature_mode]
        
        if not mode_results:
            continue
        
        # Take top 3 by explained variance
        mode_results = sorted(mode_results, key=lambda x: x['explained_var_pc1'], reverse=True)[:3]
        
        for i, result in enumerate(mode_results):
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
            
            hashtag = result['hashtag']
            window = result['window_end']
            pc_scores = result['pc_scores']
            degree = result['degree']
            weighted_degree = result['weighted_degree']
            first_seen = result['first_seen']
            
            # === Plot 1: PC1 vs Weighted Degree ===
            axes[0].scatter(weighted_degree, pc_scores[:, 0], alpha=0.4, s=20, c='steelblue')
            axes[0].set_xlabel('Weighted Degree', fontsize=12)
            axes[0].set_ylabel('PC1 Score', fontsize=12)
            axes[0].set_title(f"PC1 vs Weighted Degree\n{hashtag} ({window})", fontsize=11)
            
            # Add correlation text
            rho = result['PC1_wdeg']
            axes[0].text(0.05, 0.95, f"ρ = {rho:.3f}", 
                        transform=axes[0].transAxes, fontsize=11,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # === Plot 2: PC1 vs Degree (binary) ===
            axes[1].scatter(degree, pc_scores[:, 0], alpha=0.4, s=20, c='coral')
            axes[1].set_xlabel('Degree (binary)', fontsize=12)
            axes[1].set_ylabel('PC1 Score', fontsize=12)
            axes[1].set_title(f"PC1 vs Degree", fontsize=11)
            
            rho = result['PC1_deg']
            axes[1].text(0.05, 0.95, f"ρ = {rho:.3f}", 
                        transform=axes[1].transAxes, fontsize=11,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # === Plot 3: PC2 vs First-Seen Time ===
            valid = ~np.isnan(first_seen)
            axes[2].scatter(first_seen[valid], pc_scores[valid, 1], alpha=0.4, s=20, c='seagreen')
            axes[2].set_xlabel('First-Seen Time (normalized)', fontsize=12)
            axes[2].set_ylabel('PC2 Score', fontsize=12)
            axes[2].set_title(f"PC2 vs First-Seen Time", fontsize=11)
            
            rho = result['PC2_time']
            axes[2].text(0.05, 0.95, f"ρ = {rho:.3f}", 
                        transform=axes[2].transAxes, fontsize=11,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            # Save
            filename = f"gae_interp_{feature_mode}_{hashtag}_{window}.png"
            plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved plot: {filename}")


def make_influencer_only_plots(results, output_dir):
    """
    Create scatter plots for INFLUENCERS ONLY.
    
    Influencers have varied out-degree (# audiences reached),
    while audience nodes are mostly degree-1 (noisy).
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*80}")
    print("Creating influencer-only plots...")
    print(f"{'='*80}\n")
    
    # Group by feature_mode
    for feature_mode in ['type', 'type+time']:
        mode_results = [r for r in results if r['feature_mode'] == feature_mode]
        
        if not mode_results:
            continue
        
        # Take top 3 by explained variance
        mode_results = sorted(mode_results, key=lambda x: x['explained_var_pc1'], reverse=True)[:3]
        
        for result in mode_results:
            hashtag = result['hashtag']
            window = result['window_end']
            n_inf = result['n_inf']
            n_aud = result['n_aud']
            
            # Extract influencer indices (first n_inf nodes)
            inf_indices = slice(0, n_inf)
            
            # Influencer-only data
            inf_pc_scores = result['pc_scores'][inf_indices]
            inf_degree = result['degree'][inf_indices]
            inf_weighted_degree = result['weighted_degree'][inf_indices]
            inf_first_seen = result['first_seen'][inf_indices]
            
            # Skip if too few influencers
            if n_inf < 5:
                print(f"  ✗ Skipping {hashtag}/{window}: only {n_inf} influencers")
                continue
            
            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
            
            # === Plot 1: PC1 vs Weighted Degree ===
            axes[0].scatter(inf_weighted_degree, inf_pc_scores[:, 0], 
                          alpha=0.6, s=40, c='darkblue', edgecolors='black', linewidth=0.5)
            axes[0].set_xlabel('Weighted Degree (# audiences reached)', fontsize=12)
            axes[0].set_ylabel('PC1 Score', fontsize=12)
            axes[0].set_title(f"Influencers Only: PC1 vs Weighted Degree\n{hashtag} ({window})", 
                            fontsize=11, fontweight='bold')
            
            # Correlation
            rho = spearmanr(inf_weighted_degree, inf_pc_scores[:, 0])[0]
            axes[0].text(0.05, 0.95, f"ρ = {rho:.3f}\nn = {n_inf} influencers", 
                        transform=axes[0].transAxes, fontsize=11,
                        verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            
            # === Plot 2: PC1 vs Degree (binary) ===
            axes[1].scatter(inf_degree, inf_pc_scores[:, 0], 
                          alpha=0.6, s=40, c='darkred', edgecolors='black', linewidth=0.5)
            axes[1].set_xlabel('Degree (binary)', fontsize=12)
            axes[1].set_ylabel('PC1 Score', fontsize=12)
            axes[1].set_title(f"PC1 vs Degree", fontsize=11, fontweight='bold')
            
            rho = spearmanr(inf_degree, inf_pc_scores[:, 0])[0]
            axes[1].text(0.05, 0.95, f"ρ = {rho:.3f}", 
                        transform=axes[1].transAxes, fontsize=11,
                        verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
            
            # === Plot 3: PC2 vs First-Seen Time ===
            valid = ~np.isnan(inf_first_seen)
            if valid.sum() >= 3:
                axes[2].scatter(inf_first_seen[valid], inf_pc_scores[valid, 1], 
                              alpha=0.6, s=40, c='darkgreen', edgecolors='black', linewidth=0.5)
                axes[2].set_xlabel('First Appearance Time (normalised)', fontsize=12)
                axes[2].set_ylabel('PC2 Score', fontsize=12)
                axes[2].set_title(f"PC2 vs First Appearance", fontsize=11, fontweight='bold')
                
                rho = spearmanr(inf_first_seen[valid], inf_pc_scores[valid, 1])[0]
                axes[2].text(0.05, 0.95, f"ρ = {rho:.3f}", 
                            transform=axes[2].transAxes, fontsize=11,
                            verticalalignment='top', 
                            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            else:
                axes[2].text(0.5, 0.5, 'Insufficient temporal data', 
                            ha='center', va='center', fontsize=12)
                axes[2].set_xlabel('First Appearance Time')
                axes[2].set_ylabel('PC2 Score')
                axes[2].set_title(f"PC2 vs First Appearance", fontsize=11)
            
            plt.tight_layout()
            
            # Save
            filename = f"gae_interp_INFLUENCERS_{feature_mode}_{hashtag}_{window}.png"
            plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved influencer plot: {filename}")
    
    print(f"\n✓ Influencer-only plots complete!")


def compute_influencer_correlations(results):
    """
    Compute correlations for influencers only and add to results.
    
    Returns updated correlation summary.
    """
    
    inf_correlations = []
    
    for result in results:
        n_inf = result['n_inf']
        
        if n_inf < 5:  # Skip tiny influencer sets
            continue
        
        # Extract influencer data
        inf_pc = result['pc_scores'][:n_inf]
        inf_deg = result['degree'][:n_inf]
        inf_wdeg = result['weighted_degree'][:n_inf]
        inf_time = result['first_seen'][:n_inf]
        
        # Compute correlations
        corrs = {
            'hashtag': result['hashtag'],
            'window_end': result['window_end'],
            'feature_mode': result['feature_mode'],
            'n_inf': n_inf,
            'inf_PC1_deg': spearmanr(inf_pc[:, 0], inf_deg)[0],
            'inf_PC1_wdeg': spearmanr(inf_pc[:, 0], inf_wdeg)[0],
            'inf_PC2_deg': spearmanr(inf_pc[:, 1], inf_deg)[0],
            'inf_PC2_wdeg': spearmanr(inf_pc[:, 1], inf_wdeg)[0],
        }
        
        # First-seen time (if valid)
        valid = ~np.isnan(inf_time)
        if valid.sum() >= 3:
            corrs['inf_PC1_time'] = spearmanr(inf_pc[valid, 0], inf_time[valid])[0]
            corrs['inf_PC2_time'] = spearmanr(inf_pc[valid, 1], inf_time[valid])[0]
        else:
            corrs['inf_PC1_time'] = np.nan
            corrs['inf_PC2_time'] = np.nan
        
        inf_correlations.append(corrs)
    
    return pd.DataFrame(inf_correlations)

def main():
    """Main analysis pipeline."""
    
    emb_dir = Path('04_ml_prediction/01_features/outputs/embeddings')
    output_dir = Path('04_ml_prediction/05_scripts/gae_interpretability')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load selected windows
    selected_file = '04_ml_prediction/05_scripts/gae_analysis_windows_selected.csv'
    if not Path(selected_file).exists():
        print(f"ERROR: Run select_windows_for_gae_analysis.py first!")
        return
    
    selected = pd.read_csv(selected_file)
    print(f"Loaded {len(selected)} selected windows")
    
    # === Analyze both feature modes ===
    all_results = []
    
    for feature_mode in ['type', 'type+time']:
        print(f"\n{'='*80}")
        print(f"Analyzing: recency_weighted_{feature_mode}")
        print(f"{'='*80}\n")
        
        for _, row in selected.iterrows():
            result = analyze_one_window(
                hashtag=row['hashtag'],
                window_end=row['window_end'],
                adj_mode='recency_weighted',
                feature_mode=feature_mode,
                emb_dir=emb_dir
            )
            
            if result is not None:
                all_results.append(result)
    
    if not all_results:
        print("\n✗ No windows analyzed successfully!")
        return
    
    print(f"\n{'='*80}")
    print(f"✓ Successfully analyzed {len(all_results)} window-mode combinations")
    print(f"{'='*80}\n")
    
    # === Save correlation table ===
    corr_results = []
    for r in all_results:
        corr_dict = {k: v for k, v in r.items() 
                     if k not in ['pc_scores', 'degree', 'weighted_degree', 'first_seen']}
        corr_results.append(corr_dict)
    
    corr_df = pd.DataFrame(corr_results)
    corr_csv = output_dir / 'gae_interpretability_correlations.csv'
    corr_df.to_csv(corr_csv, index=False)
    
    print(f"✓ Saved correlation table: {corr_csv}")
    
    # === Print summary statistics (ALL nodes) ===
    print(f"\n{'='*80}")
    print("CORRELATION SUMMARY - ALL NODES (Mean ± Std)")
    print(f"{'='*80}\n")
    
    for feature_mode in ['type', 'type+time']:
        mode_df = corr_df[corr_df['feature_mode'] == feature_mode]
        
        if len(mode_df) == 0:
            continue
        
        print(f"\n*** recency_weighted_{feature_mode} (n={len(mode_df)} windows) ***\n")
        
        for col in ['PC1_deg', 'PC1_wdeg', 'PC1_time', 'PC2_deg', 'PC2_wdeg', 'PC2_time']:
            if col in mode_df.columns:
                mean = mode_df[col].mean()
                std = mode_df[col].std()
                print(f"  {col:15s}: {mean:6.3f} ± {std:5.3f}")
    
    # === NEW: Compute influencer-only correlations ===
    print(f"\n{'='*80}")
    print("Computing INFLUENCER-ONLY correlations...")
    print(f"{'='*80}\n")
    
    inf_corr_df = compute_influencer_correlations(all_results)
    
    # Save influencer correlations
    inf_corr_csv = output_dir / 'gae_interpretability_correlations_INFLUENCERS.csv'
    inf_corr_df.to_csv(inf_corr_csv, index=False)
    print(f"✓ Saved influencer correlation table: {inf_corr_csv}")
    
    # Print influencer summary
    print(f"\n{'='*80}")
    print("CORRELATION SUMMARY - INFLUENCERS ONLY (Mean ± Std)")
    print(f"{'='*80}\n")
    
    for feature_mode in ['type', 'type+time']:
        mode_df = inf_corr_df[inf_corr_df['feature_mode'] == feature_mode]
        
        if len(mode_df) == 0:
            continue
        
        print(f"\n*** recency_weighted_{feature_mode} (n={len(mode_df)} windows) ***\n")
        print(f"  Avg influencers per window: {mode_df['n_inf'].mean():.1f}")
        
        for col in ['inf_PC1_deg', 'inf_PC1_wdeg', 'inf_PC1_time', 
                    'inf_PC2_deg', 'inf_PC2_wdeg', 'inf_PC2_time']:
            if col in mode_df.columns:
                mean = mode_df[col].mean()
                std = mode_df[col].std()
                print(f"  {col:20s}: {mean:6.3f} ± {std:5.3f}")
    
    # === Create scatter plots (all nodes) ===
    print(f"\n{'='*80}")
    print("Creating scatter plots (all nodes)...")
    print(f"{'='*80}\n")
    
    make_scatter_plots(all_results, output_dir)
    
    # === NEW: Create influencer-only plots ===
    make_influencer_only_plots(all_results, output_dir)
    
    print(f"\n{'='*80}")
    print("✓ Analysis complete!")
    print(f"{'='*80}")
    print(f"\nOutputs saved to: {output_dir}/")
    print(f"  - gae_interpretability_correlations.csv (all nodes)")
    print(f"  - gae_interpretability_correlations_INFLUENCERS.csv (influencers only)")
    print(f"  - gae_interp_*.png (all nodes)")
    print(f"  - gae_interp_INFLUENCERS_*.png (influencers only)")


if __name__ == "__main__":
    main()