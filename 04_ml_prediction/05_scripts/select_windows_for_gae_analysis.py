#!/usr/bin/env python3
"""
select_windows_for_gae_analysis.py

Auto-select 8-10 diverse, high-quality windows for GAE interpretability analysis.
"""

import pandas as pd
from pathlib import Path
import numpy as np

def select_windows_for_analysis():
    """
    Select ~8-10 windows with:
    - High test AUC (GAE learned well)
    - Medium-large size (200-2000 nodes)
    - Diverse hashtags
    - Both feature modes available
    """
    
    # Load metrics
    metrics = pd.read_csv('04_ml_prediction/01_features/outputs/gae_bip_temp_wBCE_timetype_metrics.csv')
    
    print(f"Total rows in metrics: {len(metrics)}")
    
    # Filter successful runs
    successful = metrics[
        (metrics['gae_available'] == 1) &
        (metrics['test_auc'] > 0.7) &  # Reasonable GAE performance
        (metrics['n_nodes'] >= 3000) &   # Not too small
        (metrics['n_nodes'] <= 5000) &  # Not too huge
        (metrics['adj_mode'] == 'recency_weighted')  # Focus on recency_weighted
    ].copy()
    
    print(f"Successful windows: {len(successful)}")
    
    # Check which embeddings actually exist
    emb_dir = Path('04_ml_prediction/01_features/outputs/embeddings')
    
    def check_embeddings_exist(row):
        """Check if both type and type+time embeddings exist."""
        hashtag = row['hashtag']
        window = row['window_end']
        
        type_file = emb_dir / f"{hashtag}_{window}_recency_weighted_type.npz"
        type_time_file = emb_dir / f"{hashtag}_{window}_recency_weighted_type+time.npz"
        
        return type_file.exists() and type_time_file.exists()
    
    successful['both_embeddings_exist'] = successful.apply(check_embeddings_exist, axis=1)
    
    available = successful[successful['both_embeddings_exist']].copy()
    
    print(f"Windows with both embeddings: {len(available)}")
    
    if len(available) == 0:
        print("ERROR: No windows found with both embeddings!")
        return
    
    # Create a quality score
    available['quality_score'] = (
        available['test_auc'] * 0.5 +  # Performance
        (np.log10(available['n_nodes']) / 4) * 0.3 +  # Size (prefer medium-large)
        (1 - available['n_nodes'] / 2000) * 0.2  # Not too huge
    )
    
    # Sort by quality
    available = available.sort_values('quality_score', ascending=False)
    
    # Select top 10, ensuring diversity
    selected = []
    used_hashtags = set()
    
    for _, row in available.iterrows():
        if len(selected) >= 10:
            break
        
        # Prefer diverse hashtags (but allow 2 windows per hashtag max)
        hashtag = row['hashtag']
        hashtag_count = sum(1 for s in selected if s['hashtag'] == hashtag)
        
        if hashtag_count < 2:
            selected.append(row.to_dict())
            used_hashtags.add(hashtag)
    
    # Create output dataframe
    selected_df = pd.DataFrame(selected)
    
    # Save selection
    output_file = '04_ml_prediction/05_scripts/gae_analysis_windows_selected.csv'
    selected_df[['hashtag', 'window_end', 'adj_mode', 'feature_mode', 
                 'n_nodes', 'n_inf', 'n_aud', 'test_auc', 'quality_score']].to_csv(
        output_file, index=False
    )
    
    print(f"\n{'='*80}")
    print(f"✓ Selected {len(selected)} windows for analysis")
    print(f"{'='*80}")
    print(f"\nHashtags: {len(used_hashtags)} unique")
    print(f"Avg nodes: {selected_df['n_nodes'].mean():.0f}")
    print(f"Avg test AUC: {selected_df['test_auc'].mean():.3f}")
    print(f"\nSaved to: {output_file}")
    print(f"\nTop 5 selected windows:")
    print(selected_df[['hashtag', 'window_end', 'n_nodes', 'test_auc']].head())
    
    # Also check for any binary_type embeddings
    print(f"\n{'='*80}")
    print("Checking for binary_type embeddings...")
    print(f"{'='*80}")
    
    binary_available = []
    for _, row in available.head(20).iterrows():
        hashtag = row['hashtag']
        window = row['window_end']
        binary_file = emb_dir / f"{hashtag}_{window}_binary_type.npz"
        
        if binary_file.exists():
            binary_available.append({
                'hashtag': hashtag,
                'window_end': window,
                'n_nodes': row['n_nodes'],
                'test_auc': row['test_auc']
            })
    
    if binary_available:
        print(f"Found {len(binary_available)} windows with binary_type embeddings!")
        print(pd.DataFrame(binary_available))
        
        # Save these too
        pd.DataFrame(binary_available).to_csv(
            '04_ml_prediction/05_scripts/gae_analysis_windows_binary.csv',
            index=False
        )
    else:
        print("No binary_type embeddings found. Will need to rerun or skip.")


if __name__ == "__main__":
    select_windows_for_analysis()