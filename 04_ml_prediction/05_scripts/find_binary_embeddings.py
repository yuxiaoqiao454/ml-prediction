#!/usr/bin/env python3
"""
find_binary_embeddings.py

Search for any binary_type embeddings available.
"""

import pandas as pd
from pathlib import Path
import re

def find_binary_embeddings():
    """Search for all binary_type embeddings."""
    
    emb_dir = Path('04_ml_prediction/01_features/outputs/embeddings')
    
    # Find all .npz files
    all_files = list(emb_dir.glob('*.npz'))
    
    print(f"Total embedding files: {len(all_files)}")
    
    # Filter for binary_type
    binary_files = [f for f in all_files if 'binary_type' in f.name]
    
    print(f"\n{'='*80}")
    print(f"Found {len(binary_files)} binary_type embeddings")
    print(f"{'='*80}\n")
    
    if not binary_files:
        print("No binary_type embeddings found!")
        return
    
    # Parse filenames
    pattern = r'(.+)_(\d{4}-\d{2}-\d{2})_(binary)_(type(?:\+time)?).npz'
    
    binary_data = []
    for f in binary_files:
        match = re.match(pattern, f.name)
        if match:
            hashtag, window, adj_mode, feature_mode = match.groups()
            binary_data.append({
                'hashtag': hashtag,
                'window_end': window,
                'adj_mode': adj_mode,
                'feature_mode': feature_mode,
                'filename': f.name
            })
    
    if not binary_data:
        print("Could not parse binary embedding filenames!")
        return
    
    binary_df = pd.DataFrame(binary_data)
    
    # Load metrics to get node counts and AUC
    metrics = pd.read_csv('04_ml_prediction/01_features/outputs/gae_bip_struct_metrics.csv')
    
    # Merge with metrics
    binary_with_metrics = binary_df.merge(
        metrics[['hashtag', 'window_end', 'adj_mode', 'feature_mode', 'n_nodes', 'test_auc']],
        on=['hashtag', 'window_end', 'adj_mode', 'feature_mode'],
        how='left'
    )
    
    print(binary_with_metrics.to_string(index=False))
    
    # Save
    output_file = '04_ml_prediction/05_scripts/binary_embeddings_available.csv'
    binary_with_metrics.to_csv(output_file, index=False)
    
    print(f"\n✓ Saved to: {output_file}")
    
    # Pick best ones for analysis
    if len(binary_with_metrics) > 0:
        print(f"\n{'='*80}")
        print("Recommended windows for binary analysis:")
        print(f"{'='*80}\n")
        
        best = binary_with_metrics.nlargest(5, 'n_nodes')
        print(best[['hashtag', 'window_end', 'feature_mode', 'n_nodes', 'test_auc']].to_string(index=False))


if __name__ == "__main__":
    find_binary_embeddings()