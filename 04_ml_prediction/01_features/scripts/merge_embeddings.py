#!/usr/bin/env python3
"""
Merge Audience and Bipartite Embedding Features

Combines two embedding parquets into one with proper prefixes.

Usage:
    python 04_ml_prediction/01_features/scripts/merge_embeddings.py \
        --audience 04_ml_prediction/01_features/outputs/embedding_features_h550_audience.parquet \
        --bipartite 04_ml_prediction/01_features/outputs/embedding_features_h550_bipartite.parquet \
        --output 04_ml_prediction/01_features/outputs/embedding_features_h550_combined.parquet
"""

import argparse
import pandas as pd
from pathlib import Path


def merge_embeddings(audience_path, bipartite_path, output_path, verbose=False):
    """
    Merge audience and bipartite embeddings on (hashtag, window_end).
    
    Args:
        audience_path: Path to audience embedding features
        bipartite_path: Path to bipartite embedding features
        output_path: Where to save merged features
    """
    print("="*80)
    print("Merging Embedding Features")
    print("="*80)
    
    # Load data
    print(f"\nLoading audience embeddings from {audience_path}...")
    aud_df = pd.read_parquet(audience_path)
    print(f"  ✓ Loaded {len(aud_df)} windows")
    
    print(f"\nLoading bipartite embeddings from {bipartite_path}...")
    bip_df = pd.read_parquet(bipartite_path)
    print(f"  ✓ Loaded {len(bip_df)} windows")
    
    # Check columns
    aud_cols = [c for c in aud_df.columns if c not in ['hashtag', 'window_end']]
    bip_cols = [c for c in bip_df.columns if c not in ['hashtag', 'window_end']]
    
    print(f"\nAudience features: {len(aud_cols)}")
    print(f"Bipartite features: {len(bip_cols)}")
    
    if verbose:
        print(f"\nSample audience columns: {aud_cols[:5]}")
        print(f"Sample bipartite columns: {bip_cols[:5]}")
    
    # Merge on (hashtag, window_end)
    print(f"\nMerging on (hashtag, window_end)...")
    merged = aud_df.merge(
        bip_df,
        on=['hashtag', 'window_end'],
        how='outer',  # Keep all windows from both
        suffixes=('', '_dup')
    )
    
    # Remove duplicate columns (shouldn't happen but just in case)
    dup_cols = [c for c in merged.columns if c.endswith('_dup')]
    if dup_cols:
        print(f"  ⚠️  Removing {len(dup_cols)} duplicate columns")
        merged = merged.drop(columns=dup_cols)
    
    # Report merge statistics
    n_audience_only = (merged[aud_cols[0]].notna() & merged[bip_cols[0]].isna()).sum()
    n_bipartite_only = (merged[aud_cols[0]].isna() & merged[bip_cols[0]].notna()).sum()
    n_both = (merged[aud_cols[0]].notna() & merged[bip_cols[0]].notna()).sum()
    
    print(f"\n{'='*80}")
    print("Merge Statistics")
    print(f"{'='*80}")
    print(f"Total windows: {len(merged)}")
    print(f"  Both embeddings: {n_both} ({n_both/len(merged)*100:.1f}%)")
    print(f"  Audience only: {n_audience_only} ({n_audience_only/len(merged)*100:.1f}%)")
    print(f"  Bipartite only: {n_bipartite_only} ({n_bipartite_only/len(merged)*100:.1f}%)")
    
    total_features = len([c for c in merged.columns if c not in ['hashtag', 'window_end']])
    print(f"\nTotal features: {total_features}")
    print(f"  Audience: {len(aud_cols)}")
    print(f"  Bipartite: {len(bip_cols)}")
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    merged.to_parquet(output_path, index=False)
    print(f"\n✓ Saved merged embeddings to {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    print(f"\n{'='*80}")
    print("✓ Merge complete!")
    print(f"{'='*80}\n")
    
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Merge audience and bipartite embedding features"
    )
    parser.add_argument('--audience', required=True,
                       help='Path to audience embedding features')
    parser.add_argument('--bipartite', required=True,
                       help='Path to bipartite embedding features')
    parser.add_argument('--output', required=True,
                       help='Output path for merged features')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed info')
    
    args = parser.parse_args()
    
    merge_embeddings(
        args.audience,
        args.bipartite,
        args.output,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()