#!/usr/bin/env python3
"""
Split samples into train/validation/test sets.

Uses hashtag-based stratified splitting:
- Never puts same hashtag in multiple splits (prevents leakage)
- Stratifies by burst rate (keeps similar burst % in each split)
- Saves split assignments and metadata

Usage:
  python split_data.py --input samples_full_v1.parquet
  python split_data.py --train-size 0.7 --val-size 0.15
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.model_selection import train_test_split

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root))


def load_samples(samples_path):
    """Load samples parquet."""
    print(f"Loading samples from {samples_path}...")
    df = pd.read_parquet(samples_path)
    print(f"✓ Loaded {len(df)} samples")
    print(f"  Hashtags: {df['hashtag'].nunique()}")
    print(f"  Burst rate: {df['label'].mean():.1%}")
    print(f"  Features: {len([c for c in df.columns if c.startswith('ts_') or c.startswith('net_')])}\n")
    return df


def stratified_hashtag_split(df, train_size=0.7, val_size=0.15, random_state=42):
    """
    Split hashtags into train/val/test with stratification by burst rate.
    
    Strategy:
    1. Compute burst rate per hashtag
    2. Bin hashtags by burst rate (for stratification)
    3. Split hashtags (not samples) into train/val/test
    4. Assign all samples from each hashtag to its split
    
    Returns:
    --------
    train_df, val_df, test_df, split_metadata
    """
    print("Performing hashtag-based stratified split...")
    print(f"  Split: {train_size:.0%} train / {val_size:.0%} val / {1-train_size-val_size:.0%} test\n")
    
    # Compute per-hashtag burst rates
    hashtag_stats = df.groupby('hashtag').agg({
        'label': ['sum', 'count', 'mean']
    }).reset_index()
    hashtag_stats.columns = ['hashtag', 'n_bursts', 'n_samples', 'burst_rate']
    
    # Create stratification bins (quartiles of burst rate)
    n_unique = hashtag_stats['burst_rate'].nunique()
    
    if n_unique < 2:
        # Not enough variation, just use a single bin
        hashtag_stats['burst_bin'] = 'all'
    elif n_unique < 4:
        # Use cut instead of qcut for few unique values
        hashtag_stats['burst_bin'] = pd.cut(
            hashtag_stats['burst_rate'],
            bins=n_unique,
            duplicates='drop'
        )
    else:
        # Standard case: use qcut with 4 bins
        try:
            hashtag_stats['burst_bin'] = pd.qcut(
                hashtag_stats['burst_rate'], 
                q=4, 
                labels=['very_low', 'low', 'medium', 'high'],
                duplicates='drop'
            )
        except ValueError:
            # Fallback: use cut instead
            hashtag_stats['burst_bin'] = pd.cut(
                hashtag_stats['burst_rate'],
                bins=3,
                duplicates='drop'
            )
    
    def merge_rare_bins(stats_df):
        """Merge bins with <2 hashtags to enable stratification."""
        bin_counts = stats_df['burst_bin'].value_counts()
        
        # Remove empty bins from consideration
        bin_counts = bin_counts[bin_counts > 0]
        
        if (bin_counts < 2).any():
            rare_bins = bin_counts[bin_counts < 2].index
            non_rare_bins = bin_counts[bin_counts >= 2].index
            
            if len(non_rare_bins) == 0:
                # All bins are rare, merge everything into one
                stats_df['burst_bin'] = 'all'
            else:
                # Merge each rare bin into the nearest non-rare bin
                for rare_bin in rare_bins:
                    rare_mask = stats_df['burst_bin'] == rare_bin
                    
                    # Skip if no hashtags in this bin (shouldn't happen, but safety check)
                    if not rare_mask.any():
                        continue
                    
                    rare_rates = stats_df.loc[rare_mask, 'burst_rate'].values
                    
                    # Find closest non-rare bin by median burst rate
                    bin_medians = stats_df[stats_df['burst_bin'].isin(non_rare_bins)].groupby('burst_bin', observed=True)['burst_rate'].median()
                    closest_bin = bin_medians.index[(bin_medians - rare_rates[0]).abs().argmin()]
                    stats_df.loc[rare_mask, 'burst_bin'] = closest_bin
        
        return stats_df
    
    # Merge rare bins before first split
    hashtag_stats = merge_rare_bins(hashtag_stats)
    
    print("Hashtag burst rate distribution:")
    print(hashtag_stats.groupby('burst_bin', observed=True)['hashtag'].count())
    print()
    
    # Split hashtags (not samples) - stratified by burst_bin
    hashtags = hashtag_stats['hashtag'].values
    strata = hashtag_stats['burst_bin'].values
    
    # First split: train vs (val+test)
    train_hashtags, temp_hashtags = train_test_split(
        hashtags,
        train_size=train_size,
        stratify=strata,
        random_state=random_state
    )
    
    # Second split: val vs test
    # Need to merge rare bins AGAIN for the temp set
    temp_stats = hashtag_stats[hashtag_stats['hashtag'].isin(temp_hashtags)].copy()
    temp_stats = merge_rare_bins(temp_stats)
    
    temp_strata = temp_stats['burst_bin'].values
    val_ratio = val_size / (val_size + (1 - train_size - val_size))
    
    val_hashtags, test_hashtags = train_test_split(
        temp_stats['hashtag'].values,
        train_size=val_ratio,
        stratify=temp_strata,
        random_state=random_state
    )
    
    # Assign samples to splits based on hashtag
    train_df = df[df['hashtag'].isin(train_hashtags)].copy()
    val_df = df[df['hashtag'].isin(val_hashtags)].copy()
    test_df = df[df['hashtag'].isin(test_hashtags)].copy()
    
    # Add split column
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    # Create metadata
    metadata = {
        'split_method': 'hashtag_stratified',
        'split_ratios': {
            'train': float(train_size),
            'val': float(val_size),
            'test': float(1 - train_size - val_size)
        },
        'random_state': random_state,
        'created_at': datetime.now().isoformat(),
        
        'train_hashtags': sorted(train_hashtags.tolist()),
        'val_hashtags': sorted(val_hashtags.tolist()),
        'test_hashtags': sorted(test_hashtags.tolist()),
        
        'n_hashtags': {
            'train': len(train_hashtags),
            'val': len(val_hashtags),
            'test': len(test_hashtags)
        },
        
        'n_samples': {
            'train': len(train_df),
            'val': len(val_df),
            'test': len(test_df)
        },
        
        'n_bursts': {
            'train': int(train_df['label'].sum()),
            'val': int(val_df['label'].sum()),
            'test': int(test_df['label'].sum())
        },
        
        'burst_rate': {
            'train': float(train_df['label'].mean()),
            'val': float(val_df['label'].mean()),
            'test': float(test_df['label'].mean())
        }
    }
    
    # Print summary
    print("="*80)
    print("Split Summary")
    print("="*80)
    print(f"\n{'Split':<10} {'Hashtags':<12} {'Samples':<10} {'Bursts':<10} {'Burst Rate':<12}")
    print("-"*80)
    for split_name in ['train', 'val', 'test']:
        print(f"{split_name:<10} "
              f"{metadata['n_hashtags'][split_name]:<12} "
              f"{metadata['n_samples'][split_name]:<10} "
              f"{metadata['n_bursts'][split_name]:<10} "
              f"{metadata['burst_rate'][split_name]:.1%}")
    print("="*80)
    
    # Check stratification quality
    print("\nStratification quality:")
    burst_rates = [metadata['burst_rate'][s] for s in ['train', 'val', 'test']]
    print(f"  Burst rate range: {min(burst_rates):.1%} - {max(burst_rates):.1%}")
    print(f"  Std deviation: {np.std(burst_rates):.3f}")
    if np.std(burst_rates) < 0.01:
        print("  ✓ Good stratification (rates are similar)")
    else:
        print("  ⚠ Stratification could be better (consider adjusting bins)")
    
    return train_df, val_df, test_df, metadata

def save_splits(train_df, val_df, test_df, metadata, output_dir, dataset_version):
    """
    Save split datasets and metadata.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual split files
    train_path = output_dir / f"samples_train_{dataset_version}.parquet"
    val_path = output_dir / f"samples_val_{dataset_version}.parquet"
    test_path = output_dir / f"samples_test_{dataset_version}.parquet"
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"\n✓ Saved train set to {train_path}")
    print(f"✓ Saved val set to {val_path}")
    print(f"✓ Saved test set to {test_path}")
    
    # Save full dataset with split assignments
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_path = output_dir / f"samples_full_{dataset_version}.parquet"
    full_df.to_parquet(full_path, index=False)
    print(f"✓ Saved full dataset (with split assignments) to {full_path}")
    
    # Save metadata
    metadata_path = output_dir / f"split_metadata_{dataset_version}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved split metadata to {metadata_path}")
    
    print("\n" + "="*80)
    print("Split complete! Ready for model training.")
    print("="*80)


def main():
    ap = argparse.ArgumentParser(description="Split samples into train/val/test")
    ap.add_argument("--input",
                    default="04_ml_prediction/03_datasets/outputs/samples_full_v1.parquet",
                    help="Input samples file")
    ap.add_argument("--output-dir",
                    default="04_ml_prediction/03_datasets/outputs",
                    help="Output directory")
    ap.add_argument("--dataset-version",
                    default="v1",
                    help="Dataset version (for output filenames)")
    ap.add_argument("--train-size", type=float, default=0.7,
                    help="Proportion for training (default: 0.7)")
    ap.add_argument("--val-size", type=float, default=0.15,
                    help="Proportion for validation (default: 0.15)")
    ap.add_argument("--random-state", type=int, default=42,
                    help="Random seed for reproducibility")
    args = ap.parse_args()
    
    # Validate split sizes
    test_size = 1 - args.train_size - args.val_size
    if test_size <= 0:
        print(f"Error: train_size + val_size must be < 1.0")
        return
    
    print("="*80)
    print("Train/Val/Test Split")
    print("="*80)
    print(f"Input: {args.input}")
    print(f"Split: {args.train_size:.0%} / {args.val_size:.0%} / {test_size:.0%}")
    print(f"Random state: {args.random_state}")
    print("="*80 + "\n")
    
    # Load samples
    samples_df = load_samples(args.input)
    
    # Perform split
    train_df, val_df, test_df, metadata = stratified_hashtag_split(
        samples_df,
        train_size=args.train_size,
        val_size=args.val_size,
        random_state=args.random_state
    )
    
    # Save
    save_splits(train_df, val_df, test_df, metadata, args.output_dir, args.dataset_version)


if __name__ == "__main__":
    main()