#!/usr/bin/env python3
"""
Split multiple sample datasets with CONSISTENT splits.

Ensures all datasets use the same hashtag assignments for train/val/test.
This is critical when comparing:
- Different labeling rules (same features, different labels)
- Different feature sets (same labels, different features)
- Different combinations of both

Usage:
  # Single dataset (backward compatible)
  python split_data_consistent.py \
    --input samples_full_h550_0.5rel.parquet \
    --dataset-version h550_0.5rel
  
  # Multiple datasets with consistent splits
  python 04_ml_prediction/03_datasets/scripts/split_data_consistent.py \
    --inputs 04_ml_prediction/03_datasets/outputs/samples_full_h550_0.5rel.parquet 04_ml_prediction/03_datasets/outputs/samples_full_h550_0.5rel_ts_net_gae_tempbip_type.parquet 04_ml_prediction/03_datasets/outputs/samples_full_h550_0.5rel_ts_gae_tempbip_type.parquet 04_ml_prediction/03_datasets/outputs/samples_full_h550_0.5rel_ts_net_gae_tempbip_type_inf.parquet 04_ml_prediction/03_datasets/outputs/samples_full_h550_0.5rel_ts_gae_tempbip_type_inf.parquet \
    --dataset-versions h550_0.5rel h550_0.5rel_ts_net_gae_tempbip_type h550_0.5rel_ts_gae_tempbip_type h550_0.5rel_ts_net_gae_tempbip_type_inf h550_0.5rel_ts_gae_tempbip_type_inf \
    --seed-id seed10

  python 04_ml_prediction/03_datasets/scripts/split_data_consistent.py \
    --inputs 04_ml_prediction/03_datasets/outputs/samples_full_h550_0.5rel.parquet 04_ml_prediction/03_datasets/outputs/samples_full_h550_0.5rel_ts_net_gae_bip.parquet 04_ml_prediction/03_datasets/outputs/samples_full_h550_0.5rel_ts_gae_bip.parquet \
    --dataset-versions h550_0.5rel h550_0.5rel_ts_net_gae_bip h550_0.5rel_ts_gae_bip \
    --seed-id seed9


  
  # Batch mode with auto-discovery
  python split_data_consistent.py \
    --input-dir 04_ml_prediction/03_datasets/outputs \
    --pattern "samples_full_*.parquet" \
    --seed-id seed42
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
    df = pd.read_parquet(samples_path)
    return df


def compute_joint_stratification(datasets_info, train_size=0.7, val_size=0.15):
    """
    Compute stratification bins considering ALL datasets jointly.
    
    Strategy:
    1. For each hashtag, compute burst rate across ALL datasets
    2. Use average burst rate for stratification
    3. This ensures we stratify based on overall trend difficulty
    
    Args:
        datasets_info: List of dicts with 'name', 'df', 'path'
        train_size: Train split ratio
        val_size: Val split ratio
    
    Returns:
        hashtag_stats: DataFrame with stratification info
    """
    print("\n" + "="*80)
    print("COMPUTING JOINT STRATIFICATION")
    print("="*80)
    
    # Collect per-hashtag burst rates from all datasets
    all_hashtag_stats = []
    
    for info in datasets_info:
        name = info['name']
        df = info['df']
        
        stats = df.groupby('hashtag').agg({
            'label': ['sum', 'count', 'mean']
        }).reset_index()
        stats.columns = ['hashtag', 'n_bursts', 'n_samples', 'burst_rate']
        stats['dataset'] = name
        
        all_hashtag_stats.append(stats)
        
        print(f"\n{name}:")
        print(f"  Hashtags: {len(stats)}")
        print(f"  Burst rate: {stats['burst_rate'].mean():.1%} ± {stats['burst_rate'].std():.2%}")
    
    # Combine all datasets
    combined_stats = pd.concat(all_hashtag_stats, ignore_index=True)
    
    # Compute average burst rate per hashtag across datasets
    avg_burst_rates = combined_stats.groupby('hashtag')['burst_rate'].mean().reset_index()
    avg_burst_rates.columns = ['hashtag', 'avg_burst_rate']
    
    # Get unique hashtags
    unique_hashtags = avg_burst_rates.copy()
    
    print("\n" + "-"*80)
    print("JOINT STATISTICS:")
    print(f"  Total unique hashtags: {len(unique_hashtags)}")
    print(f"  Average burst rate: {unique_hashtags['avg_burst_rate'].mean():.1%}")
    print(f"  Burst rate range: {unique_hashtags['avg_burst_rate'].min():.1%} - {unique_hashtags['avg_burst_rate'].max():.1%}")
    
    # Create stratification bins based on average burst rate
    n_unique = unique_hashtags['avg_burst_rate'].nunique()
    
    if n_unique < 2:
        unique_hashtags['burst_bin'] = 'all'
    elif n_unique < 4:
        unique_hashtags['burst_bin'] = pd.cut(
            unique_hashtags['avg_burst_rate'],
            bins=n_unique,
            duplicates='drop'
        )
    else:
        try:
            unique_hashtags['burst_bin'] = pd.qcut(
                unique_hashtags['avg_burst_rate'], 
                q=4, 
                labels=['very_low', 'low', 'medium', 'high'],
                duplicates='drop'
            )
        except ValueError:
            unique_hashtags['burst_bin'] = pd.cut(
                unique_hashtags['avg_burst_rate'],
                bins=3,
                duplicates='drop'
            )
    
    # Merge rare bins
    unique_hashtags = merge_rare_bins(unique_hashtags)
    
    print("\nStratification bins:")
    print(unique_hashtags.groupby('burst_bin', observed=True)['hashtag'].count())
    
    return unique_hashtags


def merge_rare_bins(stats_df):
    """Merge bins with <2 hashtags to enable stratification."""
    bin_counts = stats_df['burst_bin'].value_counts()
    bin_counts = bin_counts[bin_counts > 0]
    
    if (bin_counts < 2).any():
        rare_bins = bin_counts[bin_counts < 2].index
        non_rare_bins = bin_counts[bin_counts >= 2].index
        
        if len(non_rare_bins) == 0:
            stats_df['burst_bin'] = 'all'
        else:
            for rare_bin in rare_bins:
                rare_mask = stats_df['burst_bin'] == rare_bin
                if not rare_mask.any():
                    continue
                
                rare_rates = stats_df.loc[rare_mask, 'avg_burst_rate'].values
                bin_medians = stats_df[stats_df['burst_bin'].isin(non_rare_bins)].groupby('burst_bin', observed=True)['avg_burst_rate'].median()
                closest_bin = bin_medians.index[(bin_medians - rare_rates[0]).abs().argmin()]
                stats_df.loc[rare_mask, 'burst_bin'] = closest_bin
    
    return stats_df


def split_hashtags(hashtag_stats, train_size=0.7, val_size=0.15, random_state=42):
    """
    Split hashtags into train/val/test using stratification.
    
    Returns:
        train_hashtags, val_hashtags, test_hashtags
    """
    print("\n" + "="*80)
    print("SPLITTING HASHTAGS")
    print("="*80)
    print(f"Split ratios: {train_size:.0%} train / {val_size:.0%} val / {1-train_size-val_size:.0%} test")
    
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
    
    print(f"\n✓ Split complete:")
    print(f"  Train: {len(train_hashtags)} hashtags")
    print(f"  Val: {len(val_hashtags)} hashtags")
    print(f"  Test: {len(test_hashtags)} hashtags")
    
    return train_hashtags, val_hashtags, test_hashtags


def apply_split_to_dataset(df, train_hashtags, val_hashtags, test_hashtags, dataset_name):
    """
    Apply pre-determined split to a dataset.
    
    Returns:
        train_df, val_df, test_df, metadata
    """
    train_df = df[df['hashtag'].isin(train_hashtags)].copy()
    val_df = df[df['hashtag'].isin(val_hashtags)].copy()
    test_df = df[df['hashtag'].isin(test_hashtags)].copy()
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    # Compute metadata
    metadata = {
        'dataset_name': dataset_name,
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
    
    return train_df, val_df, test_df, metadata


def print_split_summary(all_metadata, seed_id):
    """Print comprehensive summary for all datasets."""
    print("\n" + "="*80)
    print(f"SPLIT SUMMARY (Seed: {seed_id})")
    print("="*80)
    
    # Print per-dataset stats
    for meta in all_metadata:
        name = meta['dataset_name']
        print(f"\n{name}:")
        print(f"  {'Split':<10} {'Samples':<10} {'Bursts':<10} {'Burst Rate':<12}")
        print("  " + "-"*60)
        for split in ['train', 'val', 'test']:
            n_samples = meta['n_samples'][split]
            n_bursts = meta['n_bursts'][split]
            rate = meta['burst_rate'][split]
            print(f"  {split:<10} {n_samples:<10} {n_bursts:<10} {rate:.1%}")
    
    # Print stratification quality
    print("\n" + "="*80)
    print("STRATIFICATION QUALITY")
    print("="*80)
    
    for meta in all_metadata:
        name = meta['dataset_name']
        burst_rates = [meta['burst_rate'][s] for s in ['train', 'val', 'test']]
        std = np.std(burst_rates)
        
        print(f"\n{name}:")
        print(f"  Burst rate range: {min(burst_rates):.2%} - {max(burst_rates):.2%}")
        print(f"  Std deviation: {std:.4f}")
        if std < 0.01:
            print(f"  ✓ Excellent stratification")
        elif std < 0.02:
            print(f"  ✓ Good stratification")
        else:
            print(f"  ⚠ Moderate stratification (expected with different labels)")


def save_splits(datasets_info, train_hashtags, val_hashtags, test_hashtags, 
                all_metadata, output_dir, seed_id, random_state):
    """Save all split datasets and metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("SAVING SPLITS")
    print("="*80)
    
    saved_files = []
    
    # Save splits for each dataset
    for info in datasets_info:
        name = info['name']
        df = info['df']
        
        # Apply split
        train_df, val_df, test_df, metadata = apply_split_to_dataset(
            df, train_hashtags, val_hashtags, test_hashtags, name
        )
        
        # Generate filenames with seed suffix
        train_path = output_dir / f"samples_train_{name}_{seed_id}.parquet"
        val_path = output_dir / f"samples_val_{name}_{seed_id}.parquet"
        test_path = output_dir / f"samples_test_{name}_{seed_id}.parquet"
        full_path = output_dir / f"samples_full_{name}_{seed_id}.parquet"
        
        # Save files
        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)
        
        full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        full_df.to_parquet(full_path, index=False)
        
        saved_files.extend([train_path, val_path, test_path, full_path])
        
        print(f"\n{name}:")
        print(f"  ✓ {train_path.name}")
        print(f"  ✓ {val_path.name}")
        print(f"  ✓ {test_path.name}")
        print(f"  ✓ {full_path.name}")
    
    # Save master metadata
    master_metadata = {
        'seed_id': seed_id,
        'random_state': random_state,
        'split_method': 'consistent_hashtag_stratified',
        'created_at': datetime.now().isoformat(),
        'n_datasets': len(datasets_info),
        'dataset_names': [info['name'] for info in datasets_info],
        
        'train_hashtags': sorted(train_hashtags.tolist()),
        'val_hashtags': sorted(val_hashtags.tolist()),
        'test_hashtags': sorted(test_hashtags.tolist()),
        
        'n_hashtags': {
            'train': len(train_hashtags),
            'val': len(val_hashtags),
            'test': len(test_hashtags)
        },
        
        'per_dataset_stats': all_metadata
    }
    
    metadata_path = output_dir / f"split_metadata_{seed_id}.json"
    with open(metadata_path, 'w') as f:
        json.dump(master_metadata, f, indent=2)
    
    print(f"\n✓ Saved master metadata to {metadata_path.name}")
    
    print("\n" + "="*80)
    print(f"✓ COMPLETE: Saved {len(saved_files)} files + metadata")
    print(f"✓ All datasets use identical train/val/test hashtag splits ({seed_id})")
    print("="*80)


def discover_sample_files(input_dir, pattern):
    """Auto-discover sample files matching pattern."""
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob(pattern))
    
    if len(files) == 0:
        raise ValueError(f"No files found matching pattern: {input_dir}/{pattern}")
    
    print(f"Auto-discovered {len(files)} sample files:")
    for f in files:
        print(f"  - {f.name}")
    
    return files


def main():
    ap = argparse.ArgumentParser(
        description="Split multiple datasets with consistent hashtag assignments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single dataset
  python split_data_consistent.py \\
    --input samples_full_h550_0.5rel.parquet \\
    --dataset-version h550_0.5rel
  
  # Multiple datasets
  python split_data_consistent.py \\
    --inputs samples_full_h550_0.5rel.parquet samples_full_h550_cpinside.parquet \\
    --dataset-versions h550_0.5rel h550_cpinside \\
    --seed-id seed1
  
  # Auto-discover all samples
  python split_data_consistent.py \\
    --input-dir 04_ml_prediction/03_datasets/outputs \\
    --pattern "samples_full_*.parquet" \\
    --seed-id seed42
        """
    )
    
    # Input options
    input_group = ap.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Single input file (backward compatible)")
    input_group.add_argument("--inputs", nargs='+', help="Multiple input files")
    input_group.add_argument("--input-dir", help="Directory to search for sample files")
    
    ap.add_argument("--pattern", default="samples_full_*.parquet",
                    help="Pattern for auto-discovery (used with --input-dir)")
    ap.add_argument("--dataset-version", help="Dataset version (for single input)")
    ap.add_argument("--dataset-versions", nargs='+', help="Dataset versions (for multiple inputs)")
    
    # Output options
    ap.add_argument("--output-dir",
                    default="04_ml_prediction/03_datasets/outputs",
                    help="Output directory")
    ap.add_argument("--seed-id", default="seed1",
                    help="Seed identifier (e.g., 'seed1', 'seed42')")
    
    # Split parameters
    ap.add_argument("--train-size", type=float, default=0.7,
                    help="Train proportion (default: 0.7)")
    ap.add_argument("--val-size", type=float, default=0.15,
                    help="Val proportion (default: 0.15)")
    ap.add_argument("--random-state", type=int, default=42,
                    help="Random seed")
    
    args = ap.parse_args()
    
    # Validate split sizes
    test_size = 1 - args.train_size - args.val_size
    if test_size <= 0:
        print(f"Error: train_size + val_size must be < 1.0")
        return
    
    print("="*80)
    print("CONSISTENT SPLIT ACROSS MULTIPLE DATASETS")
    print("="*80)
    
    # Determine input files and dataset names
    if args.input:
        # Single file mode
        input_files = [Path(args.input)]
        if not args.dataset_version:
            # Extract from filename
            dataset_version = input_files[0].stem.replace('samples_full_', '')
        else:
            dataset_version = args.dataset_version
        dataset_names = [dataset_version]
        
    elif args.inputs:
        # Multiple file mode
        input_files = [Path(f) for f in args.inputs]
        if not args.dataset_versions:
            # Extract from filenames
            dataset_names = [f.stem.replace('samples_full_', '') for f in input_files]
        else:
            if len(args.dataset_versions) != len(input_files):
                print(f"Error: Number of dataset versions ({len(args.dataset_versions)}) "
                      f"must match number of inputs ({len(input_files)})")
                return
            dataset_names = args.dataset_versions
    
    else:
        # Auto-discovery mode
        input_files = discover_sample_files(args.input_dir, args.pattern)
        dataset_names = [f.stem.replace('samples_full_', '') for f in input_files]
    
    print(f"\nProcessing {len(input_files)} dataset(s):")
    for name, path in zip(dataset_names, input_files):
        print(f"  {name}: {path}")
    
    print(f"\nSeed ID: {args.seed_id}")
    print(f"Random state: {args.random_state}")
    print(f"Split: {args.train_size:.0%} / {args.val_size:.0%} / {test_size:.0%}")
    
    # Load all datasets
    datasets_info = []
    for name, path in zip(dataset_names, input_files):
        print(f"\nLoading {name}...")
        df = load_samples(path)
        print(f"  ✓ {len(df)} samples, {df['hashtag'].nunique()} hashtags, "
              f"{df['label'].mean():.1%} burst rate")
        
        datasets_info.append({
            'name': name,
            'path': path,
            'df': df
        })
    
    # Compute joint stratification
    hashtag_stats = compute_joint_stratification(
        datasets_info, 
        args.train_size, 
        args.val_size
    )
    
    # Split hashtags (ONCE for all datasets)
    train_hashtags, val_hashtags, test_hashtags = split_hashtags(
        hashtag_stats,
        train_size=args.train_size,
        val_size=args.val_size,
        random_state=args.random_state
    )
    
    # Apply split to each dataset
    all_metadata = []
    for info in datasets_info:
        _, _, _, metadata = apply_split_to_dataset(
            info['df'], train_hashtags, val_hashtags, test_hashtags, info['name']
        )
        all_metadata.append(metadata)
    
    # Print summary
    print_split_summary(all_metadata, args.seed_id)
    
    # Save all splits
    save_splits(
        datasets_info, 
        train_hashtags, val_hashtags, test_hashtags,
        all_metadata,
        args.output_dir, 
        args.seed_id, 
        args.random_state
    )


if __name__ == "__main__":
    main()