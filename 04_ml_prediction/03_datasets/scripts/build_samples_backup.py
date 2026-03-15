#!/usr/bin/env python3
"""
Build training samples by joining features and labels.

Creates samples where:
- Features come from window T (timeseries + network)
- Label comes from window T+1 (next contiguous window)

Usage:
  python 04_ml_prediction/03_datasets/scripts/build_samples.py --config 04_ml_prediction/03_datasets/configs/dataset_config.yaml
  python build_samples.py --verbose --dry-run
"""

import argparse
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root))


def load_config(config_path):
    """Load YAML config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_all_data(config):
    """
    Load timeseries features, network features, and labels.
    
    Returns:
    --------
    ts_features, net_features, labels (all DataFrames)
    """
    print("Loading input data...")
    
    ts_path = Path(config['input']['timeseries_features'])
    net_path = Path(config['input']['network_features'])
    labels_path = Path(config['input']['labels'])
    
    ts_features = pd.read_parquet(ts_path)
    net_features = pd.read_parquet(net_path)
    labels = pd.read_parquet(labels_path)
    
    # Ensure date columns are strings for matching
    for df in [ts_features, net_features, labels]:
        if 'window_end' in df.columns:
            df['window_end'] = df['window_end'].astype(str)
        if 'window_start' in df.columns:
            df['window_start'] = df['window_start'].astype(str)
    
    print(f"✓ Loaded {len(ts_features)} timeseries feature vectors")
    print(f"✓ Loaded {len(net_features)} network feature vectors")
    print(f"✓ Loaded {len(labels)} labeled windows\n")
    
    return ts_features, net_features, labels


def build_samples(ts_features, net_features, labels, config, verbose=False):
    """
    Build training samples by joining features and labels.
    
    Logic:
    1. For each window T (features), find window T+1 (label)
    2. Check if T→T+1 are contiguous (30 days apart)
    3. Join timeseries features + network features from T
    4. Attach label from T+1
    
    Returns:
    --------
    DataFrame with all samples
    """
    gap_days = config['building'].get('gap_days', 30)
    require_contiguous = config['building'].get('require_contiguous', True)
    target_label = config['building'].get('target_label', 'label_burst_comments')
    
    print("Building samples...")
    print(f"  Contiguity requirement: {gap_days}-day gap")
    print(f"  Target label: {target_label}\n")
    
    # Merge timeseries + network features on (hashtag, window_end)
    features = ts_features.merge(
        net_features,
        on=['hashtag', 'window_end'],
        how='inner',
        suffixes=('_ts', '_net')
    )
    
    print(f"✓ Merged features: {len(features)} windows with both TS and network features")
    
    # For each feature window, find its label window (next contiguous window)
    samples = []
    
    for hashtag in features['hashtag'].unique():
        hashtag_features = features[features['hashtag'] == hashtag].copy()
        hashtag_labels = labels[labels['hashtag'] == hashtag].copy()
        
        # Sort by date
        hashtag_features = hashtag_features.sort_values('window_end')
        hashtag_labels = hashtag_labels.sort_values('window_end')
        
        for idx, feature_row in hashtag_features.iterrows():
            window_T = feature_row['window_end']
            window_T_date = pd.to_datetime(window_T)
            
            # Expected next window
            expected_T1_date = window_T_date + timedelta(days=gap_days)
            expected_T1 = expected_T1_date.strftime('%Y-%m-%d')
            
            # Find actual next window in labels
            next_labels = hashtag_labels[hashtag_labels['window_end'] > window_T]
            
            if len(next_labels) == 0:
                # No next window
                continue
            
            # Get the immediate next window
            actual_T1 = next_labels.iloc[0]['window_end']
            actual_T1_date = pd.to_datetime(actual_T1)
            
            # Check contiguity
            gap = (actual_T1_date - window_T_date).days
            is_contiguous = (gap == gap_days)
            
            if require_contiguous and not is_contiguous:
                # Skip non-contiguous pairs
                if verbose:
                    print(f"  [skip] {hashtag}/{window_T}: gap={gap} days (expected {gap_days})")
                continue
            
            # Get label
            label_row = next_labels.iloc[0]
            label_value = label_row.get(target_label)
            
            if pd.isna(label_value):
                # No valid label
                if verbose:
                    print(f"  [skip] {hashtag}/{window_T}: label is NA")
                continue
            
            # Build sample
            sample = {
                'hashtag': hashtag,
                'window_end': window_T,  # Feature window (T)
                'next_window_end': actual_T1,  # Label window (T+1)
                'is_contiguous': is_contiguous,
                'gap_days': gap,
            }
            
            # Add all features from T
            feature_cols = [c for c in feature_row.index 
                          if c not in ['hashtag', 'window_end', 'window_start']]
            for col in feature_cols:
                sample[col] = feature_row[col]
            
            # Add label from T+1
            sample['label'] = int(label_value)
            
            samples.append(sample)
    
    samples_df = pd.DataFrame(samples)
    
    print(f"✓ Built {len(samples_df)} samples")
    
    return samples_df


def add_metadata(samples_df, config):
    """
    Add metadata columns: dataset_version, created_at, has_all_data.
    """
    version = config['output'].get('dataset_version', 'v1')
    timestamp = datetime.now().isoformat()
    
    samples_df['dataset_version'] = version
    samples_df['created_at'] = timestamp
    
    # Check data completeness (how many features are non-NA)
    feature_cols = [c for c in samples_df.columns 
                   if c.startswith('ts_') or c.startswith('net_')]
    
    def check_completeness(row):
        n_features = len(feature_cols)
        n_present = row[feature_cols].notna().sum()
        coverage = n_present / n_features if n_features > 0 else 0
        return coverage
    
    samples_df['feature_coverage'] = samples_df.apply(check_completeness, axis=1)
    samples_df['has_all_data'] = samples_df['feature_coverage'] >= 0.95  # 95% threshold
    
    print(f"\nMetadata added:")
    print(f"  Dataset version: {version}")
    print(f"  Created: {timestamp}")
    print(f"  Samples with complete data: {samples_df['has_all_data'].sum()} / {len(samples_df)}")
    
    return samples_df


def save_outputs(samples_df, config, verbose=False):
    """
    Save samples parquet and update dataset_versions.csv.
    """
    output_dir = Path(config['output']['samples_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    version = config['output']['dataset_version']
    
    # Save samples
    samples_path = output_dir / f"samples_full_{version}.parquet"
    samples_df.to_parquet(samples_path, index=False)
    print(f"\n✓ Saved samples to {samples_path}")
    
    # Update dataset versions tracking
    versions_csv = output_dir / "dataset_versions.csv"
    
    # Get feature columns
    feature_cols = [c for c in samples_df.columns 
                   if c.startswith('ts_') or c.startswith('net_')]
    
    # Calculate statistics
    burst_rate = samples_df['label'].mean()
    n_hashtags = samples_df['hashtag'].nunique()
    
    version_info = {
        'version': version,
        'date_created': datetime.now().strftime('%Y-%m-%d'),
        'n_hashtags': n_hashtags,
        'n_samples': len(samples_df),
        'n_features': len(feature_cols),
        'burst_rate': f"{burst_rate:.3f}",
        'n_complete': samples_df['has_all_data'].sum(),
        'notes': f"Initial dataset - {config['building']['target_label']}"
    }
    
    # Append to tracking file
    if versions_csv.exists():
        versions_df = pd.read_csv(versions_csv)
        # Check if version already exists
        if version in versions_df['version'].values:
            print(f"\n⚠️  Warning: Version {version} already exists in tracking file")
            versions_df = versions_df[versions_df['version'] != version]
        versions_df = pd.concat([versions_df, pd.DataFrame([version_info])], ignore_index=True)
    else:
        versions_df = pd.DataFrame([version_info])
    
    versions_df.to_csv(versions_csv, index=False)
    print(f"✓ Updated {versions_csv}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Dataset Summary (version {version})")
    print(f"{'='*80}")
    print(f"Samples: {len(samples_df)}")
    print(f"Hashtags: {n_hashtags}")
    print(f"Features: {len(feature_cols)}")
    print(f"  - Timeseries: {len([c for c in feature_cols if c.startswith('ts_')])}")
    print(f"  - Network: {len([c for c in feature_cols if c.startswith('net_')])}")
    print(f"Burst rate: {burst_rate:.1%}")
    print(f"Complete samples: {samples_df['has_all_data'].sum()} ({samples_df['has_all_data'].mean():.1%})")
    print(f"{'='*80}")


def main():
    ap = argparse.ArgumentParser(description="Build training samples from features and labels")
    ap.add_argument("--config",
                    default="04_ml_prediction/03_datasets/configs/dataset_config.yaml",
                    help="Path to config")
    ap.add_argument("--verbose", action="store_true",
                    help="Print detailed progress")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print actions without executing")
    args = ap.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    print(f"{'='*80}")
    print(f"Building Training Samples")
    print(f"Config: {args.config}")
    print(f"Target: {config['building']['target_label']}")
    if args.dry_run:
        print("[DRY RUN MODE]")
    print(f"{'='*80}\n")
    
    if args.dry_run:
        print("[DRY RUN] Would load data, build samples, and save outputs")
        return
    
    # Load data
    ts_features, net_features, labels = load_all_data(config)
    
    # Build samples
    samples_df = build_samples(ts_features, net_features, labels, config, args.verbose)
    
    if len(samples_df) == 0:
        print("No samples created! Check your data and config.")
        return
    
    # Add metadata
    samples_df = add_metadata(samples_df, config)
    
    # Save
    save_outputs(samples_df, config, args.verbose)


if __name__ == "__main__":
    main()