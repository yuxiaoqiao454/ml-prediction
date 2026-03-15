#!/usr/bin/env python3
"""
Build training samples by joining features and labels.

NOW SUPPORTS: Timeseries + Network + Embedding features!

Creates samples where:
- Features come from window T (timeseries + network + embeddings)
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
    Load timeseries features, network features, embeddings, and labels.
    
    Returns:
    --------
    ts_features, net_features, emb_features, labels (all DataFrames)
    """
    print("Loading input data...")
    
    ts_path = Path(config['input']['timeseries_features'])
    net_path = Path(config['input']['network_features'])
    emb_path = Path(config['input'].get('embedding_features', ''))  # Optional
    labels_path = Path(config['input']['labels'])
    
    # Load required data
    ts_features = pd.read_parquet(ts_path)
    net_features = pd.read_parquet(net_path)
    labels = pd.read_parquet(labels_path)
    
    # Load embeddings if path exists and is enabled
    use_emb = config['features'].get('use_embeddings', False)
    if use_emb and emb_path and Path(emb_path).exists():
        emb_features = pd.read_parquet(emb_path)
        print(f"✓ Loaded {len(emb_features)} embedding feature vectors")
    else:
        emb_features = None
        if use_emb:
            print(f"⚠️  Embedding features requested but not found: {emb_path}")
    
    # Ensure date columns are strings for matching
    for df in [ts_features, net_features, labels]:
        if 'window_end' in df.columns:
            df['window_end'] = df['window_end'].astype(str)
        if 'window_start' in df.columns:
            df['window_start'] = df['window_start'].astype(str)
    
    if emb_features is not None:
        if 'window_end' in emb_features.columns:
            emb_features['window_end'] = emb_features['window_end'].astype(str)
        if 'window_start' in emb_features.columns:
            emb_features['window_start'] = emb_features['window_start'].astype(str)
    
    print(f"✓ Loaded {len(ts_features)} timeseries feature vectors")
    print(f"✓ Loaded {len(net_features)} network feature vectors")
    print(f"✓ Loaded {len(labels)} labeled windows\n")
    
    return ts_features, net_features, emb_features, labels


def build_samples(ts_features, net_features, emb_features, labels, config, verbose=False):
    """
    Build training samples by joining features and labels.
    
    Logic:
    1. For each window T (features), find window T+1 (label)
    2. Check if T→T+1 are contiguous (30 days apart)
    3. Join timeseries + network + embedding features from T
    4. Attach label from T+1
    
    Returns:
    --------
    DataFrame with all samples
    """
    gap_days = config['building'].get('gap_days', 30)
    require_contiguous = config['building'].get('require_contiguous', True)
    target_label = config['building'].get('target_label', 'label_burst_comments')
    
    # Get feature type toggles
    use_ts = config['features'].get('use_timeseries', True)
    use_net = config['features'].get('use_network', True)
    use_emb = config['features'].get('use_embeddings', False)
    
    print("Building samples...")
    print(f"  Contiguity requirement: {gap_days}-day gap")
    print(f"  Target label: {target_label}")
    print(f"  Feature types: TS={use_ts}, Network={use_net}, Embeddings={use_emb}\n")
    
    # Start with one feature source as base
    if use_ts:
        features = ts_features.copy()
        print(f"✓ Starting with {len(features)} timeseries features")
    elif use_net:
        features = net_features.copy()
        print(f"✓ Starting with {len(features)} network features")
    elif use_emb and emb_features is not None:
        features = emb_features.copy()
        print(f"✓ Starting with {len(features)} embedding features")
    else:
        raise ValueError("At least one feature type must be enabled!")
    
    # Merge additional feature types
    if use_net and not use_ts:
        # Network already loaded as base
        pass
    elif use_net:
        features = features.merge(
            net_features,
            on=['hashtag', 'window_end'],
            how='inner',
            suffixes=('', '_net_dup')
        )
        print(f"✓ Merged network features: {len(features)} windows")
    
    if use_emb and emb_features is not None:
        features = features.merge(
            emb_features,
            on=['hashtag', 'window_end'],
            how='inner',
            suffixes=('', '_emb_dup')
        )
        print(f"✓ Merged embedding features: {len(features)} windows")
    
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
                   if c.startswith('ts_') or c.startswith('net_') or c.startswith('emb_')]
    
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


def filter_features(samples_df, config):
    """
    Filter feature columns based on patterns in config.
    
    Returns DataFrame with only selected features + metadata columns.
    """
    include_patterns = config['features'].get('include_patterns', [])
    exclude_patterns = config['features'].get('exclude_patterns', [])
    
    if not include_patterns and not exclude_patterns:
        # No filtering - return all features
        return samples_df
    
    # Identify feature columns (ts_*, net_*, emb_*)
    feature_cols = [c for c in samples_df.columns 
                   if c.startswith('ts_') or c.startswith('net_') or c.startswith('emb_')]
    
    # Metadata columns to always keep
    metadata_cols = [c for c in samples_df.columns 
                    if not (c.startswith('ts_') or c.startswith('net_') or c.startswith('emb_'))]
    
    # Apply include patterns (if specified)
    if include_patterns:
        import re
        selected_features = []
        for pattern in include_patterns:
            regex = re.compile(pattern)
            matched = [c for c in feature_cols if regex.search(c)]
            selected_features.extend(matched)
        selected_features = list(set(selected_features))  # Remove duplicates
    else:
        selected_features = feature_cols
    
    # Apply exclude patterns (if specified)
    if exclude_patterns:
        import re
        for pattern in exclude_patterns:
            regex = re.compile(pattern)
            selected_features = [c for c in selected_features if not regex.search(c)]
    
    # Final column list
    final_cols = metadata_cols + selected_features
    
    print(f"\nFeature filtering:")
    print(f"  Original features: {len(feature_cols)}")
    print(f"  Selected features: {len(selected_features)}")
    if include_patterns:
        print(f"  Include patterns: {include_patterns}")
    if exclude_patterns:
        print(f"  Exclude patterns: {exclude_patterns}")
    
    return samples_df[final_cols]


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
                   if c.startswith('ts_') or c.startswith('net_') or c.startswith('emb_')]
    
    # Calculate statistics
    burst_rate = samples_df['label'].mean()
    n_hashtags = samples_df['hashtag'].nunique()
    
    # Count feature types
    n_ts = len([c for c in feature_cols if c.startswith('ts_')])
    n_net = len([c for c in feature_cols if c.startswith('net_')])
    n_emb = len([c for c in feature_cols if c.startswith('emb_')])
    
    version_info = {
        'version': version,
        'date_created': datetime.now().strftime('%Y-%m-%d'),
        'n_hashtags': n_hashtags,
        'n_samples': len(samples_df),
        'n_features': len(feature_cols),
        'n_ts_features': n_ts,
        'n_net_features': n_net,
        'n_emb_features': n_emb,
        'burst_rate': f"{burst_rate:.3f}",
        'n_complete': samples_df['has_all_data'].sum(),
        'notes': f"{config['building']['target_label']} | TS={n_ts} NET={n_net} EMB={n_emb}"
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
    print(f"  - Timeseries: {n_ts}")
    print(f"  - Network: {n_net}")
    print(f"  - Embeddings: {n_emb}")
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
    ts_features, net_features, emb_features, labels = load_all_data(config)
    
    # Build samples
    samples_df = build_samples(ts_features, net_features, emb_features, labels, config, args.verbose)
    
    if len(samples_df) == 0:
        print("❌ No samples created! Check your data and config.")
        return
    
    # Add metadata
    samples_df = add_metadata(samples_df, config)

    # Filter features based on config  
    samples_df = filter_features(samples_df, config)  
    
    # Save
    save_outputs(samples_df, config, args.verbose)


if __name__ == "__main__":
    main()