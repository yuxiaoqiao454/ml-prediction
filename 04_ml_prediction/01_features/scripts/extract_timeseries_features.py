#!/usr/bin/env python3
"""
Extract time series features from windowed time series data.

Uses registry pattern for modular feature extraction.
Processes both mentions and comments metrics.

Usage:
  python 04_ml_prediction/01_features/scripts/extract_timeseries_features.py --config 04_ml_prediction/01_features/configs/timeseries_features.yaml
  python extract_timeseries_features.py --limit 10 --verbose
"""

import argparse
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

# Import feature registry
from feature_registry import TIMESERIES_FEATURE_REGISTRY


def load_config(config_path):
    """Load YAML config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_features_for_window(window_row, labels_row, config, enabled_features):
    """
    Extract all features for a single window.
    
    Parameters:
    -----------
    window_row : dict-like
        Row from window_slices parquet with dates/counts
    labels_row : dict-like or None
        Corresponding row from labels (for changepoint features)
    config : dict
        Full config
    enabled_features : list
        List of feature family names to extract
    
    Returns:
    --------
    dict of features
    """
    features = {
        'hashtag': window_row['hashtag'],
        'window_end': window_row['window_end'],
    }
    
    # Process both mentions and comments
    metrics = []
    if config.get('apply_to', {}).get('mentions', True):
        metrics.append('mentions')
    if config.get('apply_to', {}).get('comments', True):
        metrics.append('comments')
    
    for metric in metrics:
        # Get data (use raw variant)
        dates = window_row.get(f'{metric}_raw_dates', [])
        counts = window_row.get(f'{metric}_raw_counts', [])
        
        if len(dates) == 0 or len(counts) == 0:
            continue
        
        # Extract each feature family
        for feature_family in enabled_features:
            if feature_family not in TIMESERIES_FEATURE_REGISTRY:
                continue
            
            extractor = TIMESERIES_FEATURE_REGISTRY[feature_family]
            
            # Get family-specific config
            family_config = config.get(feature_family, {})
            family_config['transformations'] = config.get('transformations', {})
            
            try:
                # Special handling for changepoint features (needs labels)
                if feature_family == 'changepoint':
                    family_features = extractor(
                        dates, counts, family_config, metric, 
                        label_row=labels_row
                    )
                else:
                    family_features = extractor(
                        dates, counts, family_config, metric
                    )
                
                features.update(family_features)
                
            except Exception as e:
                # Log error but continue
                if config.get('verbose', False):
                    print(f"  [warn] Failed to extract {feature_family} for {metric}: {e}")
                continue
    
    return features


def process_hashtag(hashtag_file, labels_df, config, enabled_features, verbose=False):
    """
    Process one hashtag file.
    
    Returns:
    --------
    DataFrame with features for all windows of this hashtag
    """
    hashtag = hashtag_file.stem
    
    try:
        # Load window slices
        windows_df = pd.read_parquet(hashtag_file)
        
        if len(windows_df) == 0:
            if verbose:
                print(f"  [skip] {hashtag}: No windows")
            return None
        
        # Get labels for this hashtag (for changepoint features)
        hashtag_labels = labels_df[labels_df['hashtag'] == hashtag] if labels_df is not None else pd.DataFrame()
        
        # Extract features for each window
        feature_rows = []
        for idx, window_row in windows_df.iterrows():
            # Find corresponding label row
            if len(hashtag_labels) > 0:
                label_row = hashtag_labels[
                    hashtag_labels['window_end'] == window_row['window_end']
                ]
                label_row = label_row.iloc[0].to_dict() if len(label_row) > 0 else None
            else:
                label_row = None
            
            # Extract features
            features = extract_features_for_window(
                window_row, label_row, config, enabled_features
            )
            feature_rows.append(features)
        
        result_df = pd.DataFrame(feature_rows)
        
        if verbose:
            n_features = len([c for c in result_df.columns if c.startswith('ts_')])
            print(f"  [✓] {hashtag}: {len(result_df)} windows, {n_features} features")
        
        return result_df
        
    except Exception as e:
        if verbose:
            print(f"  [✗] {hashtag}: {str(e)}")
        return None


def main():
    ap = argparse.ArgumentParser(description="Extract time series features")
    ap.add_argument("--config", 
                    default="04_ml_prediction/01_features/configs/timeseries_features.yaml",
                    help="Path to config")
    ap.add_argument("--window-slices-dir",
                    help="Override window slices directory")
    ap.add_argument("--labels",
                    help="Override labels file path")
    ap.add_argument("--output",
                    help="Override output path")
    ap.add_argument("--limit", type=int,
                    help="Process only first N hashtags")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip if output already exists")
    ap.add_argument("--verbose", action="store_true",
                    help="Print detailed progress")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print actions without executing")
    args = ap.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override paths if specified
    window_slices_dir = Path(args.window_slices_dir or config['input']['window_slices'])
    labels_path = Path(args.labels or config['input']['labels'])
    output_path = Path(args.output or config['output']['path'])
    
    # Check skip
    if args.skip_existing and output_path.exists():
        print(f"✓ Output already exists: {output_path}")
        return
    
    # Setup
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enabled_features = config.get('enabled_features', [])
    
    print(f"{'='*80}")
    print(f"Time Series Feature Extraction")
    print(f"Config: {args.config}")
    print(f"Enabled features: {', '.join(enabled_features)}")
    print(f"Output: {output_path}")
    if args.dry_run:
        print("[DRY RUN MODE]")
    print(f"{'='*80}\n")
    
    # Load labels (for changepoint features)
    labels_df = None
    if 'changepoint' in enabled_features and labels_path.exists():
        print(f"Loading labels from {labels_path}...")
        labels_df = pd.read_parquet(labels_path)
        print(f"✓ Loaded {len(labels_df)} labeled windows\n")
    
    # Find all window slice files
    slice_files = sorted(window_slices_dir.glob("*.parquet"))
    
    if args.limit:
        slice_files = slice_files[:args.limit]
    
    print(f"Processing {len(slice_files)} hashtags...\n")
    
    # Process each hashtag
    all_features = []
    success_count = 0
    fail_count = 0
    
    for slice_file in tqdm(slice_files, desc="Extracting features", disable=args.verbose):
        if args.verbose:
            print(f"\n[{slice_file.stem}]")
        
        if not args.dry_run:
            result_df = process_hashtag(
                slice_file, labels_df, config, enabled_features, args.verbose
            )
            
            if result_df is not None and len(result_df) > 0:
                all_features.append(result_df)
                success_count += 1
            else:
                fail_count += 1
        else:
            print(f"[dry-run] Would process {slice_file.stem}")
            success_count += 1
    
    # Concatenate all features
    if not args.dry_run and all_features:
        print(f"\nCombining features from {len(all_features)} hashtags...")
        final_df = pd.concat(all_features, ignore_index=True)
        
        # Save
        final_df.to_parquet(output_path, index=False)
        
        print(f"\n{'='*80}")
        print(f"✓ Saved {len(final_df)} feature vectors to {output_path}")
        print(f"\nFeature Summary:")
        print(f"  Total features: {len([c for c in final_df.columns if c.startswith('ts_')])}")
        print(f"  Sample columns: {list(final_df.columns[:10])}")
        print(f"\nStats:")
        print(f"  Hashtags processed: {success_count}")
        print(f"  Failed: {fail_count}")
        print(f"  Total windows: {len(final_df)}")
        
        # Show sample of features
        print(f"\nSample features (first row):")
        sample_features = {k: v for k, v in final_df.iloc[0].items() 
                          if k.startswith('ts_')}
        for k, v in list(sample_features.items())[:5]:
            print(f"  {k}: {v}")
        
        print(f"{'='*80}")
    
    elif args.dry_run:
        print(f"\n[DRY RUN] Would process {success_count} hashtags")


if __name__ == "__main__":
    main()