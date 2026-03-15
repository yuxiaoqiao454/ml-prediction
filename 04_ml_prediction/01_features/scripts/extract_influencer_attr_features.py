#!/usr/bin/env python3
"""
Extract influencer attribute features from network analysis windows.

Extracts features based on influencer attributes (followers, followees, posts, category)
for influencers active in each window.

Usage:
  python extract_influencer_attr_features.py --config configs/influencer_attr_features.yaml
  python extract_influencer_attr_features.py --limit 10 --verbose
"""

import argparse
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

# Import feature registry
from influencer_attr_registry import (
    INFLUENCER_ATTR_REGISTRY,
    compute_deltas
)


def load_config(config_path):
    """Load YAML config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_influencer_attributes(parquet_path):
    """
    Load influencer attributes from parquet file.
    
    Returns:
    --------
    DataFrame with columns: username, category, followers, followees, posts
    """
    df = pd.read_parquet(parquet_path)
    
    # Ensure lowercase usernames for matching
    df['username'] = df['username'].str.lower()
    
    # Set username as index for fast lookup
    df = df.set_index('username')
    
    return df


def get_influencers_in_window(window_dir):
    """
    Get list of influencer usernames active in this window.
    
    Reads from s1_bipartite/nodes.csv, filters for partition='influencer'.
    
    Returns:
    --------
    list of usernames (lowercase)
    """
    nodes_path = window_dir / "s1_bipartite" / "nodes.csv"
    
    if not nodes_path.exists():
        return []
    
    try:
        nodes_df = pd.read_csv(nodes_path)
        
        # Filter for influencers only
        influencers = nodes_df[nodes_df['partition'] == 'influencer']
        
        # Extract usernames (lowercase)
        usernames = influencers['id'].str.lower().tolist()
        
        return usernames
        
    except Exception as e:
        return []


def get_degree_data(window_dir, projection_method='count'):
    """
    Get degree data for influencers from audience projection (side B).
    
    This is OPTIONAL for Level 5 features (network-attribute correlation).
    
    Returns:
    --------
    dict mapping username -> degree, or None if unavailable
    """
    # We want influencer degrees from the AUDIENCE projection
    # because we want to see correlation between follower count and network centrality
    
    proj_dir_B = window_dir / "s2_proj" / f"{projection_method}_B_imported"
    nodes_path = proj_dir_B / "nodes.csv"
    
    if not nodes_path.exists():
        return None
    
    try:
        nodes_df = pd.read_csv(nodes_path)
        
        # Create username -> degree mapping
        degree_dict = dict(zip(
            nodes_df['id'].str.lower(),
            nodes_df['degree']
        ))
        
        return degree_dict
        
    except Exception:
        return None


def extract_features_for_window(hashtag, window_end, window_dir, influencer_attrs_all, 
                                prev_features, config, enabled_features):
    """
    Extract all influencer attribute features for a single window.
    
    Parameters:
    -----------
    hashtag : str
    window_end : str (YYYY-MM-DD)
    window_dir : Path
    influencer_attrs_all : DataFrame (indexed by username)
    prev_features : dict or None
    config : dict
    enabled_features : list
    
    Returns:
    --------
    dict of features
    """
    features = {
        'hashtag': hashtag,
        'window_end': window_end,
    }
    
    # Get influencers active in this window
    influencer_usernames = get_influencers_in_window(window_dir)
    
    if len(influencer_usernames) == 0:
        # No influencers in this window
        features['attr_has_valid_data'] = 0
        features['attr_has_prev_window'] = 0
        return features
    
    # Match to attribute data
    # Keep only influencers with valid attributes (no missing followers/followees/posts)
    matched_influencers = []
    for username in influencer_usernames:
        if username in influencer_attrs_all.index:
            row = influencer_attrs_all.loc[username]
            # Check all required fields are present and valid
            if (pd.notna(row['followers']) and 
                pd.notna(row['followees']) and 
                pd.notna(row['posts']) and
                row['followers'] >= 0 and
                row['followees'] >= 0 and
                row['posts'] >= 0):
                matched_influencers.append({
                    'username': username,
                    'category': row['category'],
                    'followers': row['followers'],
                    'followees': row['followees'],
                    'posts': row['posts']
                })
    
    # Create DataFrame of matched influencers
    influencer_attrs_df = pd.DataFrame(matched_influencers)
    
    # Check if we have valid data
    N = len(influencer_attrs_df)
    
    if N == 0:
        # No valid influencer attributes for this window
        features['attr_has_valid_data'] = 0
        features['attr_has_prev_window'] = 0
        return features
    
    # Mark as having valid data
    features['attr_has_valid_data'] = 1
    
    # ========================================================================
    # Extract features from each enabled family
    # ========================================================================
    
    for feature_family in enabled_features:
        if feature_family not in INFLUENCER_ATTR_REGISTRY:
            continue
        
        extractor = INFLUENCER_ATTR_REGISTRY[feature_family]
        family_config = config.get(feature_family, {})
        
        try:
            if feature_family == 'level5_network_interaction':
                # Level 5 needs degree data
                degree_data = get_degree_data(window_dir, config.get('projection_method', 'count'))
                if degree_data is not None:
                    family_features = extractor(influencer_attrs_df, degree_data, family_config)
                else:
                    # No degree data available
                    family_features = {
                        'attr_corr_followers_degree': np.nan,
                        'attr_corr_posts_degree': np.nan,
                    }
            else:
                # Other levels don't need extra data
                family_features = extractor(influencer_attrs_df, family_config)
            
            features.update(family_features)
            
        except Exception as e:
            # If extraction fails, continue (features will be missing/NaN)
            continue
    
    # ========================================================================
    # Temporal Deltas
    # ========================================================================
    
    has_prev_window = prev_features is not None
    features['attr_has_prev_window'] = int(has_prev_window)
    
    if has_prev_window and prev_features.get('attr_has_valid_data', 0) == 1:
        # Compute deltas
        delta_features = compute_deltas(features, prev_features)
        features.update(delta_features)
    
    return features


def process_hashtag(hashtag, network_base_dir, influencer_attrs_all, config, 
                   enabled_features, verbose=False):
    """
    Process all windows for one hashtag.
    
    Returns:
    --------
    DataFrame with features for all windows
    """
    hashtag_dir = network_base_dir / f"windows_{hashtag}"
    
    if not hashtag_dir.exists():
        if verbose:
            print(f"  [skip] {hashtag}: No network directory")
        return None
    
    # Find all window directories (YYYY-MM-DD format)
    window_dirs = []
    for item in hashtag_dir.iterdir():
        if item.is_dir() and len(item.name) == 10 and item.name[4] == '-' and item.name[7] == '-':
            window_dirs.append(item)
    
    if len(window_dirs) == 0:
        if verbose:
            print(f"  [skip] {hashtag}: No window folders")
        return None
    
    # Sort chronologically
    window_dirs = sorted(window_dirs, key=lambda x: x.name)
    
    # Extract features for each window
    feature_rows = []
    prev_features = None
    prev_window_date = None
    
    for window_dir in window_dirs:
        window_end = window_dir.name
        window_date = pd.to_datetime(window_end)
        
        # Check if this is contiguous with previous window (exactly 30 days apart)
        is_contiguous = False
        if prev_window_date is not None:
            gap_days = (window_date - prev_window_date).days
            is_contiguous = (gap_days == 30)
        
        # If not contiguous, reset prev_features
        if not is_contiguous:
            prev_features = None
        
        try:
            features = extract_features_for_window(
                hashtag, window_end, window_dir, influencer_attrs_all,
                prev_features, config, enabled_features
            )
            
            feature_rows.append(features)
            
            # Update prev_features only if current window has valid data
            if features.get('attr_has_valid_data', 0) == 1:
                prev_features = features
            else:
                prev_features = None  # Don't use invalid windows for deltas
            
            prev_window_date = window_date
            
        except Exception as e:
            if verbose:
                print(f"  [warn] {hashtag}/{window_end}: {str(e)}")
            continue
    
    if len(feature_rows) == 0:
        return None
    
    result_df = pd.DataFrame(feature_rows)
    
    if verbose:
        n_valid = result_df['attr_has_valid_data'].sum()
        n_with_deltas = result_df['attr_has_prev_window'].sum()
        n_features = len([c for c in result_df.columns if c.startswith('attr_')])
        print(f"  [✓] {hashtag}: {len(result_df)} windows, {n_valid} valid, {n_with_deltas} with deltas, {n_features} features")
    
    return result_df


def main():
    ap = argparse.ArgumentParser(description="Extract influencer attribute features")
    ap.add_argument("--config",
                    default="04_ml_prediction/01_features/configs/influencer_attr_features.yaml",
                    help="Path to config")
    ap.add_argument("--influencer-attrs",
                    help="Override path to influencer attributes parquet")
    ap.add_argument("--network-dir",
                    help="Override network windows directory")
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
    
    # Paths
    influencer_attrs_path = Path(args.influencer_attrs or config['input']['influencer_attributes'])
    network_base_dir = Path(args.network_dir or config['input']['network_windows'])
    output_path = Path(args.output or config['output']['path'])
    
    # Check skip
    if args.skip_existing and output_path.exists():
        print(f"✓ Output already exists: {output_path}")
        return
    
    # Setup
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enabled_features = config.get('enabled_features', [])
    
    print(f"{'='*80}")
    print(f"Influencer Attribute Feature Extraction")
    print(f"Config: {args.config}")
    print(f"Enabled features: {', '.join(enabled_features)}")
    print(f"Projection method: {config.get('projection_method', 'count')}")
    print(f"Output: {output_path}")
    if args.dry_run:
        print("[DRY RUN MODE]")
    print(f"{'='*80}\n")
    
    # Load influencer attributes
    print(f"Loading influencer attributes from {influencer_attrs_path}...")
    influencer_attrs_all = load_influencer_attributes(influencer_attrs_path)
    print(f"✓ Loaded {len(influencer_attrs_all)} influencers\n")
    
    # Find all hashtag directories
    hashtag_dirs = [d for d in network_base_dir.iterdir() 
                   if d.is_dir() and d.name.startswith('windows_')]
    hashtags = [d.name.replace('windows_', '') for d in hashtag_dirs]
    hashtags = sorted(hashtags)
    
    if args.limit:
        hashtags = hashtags[:args.limit]
    
    print(f"Processing {len(hashtags)} hashtags...\n")
    
    # Process each hashtag
    all_features = []
    success_count = 0
    fail_count = 0
    no_valid_influencers_count = 0
    
    for hashtag in tqdm(hashtags, desc="Extracting influencer features", disable=args.verbose):
        if args.verbose:
            print(f"\n[{hashtag}]")
        
        if not args.dry_run:
            result_df = process_hashtag(
                hashtag, network_base_dir, influencer_attrs_all, config,
                enabled_features, args.verbose
            )
            
            if result_df is not None and len(result_df) > 0:
                all_features.append(result_df)
                success_count += 1
                
                # Count windows with no valid influencers
                no_valid = (result_df['attr_has_valid_data'] == 0).sum()
                no_valid_influencers_count += no_valid
            else:
                fail_count += 1
        else:
            print(f"[dry-run] Would process {hashtag}")
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
        print(f"  Total influencer attribute features: {len([c for c in final_df.columns if c.startswith('attr_')])}")
        print(f"\nStats:")
        print(f"  Hashtags processed: {success_count}")
        print(f"  Failed: {fail_count}")
        print(f"  Total windows: {len(final_df)}")
        print(f"  Windows with valid data: {final_df['attr_has_valid_data'].sum()}")
        print(f"  Windows without valid influencers: {no_valid_influencers_count}")
        print(f"  Windows with prev window: {final_df['attr_has_prev_window'].sum()}")
        
        # Show sample
        print(f"\nSample features (first valid window):")
        valid_rows = final_df[final_df['attr_has_valid_data'] == 1]
        if len(valid_rows) > 0:
            sample_features = {k: v for k, v in valid_rows.iloc[0].items() 
                              if k.startswith('attr_') and not k.startswith('attr_has') and not k.startswith('attr_delta')}
            for k, v in list(sample_features.items())[:10]:
                print(f"  {k}: {v}")
        
        print(f"{'='*80}")
        
        # Generate summary CSV
        summary_path = output_path.parent / "influencer_attribute_feature_extraction_summary.csv"
        summary_data = []
        
        for hashtag in hashtags[:len(all_features)]:
            hashtag_df = final_df[final_df['hashtag'] == hashtag]
            summary_data.append({
                'hashtag': hashtag,
                'n_windows_total': len(hashtag_df),
                'n_windows_no_valid_influencers': (hashtag_df['attr_has_valid_data'] == 0).sum(),
                'n_windows_with_deltas': (hashtag_df['attr_has_prev_window'] == 1).sum(),
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✓ Summary saved to {summary_path}")
    
    elif args.dry_run:
        print(f"\n[DRY RUN] Would process {success_count} hashtags")


if __name__ == "__main__":
    main()