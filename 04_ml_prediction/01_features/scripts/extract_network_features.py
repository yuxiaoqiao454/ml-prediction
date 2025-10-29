#!/usr/bin/env python3
"""
Extract network features from network analysis windows.

Extracts features from:
- Bipartite networks
- Projected networks (influencer and audience sides)
- Clustered networks (with community structure)

Usage:
  python extract_network_features.py --config configs/network_features.yaml
  python extract_network_features.py --limit 10 --verbose
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
from network_feature_registry import NETWORK_FEATURE_REGISTRY, load_graph_from_edges


def load_config(config_path):
    """Load YAML config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_features_for_window(hashtag, window_end, window_dir, prev_features, config, enabled_features):
    """
    Extract all network features for a single window.
    
    Parameters:
    -----------
    hashtag : str
    window_end : str (YYYY-MM-DD)
    window_dir : Path
        Directory containing s1_bipartite, s2_proj, s3_cluster folders
    prev_features : dict or None
        Features from previous window (for temporal deltas)
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
    
    proj_method = config.get('projection_method', 'count')
    cluster_method = config.get('clustering_method', 'infomap')
    
    # Track what exists
    has_influencer_proj = False
    has_audience_proj = False
    
    # ========================================================================
    # 1. Bipartite Network
    # ========================================================================
    if config.get('compute_on', {}).get('bipartite', True):
        bip_edges = window_dir / "s1_bipartite" / "edges.csv"
        G_bip = load_graph_from_edges(bip_edges, directed=False)
        
        if G_bip is not None:
            for feature_family in enabled_features:
                if feature_family == 'temporal_delta':
                    continue  # Skip temporal for bipartite
                
                if feature_family not in NETWORK_FEATURE_REGISTRY:
                    continue
                
                extractor = NETWORK_FEATURE_REGISTRY[feature_family]
                family_config = config.get(feature_family, {})
                
                try:
                    if feature_family == 'community_basic':
                        # Bipartite doesn't have clustering
                        continue
                    else:
                        family_features = extractor(G_bip, family_config, 'bipartite')
                    
                    features.update(family_features)
                except Exception:
                    continue
    
    # ========================================================================
    # 2. Influencer Projection (Side A)
    # ========================================================================
    if config.get('compute_on', {}).get('projection_influencer', True):
        proj_dir_A = window_dir / "s2_proj" / f"{proj_method}_A_imported"
        edges_A = proj_dir_A / "edges.csv"
        G_A = load_graph_from_edges(edges_A, directed=False)
        
        if G_A is not None:
            has_influencer_proj = True
            
            for feature_family in enabled_features:
                if feature_family == 'temporal_delta':
                    continue
                
                if feature_family not in NETWORK_FEATURE_REGISTRY:
                    continue
                
                extractor = NETWORK_FEATURE_REGISTRY[feature_family]
                family_config = config.get(feature_family, {})
                
                try:
                    if feature_family == 'community_basic':
                        # Need clustering labels
                        cluster_dir = window_dir / "s3_cluster" / f"{proj_method}_A_imported" / cluster_method
                        labels_path = cluster_dir / "labels.csv"
                        family_features = extractor(G_A, labels_path, family_config, 'influencer')
                    else:
                        family_features = extractor(G_A, family_config, 'influencer')
                    
                    features.update(family_features)
                except Exception:
                    continue
    
    features['net_has_influencer_proj'] = int(has_influencer_proj)
    
    # ========================================================================
    # 3. Audience Projection (Side B)
    # ========================================================================
    if config.get('compute_on', {}).get('projection_audience', True):
        proj_dir_B = window_dir / "s2_proj" / f"{proj_method}_B_imported"
        edges_B = proj_dir_B / "edges.csv"
        G_B = load_graph_from_edges(edges_B, directed=False)
        
        if G_B is not None:
            has_audience_proj = True
            
            for feature_family in enabled_features:
                if feature_family == 'temporal_delta':
                    continue
                
                if feature_family not in NETWORK_FEATURE_REGISTRY:
                    continue
                
                extractor = NETWORK_FEATURE_REGISTRY[feature_family]
                family_config = config.get(feature_family, {})
                
                try:
                    if feature_family == 'community_basic':
                        cluster_dir = window_dir / "s3_cluster" / f"{proj_method}_B_imported" / cluster_method
                        labels_path = cluster_dir / "labels.csv"
                        family_features = extractor(G_B, labels_path, family_config, 'audience')
                    else:
                        family_features = extractor(G_B, family_config, 'audience')
                    
                    features.update(family_features)
                except Exception:
                    continue
    
    features['net_has_audience_proj'] = int(has_audience_proj)
    
    # ========================================================================
    # 4. Temporal Deltas (vs previous window)
    # ========================================================================
    has_prev_window = prev_features is not None
    features['net_has_prev_window'] = int(has_prev_window)
    
    if 'temporal_delta' in enabled_features and has_prev_window:
        delta_config = config.get('temporal_delta', {})
        extractor = NETWORK_FEATURE_REGISTRY['temporal_delta']
        
        # Compute deltas for each graph type
        for graph_name in ['bipartite', 'influencer', 'audience']:
            try:
                delta_features = extractor(features, prev_features, delta_config, graph_name)
                features.update(delta_features)
            except Exception:
                continue
    
    return features


def process_hashtag(hashtag, network_base_dir, config, enabled_features, verbose=False):
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
    
    for window_dir in window_dirs:
        window_end = window_dir.name
        
        try:
            features = extract_features_for_window(
                hashtag, window_end, window_dir, prev_features, config, enabled_features
            )
            
            feature_rows.append(features)
            prev_features = features  # Save for next window
            
        except Exception as e:
            if verbose:
                print(f"  [warn] {hashtag}/{window_end}: {str(e)}")
            continue
    
    if len(feature_rows) == 0:
        return None
    
    result_df = pd.DataFrame(feature_rows)
    
    if verbose:
        n_features = len([c for c in result_df.columns if c.startswith('net_')])
        print(f"  [✓] {hashtag}: {len(result_df)} windows, {n_features} features")
    
    return result_df


def main():
    ap = argparse.ArgumentParser(description="Extract network features")
    ap.add_argument("--config",
                    default="04_ml_prediction/01_features/configs/network_features.yaml",
                    help="Path to config")
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
    print(f"Network Feature Extraction")
    print(f"Config: {args.config}")
    print(f"Enabled features: {', '.join(enabled_features)}")
    print(f"Projection: {config.get('projection_method')}")
    print(f"Clustering: {config.get('clustering_method')}")
    print(f"Output: {output_path}")
    if args.dry_run:
        print("[DRY RUN MODE]")
    print(f"{'='*80}\n")
    
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
    
    for hashtag in tqdm(hashtags, desc="Extracting network features", disable=args.verbose):
        if args.verbose:
            print(f"\n[{hashtag}]")
        
        if not args.dry_run:
            result_df = process_hashtag(
                hashtag, network_base_dir, config, enabled_features, args.verbose
            )
            
            if result_df is not None and len(result_df) > 0:
                all_features.append(result_df)
                success_count += 1
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
        print(f"  Total network features: {len([c for c in final_df.columns if c.startswith('net_')])}")
        print(f"\nStats:")
        print(f"  Hashtags processed: {success_count}")
        print(f"  Failed: {fail_count}")
        print(f"  Total windows: {len(final_df)}")
        
        # Check missing data
        print(f"\nMissing Data Summary:")
        print(f"  Windows without influencer proj: {(final_df['net_has_influencer_proj']==0).sum()}")
        print(f"  Windows without audience proj: {(final_df['net_has_audience_proj']==0).sum()}")
        print(f"  Windows without prev window: {(final_df['net_has_prev_window']==0).sum()}")
        
        # Show sample
        print(f"\nSample features (first row):")
        sample_features = {k: v for k, v in final_df.iloc[0].items() 
                          if k.startswith('net_')}
        for k, v in list(sample_features.items())[:5]:
            print(f"  {k}: {v}")
        
        print(f"{'='*80}")
    
    elif args.dry_run:
        print(f"\n[DRY RUN] Would process {success_count} hashtags")


if __name__ == "__main__":
    main()