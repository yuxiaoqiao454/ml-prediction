#!/usr/bin/env python3
"""
Extract GAE embedding features from influencer projection with node attributes.

Graph: count_A_imported (influencer projection)
Node features: [followers, followees, posts, audience_size, first_post_time, has_profile]
"""

import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# Import feature extraction functions
from gae_infproj_attr_feature_registry import (
    extract_gae_infproj_attr_features,
    compute_delta_features
)


def load_config(config_path: str) -> dict:
    """Load YAML config with type conversions."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Type conversions
    cfg['computation']['min_graph_size'] = int(cfg['computation'].get('min_graph_size', 10))
    cfg['computation']['max_graph_size'] = int(cfg['computation'].get('max_graph_size', 10000))
    cfg['computation']['hidden_dim'] = int(cfg['computation'].get('hidden_dim', 32))
    cfg['computation']['embedding_dim'] = int(cfg['computation'].get('embedding_dim', 16))
    cfg['computation']['epochs'] = int(cfg['computation'].get('epochs', 200))
    cfg['computation']['learning_rate'] = float(cfg['computation'].get('learning_rate', 0.01))
    cfg['computation']['dropout'] = float(cfg['computation'].get('dropout', 0.0))
    
    # Feature toggles
    for key in ['tier1_core', 'tier1_pca', 'tier1_entropy', 'tier2_extended', 
                'tier3_advanced', 'context', 'temporal_deltas']:
        cfg['features'][key] = bool(cfg['features'].get(key, True))
    
    return cfg


def discover_all_hashtags(base_dir: Path) -> List[str]:
    """Find all hashtags with network windows."""
    hashtags = []
    for folder in sorted(base_dir.iterdir()):
        if folder.is_dir() and folder.name.startswith('windows_'):
            hashtag = folder.name.replace('windows_', '')
            hashtags.append(hashtag)
    return hashtags


def discover_hashtag_windows(hashtag: str, base_dir: Path) -> List[str]:
    """Find all window folders for a hashtag (YYYY-MM-DD format)."""
    windows_dir = base_dir / f"windows_{hashtag}"
    if not windows_dir.exists():
        return []
    
    windows = []
    for folder in sorted(windows_dir.iterdir()):
        if folder.is_dir() and len(folder.name) == 10 and folder.name.count('-') == 2:
            windows.append(folder.name)
    return windows


def load_projection_edges(edges_path: Path) -> pd.DataFrame:
    """Load projection edges from CSV."""
    if not edges_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(edges_path)
        return df
    except:
        return pd.DataFrame()


def load_bipartite_edges(edges_path: Path) -> pd.DataFrame:
    """Load bipartite edges from CSV."""
    if not edges_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(edges_path)
        return df
    except:
        return pd.DataFrame()


def process_hashtag_windows(
    hashtag: str,
    windows: List[str],
    base_dir: Path,
    influencer_attrs_path: Path,
    config: dict,
    verbose: bool = False
) -> List[Dict]:
    """
    Process all windows for a hashtag and extract GAE features.
    
    Returns list of feature dicts (one per window).
    """
    results = []
    prev_features = None
    
    for i, window_end in enumerate(windows):
        if verbose:
            print(f"    [{i+1}/{len(windows)}] Window {window_end}:")
        
        # Build paths
        proj_path = base_dir / f"windows_{hashtag}" / window_end / "s2_proj" / "count_A_imported" / "edges.csv"
        bip_path = base_dir / f"windows_{hashtag}" / window_end / "s1_bipartite" / "edges.csv"
        
        # Check if projection exists
        if not proj_path.exists():
            if verbose:
                print(f"      [SKIP] No projection found")
            prev_features = None  # Reset temporal chain
            continue
        
        # Extract features
        features, metadata = extract_gae_infproj_attr_features(
            projection_edges_path=proj_path,
            bipartite_edges_path=bip_path,
            influencer_attrs_path=influencer_attrs_path,
            config=config['computation'],
            verbose=verbose
        )
        
        # Compute temporal deltas
        if config['features'].get('temporal_deltas', True) and prev_features is not None:
            delta_features = compute_delta_features(features, prev_features)
            features.update(delta_features)
            features['gae_has_prev_window'] = 1
        else:
            features['gae_has_prev_window'] = 0
        
        # Add flags
        features['gae_too_small'] = metadata.get('gae_too_small', 0)
        features['gae_train_failed'] = metadata.get('gae_train_failed', 0)
        
        # Store result
        result = {
            'hashtag': hashtag,
            'window_end': window_end,
            **features
        }
        results.append(result)
        
        # Update prev_features for next iteration
        if metadata.get('success', False):
            prev_features = features.copy()
        else:
            prev_features = None
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract GAE features from influencer projection with attributes"
    )
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--limit', type=int, help='Limit to N hashtags')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--skip-existing', action='store_true', help='Skip if output exists')
    parser.add_argument('--dry-run', action='store_true', help='Print plan without execution')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    base_dir = Path(config['input']['network_windows_base'])
    influencer_attrs_path = Path(config['input']['influencer_attributes'])
    output_file = Path(config['output']['features_file'])
    summary_file = Path(config['output']['summary_file'])
    
    # Check influencer attributes exist
    if not influencer_attrs_path.exists():
        print(f"[ERROR] Influencer attributes not found: {influencer_attrs_path}")
        return
    
    # Skip if exists
    if args.skip_existing and output_file.exists():
        print(f"✓ Output exists, skipping: {output_file}")
        return
    
    # Header
    print("=" * 80)
    print("GAE Influencer Projection + Attributes Feature Extraction")
    print(f"Config: {args.config}")
    print(f"Output: {output_file}")
    print("=" * 80)
    
    # Discover hashtags
    print("Discovering hashtags...")
    hashtags = discover_all_hashtags(base_dir)
    
    if args.limit:
        hashtags = hashtags[:args.limit]
        print(f"  Limited to {args.limit} hashtags")
    
    print(f"✓ Found {len(hashtags)} hashtags to process")
    
    # Show enabled features
    print("\nEnabled feature tiers:")
    for key, val in config['features'].items():
        if val:
            print(f"  ✓ {key}")
    
    # Show GAE config
    print("\nGAE Configuration:")
    print(f"  Hidden dim: {config['computation']['hidden_dim']}")
    print(f"  Embedding dim: {config['computation']['embedding_dim']}")
    print(f"  Epochs: {config['computation']['epochs']}")
    print(f"  Learning rate: {config['computation']['learning_rate']}")
    print(f"  Max graph size: {config['computation']['max_graph_size']}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would process these hashtags:")
        for tag in hashtags[:10]:
            print(f"  - {tag}")
        if len(hashtags) > 10:
            print(f"  ... and {len(hashtags) - 10} more")
        return
    
    # Process each hashtag
    all_results = []
    failed_hashtags = []
    
    for hashtag in hashtags:
        print(f"\n[{hashtag}]")
        
        try:
            # Discover windows
            windows = discover_hashtag_windows(hashtag, base_dir)
            if len(windows) == 0:
                print(f"  No windows found, skipping")
                continue
            
            print(f"  Found {len(windows)} windows")
            
            # Process windows
            results = process_hashtag_windows(
                hashtag=hashtag,
                windows=windows,
                base_dir=base_dir,
                influencer_attrs_path=influencer_attrs_path,
                config=config,
                verbose=args.verbose
            )
            
            all_results.extend(results)
            
            n_success = sum(1 for r in results if r.get('gae_train_failed', 1) == 0 
                           and r.get('gae_too_small', 1) == 0)
            print(f"  ✓ {n_success}/{len(results)} windows succeeded")
        
        except Exception as e:
            print(f"  [ERROR] Failed: {e}")
            failed_hashtags.append(hashtag)
    
    # Combine results
    if len(all_results) == 0:
        print("\n[WARNING] No features extracted!")
        return
    
    print(f"\nCombining features from {len(hashtags)} hashtags...")
    df = pd.DataFrame(all_results)
    
    # Save features
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)
    print(f"✓ Saved {len(df)} feature vectors to {output_file}")
    
    # Save summary
    summary = []
    for hashtag in hashtags:
        hashtag_results = [r for r in all_results if r['hashtag'] == hashtag]
        if len(hashtag_results) == 0:
            continue
        
        n_total = len(hashtag_results)
        n_success = sum(1 for r in hashtag_results if r.get('gae_train_failed', 1) == 0 
                       and r.get('gae_too_small', 1) == 0)
        n_too_small = sum(1 for r in hashtag_results if r.get('gae_too_small', 0) == 1)
        n_failed = sum(1 for r in hashtag_results if r.get('gae_train_failed', 0) == 1)
        
        summary.append({
            'hashtag': hashtag,
            'n_windows': n_total,
            'n_success': n_success,
            'n_too_small': n_too_small,
            'n_failed': n_failed
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Saved summary to {summary_file}")
    
    # Final stats
    print("\n" + "=" * 80)
    print("Extraction Summary")
    print("=" * 80)
    print(f"Total windows processed: {len(df)}")
    print(f"Hashtags: {len(hashtags)}")
    
    n_features = len([c for c in df.columns if c.startswith('gae_')])
    print(f"GAE features extracted: {n_features}")
    
    print("\nDiagnostics:")
    print(f"  Successful: {df['gae_train_failed'].eq(0).sum()}")
    print(f"  Too small: {df['gae_too_small'].eq(1).sum()}")
    print(f"  Training failed: {df['gae_train_failed'].eq(1).sum()}")
    print(f"  Windows with deltas: {df['gae_has_prev_window'].eq(1).sum()}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()