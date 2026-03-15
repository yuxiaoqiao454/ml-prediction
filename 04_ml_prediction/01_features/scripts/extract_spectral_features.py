#!/usr/bin/env python3
"""
Extract Spectral Network Features

Computes spectral features (eigenvalues, IPR, entropy) from network windows.
"""

import argparse
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

# Import feature registry
from spectral_feature_registry import (
    SPECTRAL_FEATURE_REGISTRY,
    build_graph_from_edges,
    extract_largest_cc,
    compute_normalized_laplacian,
    compute_eigenvalues,
    compute_delta_features,
    create_flags
)


def load_config(config_path):
    """Load YAML config with type conversion."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Convert numeric parameters
    comp = cfg['computation']
    comp['epsilon'] = float(comp['epsilon'])
    comp['min_graph_size'] = int(comp['min_graph_size'])
    comp['k_small'] = int(comp['k_small'])
    comp['k_tail'] = int(comp['k_tail'])
    comp['max_dense_size'] = int(comp['max_dense_size'])
    comp['cc2_min_size'] = int(comp['cc2_min_size'])
    
    return cfg


def discover_all_hashtags(base_dir):
    """Find all hashtag directories."""
    hashtag_dirs = [d for d in base_dir.iterdir() 
                   if d.is_dir() and d.name.startswith('windows_')]
    
    hashtags = [d.name.replace('windows_', '') for d in hashtag_dirs]
    
    return sorted(hashtags)


def discover_hashtag_windows(hashtag, base_dir):
    """Discover all window folders for a hashtag."""
    windows_dir = base_dir / f"windows_{hashtag}"
    
    if not windows_dir.exists():
        return []
    
    windows = []
    for folder in windows_dir.iterdir():
        if folder.is_dir() and len(folder.name) == 10 and folder.name.count('-') == 2:
            try:
                pd.to_datetime(folder.name)
                windows.append(folder.name)
            except:
                continue
    
    return sorted(windows)


def load_edges_for_window(base_dir, hashtag, window_end, graph_type, config):
    """Load edges CSV for a window."""
    graph_config = config['graph_types'][graph_type]
    
    if not graph_config['enabled']:
        return None
    
    path_pattern = graph_config['path_pattern']
    path = Path(base_dir) / path_pattern.format(hashtag=hashtag, window_end=window_end)
    
    if not path.exists():
        return None
    
    try:
        edges = pd.read_csv(path)
        return edges
    except Exception as e:
        return None


def extract_features_for_window(edges_df, graph_type, prefix, config, verbose=False):
    """
    Extract all spectral features for one graph.
    
    Returns:
    --------
    features : dict
    metadata : dict with computation info
    """

    import time
    t_total = time.time()

    features = {}
    metadata = {
        'success': False,
        'n_nodes': 0,
        'n_edges': 0,
        'n_components': 0,
        'too_small': False,
        'eig_failed': False
    }
    
    # Build graph
    t0 = time.time()
    use_weights = config['computation']['use_weighted_edges']
    remove_isolated = config['computation']['remove_isolated_nodes']
    
    G, build_meta = build_graph_from_edges(edges_df, use_weights, remove_isolated)
    
    metadata['n_nodes'] = build_meta['n_nodes_final']
    metadata['n_edges'] = build_meta['n_edges_final']
    print(f"[DEBUG] build_graph: {time.time()-t0:.3f}s")
    
    # Check minimum size
    t0 = time.time()
    min_size = config['computation']['min_graph_size']
    if G.number_of_nodes() < min_size:
        metadata['too_small'] = True
        flags = create_flags(G, G, [], False, False, prefix)
        features.update(flags)
        return features, metadata
    print(f"[DEBUG] check_min_size: {time.time()-t0:.3f}s")
    
    # Extract largest CC + components
    t0 = time.time()
    G_lcc, components = extract_largest_cc(G)
    
    metadata['n_components'] = len(components)
    print(f"[DEBUG] extract_lcc: {time.time()-t0:.3f}s")
    
    # Compute Laplacian
    t0 = time.time()
    L, node_list = compute_normalized_laplacian(G_lcc, epsilon=config['computation']['epsilon'])
    
    if L is None:
        metadata['too_small'] = True
        flags = create_flags(G, G_lcc, components, False, False, prefix)
        features.update(flags)
        return features, metadata
    print(f"[DEBUG] laplacian: {time.time()-t0:.3f}s")
    
    # Compute eigenvalues
    t0 = time.time()
    eigenvalues, eig_success, eig_method = compute_eigenvalues(
        L,
        k_small=config['computation']['k_small'],
        k_tail=config['computation']['k_tail'],
        max_dense_size=config['computation']['max_dense_size'],
        arpack_timeout=config['computation'].get('arpack_timeout', 300),
        arpack_size_limit=config['computation'].get('arpack_size_limit', 15000),
        lobpcg_tol=config['computation'].get('lobpcg_tol', 1e-3),
        lobpcg_maxiter=config['computation'].get('lobpcg_maxiter', 200)
    )

    if not eig_success:
        if verbose:
            print(f"      [EIGEN FAILED] method={eig_method}, n={metadata['n_nodes']}")
        flags = create_flags(G, G_lcc, components, False, False, prefix)
        features.update(flags)
        return features, metadata
        
    metadata['eig_failed'] = not eig_success
    
    if not eig_success:
        flags = create_flags(G, G_lcc, components, False, False, prefix)
        features.update(flags)
        return features, metadata
    print(f"[DEBUG] eigenvalues: {time.time()-t0:.3f}s")
    
    # Extract feature families
    
    feature_config = config['features']
    
    if feature_config.get('lcc_spectral', True):
        t0 = time.time()
        lcc_features = SPECTRAL_FEATURE_REGISTRY['lcc_spectral'](
            eigenvalues, prefix, config['computation']
        )
        features.update(lcc_features)
        print(f"[DEBUG] lcc_spectral: {time.time()-t0:.3f}s")
    
    if feature_config.get('cc_summary', True):
        t0 = time.time()
        cc_features = SPECTRAL_FEATURE_REGISTRY['cc_summary'](
            components, G, prefix, config['computation']
        )
        features.update(cc_features)
        print(f"[DEBUG] cc_summary: {time.time()-t0:.3f}s")
    
    if feature_config.get('cc2_features', True):
        t0 = time.time()
        cc2_features = SPECTRAL_FEATURE_REGISTRY['cc2_features'](
            components, G, prefix, config['computation']
        )
        features.update(cc2_features)
        print(f"[DEBUG] cc2_features: {time.time()-t0:.3f}s")
    
    if feature_config.get('global_indicators', True):
        t0 = time.time()
        global_features = SPECTRAL_FEATURE_REGISTRY['global_indicators'](
            G, components, prefix, config['computation']
        )
        features.update(global_features)
        print(f"[DEBUG] global_indicators: {time.time()-t0:.3f}s")
        
        # ✅ FIXED: Fill placeholder with actual value
        if len(components) == 1 and f'{prefix}_lcc_lambda2' in features:
            features[f'{prefix}_global_lambda2_full'] = features[f'{prefix}_lcc_lambda2']
    
    # Add flags
    flags = create_flags(G, G_lcc, components, True, False, prefix)
    features.update(flags)
    
    metadata['success'] = True

    print(f"[DEBUG] TOTAL FUNCTION TIME: {time.time()-t_total:.3f}s")
    
    return features, metadata


def process_hashtag_windows(hashtag, windows, base_dir, config, verbose=False):
    """Process all windows for one hashtag."""
    if verbose:
        print(f"  Processing {len(windows)} windows...")
    
    features_list = []
    stats = {
        'n_windows_total': len(windows),
        'n_windows_too_small': 0,
        'n_windows_eig_failed': 0,
        'n_windows_disconnected': 0,
        'n_windows_with_deltas': 0
    }
    
    # Store previous features for delta computation
    previous_features = {
        'bipartite': None,
        'audience': None,
        'influencer': None
    }
    
    # Progress
    window_iter = enumerate(windows)
    if not verbose:
        window_iter = enumerate(tqdm(windows, desc=f"  [{hashtag}]", leave=False))
    
    # Process each window
    for idx, window_end in window_iter:
        if verbose:
            print(f"    [{idx+1}/{len(windows)}] Window {window_end}:")
        
        window_features = {
            'hashtag': hashtag,
            'window_end': window_end
        }
        
        # Process each graph type
        for graph_type in ['bipartite', 'audience', 'influencer']:
            graph_config = config['graph_types'][graph_type]
            
            if not graph_config['enabled']:
                continue
            
            prefix = graph_config['prefix']
            
            # Load edges
            edges_df = load_edges_for_window(base_dir, hashtag, window_end, graph_type, config)
            
            if edges_df is None or len(edges_df) == 0:
                # Graph doesn't exist - add flags
                flags = {
                    f'{prefix}_graph_too_small': 1,
                    f'{prefix}_eig_failed': 0,
                    f'{prefix}_g_was_disconnected': 0,
                    f'{prefix}_has_prev_window': 0
                }
                window_features.update(flags)
                
                if verbose:
                    print(f"      [{graph_type}] No edges found")
                
                continue
            
            # Extract features
            try:
                features, metadata = extract_features_for_window(
                    edges_df, graph_type, prefix, config, verbose
                )
                
                window_features.update(features)
                
                # Update stats
                if metadata['too_small']:
                    stats['n_windows_too_small'] += 1
                if metadata['eig_failed']:
                    stats['n_windows_eig_failed'] += 1
                if metadata['n_components'] > 1:
                    stats['n_windows_disconnected'] += 1
                
                # ✅ FIXED: Compute deltas with correct config parameter
                if config['features'].get('temporal_deltas', True) and previous_features[graph_type] is not None:
                    delta_features = compute_delta_features(
                        window_features, 
                        previous_features[graph_type], 
                        prefix, 
                        config['features']  # ✅ FIXED: Was config['computation']
                    )
                    
                    if len(delta_features) > 0:
                        window_features.update(delta_features)
                        # ✅ FIXED: Update has_prev flag
                        window_features[f'{prefix}_has_prev_window'] = 1
                        stats['n_windows_with_deltas'] += 1
                
                # Store for next iteration
                previous_features[graph_type] = window_features.copy()
                
                if verbose:
                    status = "✓" if metadata['success'] else "✗"
                    print(f"      [{graph_type}] {status} n={metadata['n_nodes']}, "
                          f"e={metadata['n_edges']}, cc={metadata['n_components']}")
                
            except Exception as e:
                if verbose:
                    print(f"      [{graph_type}] ✗ Failed: {str(e)}")
                # Add error flags
                flags = {
                    f'{prefix}_graph_too_small': 0,
                    f'{prefix}_eig_failed': 1,
                    f'{prefix}_g_was_disconnected': 0,
                    f'{prefix}_has_prev_window': 0
                }
                window_features.update(flags)
        
        features_list.append(window_features)
    
    return features_list, stats


def main():
    ap = argparse.ArgumentParser(description="Extract spectral network features")
    ap.add_argument("--config",
                    default="04_ml_prediction/01_features/configs/spectral_features.yaml",
                    help="Path to config")
    ap.add_argument("--limit", type=int,
                    help="Process only first N hashtags")
    ap.add_argument("--verbose", action="store_true",
                    help="Print detailed progress")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip if output exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print actions without executing")
    args = ap.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    output_file = Path(config['output']['features_file'])
    base_dir = Path(config['input']['network_windows_base'])
    
    print(f"{'='*80}")
    print(f"Spectral Feature Extraction")
    print(f"Config: {args.config}")
    print(f"Output: {output_file}")
    print(f"{'='*80}\n")
    
    # Check skip-existing
    if args.skip_existing and output_file.exists():
        print(f"✓ Output already exists: {output_file}")
        print(f"  Skipping (remove --skip-existing to rerun)")
        return
    
    if args.dry_run:
        print("[DRY RUN] Would extract spectral features")
        return
    
    # Discover hashtags
    print("Discovering hashtags...")
    all_hashtags = discover_all_hashtags(base_dir)
    
    if args.limit:
        all_hashtags = all_hashtags[:args.limit]
        print(f"  Limited to {args.limit} hashtags")
    
    print(f"✓ Found {len(all_hashtags)} hashtags to process\n")
    
    # Print enabled features
    print("Enabled features:")
    for family, enabled in config['features'].items():
        if enabled:
            print(f"  ✓ {family}")
    
    print("\nEnabled graph types:")
    for graph_type, graph_config in config['graph_types'].items():
        if graph_config['enabled']:
            print(f"  ✓ {graph_type} → {graph_config['prefix']}")
    print()
    
    # Process each hashtag
    all_features = []
    all_stats = []
    
    for hashtag in tqdm(all_hashtags, desc="Extracting spectral features", disable=args.verbose):
        if args.verbose:
            print(f"\n[{hashtag}]")
        
        # Discover windows
        windows = discover_hashtag_windows(hashtag, base_dir)
        
        if len(windows) == 0:
            if args.verbose:
                print(f"  [skip] No windows found")
            continue
        
        if args.verbose:
            print(f"  Found {len(windows)} windows")
        
        try:
            # Process windows
            features_list, stats = process_hashtag_windows(
                hashtag, windows, base_dir, config, args.verbose
            )
            
            all_features.extend(features_list)
            
            stats['hashtag'] = hashtag
            all_stats.append(stats)
            
            if args.verbose:
                print(f"  ✓ {stats['n_windows_total']} windows processed")
            
        except Exception as e:
            print(f"\n[✗] {hashtag}: {str(e)}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            continue
    
    # Combine features
    print(f"\nCombining features from {len(all_hashtags)} hashtags...")
    features_df = pd.DataFrame(all_features)
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_file, index=False)
    print(f"✓ Saved {len(features_df)} feature vectors to {output_file}")
    
    # Save summary
    summary_file = Path(config['output']['summary_file'])
    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(summary_file, index=False)
    print(f"✓ Saved summary to {summary_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("Extraction Summary")
    print(f"{'='*80}")
    print(f"Total windows processed: {len(features_df)}")
    print(f"Hashtags: {len(all_hashtags)}")
    
    if len(features_df) == 0:
        print("\n⚠️  No features extracted!")
        print(f"{'='*80}")
        return
    
    # Count features
    feature_cols = [c for c in features_df.columns 
                   if c.startswith('net_') and 'spec' in c]
    print(f"Spectral features extracted: {len(feature_cols)}")
    
    # Aggregate stats
    if len(stats_df) > 0:
        total_too_small = stats_df['n_windows_too_small'].sum()
        total_eig_failed = stats_df['n_windows_eig_failed'].sum()
        total_disconnected = stats_df['n_windows_disconnected'].sum()
        total_with_deltas = stats_df['n_windows_with_deltas'].sum()
        
        print(f"\nDiagnostics:")
        print(f"  Windows too small: {total_too_small}")
        print(f"  Eigenvalue computation failed: {total_eig_failed}")
        print(f"  Disconnected graphs: {total_disconnected}")
        print(f"  Windows with deltas: {total_with_deltas}")
    
    print(f"{'='*80}")


if __name__ == "__main__":
    main()