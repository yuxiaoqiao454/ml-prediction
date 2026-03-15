#!/usr/bin/env python3
"""
Extract GAE Bipartite Structural Features

Trains Graph Autoencoders on bipartite networks and extracts
graph-level features from learned embeddings.

Usage:
    python 04_ml_prediction/01_features/scripts/extract_gae_bip_struct_features.py --config 04_ml_prediction/01_features/configs/gae_bip_struct_features.yaml --limit 5 --verbose
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
import csv
from datetime import datetime

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

# Import feature registry
from gae_bip_struct_feature_registry import (
    extract_gae_bip_struct_features
)


def load_config(config_path):
    """Load YAML config with type conversion."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Convert numeric parameters
    comp = cfg['computation']
    comp['min_graph_size'] = int(comp['min_graph_size'])
    comp['hidden_dim'] = int(comp['hidden_dim'])
    comp['embedding_dim'] = int(comp['embedding_dim'])
    comp['max_epochs'] = int(comp['max_epochs'])  # ← Changed from 'epochs'
    comp['learning_rate'] = float(comp['learning_rate'])
    comp['dropout'] = float(comp['dropout'])
    
    # New parameters for train/val/test split
    comp['train_ratio'] = float(comp.get('train_ratio', 0.85))
    comp['val_ratio'] = float(comp.get('val_ratio', 0.05))
    comp['test_ratio'] = float(comp.get('test_ratio', 0.10))
    comp['patience'] = int(comp.get('patience', 20))
    comp['check_every'] = int(comp.get('check_every', 5))
    comp['neg_sample_ratio'] = float(comp.get('neg_sample_ratio', 1.0))
    comp['random_seed'] = int(comp.get('random_seed', 42))

    # Convert recency config if present  ← ADD THIS
    if 'recency' in comp and comp['recency'] is not None:
        rec = comp['recency']
        rec['tau_days'] = float(rec.get('tau_days', 14.0))
    
    return cfg

def write_metrics_row(metrics_file, row_data, is_new_file=False):
    """
    Write a single row to the metrics CSV with all enhanced columns.
    
    Parameters:
    -----------
    metrics_file : Path
        Path to metrics CSV
    row_data : dict
        Row data to write
    is_new_file : bool
        Whether to write header
    """
    fieldnames = [
        'hashtag', 
        'window_end',
        'gae_available',      # ← ADD THIS
        'gae_skip_reason',    # ← ADD THIS
        'n_nodes',
        'n_edges',
        'n_inf',
        'n_aud',
        'n_train_edges',
        'n_val_edges', 
        'n_test_edges',
        'test_auc',
        'test_ap',
        'best_epoch',
        'final_train_loss',
        'score_weight_corr',      # Spearman(score, weight) on pos test edges
        'top20_auc',              # AUC(top 20% weighted vs neg)
        'top20_n_edges',          # How many edges in top 20%
        'weight_median',          # Median weight
        'weight_p90',             # 90th percentile
        'weight_p99',             # 99th percentile
        'weight_max',             # Max weight
        'weight_top1pct_mass',    # Top 1% mass fraction
        'adj_mode',               # binary or recency_weighted
        'feature_mode',           # type+time, etc.
        'use_weighted_bce',       # Whether weighted loss used
        'failure_reason',         # Why failed if applicable
        'timestamp'
    ]
    
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if is_new_file:
            writer.writeheader()
        
        writer.writerow(row_data)

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


def load_bipartite_edges(base_dir, hashtag, window_end, config):
    """
    Load bipartite edges for one window.
    
    Returns:
    --------
    edges_df : pd.DataFrame or None
        Columns: ['source', 'target', 'weight' (optional)]
    """
    path_pattern = config['graph_source']['path_pattern']
    path = Path(base_dir) / path_pattern.format(hashtag=hashtag, window_end=window_end)
    
    if not path.exists():
        return None
    
    try:
        edges = pd.read_csv(path)
        
        # Check required columns
        if 'source' not in edges.columns or 'target' not in edges.columns:
            # Assume first two columns are source, target
            edges.columns = ['source', 'target'] + list(edges.columns[2:])
        
        return edges
    except Exception as e:
        return None


def extract_features_for_window(edges_df, prefix, config, window_dir, verbose=False):
    """
    Extract GAE features for one window.
    
    Returns:
    --------
    features : dict
    metadata : dict
    """
    features, metadata = extract_gae_bip_struct_features(
        edges_df=edges_df,
        prefix=prefix,
        comp_config=config['computation'],
        feature_config=config['features'],
        window_dir=window_dir,  # ADD THIS
        verbose=verbose
    )
    
    return features, metadata


def compute_delta_features(current_features, previous_features, prefix, feature_config):
    """
    Compute temporal change features.
    
    Only computes deltas for scalar graph-level features (not per-dimension).
    
    Parameters:
    -----------
    current_features : dict
    previous_features : dict
    prefix : str
    feature_config : dict
    
    Returns:
    --------
    delta_features : dict
    """
    delta_features = {}
    
    if not feature_config.get('temporal_deltas', True):
        return delta_features
    
    # Define which features to compute deltas for
    # (Only scalar features, not per-dimension embeddings)
    delta_keys = [
        f'{prefix}_mean_norm',
        f'{prefix}_std_norm',
        f'{prefix}_energy',
        f'{prefix}_pca1',
        f'{prefix}_pca2',
        f'{prefix}_pca3',
        f'{prefix}_norm_entropy',
        f'{prefix}_max_norm',
        f'{prefix}_median_norm',
        f'{prefix}_global_mean',
        f'{prefix}_global_std',
        f'{prefix}_l2_mean',
        f'{prefix}_l2_std',
        f'{prefix}_cov_frob',
        f'{prefix}_effective_rank',
    ]
    
    for key in delta_keys:
        if key in current_features and key in previous_features:
            curr_val = current_features[key]
            prev_val = previous_features[key]
            
            # Check for valid values
            if not (np.isnan(curr_val) or np.isnan(prev_val)):
                delta = curr_val - prev_val
                delta_key = key.replace(prefix, f'{prefix}_delta')
                delta_features[delta_key] = delta
            else:
                delta_key = key.replace(prefix, f'{prefix}_delta')
                delta_features[delta_key] = np.nan
    
    return delta_features


def process_hashtag_windows(hashtag, windows, base_dir, config, curves_dir, metrics_file, verbose=False):
    """
    Process all windows for one hashtag.
    
    Returns:
    --------
    features_list : list of dict
    stats : dict
    """
    features_list = []
    stats = {
        'n_windows_total': len(windows),
        'n_windows_success': 0,
        'n_windows_too_small': 0,
        'n_windows_train_failed': 0,
        'n_windows_with_deltas': 0
    }
    
    # Store previous window features for deltas
    previous_features = None
    
    prefix = config['graph_source']['prefix']
    
    # Check if metrics file exists to determine if we need header
    is_new_metrics_file = not metrics_file.exists()
    
    # Process each window
    window_iter = enumerate(windows)
    if not verbose:
        window_iter = enumerate(tqdm(windows, desc=f"  [{hashtag}]", leave=False))
    
    for idx, window_end in window_iter:
        if verbose:
            print(f"    [{idx+1}/{len(windows)}] Window {window_end}:")
        
        window_features = {
            'hashtag': hashtag,
            'window_end': window_end
        }
        
        # Load bipartite edges
        edges_df = load_bipartite_edges(base_dir, hashtag, window_end, config)
        
        if edges_df is None or len(edges_df) == 0:
            # No edges - mark as too small
            window_features[f'{prefix}_too_small'] = 1
            window_features[f'{prefix}_train_failed'] = 0
            window_features[f'{prefix}_has_prev_window'] = 0
            
            # Log to metrics CSV
            write_metrics_row(metrics_file, {
                'hashtag': hashtag,
                'window_end': window_end,
                'gae_available': 0,
                'gae_skip_reason': 'no_edges',  # ← FIXED: literal string
                'n_nodes': 0,
                'n_edges': 0,
                'n_inf': 0,
                'n_aud': 0,
                'n_train_edges': np.nan,
                'n_val_edges': np.nan,
                'n_test_edges': np.nan,
                'test_auc': np.nan,
                'test_ap': np.nan,
                'best_epoch': np.nan,
                'final_train_loss': np.nan,
                'score_weight_corr': np.nan,
                'top20_auc': np.nan,
                'top20_n_edges': 0,
                'weight_median': np.nan,
                'weight_p90': np.nan,
                'weight_p99': np.nan,
                'weight_max': np.nan,
                'weight_top1pct_mass': np.nan,
                'adj_mode': config['computation'].get('adj_mode', 'binary'),
                'feature_mode': config['computation'].get('feature_mode', 'type'),
                'use_weighted_bce': config['computation'].get('use_weighted_bce', False),
                'failure_reason': 'no_edges',
                'timestamp': datetime.now().isoformat()
            }, is_new_file=is_new_metrics_file)
            is_new_metrics_file = False  # Header written
            
            if verbose:
                print(f"      No edges found")
            
            features_list.append(window_features)
            stats['n_windows_too_small'] += 1
            continue
        
        # Extract features
        try:
            # Construct window directory path
            window_dir = base_dir / f"windows_{hashtag}" / window_end

            features, metadata = extract_features_for_window(
                edges_df, prefix, config, window_dir, verbose
            )
            
            window_features.update(features)
            
            # Update stats
            if metadata['success']:
                stats['n_windows_success'] += 1
                # Save training curve
                if 'training_curve' in metadata and len(metadata['training_curve']) > 0:
                    curve_file = curves_dir / f"{hashtag}_{window_end}_curve.csv"
                    pd.DataFrame(metadata['training_curve']).to_csv(curve_file, index=False)
                
                # Log successful metrics to CSV
                write_metrics_row(metrics_file, {
                    'hashtag': hashtag,
                    'window_end': window_end,
                    'gae_available': 1,  # ← FIXED: 1 for success!
                    'gae_skip_reason': '',  # ← FIXED: empty string for success
                    'n_nodes': metadata.get('n_nodes', np.nan),
                    'n_edges': metadata.get('n_edges', np.nan),
                    'n_inf': metadata.get('n_inf', np.nan),
                    'n_aud': metadata.get('n_aud', np.nan),
                    'n_train_edges': metadata.get('n_train_edges', np.nan),
                    'n_val_edges': metadata.get('n_val_edges', np.nan),
                    'n_test_edges': metadata.get('n_test_edges', np.nan),
                    'test_auc': metadata.get('test_auc', np.nan),
                    'test_ap': metadata.get('test_ap', np.nan),
                    'best_epoch': metadata.get('best_epoch', np.nan),
                    'final_train_loss': metadata.get('final_train_loss', np.nan),
                    'score_weight_corr': metadata.get('score_weight_corr', np.nan),
                    'top20_auc': metadata.get('top20_auc', np.nan),
                    'top20_n_edges': metadata.get('top20_n_edges', 0),
                    'weight_median': metadata.get('weight_median', np.nan),
                    'weight_p90': metadata.get('weight_p90', np.nan),
                    'weight_p99': metadata.get('weight_p99', np.nan),
                    'weight_max': metadata.get('weight_max', np.nan),
                    'weight_top1pct_mass': metadata.get('weight_top1pct_mass', np.nan),
                    'adj_mode': config['computation'].get('adj_mode', 'binary'),
                    'feature_mode': config['computation'].get('feature_mode', 'type'),
                    'use_weighted_bce': config['computation'].get('use_weighted_bce', False),
                    'failure_reason': '',  # Empty for success
                    'timestamp': datetime.now().isoformat()
                }, is_new_file=is_new_metrics_file)
                is_new_metrics_file = False
                
            if metadata['too_small']:
                stats['n_windows_too_small'] += 1
            if metadata['train_failed']:
                stats['n_windows_train_failed'] += 1
            
            # Compute temporal deltas
            if config['features'].get('temporal_deltas', True) and previous_features is not None:
                delta_features = compute_delta_features(
                    window_features,
                    previous_features,
                    prefix,
                    config['features']
                )
                
                if len(delta_features) > 0:
                    window_features.update(delta_features)
                    window_features[f'{prefix}_has_prev_window'] = 1
                    stats['n_windows_with_deltas'] += 1
            else:
                window_features[f'{prefix}_has_prev_window'] = 0
            
            # Store for next iteration
            previous_features = window_features.copy()
            
            if verbose:
                status = "✓" if metadata['success'] else "✗"
                print(f"      {status} n={metadata['n_nodes']}, e={metadata['n_edges']}")
        
        except Exception as e:
            # Determine failure reason
            error_msg = str(e)
            failure_reason = 'unknown_error'  # ← ADD DEFAULT VALUE FIRST
            
            if 'Insufficient train edges' in error_msg:
                failure_reason = 'insufficient_train_edges'
            elif 'Insufficient val edges' in error_msg:
                failure_reason = 'insufficient_val_edges'
            elif 'Insufficient test edges' in error_msg:
                failure_reason = 'insufficient_test_edges'
            elif 'CUDA' in error_msg or 'GPU' in error_msg:
                failure_reason = 'gpu_error'
            elif 'singular' in error_msg.lower():
                failure_reason = 'singular_matrix'
            elif 'cannot access local variable' in error_msg:
                failure_reason = 'scope_error'
            else:
                failure_reason = f'other: {error_msg[:50]}'
            
            if verbose:
                print(f"      ✗ Failed: {failure_reason}")
            
            # Add error flags
            window_features[f'{prefix}_too_small'] = 0
            window_features[f'{prefix}_train_failed'] = 1
            window_features[f'{prefix}_has_prev_window'] = 0
            
            # Log failure to metrics CSV
            write_metrics_row(metrics_file, {
                'hashtag': hashtag,
                'window_end': window_end,
                'gae_available': 0,  # or 1 for success
                'gae_skip_reason': failure_reason,
                'n_nodes': len(edges_df['source'].unique()) + len(edges_df['target'].unique()) if edges_df is not None else np.nan,
                'n_edges': len(edges_df) if edges_df is not None else np.nan,
                'n_inf': np.nan,
                'n_aud': np.nan,
                'n_train_edges': np.nan,
                'n_val_edges': np.nan,
                'n_test_edges': np.nan,
                'test_auc': np.nan,
                'test_ap': np.nan,
                'best_epoch': np.nan,
                'final_train_loss': np.nan,
                'score_weight_corr': np.nan,
                'top20_auc': np.nan,
                'top20_n_edges': 0,
                'weight_median': np.nan,
                'weight_p90': np.nan,
                'weight_p99': np.nan,
                'weight_max': np.nan,
                'weight_top1pct_mass': np.nan,
                'adj_mode': config['computation'].get('adj_mode', 'binary'),
                'feature_mode': config['computation'].get('feature_mode', 'type'),
                'use_weighted_bce': config['computation'].get('use_weighted_bce', False),
                'failure_reason': failure_reason,
                'timestamp': datetime.now().isoformat()
            }, is_new_file=is_new_metrics_file)
            is_new_metrics_file = False
            
            stats['n_windows_train_failed'] += 1
        
        features_list.append(window_features)
    
    return features_list, stats


def main():
    ap = argparse.ArgumentParser(description="Extract GAE bipartite structural features")
    ap.add_argument("--config",
                    default="04_ml_prediction/01_features/configs/gae_bip_struct_features.yaml",
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

    # ADD THIS DEBUG PRINT
    print(f"\n[DEBUG] Loaded config:")
    print(f"  adj_mode: {config['computation'].get('adj_mode', 'NOT FOUND')}")
    print(f"  recency config: {config['computation'].get('recency', 'NOT FOUND')}")
    print()
        
    output_file = Path(config['output']['features_file'])
    base_dir = Path(config['input']['network_windows_base'])

    # Create output directories
    output_file.parent.mkdir(parents=True, exist_ok=True)
    curves_dir = Path(config['output']['training_curves_dir'])
    curves_dir.mkdir(parents=True, exist_ok=True)
    
    
    print(f"{'='*80}")
    print(f"GAE Bipartite Structural Feature Extraction")
    print(f"Config: {args.config}")
    print(f"Output: {output_file}")
    print(f"{'='*80}\n")
    
    # Check skip-existing
    if args.skip_existing and output_file.exists():
        print(f"✓ Output already exists: {output_file}")
        print(f"  Skipping (remove --skip-existing to rerun)")
        return
    
    if args.dry_run:
        print("[DRY RUN] Would extract GAE features")
        return
    
    # Discover hashtags
    print("Discovering hashtags...")
    all_hashtags = discover_all_hashtags(base_dir)
    
    if args.limit:
        all_hashtags = all_hashtags[:args.limit]
        print(f"  Limited to {args.limit} hashtags")
    
    print(f"✓ Found {len(all_hashtags)} hashtags to process\n")
    
    # Print enabled features
    print("Enabled feature tiers:")
    for family, enabled in config['features'].items():
        if enabled and family != 'temporal_deltas':
            print(f"  ✓ {family}")
    print()
    
    # Print GAE config
    print("GAE Configuration:")
    print(f"  Hidden dim: {config['computation']['hidden_dim']}")
    print(f"  Embedding dim: {config['computation']['embedding_dim']}")
    print(f"  Max epochs: {config['computation']['max_epochs']}")
    print(f"  Learning rate: {config['computation']['learning_rate']}")
    print()
    
    # Process each hashtag
    all_features = []
    all_stats = []

    # Create metrics file path
    metrics_file = Path(config['output']['metrics_file'])
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    for hashtag in tqdm(all_hashtags, desc="Extracting GAE features", disable=args.verbose):
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
                hashtag, windows, base_dir, config, curves_dir, metrics_file, args.verbose
            )
            
            all_features.extend(features_list)
            
            stats['hashtag'] = hashtag
            all_stats.append(stats)
            
            if args.verbose:
                print(f"  ✓ {stats['n_windows_success']}/{stats['n_windows_total']} windows succeeded")
        
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

    # # Save metrics summary
    # print(f"\nSaving metrics summary...")
    # metrics_rows = []
    # for feat_dict in all_features:
    #     if f"{config['graph_source']['prefix']}_test_auc" in feat_dict:
    #         metrics_rows.append({
    #             'hashtag': feat_dict['hashtag'],
    #             'window_end': feat_dict['window_end'],
    #             'test_auc': feat_dict[f"{config['graph_source']['prefix']}_test_auc"],
    #             'test_ap': feat_dict[f"{config['graph_source']['prefix']}_test_ap"],
    #             'best_epoch': feat_dict[f"{config['graph_source']['prefix']}_best_epoch"]
    #         })
    
    # if len(metrics_rows) > 0:
    #     metrics_df = pd.DataFrame(metrics_rows)
    #     metrics_file = Path(config['output']['metrics_file'])
    #     metrics_df.to_csv(metrics_file, index=False)
    #     print(f"✓ Saved {len(metrics_df)} model metrics to {metrics_file}")
        
    #     # Print aggregate stats
    #     print(f"\nModel Performance (Test Set):")
    #     print(f"  Mean AUC: {metrics_df['test_auc'].mean():.4f} ± {metrics_df['test_auc'].std():.4f}")
    #     print(f"  Mean AP:  {metrics_df['test_ap'].mean():.4f} ± {metrics_df['test_ap'].std():.4f}")
    #     print(f"  Mean epochs: {metrics_df['best_epoch'].mean():.1f}")
    
    # Print summary with failure breakdown
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
                if c.startswith('gae_bip_struct') and not c.endswith(('_too_small', '_train_failed', '_has_prev_window'))]
    print(f"GAE features extracted: {len(feature_cols)}")

    # Aggregate stats
    if len(stats_df) > 0:
        total_success = stats_df['n_windows_success'].sum()
        total_too_small = stats_df['n_windows_too_small'].sum()
        total_train_failed = stats_df['n_windows_train_failed'].sum()
        total_with_deltas = stats_df['n_windows_with_deltas'].sum()
        
        print(f"\nDiagnostics:")
        print(f"  Successful: {total_success}")
        print(f"  Too small: {total_too_small}")
        print(f"  Training failed: {total_train_failed}")
        
        # Show failure breakdown from metrics CSV
        if total_train_failed > 0 and metrics_file.exists():
            try:
                metrics_df_full = pd.read_csv(metrics_file)
                failed_df = metrics_df_full[metrics_df_full['failure_reason'].notna() & (metrics_df_full['failure_reason'] != '')]
                
                if len(failed_df) > 0:
                    print("\n  Failure breakdown:")
                    failure_counts = failed_df['failure_reason'].value_counts()
                    for reason, count in failure_counts.items():
                        print(f"    - {reason}: {count}")
            except Exception as e:
                pass  # Silently skip if can't read
        
        print(f"  Windows with deltas: {total_with_deltas}")

    # Metrics summary from enhanced CSV
    if metrics_file.exists():
        try:
            metrics_df_full = pd.read_csv(metrics_file)
            successful_metrics = metrics_df_full[metrics_df_full['test_auc'].notna()]
            
            if len(successful_metrics) > 0:
                print(f"\nMetrics Summary (successful windows):")
                print(f"  Test AUC: {successful_metrics['test_auc'].mean():.4f} ± {successful_metrics['test_auc'].std():.4f}")
                print(f"  Test AP: {successful_metrics['test_ap'].mean():.4f} ± {successful_metrics['test_ap'].std():.4f}")
                
                if 'score_weight_corr' in successful_metrics.columns:
                    corr_valid = successful_metrics['score_weight_corr'].dropna()
                    if len(corr_valid) > 0:
                        print(f"  Score-weight corr: {corr_valid.mean():.4f} ± {corr_valid.std():.4f}")
                
                if 'top20_auc' in successful_metrics.columns:
                    top20_valid = successful_metrics['top20_auc'].dropna()
                    if len(top20_valid) > 0:
                        print(f"  Top20% AUC: {top20_valid.mean():.4f} ± {top20_valid.std():.4f}")
        except Exception as e:
            pass  # Silently skip if can't read

    print(f"{'='*80}")

    
if __name__ == "__main__":
    main()







# def process_hashtag_windows(hashtag, windows, base_dir, config, curves_dir, verbose=False):
#     """
#     Process all windows for one hashtag.
    
#     Returns:
#     --------
#     features_list : list of dict
#     stats : dict
#     """
#     features_list = []
#     stats = {
#         'n_windows_total': len(windows),
#         'n_windows_success': 0,
#         'n_windows_too_small': 0,
#         'n_windows_train_failed': 0,
#         'n_windows_with_deltas': 0
#     }
    
#     # Store previous window features for deltas
#     previous_features = None
    
#     prefix = config['graph_source']['prefix']
    
#     # Process each window
#     window_iter = enumerate(windows)
#     if not verbose:
#         window_iter = enumerate(tqdm(windows, desc=f"  [{hashtag}]", leave=False))
    
#     for idx, window_end in window_iter:
#         if verbose:
#             print(f"    [{idx+1}/{len(windows)}] Window {window_end}:")
        
#         window_features = {
#             'hashtag': hashtag,
#             'window_end': window_end
#         }
        
#         # Load bipartite edges
#         edges_df = load_bipartite_edges(base_dir, hashtag, window_end, config)
        
#         if edges_df is None or len(edges_df) == 0:
#             # No edges - mark as too small
#             window_features[f'{prefix}_too_small'] = 1
#             window_features[f'{prefix}_train_failed'] = 0
#             window_features[f'{prefix}_has_prev_window'] = 0
            
#             if verbose:
#                 print(f"      No edges found")
            
#             features_list.append(window_features)
#             stats['n_windows_too_small'] += 1
#             continue
        
#         # Extract features
#         try:
#             # Construct window directory path
#             window_dir = base_dir / f"windows_{hashtag}" / window_end

#             features, metadata = extract_features_for_window(
#                 edges_df, prefix, config, window_dir, verbose
#             )
            
#             window_features.update(features)
            
#             # Update stats
#             if metadata['success']:
#                 stats['n_windows_success'] += 1
#                 # Save training curve
#                 if 'training_curve' in metadata and len(metadata['training_curve']) > 0:
#                     curve_file = curves_dir / f"{hashtag}_{window_end}_curve.csv"
#                     pd.DataFrame(metadata['training_curve']).to_csv(curve_file, index=False)
#             if metadata['too_small']:
#                 stats['n_windows_too_small'] += 1
#             if metadata['train_failed']:
#                 stats['n_windows_train_failed'] += 1
            
#             # Compute temporal deltas
#             if config['features'].get('temporal_deltas', True) and previous_features is not None:
#                 delta_features = compute_delta_features(
#                     window_features,
#                     previous_features,
#                     prefix,
#                     config['features']
#                 )
                
#                 if len(delta_features) > 0:
#                     window_features.update(delta_features)
#                     window_features[f'{prefix}_has_prev_window'] = 1
#                     stats['n_windows_with_deltas'] += 1
#             else:
#                 window_features[f'{prefix}_has_prev_window'] = 0
            
#             # Store for next iteration
#             previous_features = window_features.copy()
            
#             if verbose:
#                 status = "✓" if metadata['success'] else "✗"
#                 print(f"      {status} n={metadata['n_nodes']}, e={metadata['n_edges']}")
        
#         except Exception as e:
#             if verbose:
#                 print(f"      ✗ Failed: {e}")
            
#             # Add error flags
#             window_features[f'{prefix}_too_small'] = 0
#             window_features[f'{prefix}_train_failed'] = 1
#             window_features[f'{prefix}_has_prev_window'] = 0
            
#             stats['n_windows_train_failed'] += 1
        
#         features_list.append(window_features)
    
#     return features_list, stats