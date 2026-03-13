#!/usr/bin/env python3
"""
Label hashtag windows with burst/non-burst labels using change-point detection.

Uses PELT (Pruned Exact Linear Time) algorithm to detect change-points in
time series, then labels windows based on:
- Presence of change-point near window boundary
- Significant jump in mean between current and previous window
- Configurable thresholds for relative and absolute jumps

Supports multiple labeling methods via registry pattern.
Currently implements: PELT

Usage:
  python 04_ml_prediction/02_labels/scripts/run_labeling.py --config 04_ml_prediction/02_labels/configs/labeling_cpinside.yaml --output 04_ml_prediction/02_labels/labels_h550_cpinside.parquet
  python 04_ml_prediction/02_labels/scripts/run_labeling.py --limit 5 --dry-run
"""

import argparse
import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm
import ruptures as rpt
import json
import subprocess

warnings.filterwarnings('ignore')

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))


# ============================================================================
# Registry Pattern for Labeling Methods
# ============================================================================

LABELING_REGISTRY = {}

def register_labeling_method(name):
    """Decorator to register a labeling method."""
    def decorator(fn):
        LABELING_REGISTRY[name] = fn
        return fn
    return decorator


# ============================================================================
# PELT Labeling Implementation
# ============================================================================

@register_labeling_method("pelt")
def run_pelt_labeling(hashtag, windows_df, timeseries_df, config):
    """
    Label windows using PELT change-point detection.
    
    Parameters:
    -----------
    hashtag : str
        Hashtag being processed
    windows_df : DataFrame
        Window metadata from slice_timeseries_windows.py
    timeseries_df : DataFrame
        Full daily time series for the hashtag (date, mentions, comments)
    config : dict
        Configuration parameters
    
    Returns:
    --------
    DataFrame with columns:
        hashtag, window_end, window_start,
        label_burst_mentions, mean_mentions_prev, mean_mentions_curr,
        jump_ratio_mentions, jump_absolute_mentions,
        has_cp_near_boundary_mentions, cp_position_mentions,
        [same for comments],
        n_cps_total_mentions, n_cps_total_comments,
        method, timeseries_variant
    """
    
    variant = config.get('timeseries_variant', 'raw')
    model = config.get('model', 'l2')
    penalty = config.get('penalty', 'bic')
    min_seg_len = config.get('min_seg_len', 7)
    guard_days = config.get('guard_days', 7)
    rel_jump = config.get('rel_jump', 1.0)
    abs_jump = config.get('abs_jump', 10)
    min_coverage = config.get('min_coverage_pct', 30) / 100.0
    
    # Prepare time series based on variant
    ts = timeseries_df.copy()
    ts['date'] = pd.to_datetime(ts['date'])
    ts = ts.sort_values('date')
    
    mentions_col = f'mentions_{variant}_counts' if variant == 'raw' else f'mentions_{variant}_values'
    comments_col = f'comments_{variant}_counts' if variant == 'raw' else f'comments_{variant}_values'
    
    # Get values
    mentions_series = ts[mentions_col].values if mentions_col in ts.columns else None
    comments_series = ts[comments_col].values if comments_col in ts.columns else None
    # dates = ts['date'].values
    dates = pd.to_datetime(ts['date'])
    date_to_idx = {date: i for i, date in enumerate(dates)}  # Map dates to indices
    
    # Run PELT on full series
    cps_mentions = None
    cps_comments = None
    n_cps_mentions = 0
    n_cps_comments = 0
    
    if mentions_series is not None and len(mentions_series) > min_seg_len * 2:
        try:
            algo_m = rpt.Pelt(model=model, min_size=min_seg_len).fit(mentions_series)
            cps_mentions = algo_m.predict(pen=penalty if isinstance(penalty, (int, float)) else np.log(len(mentions_series)))
            n_cps_mentions = len(cps_mentions) - 1  # Exclude endpoint
        except Exception as e:
            print(f"  [warn] PELT failed for {hashtag} mentions: {e}")
    
    if comments_series is not None and len(comments_series) > min_seg_len * 2:
        try:
            algo_c = rpt.Pelt(model=model, min_size=min_seg_len).fit(comments_series)
            cps_comments = algo_c.predict(pen=penalty if isinstance(penalty, (int, float)) else np.log(len(comments_series)))
            n_cps_comments = len(cps_comments) - 1  # Exclude endpoint
        except Exception as e:
            print(f"  [warn] PELT failed for {hashtag} comments: {e}")
    
    # Label each window
    rows = []
    
    for idx, window in windows_df.iterrows():
        window_end = pd.to_datetime(window['window_end'])
        window_start = pd.to_datetime(window['window_start'])
        
        # Get data for current window
        mask_curr = (ts['date'] > window_start) & (ts['date'] <= window_end)
        curr_data = ts[mask_curr]
        
        # Check coverage
        expected_days = (window_end - window_start).days
        actual_days = len(curr_data)
        coverage = actual_days / expected_days if expected_days > 0 else 0
        
        if coverage < min_coverage:
            # Skip - insufficient data
            continue
        
        # Get previous window data (for computing "previous mean")
        prev_window_end = window_start
        prev_window_start = prev_window_end - timedelta(days=30)
        mask_prev = (ts['date'] > prev_window_start) & (ts['date'] <= prev_window_end)
        prev_data = ts[mask_prev]
        
        # Initialize row
        row = {
            'hashtag': hashtag,
            'window_end': window_end.strftime('%Y-%m-%d'),
            'window_start': window_start.strftime('%Y-%m-%d'),
        }
        
        # ---- Label mentions ----
        if mentions_series is not None and len(curr_data) > 0:
            curr_mentions = curr_data[mentions_col].values
            prev_mentions = prev_data[mentions_col].values if len(prev_data) > 0 else np.array([0])
            
            mean_curr = np.mean(curr_mentions)
            mean_prev = np.mean(prev_mentions)
            
            # Check for change-point near boundary
            has_cp_inside = False
            cp_position = None
            
            if cps_mentions is not None:
                # Find CPs near window start (boundary between prev and curr)
                # boundary_idx = np.searchsorted(dates, window_start)
                # boundary_idx = date_to_idx.get(window_start, len(dates))  # Get index of boundary
                # for cp_idx in cps_mentions[:-1]:  # Exclude endpoint
                #     if abs(cp_idx - boundary_idx) <= guard_days:
                #         has_cp_near = True
                #         cp_position = dates[cp_idx]
                #         break
                window_start_date = pd.to_datetime(window_start)
                window_end_date = pd.to_datetime(window_end)
                
                for cp_idx in cps_mentions[:-1]:  # Exclude endpoint
                    cp_date = dates[cp_idx]
                    
                    # Is CP inside window T? (exclusive start, inclusive end)
                    if window_start_date < cp_date <= window_end_date:
                        has_cp_inside = True
                        cp_position = cp_date
                        break  # Found one, that's enough
            
            # Compute jump metrics
            jump_abs = mean_curr - mean_prev
            jump_ratio = mean_curr / mean_prev if mean_prev > 0 else np.inf
            
            # Label logic
            label = 0
            if has_cp_inside and jump_abs >= abs_jump and jump_ratio >= (1 + rel_jump):
                label = 1
            
            row.update({
                'label_burst_mentions': label,
                'mean_mentions_prev': float(mean_prev),
                'mean_mentions_curr': float(mean_curr),
                'jump_ratio_mentions': float(jump_ratio),
                'jump_absolute_mentions': float(jump_abs),
                'has_cp_inside_boundary_mentions': has_cp_inside,
                'cp_position_mentions': pd.Timestamp(cp_position).strftime('%Y-%m-%d') if cp_position is not None else None,
            })
        else:
            row.update({
                'label_burst_mentions': None,
                'mean_mentions_prev': None,
                'mean_mentions_curr': None,
                'jump_ratio_mentions': None,
                'jump_absolute_mentions': None,
                'has_cp_inside_boundary_mentions': None,
                'cp_position_mentions': None,
            })
        
        # ---- Label comments (same logic) ----
        if comments_series is not None and len(curr_data) > 0:
            curr_comments = curr_data[comments_col].values
            prev_comments = prev_data[comments_col].values if len(prev_data) > 0 else np.array([0])
            
            mean_curr_c = np.mean(curr_comments)
            mean_prev_c = np.mean(prev_comments)
            
            has_cp_inside_c = False
            cp_position_c = None
            
            if cps_comments is not None:
                # boundary_idx = np.searchsorted(dates, window_start)
                # boundary_idx = date_to_idx.get(window_start, len(dates))
                # for cp_idx in cps_comments[:-1]:
                #     if abs(cp_idx - boundary_idx) <= guard_days:
                #         has_cp_near_c = True
                #         cp_position_c = dates[cp_idx]
                #         break
                window_start_date = pd.to_datetime(window_start)
                window_end_date = pd.to_datetime(window_end)
                
                for cp_idx in cps_comments[:-1]:
                    cp_date = dates[cp_idx]
                    
                    if window_start_date < cp_date <= window_end_date:
                        has_cp_inside_c = True
                        cp_position_c = cp_date
                        break
            
            jump_abs_c = mean_curr_c - mean_prev_c
            jump_ratio_c = mean_curr_c / mean_prev_c if mean_prev_c > 0 else np.inf
            
            label_c = 0
            if has_cp_inside_c and jump_abs_c >= abs_jump and jump_ratio_c >= (1 + rel_jump):
                label_c = 1
            
            row.update({
                'label_burst_comments': label_c,
                'mean_comments_prev': float(mean_prev_c),
                'mean_comments_curr': float(mean_curr_c),
                'jump_ratio_comments': float(jump_ratio_c),
                'jump_absolute_comments': float(jump_abs_c),
                'has_cp_inside_boundary_comments': has_cp_inside_c,
                'cp_position_comments': pd.Timestamp(cp_position_c).strftime('%Y-%m-%d') if cp_position_c is not None else None,
            })
        else:
            row.update({
                'label_burst_comments': None,
                'mean_comments_prev': None,
                'mean_comments_curr': None,
                'jump_ratio_comments': None,
                'jump_absolute_comments': None,
                'has_cp_inside_boundary_comments': None,
                'cp_position_comments': None,
            })
        
        # Metadata
        row.update({
            'n_cps_total_mentions': n_cps_mentions,
            'n_cps_total_comments': n_cps_comments,
            'method': 'pelt',
            'timeseries_variant': variant,
        })
        
        rows.append(row)
    
    return pd.DataFrame(rows)


# ============================================================================
# Main Orchestration
# ============================================================================

def load_config(config_path):
    """Load YAML config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_timeseries_for_hashtag(hashtag, mentions_path, comments_path, variant='raw'):
    """
    Load time series data for one hashtag.
    
    Returns DataFrame with columns: date, mentions_*_counts/values, comments_*_counts/values
    """
    # Load mentions
    mentions_df = pd.read_csv(mentions_path)
    mentions_df['hashtag'] = mentions_df['hashtag'].astype(str).str.lower()
    mentions_df = mentions_df[mentions_df['hashtag'] == hashtag].copy()
    
    # Load comments
    comments_df = pd.read_csv(comments_path)
    comments_df['hashtag'] = comments_df['hashtag'].astype(str).str.lower()
    comments_df = comments_df[comments_df['hashtag'] == hashtag].copy()
    
    # Merge on date
    merged = pd.merge(
        mentions_df[['date', 'mentions', 'mentions_per_1000posts', 'mentions_per_1000posts_ma7']],
        comments_df[['date', 'total_comments', 'norm_platform', 'smoothed_norm_platform']],
        on='date',
        how='outer'
    ).sort_values('date')
    
    # Rename for consistency
    merged = merged.rename(columns={
        'mentions': 'mentions_raw_counts',
        'mentions_per_1000posts': 'mentions_norm_values',
        'mentions_per_1000posts_ma7': 'mentions_smooth_values',
        'total_comments': 'comments_raw_counts',
        'norm_platform': 'comments_norm_values',
        'smoothed_norm_platform': 'comments_smooth_values',
    })
    
    # Fill missing with 0
    for col in merged.columns:
        if col != 'date':
            merged[col] = merged[col].fillna(0)
    
    return merged

def log_labeling_run(config_path, output_path, labeling_config, summary_stats):
    """
    Automatically log labeling run metadata to JSON file.
    
    Parameters:
    -----------
    config_path : Path
        Path to config file used
    output_path : Path
        Path to output labels file
    labeling_config : dict
        Labeling configuration parameters
    summary_stats : dict
        Summary statistics from the run
    """
    import json
    import subprocess
    from pathlib import Path
    
    log_file = output_path.parent / "labeling_runs.json"
    
    # Get git commit (if available)
    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
    except:
        git_commit = None
    
    # Load existing log
    if log_file.exists():
        with open(log_file, 'r') as f:
            log_data = json.load(f)
    else:
        log_data = {"runs": []}
    
    # Extract run_id from output filename
    run_id = output_path.stem.replace('labels_', '')
    
    # Create new entry
    run_entry = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config_file": str(config_path),
        "output_file": str(output_path),
        
        # Parameters from config
        "parameters": {
            "method": labeling_config.get('method', 'pelt'),
            "timeseries_variant": labeling_config.get('timeseries_variant', 'raw'),
            "model": labeling_config.get('model', 'l2'),
            "penalty": labeling_config.get('penalty', 'bic'),
            "min_seg_len": labeling_config.get('min_seg_len', 7),
            "use_guard_days": labeling_config.get('use_guard_days', labeling_config.get('guard_days') is not None),
            "guard_days": labeling_config.get('guard_days'),
            "rel_jump": labeling_config.get('rel_jump', 1.0),
            "abs_jump": labeling_config.get('abs_jump', 10),
            "min_coverage_pct": labeling_config.get('min_coverage_pct', 30),
        },
        
        # Summary statistics
        "data": {
            "n_hashtags_processed": summary_stats['n_hashtags'],
            "n_hashtags_success": summary_stats['n_success'],
            "n_hashtags_failed": summary_stats['n_failed'],
            "n_windows_total": summary_stats['n_windows_total'],
            "n_windows_labeled": summary_stats['n_windows_labeled'],
            "n_bursts_mentions": summary_stats['n_bursts_mentions'],
            "n_bursts_comments": summary_stats['n_bursts_comments'],
            "burst_rate_mentions": summary_stats['burst_rate_mentions'],
            "burst_rate_comments": summary_stats['burst_rate_comments'],
        },
        
        # Version control
        "git_commit": git_commit,
        
        # Notes
        "notes": labeling_config.get('notes', '')
    }
    
    # Check if run_id already exists (overwrite if re-running)
    existing_runs = [r for r in log_data['runs'] if r['run_id'] != run_id]
    existing_runs.append(run_entry)
    log_data['runs'] = existing_runs
    
    # Save
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\n✓ Logged run to {log_file}")
    print(f"  Run ID: {run_id}")


def main():
    ap = argparse.ArgumentParser(description="Label hashtag windows using change-point detection")
    ap.add_argument("--config", default="04_ml_prediction/configs/labeling.yaml",
                    help="Path to labeling config")
    ap.add_argument("--window-slices-dir", default="02t_timeseries/window_slices",
                    help="Directory with window slice parquets")
    ap.add_argument("--mentions-csv", default="02t_timeseries/csvs/hashtag_timeseries_mentions_norm_smooth.csv",
                    help="Path to mentions CSV")
    ap.add_argument("--comments-csv", default="02t_timeseries/csvs/hashtag_timeseries_comments_norm_smooth.csv",
                    help="Path to comments CSV")
    ap.add_argument("--output", default="04_ml_prediction/02_labels/labels_all.parquet",
                    help="Output parquet file")
    ap.add_argument("--limit", type=int, help="Process only first N hashtags")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip if output already exists")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    args = ap.parse_args()
    
    # Load config
    config = load_config(args.config)
    labeling_config = config.get('labeling', {})
    method = labeling_config.get('method', 'pelt')
    
    if method not in LABELING_REGISTRY:
        print(f"❌ Unknown labeling method: {method}")
        print(f"Available methods: {list(LABELING_REGISTRY.keys())}")
        sys.exit(1)
    
    labeling_fn = LABELING_REGISTRY[method]
    
    # Setup paths
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    log_dir = output_path.parent
    log_file = log_dir / "labeling_summary.log"
    summary_csv = log_dir / "labeling_summary.csv"
    
    # Check skip
    if args.skip_existing and output_path.exists():
        print(f"✓ Output already exists: {output_path}")
        return
    
    # Find all window slice files
    slices_dir = Path(args.window_slices_dir)
    slice_files = sorted(slices_dir.glob("*.parquet"))
    
    if args.limit:
        slice_files = slice_files[:args.limit]
    
    print(f"{'='*80}")
    print(f"Labeling {len(slice_files)} hashtags using method: {method}")
    print(f"Config: {args.config}")
    print(f"Output: {output_path}")
    if args.dry_run:
        print("[DRY RUN MODE]")
    print(f"{'='*80}\n")
    
    # Process each hashtag
    all_labels = []
    summary_rows = []
    
    for slice_file in tqdm(slice_files, desc="Labeling"):
        hashtag = slice_file.stem
        
        try:
            # Load window slices
            windows_df = pd.read_parquet(slice_file)
            
            # Load full time series
            timeseries_df = load_timeseries_for_hashtag(
                hashtag,
                args.mentions_csv,
                args.comments_csv,
                variant=labeling_config.get('timeseries_variant', 'raw')
            )
            
            if len(timeseries_df) == 0:
                tqdm.write(f"[skip] {hashtag}: No time series data")
                continue
            
            # Run labeling
            if not args.dry_run:
                labels_df = labeling_fn(hashtag, windows_df, timeseries_df, labeling_config)
                all_labels.append(labels_df)
            
            # Summary stats
            n_windows = len(windows_df)
            if not args.dry_run and len(labels_df) > 0:
                n_labeled = len(labels_df)
                pct_burst_m = labels_df['label_burst_mentions'].sum() / n_labeled * 100 if 'label_burst_mentions' in labels_df else 0
                pct_burst_c = labels_df['label_burst_comments'].sum() / n_labeled * 100 if 'label_burst_comments' in labels_df else 0
            else:
                n_labeled = 0
                pct_burst_m = 0
                pct_burst_c = 0
            
            summary_rows.append({
                'hashtag': hashtag,
                'n_windows_total': n_windows,
                'n_windows_labeled': n_labeled,
                'pct_burst_mentions': pct_burst_m,
                'pct_burst_comments': pct_burst_c,
            })
            
            tqdm.write(f"[✓] {hashtag}: {n_labeled}/{n_windows} windows labeled "
                      f"({pct_burst_m:.1f}% burst mentions, {pct_burst_c:.1f}% burst comments)")
            
        except Exception as e:
            tqdm.write(f"[✗] {hashtag}: {str(e)}")
            summary_rows.append({
                'hashtag': hashtag,
                'n_windows_total': 0,
                'n_windows_labeled': 0,
                'pct_burst_mentions': 0,
                'pct_burst_comments': 0,
            })
    
    # # Concatenate all labels
    # if not args.dry_run and all_labels:
    #     final_df = pd.concat(all_labels, ignore_index=True)
    #     final_df.to_parquet(output_path, index=False)
    #     print(f"\n✓ Saved {len(final_df)} labeled windows to {output_path}")
    
    # # Save summary
    # if not args.dry_run:
    #     summary_df = pd.DataFrame(summary_rows)
    #     summary_df.to_csv(summary_csv, index=False)
        
    #     with open(log_file, 'w') as f:
    #         f.write(f"Labeling Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    #         f.write(f"{'='*80}\n")
    #         f.write(f"Method: {method}\n")
    #         f.write(f"Total hashtags: {len(summary_df)}\n")
    #         f.write(f"Total windows labeled: {summary_df['n_windows_labeled'].sum()}\n")
    #         f.write(f"\nPer-hashtag statistics:\n")
    #         f.write(summary_df.to_string(index=False))
        
    #     print(f"✓ Saved summary to {summary_csv}")
    #     print(f"✓ Saved log to {log_file}")
    
    # print(f"\n{'='*80}")
    # Concatenate all labels
    success_count = 0
    fail_count = 0
    
    if not args.dry_run and all_labels:
        final_df = pd.concat(all_labels, ignore_index=True)
        final_df.to_parquet(output_path, index=False)
        print(f"\n✓ Saved {len(final_df)} labeled windows to {output_path}")
        
        # Compute summary statistics
        n_bursts_mentions = final_df['label_burst_mentions'].sum() if 'label_burst_mentions' in final_df else 0
        n_bursts_comments = final_df['label_burst_comments'].sum() if 'label_burst_comments' in final_df else 0
        burst_rate_mentions = n_bursts_mentions / len(final_df) if len(final_df) > 0 else 0
        burst_rate_comments = n_bursts_comments / len(final_df) if len(final_df) > 0 else 0
        
        # Count successes and failures
        summary_df = pd.DataFrame(summary_rows)
        success_count = (summary_df['n_windows_labeled'] > 0).sum()
        fail_count = len(summary_df) - success_count
        
        # Prepare summary stats for logging
        summary_stats = {
            'n_hashtags': len(slice_files),
            'n_success': success_count,
            'n_failed': fail_count,
            'n_windows_total': summary_df['n_windows_total'].sum(),
            'n_windows_labeled': len(final_df),
            'n_bursts_mentions': int(n_bursts_mentions),
            'n_bursts_comments': int(n_bursts_comments),
            'burst_rate_mentions': float(burst_rate_mentions),
            'burst_rate_comments': float(burst_rate_comments),
        }
        
        # Log this run
        log_labeling_run(
            config_path=Path(args.config),
            output_path=output_path,
            labeling_config=labeling_config,
            summary_stats=summary_stats
        )
    
    # Save per-hashtag summary
    if not args.dry_run:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(summary_csv, index=False)
        
        with open(log_file, 'w') as f:
            f.write(f"Labeling Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Method: {method}\n")
            f.write(f"Total hashtags: {len(summary_df)}\n")
            f.write(f"Total windows labeled: {summary_df['n_windows_labeled'].sum()}\n")
            f.write(f"\nPer-hashtag statistics:\n")
            f.write(summary_df.to_string(index=False))
        
        print(f"✓ Saved summary to {summary_csv}")
        print(f"✓ Saved log to {log_file}")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()