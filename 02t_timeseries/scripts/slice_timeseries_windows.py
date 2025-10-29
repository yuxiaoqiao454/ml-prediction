#!/usr/bin/env python3
"""
Slice time series data into windows aligned with network analysis windows.

For each hashtag, creates a parquet file with one row per window containing:
- Window metadata (dates, gaps, contiguity)
- Time series slices (raw, normalized, smoothed variants)
- Aggregated statistics
- Data availability flags

Output: 02t_timeseries/window_slices/<hashtag>.parquet

Usage:
  python slice_timeseries_windows.py --hashtags-list 03_networks/data/_meta/hashtags_masterlist.csv
  python slice_timeseries_windows.py --limit 5 --skip-existing
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import yaml
from tqdm import tqdm

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))


def load_config(config_path):
    """Load YAML config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_timeseries_data(mentions_path, comments_path):
    """
    Load both time series CSV files.
    
    Returns:
    --------
    mentions_df : DataFrame with columns [hashtag, date, mentions, mentions_per_1000posts, mentions_per_1000posts_ma7]
    comments_df : DataFrame with columns [hashtag, date, total_comments, norm_platform, smoothed_norm_platform]
    """
    # Load mentions
    mentions = pd.read_csv(mentions_path)
    mentions['hashtag'] = mentions['hashtag'].astype(str).str.lower().str.strip()
    mentions['date'] = pd.to_datetime(mentions['date'])
    mentions = mentions[['hashtag', 'date', 'mentions', 'mentions_per_1000posts', 'mentions_per_1000posts_ma7']]
    
    # Load comments
    comments = pd.read_csv(comments_path)
    comments['hashtag'] = comments['hashtag'].astype(str).str.lower().str.strip()
    comments['date'] = pd.to_datetime(comments['date'])
    comments = comments[['hashtag', 'date', 'total_comments', 'norm_platform', 'smoothed_norm_platform']]
    
    return mentions, comments


def detect_windows(base_dir, hashtag):
    """
    Detect all valid window folders for a hashtag.
    
    Returns sorted list of window end dates as datetime objects.
    """
    windows_dir = Path(base_dir) / f"windows_{hashtag}"
    
    if not windows_dir.exists():
        return []
    
    windows = []
    for item in windows_dir.iterdir():
        if not item.is_dir():
            continue
        name = item.name
        # Only YYYY-MM-DD folders (10 chars, 2 hyphens at positions 4 and 7)
        if len(name) == 10 and name[4] == '-' and name[7] == '-':
            try:
                date = datetime.strptime(name, '%Y-%m-%d')
                windows.append(date)
            except ValueError:
                continue
    
    return sorted(windows)


def slice_series(df, start_date, end_date):
    """
    Slice time series between start and end dates (inclusive).
    
    Returns:
    --------
    dates : list of strings (YYYY-MM-DD)
    values : list of numbers
    """
    mask = (df['date'] > start_date) & (df['date'] <= end_date)
    subset = df[mask].sort_values('date')
    
    if len(subset) == 0:
        return [], []
    
    dates = subset['date'].dt.strftime('%Y-%m-%d').tolist()
    return dates, subset


def process_hashtag(hashtag, mentions_df, comments_df, base_dir, step_days, dry_run=False):
    """
    Process one hashtag: detect windows and slice time series.
    
    Returns DataFrame with one row per window.
    """
    # Detect windows
    windows = detect_windows(base_dir, hashtag)
    
    if not windows:
        return None, f"No windows found"
    
    # Filter time series for this hashtag
    tag_mentions = mentions_df[mentions_df['hashtag'] == hashtag].copy()
    tag_comments = comments_df[comments_df['hashtag'] == hashtag].copy()
    
    has_mentions = len(tag_mentions) > 0
    has_comments = len(tag_comments) > 0
    
    if not has_mentions and not has_comments:
        return None, f"No time series data"
    
    # Process each window
    rows = []
    
    for i, window_end in enumerate(windows):
        # Infer window size from actual data (typically 30 days)
        # We'll compute it from the earliest mention/comment in the window
        # For now, assume 90 days as standard
        window_start = window_end - timedelta(days=30)
        
        # Next window info
        has_next = i < len(windows) - 1
        next_window_end = windows[i + 1] if has_next else None
        next_gap = (next_window_end - window_end).days if has_next else None
        is_contiguous = (next_gap == step_days) if has_next else False
        
        # Slice mentions (3 variants: raw, normalized, smoothed)
        mentions_raw_dates = []
        mentions_raw_counts = []
        mentions_norm_dates = []
        mentions_norm_values = []
        mentions_smooth_dates = []
        mentions_smooth_values = []
        
        if has_mentions:
            dates, subset = slice_series(tag_mentions, window_start, window_end)
            if len(dates) > 0:
                mentions_raw_dates = dates
                mentions_raw_counts = subset['mentions'].tolist()
                mentions_norm_dates = dates
                mentions_norm_values = subset['mentions_per_1000posts'].tolist()
                mentions_smooth_dates = dates
                mentions_smooth_values = subset['mentions_per_1000posts_ma7'].tolist()
        
        # Slice comments (3 variants: raw, normalized, smoothed)
        comments_raw_dates = []
        comments_raw_counts = []
        comments_norm_dates = []
        comments_norm_values = []
        comments_smooth_dates = []
        comments_smooth_values = []
        
        if has_comments:
            dates, subset = slice_series(tag_comments, window_start, window_end)
            if len(dates) > 0:
                comments_raw_dates = dates
                comments_raw_counts = subset['total_comments'].tolist()
                comments_norm_dates = dates
                comments_norm_values = subset['norm_platform'].tolist()
                comments_smooth_dates = dates
                comments_smooth_values = subset['smoothed_norm_platform'].tolist()
        
        # Compute aggregates
        mentions_raw_sum = sum(mentions_raw_counts) if mentions_raw_counts else 0
        mentions_raw_mean = mentions_raw_sum / len(mentions_raw_counts) if mentions_raw_counts else 0.0
        mentions_norm_mean = sum(mentions_norm_values) / len(mentions_norm_values) if mentions_norm_values else 0.0
        mentions_smooth_mean = sum(mentions_smooth_values) / len(mentions_smooth_values) if mentions_smooth_values else 0.0
        
        comments_raw_sum = sum(comments_raw_counts) if comments_raw_counts else 0
        comments_raw_mean = comments_raw_sum / len(comments_raw_counts) if comments_raw_counts else 0.0
        comments_norm_mean = sum(comments_norm_values) / len(comments_norm_values) if comments_norm_values else 0.0
        comments_smooth_mean = sum(comments_smooth_values) / len(comments_smooth_values) if comments_smooth_values else 0.0
        
        # Build row
        row = {
            'hashtag': hashtag,
            'window_end': window_end.strftime('%Y-%m-%d'),
            'window_start': window_start.strftime('%Y-%m-%d'),
            'window_size_days': 30,
            
            # Next window metadata
            'has_next_window': has_next,
            'next_window_end': next_window_end.strftime('%Y-%m-%d') if next_window_end else None,
            'next_window_gap_days': next_gap,
            'is_contiguous': is_contiguous,
            
            # Mentions time series (raw)
            'mentions_raw_dates': mentions_raw_dates,
            'mentions_raw_counts': mentions_raw_counts,
            'mentions_raw_sum': mentions_raw_sum,
            'mentions_raw_mean': mentions_raw_mean,
            'mentions_raw_count': len(mentions_raw_counts),
            
            # Mentions time series (normalized)
            'mentions_norm_dates': mentions_norm_dates,
            'mentions_norm_values': mentions_norm_values,
            'mentions_norm_mean': mentions_norm_mean,
            
            # Mentions time series (smoothed)
            'mentions_smooth_dates': mentions_smooth_dates,
            'mentions_smooth_values': mentions_smooth_values,
            'mentions_smooth_mean': mentions_smooth_mean,
            
            # Comments time series (raw)
            'comments_raw_dates': comments_raw_dates,
            'comments_raw_counts': comments_raw_counts,
            'comments_raw_sum': comments_raw_sum,
            'comments_raw_mean': comments_raw_mean,
            'comments_raw_count': len(comments_raw_counts),
            
            # Comments time series (normalized)
            'comments_norm_dates': comments_norm_dates,
            'comments_norm_values': comments_norm_values,
            'comments_norm_mean': comments_norm_mean,
            
            # Comments time series (smoothed)
            'comments_smooth_dates': comments_smooth_dates,
            'comments_smooth_values': comments_smooth_values,
            'comments_smooth_mean': comments_smooth_mean,
            
            # Data availability flags
            'has_mentions_data': len(mentions_raw_counts) > 0,
            'has_comments_data': len(comments_raw_counts) > 0,
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Summary stats
    n_windows = len(df)
    n_contiguous = df['is_contiguous'].sum()
    n_with_next = df['has_next_window'].sum()
    
    summary = f"{n_windows} windows ({n_contiguous} contiguous, {n_with_next} with next)"
    
    return df, summary


def main():
    ap = argparse.ArgumentParser(description="Slice time series into network-aligned windows")
    ap.add_argument("--hashtags-list", default="03_networks/data/_meta/hashtags_masterlist.csv",
                    help="Path to hashtags masterlist")
    ap.add_argument("--config", default="03_networks/configs/default.yaml",
                    help="Path to network config (for step_days)")
    ap.add_argument("--mentions-csv", default="02t_timeseries/csvs/hashtag_timeseries_mentions_norm_smooth.csv",
                    help="Path to mentions time series CSV")
    ap.add_argument("--comments-csv", default="02t_timeseries/csvs/hashtag_timeseries_comments_norm_smooth.csv",
                    help="Path to comments time series CSV")
    ap.add_argument("--out-dir", default="02t_timeseries/window_slices",
                    help="Output directory for parquet files")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip hashtags with existing output files")
    ap.add_argument("--limit", type=int, help="Process only first N hashtags")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    args = ap.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    step_days = cfg.get('step_days', 30)
    base_dir = cfg.get('base_dir', '03_networks/data')
    
    # Setup output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load masterlist
    masterlist = pd.read_csv(args.hashtags_list)
    hashtags = masterlist['hashtag'].dropna().str.lower().str.strip().tolist()
    
    if args.limit:
        hashtags = hashtags[:args.limit]
    
    print(f"{'='*80}")
    print(f"Slicing time series for {len(hashtags)} hashtags")
    print(f"Step days: {step_days}")
    print(f"Output: {out_dir}")
    if args.dry_run:
        print("[DRY RUN MODE]")
    print(f"{'='*80}\n")
    
    # Load time series data once
    print("Loading time series data...")
    mentions_df, comments_df = load_timeseries_data(args.mentions_csv, args.comments_csv)
    print(f"✓ Loaded {len(mentions_df)} mention records, {len(comments_df)} comment records\n")
    
    # Process each hashtag
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for hashtag in tqdm(hashtags, desc="Processing"):
        out_file = out_dir / f"{hashtag}.parquet"
        
        # Skip if exists
        if args.skip_existing and out_file.exists():
            skip_count += 1
            tqdm.write(f"[skip] {hashtag}: already exists")
            continue
        
        try:
            df, summary = process_hashtag(hashtag, mentions_df, comments_df, base_dir, step_days, args.dry_run)
            
            if df is None:
                tqdm.write(f"[skip] {hashtag}: {summary}")
                skip_count += 1
                continue
            
            # Save parquet
            if not args.dry_run:
                df.to_parquet(out_file, index=False)
            
            tqdm.write(f"[✓] {hashtag}: {summary}")
            success_count += 1
            
        except Exception as e:
            tqdm.write(f"[✗] {hashtag}: {str(e)}")
            fail_count += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  ✓ Success: {success_count}")
    print(f"  ⊘ Skipped: {skip_count}")
    print(f"  ✗ Failed: {fail_count}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()