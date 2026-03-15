#!/usr/bin/env python3
"""
Analyze test predictions per hashtag.

Computes metrics for each hashtag and saves analysis to the same directory.

Usage:
  python analyse_predictions.py --input path/to/test_predictions.csv
  python analyse_predictions.py --input test_predictions.csv --sort-by recall
  python 04_ml_prediction/05_scripts/analyse_predictions.py --input 04_ml_prediction/04_models/trained/exp_041_xgboost_h550_cpinside/test_predictions.csv --sort-by precision 
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime


def load_predictions(predictions_path):
    """Load predictions CSV."""
    print(f"Loading predictions from {predictions_path}...")
    df = pd.read_csv(predictions_path)
    
    # Validate required columns
    required_cols = ['hashtag', 'window_end', 'label', 'y_pred', 'prediction_type']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Convert window_end to datetime for timespan calculation
    df['window_end'] = pd.to_datetime(df['window_end'])
    
    print(f"✓ Loaded {len(df)} predictions for {df['hashtag'].nunique()} hashtags\n")
    return df


def calculate_metrics_per_hashtag(df):
    """
    Calculate metrics for each hashtag.
    
    Returns DataFrame with one row per hashtag.
    """
    print("Calculating per-hashtag metrics...")
    
    results = []
    
    for hashtag in sorted(df['hashtag'].unique()):
        hashtag_df = df[df['hashtag'] == hashtag].copy()
        
        # Sort by window_end for consecutive burst calculation
        hashtag_df = hashtag_df.sort_values('window_end')
        
        # Count confusion matrix elements
        tp = (hashtag_df['prediction_type'] == 'TP').sum()
        tn = (hashtag_df['prediction_type'] == 'TN').sum()
        fp = (hashtag_df['prediction_type'] == 'FP').sum()
        fn = (hashtag_df['prediction_type'] == 'FN').sum()
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Total windows (samples)
        n_windows = len(hashtag_df)
        
        # Timespan (earliest to latest window)
        earliest = hashtag_df['window_end'].min()
        latest = hashtag_df['window_end'].max()
        timespan_days = (latest - earliest).days
        
        # Total bursts (actual positive labels)
        n_bursts = (hashtag_df['label'] == 1).sum()
        
        # Max consecutive bursts
        max_consecutive_bursts = calculate_max_consecutive_bursts(hashtag_df['label'].values)
        
        # Accuracy
        accuracy = (tp + tn) / n_windows if n_windows > 0 else 0.0
        
        results.append({
            'hashtag': hashtag,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'n_windows': n_windows,
            'timespan_days': timespan_days,
            'n_bursts': n_bursts,
            'max_consecutive_bursts': max_consecutive_bursts,
            'earliest_window': earliest.strftime('%Y-%m-%d'),
            'latest_window': latest.strftime('%Y-%m-%d')
        })
    
    results_df = pd.DataFrame(results)
    print(f"✓ Computed metrics for {len(results_df)} hashtags\n")
    
    return results_df


def calculate_max_consecutive_bursts(labels):
    """
    Calculate maximum number of consecutive burst labels (1s).
    
    Args:
        labels: array of 0s and 1s
    
    Returns:
        int: maximum consecutive count of 1s
    """
    if len(labels) == 0:
        return 0
    
    max_consecutive = 0
    current_consecutive = 0
    
    for label in labels:
        if label == 1:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    return max_consecutive


def print_summary_stats(results_df):
    """Print summary statistics across all hashtags."""
    print("="*80)
    print("Summary Statistics (Across All Hashtags)")
    print("="*80)
    
    print(f"\nTotal hashtags: {len(results_df)}")
    print(f"Total windows: {results_df['n_windows'].sum()}")
    print(f"Total bursts: {results_df['n_bursts'].sum()}")
    
    print(f"\nConfusion Matrix Totals:")
    print(f"  TP: {results_df['tp'].sum()}")
    print(f"  TN: {results_df['tn'].sum()}")
    print(f"  FP: {results_df['fp'].sum()}")
    print(f"  FN: {results_df['fn'].sum()}")
    
    print(f"\nMetric Averages (across hashtags):")
    print(f"  Precision: {results_df['precision'].mean():.3f} ± {results_df['precision'].std():.3f}")
    print(f"  Recall:    {results_df['recall'].mean():.3f} ± {results_df['recall'].std():.3f}")
    print(f"  F1:        {results_df['f1'].mean():.3f} ± {results_df['f1'].std():.3f}")
    print(f"  Accuracy:  {results_df['accuracy'].mean():.3f} ± {results_df['accuracy'].std():.3f}")
    
    print(f"\nBurst Statistics:")
    print(f"  Avg bursts per hashtag: {results_df['n_bursts'].mean():.1f}")
    print(f"  Avg windows per hashtag: {results_df['n_windows'].mean():.1f}")
    print(f"  Avg timespan: {results_df['timespan_days'].mean():.0f} days")
    print(f"  Max consecutive bursts (max across hashtags): {results_df['max_consecutive_bursts'].max()}")
    
    print("\n" + "="*80 + "\n")


def save_results(results_df, predictions_path, sort_by='precision', ascending=False, output_dir=None):
    """
    Save analysis results to CSV in the same directory as input.
    
    Args:
        results_df: DataFrame with per-hashtag metrics
        predictions_path: Path to input predictions file
        sort_by: Column to sort by
        ascending: Sort order
        output_dir: Optional output directory (defaults to input file's directory)
    """
    # Sort results
    if sort_by in results_df.columns:
        results_df = results_df.sort_values(sort_by, ascending=ascending)
        print(f"Sorted by: {sort_by} ({'ascending' if ascending else 'descending'})")
    else:
        print(f"Warning: '{sort_by}' not found in results. Using default order.")
    
    # Generate output path
    input_path = Path(predictions_path)
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename = input_path.stem + '_analysis.csv'
    output_path = output_dir / output_filename
    
    # Save
    results_df.to_csv(output_path, index=False)
    print(f"✓ Saved analysis to {output_path}")
    
    # Also save a formatted version for easy reading
    formatted_path = output_dir / (input_path.stem + '_analysis_formatted.txt')
    with open(formatted_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write(f"Per-Hashtag Prediction Analysis\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Sorted by: {sort_by} ({'ascending' if ascending else 'descending'})\n")
        f.write("="*100 + "\n\n")
        
        # Format DataFrame for display
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 20)
        
        # Round numeric columns for display
        display_df = results_df.copy()
        for col in ['precision', 'recall', 'f1', 'accuracy']:
            display_df[col] = display_df[col].round(3)
        
        f.write(display_df.to_string(index=False))
        f.write("\n\n" + "="*100 + "\n")
    
    print(f"✓ Saved formatted report to {formatted_path}")
    
    return output_path


def print_top_bottom(results_df, metric='precision', n=5):
    """Print top and bottom N hashtags by a metric."""
    print(f"\n{'='*80}")
    print(f"Top {n} Hashtags by {metric.upper()}")
    print(f"{'='*80}")
    
    top = results_df.nlargest(n, metric)[['hashtag', metric, 'n_bursts', 'n_windows', 'tp', 'fp', 'fn']]
    print(top.to_string(index=False))
    
    print(f"\n{'='*80}")
    print(f"Bottom {n} Hashtags by {metric.upper()}")
    print(f"{'='*80}")
    
    bottom = results_df.nsmallest(n, metric)[['hashtag', metric, 'n_bursts', 'n_windows', 'tp', 'fp', 'fn']]
    print(bottom.to_string(index=False))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze test predictions per hashtag",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python analyse_predictions.py --input test_predictions.csv
  
  # Sort by recall (descending)
  python analyse_predictions.py --input test_predictions.csv --sort-by recall
  
  # Sort by number of bursts (ascending)
  python analyse_predictions.py --input test_predictions.csv --sort-by n_bursts --ascending
  
  # Sort by F1 score
  python analyse_predictions.py --input test_predictions.csv --sort-by f1
        """
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Path to test_predictions.csv file'
    )
    
    parser.add_argument(
        '--sort-by',
        default='precision',
        help='Column to sort results by (default: precision)'
    )
    
    parser.add_argument(
        '--ascending',
        action='store_true',
        help='Sort in ascending order (default: descending)'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=5,
        help='Number of top/bottom hashtags to display (default: 5)'
    )
    
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory (default: same as input file directory)'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    
    print("="*80)
    print("Per-Hashtag Prediction Analysis")
    print("="*80)
    print(f"Input: {args.input}\n")
    
    # Load predictions
    df = load_predictions(args.input)
    
    # Calculate metrics
    results_df = calculate_metrics_per_hashtag(df)
    
    # Print summary
    print_summary_stats(results_df)
    
    # Save results
    output_path = save_results(
        results_df, 
        args.input, 
        sort_by=args.sort_by, 
        ascending=args.ascending,
        output_dir=args.output_dir
    )
    
    # Print top/bottom performers
    print_top_bottom(results_df, metric=args.sort_by, n=args.top_n)
    
    print("="*80)
    print("Analysis complete!")
    print(f"Results saved to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()