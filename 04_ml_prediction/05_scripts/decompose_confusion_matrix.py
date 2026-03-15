#!/usr/bin/env python3
"""
Decompose Confusion Matrix into Error Categories

Analyzes prediction errors (FPs and FNs) and classifies them as:
- Too early (within 60 days before actual burst)
- Too late (within 60 days after actual burst)  
- Totally wrong / Total miss (no burst within ±60 days)

Usage:
    python decompose_confusion_matrix.py --input test_predictions.csv
    
    python 04_ml_prediction/05_scripts/decompose_confusion_matrix.py \
        --input 04_ml_prediction/04_models/trained/exp_088_lightgbm_h550_0.5rel_75abs_seed1/test_predictions.csv \
        --output-dir 04_ml_prediction/04_models/trained/exp_088_lightgbm_h550_0.5rel_75abs_seed1
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from sklearn.metrics import confusion_matrix


def load_predictions(input_path):
    """Load test predictions CSV."""
    df = pd.read_csv(input_path)
    
    # Handle different column naming
    if 'label' in df.columns:
        df = df.rename(columns={'label': 'y_true'})
    if 'y_pred' not in df.columns and 'prediction_type' in df.columns:
        df['y_pred'] = df['prediction_type'].isin(['TP', 'FP']).astype(int)
    
    df['window_end'] = pd.to_datetime(df['window_end'])
    df = df.sort_values(['hashtag', 'window_end']).reset_index(drop=True)
    
    return df


def decompose_false_positives(df, lookback_days=60, lookahead_days=60):
    """
    Decompose false positives into: too early, too late, totally wrong.
    
    Args:
        df: DataFrame with predictions
        lookback_days: Days to look back for "too late"
        lookahead_days: Days to look ahead for "too early"
    
    Returns:
        DataFrame with FP classifications
    """
    fps = df[(df['y_true'] == 0) & (df['y_pred'] == 1)].copy()
    
    if len(fps) == 0:
        return pd.DataFrame()
    
    fps['fp_category'] = 'totally_wrong'  # Default
    
    for hashtag in fps['hashtag'].unique():
        hashtag_mask = df['hashtag'] == hashtag
        hashtag_df = df[hashtag_mask].copy()
        
        # Get all true bursts for this hashtag
        true_bursts = hashtag_df[hashtag_df['y_true'] == 1]['window_end'].values
        
        # Classify each FP
        fp_mask = (fps['hashtag'] == hashtag)
        
        for idx in fps[fp_mask].index:
            fp_date = fps.loc[idx, 'window_end']
            
            # Check if within 60 days BEFORE any true burst (too early)
            for burst_date in true_bursts:
                days_before = (burst_date - fp_date).days
                if 0 < days_before <= lookahead_days:
                    fps.loc[idx, 'fp_category'] = 'too_early'
                    break
            
            # Check if within 60 days AFTER any true burst (too late)
            if fps.loc[idx, 'fp_category'] == 'totally_wrong':  # Only if not already too_early
                for burst_date in true_bursts:
                    days_after = (fp_date - burst_date).days
                    if 0 < days_after <= lookback_days:
                        fps.loc[idx, 'fp_category'] = 'too_late'
                        break
    
    return fps


def decompose_false_negatives(df, lookback_days=60, lookahead_days=60):
    """
    Decompose false negatives into: too early, too late, total miss.
    
    Args:
        df: DataFrame with predictions
        lookback_days: Days to look back for "too early"
        lookahead_days: Days to look ahead for "too late"
    
    Returns:
        DataFrame with FN classifications
    """
    fns = df[(df['y_true'] == 1) & (df['y_pred'] == 0)].copy()
    
    if len(fns) == 0:
        return pd.DataFrame()
    
    fns['fn_category'] = 'total_miss'  # Default
    
    for hashtag in fns['hashtag'].unique():
        hashtag_mask = df['hashtag'] == hashtag
        hashtag_df = df[hashtag_mask].copy()
        
        # Get all positive predictions for this hashtag
        positive_preds = hashtag_df[hashtag_df['y_pred'] == 1]['window_end'].values
        
        # Classify each FN
        fn_mask = (fns['hashtag'] == hashtag)
        
        for idx in fns[fn_mask].index:
            fn_date = fns.loc[idx, 'window_end']
            
            # Check if positive prediction within 60 days BEFORE this missed burst (too early)
            for pred_date in positive_preds:
                days_before = (fn_date - pred_date).days
                if 0 < days_before <= lookback_days:
                    fns.loc[idx, 'fn_category'] = 'too_early'
                    break
            
            # Check if positive prediction within 60 days AFTER this missed burst (too late)
            if fns.loc[idx, 'fn_category'] == 'total_miss':  # Only if not already too_early
                for pred_date in positive_preds:
                    days_after = (pred_date - fn_date).days
                    if 0 < days_after <= lookahead_days:
                        fns.loc[idx, 'fn_category'] = 'too_late'
                        break
    
    return fns


def print_decomposition_results(df, fps_decomposed, fns_decomposed):
    """Print comprehensive decomposition analysis."""
    
    # Standard confusion matrix
    y_true = df['y_true']
    y_pred = df['y_pred']
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    total = len(df)
    
    print("\n" + "="*80)
    print("CONFUSION MATRIX DECOMPOSITION")
    print("="*80)
    
    print("\n" + "-"*80)
    print("STANDARD CONFUSION MATRIX")
    print("-"*80)
    print(f"{'Category':<20} {'Count':<15} {'Percentage':<15}")
    print("-"*80)
    print(f"{'True Positives':<20} {tp:<15} {tp/total*100:>6.2f}%")
    print(f"{'False Positives':<20} {fp:<15} {fp/total*100:>6.2f}%")
    print(f"{'True Negatives':<20} {tn:<15} {tn/total*100:>6.2f}%")
    print(f"{'False Negatives':<20} {fn:<15} {fn/total*100:>6.2f}%")
    print("-"*80)
    print(f"{'Total':<20} {total:<15} {100.0:>6.2f}%")
    print()
    
    # FALSE POSITIVES DECOMPOSITION
    print("\n" + "="*80)
    print("FALSE POSITIVES DECOMPOSITION")
    print("="*80)
    
    if len(fps_decomposed) > 0:
        fp_counts = fps_decomposed['fp_category'].value_counts()
        
        print(f"\n{'Category':<20} {'Count':<15} {'% of FPs':<15} {'% of Total':<15}")
        print("-"*80)
        
        categories = ['too_early', 'too_late', 'totally_wrong']
        category_names = {
            'too_early': 'Too Early',
            'too_late': 'Too Late', 
            'totally_wrong': 'Totally Wrong'
        }
        
        for cat in categories:
            count = fp_counts.get(cat, 0)
            pct_fp = count / fp * 100 if fp > 0 else 0
            pct_total = count / total * 100
            print(f"{category_names[cat]:<20} {count:<15} {pct_fp:>6.2f}%        {pct_total:>6.2f}%")
        
        print("-"*80)
        print(f"{'Total FPs':<20} {fp:<15} {100.0:>6.2f}%        {fp/total*100:>6.2f}%")
        
        # Detailed breakdown
        print("\nInterpretation:")
        too_early = fp_counts.get('too_early', 0)
        too_late = fp_counts.get('too_late', 0)
        totally_wrong = fp_counts.get('totally_wrong', 0)
        
        if too_early > 0:
            print(f"  • {too_early} FPs ({too_early/fp*100:.1f}%) are TOO EARLY")
            print(f"    → Model predicted bursts 1-60 days before they occurred")
        
        if too_late > 0:
            print(f"  • {too_late} FPs ({too_late/fp*100:.1f}%) are TOO LATE")
            print(f"    → Model predicted bursts 1-60 days after they occurred")
        
        if totally_wrong > 0:
            print(f"  • {totally_wrong} FPs ({totally_wrong/fp*100:.1f}%) are TOTALLY WRONG")
            print(f"    → Random false alarms with no nearby bursts (±60 days)")
    else:
        print("\nNo false positives to decompose.")
    
    # FALSE NEGATIVES DECOMPOSITION
    print("\n" + "="*80)
    print("FALSE NEGATIVES DECOMPOSITION")
    print("="*80)
    
    if len(fns_decomposed) > 0:
        fn_counts = fns_decomposed['fn_category'].value_counts()
        
        print(f"\n{'Category':<20} {'Count':<15} {'% of FNs':<15} {'% of Total':<15}")
        print("-"*80)
        
        categories = ['too_early', 'too_late', 'total_miss']
        category_names = {
            'too_early': 'Too Early',
            'too_late': 'Too Late',
            'total_miss': 'Total Miss'
        }
        
        for cat in categories:
            count = fn_counts.get(cat, 0)
            pct_fn = count / fn * 100 if fn > 0 else 0
            pct_total = count / total * 100
            print(f"{category_names[cat]:<20} {count:<15} {pct_fn:>6.2f}%        {pct_total:>6.2f}%")
        
        print("-"*80)
        print(f"{'Total FNs':<20} {fn:<15} {100.0:>6.2f}%        {fn/total*100:>6.2f}%")
        
        # Detailed breakdown
        print("\nInterpretation:")
        too_early = fn_counts.get('too_early', 0)
        too_late = fn_counts.get('too_late', 0)
        total_miss = fn_counts.get('total_miss', 0)
        
        if too_early > 0:
            print(f"  • {too_early} FNs ({too_early/fn*100:.1f}%) are TOO EARLY")
            print(f"    → Model predicted 1-60 days before the actual burst")
        
        if too_late > 0:
            print(f"  • {too_late} FNs ({too_late/fn*100:.1f}%) are TOO LATE")
            print(f"    → Model predicted 1-60 days after the actual burst")
        
        if total_miss > 0:
            print(f"  • {total_miss} FNs ({total_miss/fn*100:.1f}%) are TOTAL MISSES")
            print(f"    → Model failed completely (no predictions within ±60 days)")
    else:
        print("\nNo false negatives to decompose.")
    
    # SUMMARY
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\nError Distribution:")
    print(f"  Total Predictions: {total}")
    print(f"  Correct: {tp + tn} ({(tp+tn)/total*100:.1f}%)")
    print(f"  Errors: {fp + fn} ({(fp+fn)/total*100:.1f}%)")
    
    if fp > 0:
        print(f"\n  False Positives: {fp} ({fp/total*100:.1f}% of total)")
        if len(fps_decomposed) > 0:
            fp_counts = fps_decomposed['fp_category'].value_counts()
            timing_errors = fp_counts.get('too_early', 0) + fp_counts.get('too_late', 0)
            print(f"    - Timing errors: {timing_errors} ({timing_errors/fp*100:.1f}% of FPs)")
            print(f"    - Random errors: {fp_counts.get('totally_wrong', 0)} ({fp_counts.get('totally_wrong', 0)/fp*100:.1f}% of FPs)")
    
    if fn > 0:
        print(f"\n  False Negatives: {fn} ({fn/total*100:.1f}% of total)")
        if len(fns_decomposed) > 0:
            fn_counts = fns_decomposed['fn_category'].value_counts()
            timing_errors = fn_counts.get('too_early', 0) + fn_counts.get('too_late', 0)
            print(f"    - Timing errors: {timing_errors} ({timing_errors/fn*100:.1f}% of FNs)")
            print(f"    - Complete misses: {fn_counts.get('total_miss', 0)} ({fn_counts.get('total_miss', 0)/fn*100:.1f}% of FNs)")
    
    print("\n" + "="*80 + "\n")


def save_detailed_results(df, fps_decomposed, fns_decomposed, output_dir):
    """Save detailed decomposition results to CSV files."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save FP decomposition
    if len(fps_decomposed) > 0:
        fp_output = output_dir / "false_positives_decomposed.csv"
        fps_decomposed.to_csv(fp_output, index=False)
        print(f"✓ Saved FP decomposition to: {fp_output}")
    
    # Save FN decomposition
    if len(fns_decomposed) > 0:
        fn_output = output_dir / "false_negatives_decomposed.csv"
        fns_decomposed.to_csv(fn_output, index=False)
        print(f"✓ Saved FN decomposition to: {fn_output}")
    
    # Save summary statistics
    summary = []
    
    y_true = df['y_true']
    y_pred = df['y_pred']
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = len(df)
    
    summary.append({
        'category': 'TP',
        'count': tp,
        'pct_total': tp/total*100
    })
    
    if len(fps_decomposed) > 0:
        fp_counts = fps_decomposed['fp_category'].value_counts()
        for cat in ['too_early', 'too_late', 'totally_wrong']:
            count = fp_counts.get(cat, 0)
            summary.append({
                'category': f'FP_{cat}',
                'count': count,
                'pct_total': count/total*100,
                'pct_of_error_type': count/fp*100 if fp > 0 else 0
            })
    
    summary.append({
        'category': 'TN',
        'count': tn,
        'pct_total': tn/total*100
    })
    
    if len(fns_decomposed) > 0:
        fn_counts = fns_decomposed['fn_category'].value_counts()
        for cat in ['too_early', 'too_late', 'total_miss']:
            count = fn_counts.get(cat, 0)
            summary.append({
                'category': f'FN_{cat}',
                'count': count,
                'pct_total': count/total*100,
                'pct_of_error_type': count/fn*100 if fn > 0 else 0
            })
    
    summary_df = pd.DataFrame(summary)
    summary_output = output_dir / "decomposition_summary.csv"
    summary_df.to_csv(summary_output, index=False)
    print(f"✓ Saved summary to: {summary_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Decompose confusion matrix into interpretable error categories"
    )
    parser.add_argument('--input', required=True,
                       help='Path to test_predictions.csv')
    parser.add_argument('--output-dir', default=None,
                       help='Directory for detailed output CSVs (optional)')
    parser.add_argument('--lookback', type=int, default=60,
                       help='Days to look back for "too late" classification (default: 60)')
    parser.add_argument('--lookahead', type=int, default=60,
                       help='Days to look ahead for "too early" classification (default: 60)')
    
    args = parser.parse_args()
    
    print(f"\nLoading predictions from: {args.input}")
    df = load_predictions(args.input)
    print(f"✓ Loaded {len(df)} predictions ({df['hashtag'].nunique()} hashtags)")
    
    print(f"\nDecomposing errors (±{args.lookback} day window)...")
    
    # Decompose FPs
    fps_decomposed = decompose_false_positives(df, args.lookback, args.lookahead)
    print(f"  ✓ Analyzed {len(fps_decomposed)} false positives")
    
    # Decompose FNs
    fns_decomposed = decompose_false_negatives(df, args.lookback, args.lookahead)
    print(f"  ✓ Analyzed {len(fns_decomposed)} false negatives")
    
    # Print results
    print_decomposition_results(df, fps_decomposed, fns_decomposed)
    
    # Save detailed results if requested
    if args.output_dir:
        save_detailed_results(df, fps_decomposed, fns_decomposed, args.output_dir)


if __name__ == "__main__":
    main()