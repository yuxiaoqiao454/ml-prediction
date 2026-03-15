#!/usr/bin/env python3
"""
Relaxed Burst Prediction Evaluation

Evaluates predictions with temporal tolerance:
- TP if positive prediction within N months before or M months after actual burst

Usage:
    python evaluate_relaxed.py --input test_predictions.csv --before 1 --after 1
    python evaluate_relaxed.py --input test_predictions.csv --verbose
    python 04_ml_prediction/05_scripts/evaluate_relaxed.py \
    --input 04_ml_prediction/04_models/trained/exp_100_lightgbm_h550_0.5rel_ts_net_gae_inflattr_spec_seed4/test_predictions.csv \
    --output 04_ml_prediction/04_models/trained/exp_100_lightgbm_h550_0.5rel_ts_net_gae_inflattr_spec_seed4/test_predictions_relaxed.csv \
    --before 3 \
    --after 1
"""

import argparse
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def load_predictions(input_path):
    """Load test predictions CSV."""
    df = pd.read_csv(input_path)
    
    # Handle both column naming conventions
    if 'label' in df.columns:
        df = df.rename(columns={'label': 'y_true'})
    if 'y_pred' not in df.columns and 'prediction_type' in df.columns:
        df['y_pred'] = df['prediction_type'].isin(['TP', 'FP']).astype(int)
    
    df['window_end'] = pd.to_datetime(df['window_end'])
    return df.sort_values(['hashtag', 'window_end']).reset_index(drop=True)


def apply_relaxed_criteria(df, months_before=2, months_after=1):
    """
    Apply relaxed evaluation with temporal tolerance.
    
    Logic:
    - For each actual burst, look for positive predictions in window
    - Window = [burst_date - months_before, burst_date + months_after]
    - If found: those predictions become TP
    - If not found: burst remains FN
    """
    df = df.copy()
    df['matched_to_burst'] = False
    df['match_type'] = 'none'
    df['result_relaxed'] = 'TN'
    
    for hashtag in df['hashtag'].unique():
        mask = df['hashtag'] == hashtag
        hashtag_df = df[mask].copy()
        actual_bursts = hashtag_df[hashtag_df['y_true'] == 1]
        
        for burst_idx, burst_row in actual_bursts.iterrows():
            burst_date = burst_row['window_end']
            window_start = burst_date - pd.DateOffset(months=months_before)
            window_end = burst_date + pd.DateOffset(months=months_after)
            
            in_window = (
                (hashtag_df['window_end'] >= window_start) &
                (hashtag_df['window_end'] <= window_end)
            )
            
            predictions_in_window = hashtag_df[in_window]
            has_positive_pred = (predictions_in_window['y_pred'] == 1).any()
            
            if has_positive_pred:
                positive_preds = predictions_in_window[predictions_in_window['y_pred'] == 1]
                for pred_idx in positive_preds.index:
                    pred_date = hashtag_df.loc[pred_idx, 'window_end']
                    df.loc[pred_idx, 'matched_to_burst'] = True
                    
                    if pred_date == burst_date:
                        df.loc[pred_idx, 'match_type'] = 'exact'
                    elif pred_date < burst_date:
                        df.loc[pred_idx, 'match_type'] = 'before'
                    else:
                        df.loc[pred_idx, 'match_type'] = 'after'
            else:
                df.loc[burst_idx, 'result_relaxed'] = 'FN'
    
    df.loc[(df['y_pred'] == 1) & (df['matched_to_burst']), 'result_relaxed'] = 'TP'
    df.loc[(df['y_pred'] == 1) & (~df['matched_to_burst']), 'result_relaxed'] = 'FP'
    df.loc[(df['y_true'] == 0) & (df['y_pred'] == 0), 'result_relaxed'] = 'TN'
    
    return df


def compute_metrics(df, use_relaxed=False):
    """Compute classification metrics."""
    if use_relaxed:
        y_pred = ((df['result_relaxed'] == 'TP') | (df['result_relaxed'] == 'FP')).astype(int)
        y_true = ((df['result_relaxed'] == 'TP') | (df['result_relaxed'] == 'FN')).astype(int)
    else:
        y_true = df['y_true']
        y_pred = df['y_pred']
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
    }


def print_comparison(orig, relax, before, after):
    """Print side-by-side metric comparison."""
    print("\n" + "="*80)
    print(f"RELAXED EVALUATION RESULTS")
    print(f"Tolerance: {before} months before, {after} month(s) after actual burst")
    print("="*80)
    
    print(f"\n{'Metric':<20} {'Original':<15} {'Relaxed':<15} {'Improvement':<15}")
    print("-"*80)
    
    for m in ['accuracy', 'precision', 'recall', 'f1']:
        o, r = orig[m], relax[m]
        print(f"{m.capitalize():<20} {o:>7.1%}        {r:>7.1%}        {r-o:>+7.1%}")
    
    print("\n" + "-"*80)
    print(f"{'Confusion Matrix':<20} {'Original':<15} {'Relaxed':<15} {'Change':<15}")
    print("-"*80)
    
    for k in ['tp', 'fp', 'tn', 'fn']:
        o, r = orig[k], relax[k]
        print(f"{k.upper():<20} {o:>7}        {r:>7}        {r-o:>+7}")
    
    print("="*80)
    
    print("\nKey Changes:")
    fp_to_tp = orig['fp'] - relax['fp']
    fn_to_tp = orig['fn'] - relax['fn']
    if fp_to_tp > 0:
        print(f"  • {fp_to_tp} False Positives → True Positives (early warnings)")
    if fn_to_tp > 0:
        print(f"  • {fn_to_tp} False Negatives → True Positives (late detections)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate with temporal tolerance")
    parser.add_argument('--input', required=True, help='Path to test_predictions.csv')
    parser.add_argument('--output', default=None, help='Output path')
    parser.add_argument('--before', type=int, default=2, help='Months before (default: 2)')
    parser.add_argument('--after', type=int, default=1, help='Months after (default: 1)')
    parser.add_argument('--verbose', action='store_true', help='Detailed stats')
    args = parser.parse_args()
    
    if args.output is None:
        p = Path(args.input)
        args.output = str(p.parent / f"{p.stem}_relaxed.csv")
    
    print("="*80)
    print("RELAXED BURST PREDICTION EVALUATION")
    print("="*80)
    print(f"\nInput:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Tolerance: {args.before} months before, {args.after} month(s) after")
    
    df = load_predictions(args.input)
    print(f"\n✓ Loaded {len(df)} predictions")
    print(f"  Hashtags: {df['hashtag'].nunique()}")
    print(f"  Actual bursts: {(df['y_true']==1).sum()}")
    print(f"  Predicted bursts: {(df['y_pred']==1).sum()}")
    
    orig_metrics = compute_metrics(df, use_relaxed=False)
    
    print(f"\nApplying relaxed criteria...")
    df_relaxed = apply_relaxed_criteria(df, args.before, args.after)
    
    relax_metrics = compute_metrics(df_relaxed, use_relaxed=True)
    
    print_comparison(orig_metrics, relax_metrics, args.before, args.after)
    
    if args.verbose:
        print("="*80)
        print("DETAILED BREAKDOWN")
        print("="*80)
        print("\nMatch types:")
        print(df_relaxed['match_type'].value_counts())
        print("\nRelaxed results:")
        print(df_relaxed['result_relaxed'].value_counts())
        
        matched = df_relaxed[(df_relaxed['y_pred'] == 1) & (df_relaxed['matched_to_burst'])]
        print(f"\nPositive predictions matched: {len(matched)}")
        print(f"  Exact: {(matched['match_type']=='exact').sum()}")
        print(f"  Before: {(matched['match_type']=='before').sum()}")
        print(f"  After: {(matched['match_type']=='after').sum()}")
        print()
    
    df_relaxed.to_csv(args.output, index=False)
    print(f"✓ Saved to: {args.output}\n")


if __name__ == "__main__":
    main()