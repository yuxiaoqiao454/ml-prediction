#!/usr/bin/env python3
"""
Visualize and compare labeling rules and prediction models.

Creates comparison matrix showing:
- Row 1: Labels from different labeling rules
- Rows 2+: Predictions from different models

Usage:
    # All rules and models for a hashtag
    python plot_labeling_predictions.py --hashtag 4thofjuly --timeseries mentions
    
    # Specific rules and models
    python plot_labeling_predictions.py \
        --hashtag inmyfeelings \
        --timeseries comments \
        --rules h550_0.5rel h550_cpinside \
        --models exp_040_random_forest_h550_0.5rel exp_041_xgboost_h550_0.5rel
    
    # Labels only (no predictions)
    python plot_labeling_predictions.py \
        --hashtag christmas \
        --rules h550_0.5rel h550_1.0rel \
        --output plots/christmas_labels.png
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_timeseries(hashtag: str, timeseries_type: str = 'mentions') -> pd.DataFrame:
    """
    Load timeseries data for a hashtag.
    
    Args:
        hashtag: Hashtag name
        timeseries_type: 'mentions' or 'comments'
    
    Returns:
        DataFrame with columns: date, value (raw counts)
    """
    if timeseries_type == 'mentions':
        csv_path = Path("02t_timeseries/csvs/hashtag_timeseries_mentions_norm_smooth.csv")
        value_col = 'mentions'  # Raw count column
    else:
        csv_path = Path("02t_timeseries/csvs/hashtag_timeseries_comments_norm_smooth.csv")
        value_col = 'total_comments'  # Raw count column
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Timeseries file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df['hashtag'] = df['hashtag'].astype(str).str.lower()
    df = df[df['hashtag'] == hashtag.lower()].copy()
    
    if len(df) == 0:
        raise ValueError(f"No timeseries data found for hashtag: {hashtag}")
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Return simplified dataframe
    return df[['date', value_col]].rename(columns={value_col: 'value'})


def load_labels(hashtag: str, labeling_rule: str) -> pd.DataFrame:
    """
    Load labels for a specific labeling rule.
    
    Args:
        hashtag: Hashtag name
        labeling_rule: Label file stem (e.g., 'h550_0.5rel')
    
    Returns:
        DataFrame with labels for this hashtag
    """
    labels_path = Path(f"04_ml_prediction/02_labels/labels_{labeling_rule}.parquet")
    
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    df = pd.read_parquet(labels_path)
    df['hashtag'] = df['hashtag'].astype(str).str.lower()
    df = df[df['hashtag'] == hashtag.lower()].copy()
    
    if len(df) == 0:
        return pd.DataFrame()  # Empty - no labels for this hashtag
    
    # Convert dates
    df['window_start'] = pd.to_datetime(df['window_start'])
    df['window_end'] = pd.to_datetime(df['window_end'])
    
    return df


def load_predictions(hashtag: str, experiment_id: str, use_relaxed: bool = False) -> pd.DataFrame:
    """
    Load predictions from a trained model.
    
    Args:
        hashtag: Hashtag name
        experiment_id: Full experiment ID (e.g., 'exp_040_random_forest_h550_0.5rel')
        use_relaxed: If True, load relaxed predictions (test_predictions_relaxed.csv)
    
    Returns:
        DataFrame with predictions for this hashtag
    """
    if use_relaxed:
        pred_filename = "test_predictions_relaxed.csv"
    else:
        pred_filename = "test_predictions.csv"
    
    pred_path = Path(f"04_ml_prediction/04_models/trained/{experiment_id}/{pred_filename}")
    
    if not pred_path.exists():
        if use_relaxed:
            raise FileNotFoundError(
                f"Relaxed predictions not found: {pred_path}\n"
                f"Hint: Run evaluate_relaxed.py first to generate relaxed predictions"
            )
        else:
            raise FileNotFoundError(f"Predictions file not found: {pred_path}")
    
    df = pd.read_csv(pred_path)
    df['hashtag'] = df['hashtag'].astype(str).str.lower()
    df = df[df['hashtag'] == hashtag.lower()].copy()
    
    if len(df) == 0:
        return pd.DataFrame()
    
    df['window_end'] = pd.to_datetime(df['window_end'])
    
    return df

def load_all_windows(hashtag: str) -> pd.DataFrame:
    """
    Load ALL windows for a hashtag (including unlabeled ones).
    
    Args:
        hashtag: Hashtag name
    
    Returns:
        DataFrame with all windows
    """
    window_path = Path(f"02t_timeseries/window_slices/{hashtag.lower()}.parquet")
    
    if not window_path.exists():
        return pd.DataFrame()
    
    df = pd.read_parquet(window_path)
    df['window_start'] = pd.to_datetime(df['window_start'])
    df['window_end'] = pd.to_datetime(df['window_end'])
    
    return df


# ============================================================================
# Discovery Functions
# ============================================================================

def discover_labeling_rules() -> List[str]:
    """Find all available labeling rules."""
    labels_dir = Path("04_ml_prediction/02_labels/")
    
    if not labels_dir.exists():
        return []
    
    rules = []
    for f in labels_dir.glob("labels_*.parquet"):
        rule = f.stem.replace('labels_', '')
        rules.append(rule)
    
    return sorted(rules)


def discover_models(labeling_rule: str = None) -> List[str]:
    """
    Find all available trained models.
    
    Args:
        labeling_rule: If specified, only return models trained on this rule
    
    Returns:
        List of experiment IDs
    """
    trained_dir = Path("04_ml_prediction/04_models/trained/")
    
    if not trained_dir.exists():
        return []
    
    models = []
    for exp_dir in trained_dir.glob("exp_*"):
        if not exp_dir.is_dir():
            continue
        
        exp_id = exp_dir.name
        
        # Filter by labeling rule if specified
        if labeling_rule and labeling_rule not in exp_id:
            continue
        
        # Check if test_predictions.csv exists
        if (exp_dir / "test_predictions.csv").exists():
            models.append(exp_id)
    
    return sorted(models)


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_label_subplot(
    ax,
    hashtag: str,
    timeseries_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    all_windows_df: pd.DataFrame,
    timeseries_type: str,
    labeling_rule: str
):
    """
    Plot a single label subplot.
    
    Shows timeseries with:
    - Window boundaries (vertical dashed lines)
    - Change points (red X)
    - Burst windows (blue shading)
    - Unlabeled windows (gray shading + "NA")
    """
    # Plot timeseries
    ax.plot(timeseries_df['date'], timeseries_df['value'], 
            'k-', linewidth=1.5, zorder=2)
    
    # Determine which column to use for labels and CPs
    if timeseries_type == 'mentions':
        label_col = 'label_burst_mentions'
        cp_col = 'cp_position_mentions'
        has_cp_col = 'has_cp_inside_boundary_mentions'
    else:
        label_col = 'label_burst_comments'
        cp_col = 'cp_position_comments'
        has_cp_col = 'has_cp_inside_boundary_comments'
    
    # Get y-limits for shading
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    
    # Track which windows have labels (convert to datetime for comparison)
    if len(labels_df) > 0:
        labeled_window_ends = set(pd.to_datetime(labels_df['window_end']))
    else:
        labeled_window_ends = set()
    
    # Draw all windows
    for _, window in all_windows_df.iterrows():
        window_end = window['window_end']
        window_start = window['window_start']
        
        # Vertical line at window_end
        ax.axvline(window_end, color='black', linestyle='--', 
                  linewidth=0.8, alpha=0.5, zorder=1)
        
        # Check if this window has a label
        if window_end in labeled_window_ends:
            # Get label for this window (match by converting window_end to string for lookup)
            window_end_str = window_end.strftime('%Y-%m-%d')
            label_matches = labels_df[labels_df['window_end'] == window_end_str]
            
            if len(label_matches) == 0:
                # Shouldn't happen, but handle gracefully
                continue
            
            label_row = label_matches.iloc[0]
            label = label_row[label_col] if label_col in label_row else None
            
            if pd.notna(label) and label == 1:
                # Burst window - blue shading
                ax.axvspan(window_start, window_end, 
                          color='blue', alpha=0.25, zorder=0)
                
                # Add "Burst" text if space allows
                mid_date = window_start + (window_end - window_start) / 2
                text_y = ymax - y_range * 0.1
                ax.text(mid_date, text_y, 'Burst', 
                       ha='center', va='top', fontsize=8, 
                       color='blue', weight='bold')
            
            # Plot change point if present
            if has_cp_col in label_row and label_row[has_cp_col]:
                cp_date = label_row[cp_col]
                if pd.notna(cp_date):
                    cp_date = pd.to_datetime(cp_date)
                    
                    # Get y-value at this date (interpolate)
                    mask = (timeseries_df['date'] >= cp_date - pd.Timedelta(days=1)) & \
                           (timeseries_df['date'] <= cp_date + pd.Timedelta(days=1))
                    nearby = timeseries_df[mask]
                    
                    if len(nearby) > 0:
                        cp_y = nearby['value'].mean()
                    else:
                        cp_y = ymax * 0.5
                    
                    # Plot red X
                    ax.scatter(cp_date, cp_y, marker='x', s=150, 
                             color='red', linewidths=3, zorder=5)
        
        else:
            # Unlabeled window - gray shading
            ax.axvspan(window_start, window_end, 
                      color='gray', alpha=0.15, zorder=0)
            
            # Add "NA" text
            mid_date = window_start + (window_end - window_start) / 2
            text_y = ymax - y_range * 0.1
            ax.text(mid_date, text_y, 'NA', 
                   ha='center', va='top', fontsize=8, 
                   color='gray', style='italic')
    
    # Formatting
    ax.set_ylabel(f'{timeseries_type.capitalize()}\n(raw count)', fontsize=10)
    ax.set_title(f'Labels: {labeling_rule}', fontsize=11, weight='bold')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # X-axis: only show window_end dates
    ax.set_xticks(all_windows_df['window_end'].values)
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='blue', alpha=0.25, label='Burst (label=1)'),
        mpatches.Patch(color='gray', alpha=0.15, label='No label (NA)'),
        plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='red', 
                  markersize=10, markeredgewidth=2, label='Change point')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8, framealpha=0.9)


def plot_prediction_subplot(
    ax,
    hashtag: str,
    timeseries_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    all_windows_df: pd.DataFrame,
    timeseries_type: str,
    model_name: str,
    use_relaxed: bool = False  # NEW PARAMETER
):
    """
    Plot a single prediction subplot.
    
    Shows timeseries with:
    - Window boundaries
    - Color-coded predictions (TP=green, FP=red, FN=yellow, TN=none)
    - Text labels (TP/FP/FN/TN/NA)
    - Relaxed mode: Shows match_type annotations
    """
    # Plot timeseries
    ax.plot(timeseries_df['date'], timeseries_df['value'], 
            'k-', linewidth=1.5, zorder=2)
    
    # Get y-limits
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    
    # Track which windows have predictions (convert to datetime)
    if len(predictions_df) > 0:
        predicted_window_ends = set(pd.to_datetime(predictions_df['window_end']))
    else:
        predicted_window_ends = set()
    
    # Color mapping
    colors = {
        'TP': ('green', 0.25),
        'FP': ('red', 0.25),
        'FN': ('yellow', 0.25),
        'TN': (None, 0),  # No shading
    }
    
    # Draw all windows
    for _, window in all_windows_df.iterrows():
        window_end = window['window_end']
        window_start = window['window_start']
        
        # Vertical line
        ax.axvline(window_end, color='black', linestyle='--', 
                  linewidth=0.8, alpha=0.5, zorder=1)
        
        # Check if prediction exists
        if window_end in predicted_window_ends:
            window_end_str = window_end.strftime('%Y-%m-%d')
            pred_matches = predictions_df[predictions_df['window_end'] == window_end_str]
            
            if len(pred_matches) == 0:
                continue
            
            pred_row = pred_matches.iloc[0]
            
            # Get prediction type based on mode
            if use_relaxed and 'result_relaxed' in pred_row:
                pred_type = pred_row['result_relaxed']
                match_type = pred_row.get('match_type', 'none')
            else:
                pred_type = pred_row['prediction_type']
                match_type = None
            
            # Shade window
            color, alpha = colors.get(pred_type, (None, 0))
            if color is not None:
                ax.axvspan(window_start, window_end, 
                          color=color, alpha=alpha, zorder=0)
            
            # Add text label
            mid_date = window_start + (window_end - window_start) / 2
            text_y = ymax - y_range * 0.1
            
            # For relaxed mode, show match type for TPs
            if use_relaxed and pred_type == 'TP' and match_type and match_type != 'none':
                label_text = f"{pred_type}\n({match_type})"
                fontsize = 7
            else:
                label_text = pred_type
                fontsize = 8
            
            ax.text(mid_date, text_y, label_text, 
                   ha='center', va='top', fontsize=fontsize, 
                   weight='bold', color='black')
        
        else:
            # No prediction - gray shading
            ax.axvspan(window_start, window_end, 
                      color='gray', alpha=0.15, zorder=0)
            
            mid_date = window_start + (window_end - window_start) / 2
            text_y = ymax - y_range * 0.1
            ax.text(mid_date, text_y, 'NA', 
                   ha='center', va='top', fontsize=8, 
                   color='gray', style='italic')
    
    # Formatting
    ax.set_ylabel(f'{timeseries_type.capitalize()}\n(raw count)', fontsize=10)
    
    # Parse model name for title
    model_display = model_name.replace('_', ' ').replace('exp ', 'Exp')
    if use_relaxed:
        model_display += " (Relaxed)"
    ax.set_title(f'Predictions: {model_display}', fontsize=11, weight='bold')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # X-axis
    ax.set_xticks(all_windows_df['window_end'].values)
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='green', alpha=0.25, label='True Positive (TP)'),
        mpatches.Patch(color='red', alpha=0.25, label='False Positive (FP)'),
        mpatches.Patch(color='yellow', alpha=0.25, label='False Negative (FN)'),
        mpatches.Patch(color='white', edgecolor='black', label='True Negative (TN)'),
        mpatches.Patch(color='gray', alpha=0.15, label='No prediction (NA)')
    ]
    
    if use_relaxed:
        legend_elements.append(
            mpatches.Patch(facecolor='none', edgecolor='none', 
                          label='(exact/before/after)')
        )
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8, 
             framealpha=0.9, ncol=2)

def plot_labeling_comparison(
    hashtag: str,
    timeseries_type: str = 'mentions',
    labeling_rules: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    figsize_per_subplot: Tuple[int, int] = (12, 3),
    dpi: int = 150,
    use_relaxed: bool = False 
):
    """
    Generate comparison matrix of labels and predictions.
    
    Args:
        hashtag: Hashtag to plot (e.g., '4thofjuly')
        timeseries_type: 'mentions' or 'comments'
        labeling_rules: List of label file stems. If None, auto-discover
        models: List of experiment IDs. If None, labels only
        output_path: Where to save figure
        figsize_per_subplot: Size of each subplot (width, height)
        dpi: Output resolution
        use_relaxed: If True, load relaxed predictions instead of exact
    
    Example:
        plot_labeling_comparison(
            hashtag='4thofjuly',
            timeseries_type='mentions',
            labeling_rules=['h550_0.5rel'],
            models=['exp_040_random_forest_h550_0.5rel'],
            output_path='plots/4thofjuly.png'
        )
    """
    print(f"\n{'='*80}")
    print(f"Generating Comparison Plot for: {hashtag}")
    if use_relaxed:
        print(f"Mode: RELAXED predictions")
    print(f"{'='*80}\n")
    
    # Auto-discover if needed
    if labeling_rules is None:
        labeling_rules = discover_labeling_rules()
        print(f"Auto-discovered {len(labeling_rules)} labeling rules")
    
    if models is None:
        models = []
        print("No models specified - plotting labels only")
    
    # Load data
    print(f"\nLoading timeseries ({timeseries_type})...")
    timeseries_df = load_timeseries(hashtag, timeseries_type)
    print(f"  ✓ Loaded {len(timeseries_df)} time points")
    
    print(f"\nLoading all windows...")
    all_windows_df = load_all_windows(hashtag)
    if len(all_windows_df) == 0:
        raise ValueError(f"No windows found for hashtag: {hashtag}")
    print(f"  ✓ Loaded {len(all_windows_df)} windows")
    
    # Load labels for each rule
    labels_data = {}
    for rule in labeling_rules:
        print(f"\nLoading labels: {rule}...")
        labels_df = load_labels(hashtag, rule)
        labels_data[rule] = labels_df
        print(f"  ✓ Loaded {len(labels_df)} labeled windows")
    
    # Load predictions for each model
    predictions_data = {}
    for model in models:
        if use_relaxed:
            print(f"\nLoading RELAXED predictions: {model}...")
        else:
            print(f"\nLoading predictions: {model}...")
        pred_df = load_predictions(hashtag, model, use_relaxed=use_relaxed)  # PASS FLAG
        predictions_data[model] = pred_df
        print(f"  ✓ Loaded {len(pred_df)} predictions")
    
    # Create figure
    n_rows = len(labeling_rules)
    n_cols = 1 + len(models)  # Labels + predictions
    
    figwidth = figsize_per_subplot[0] * n_cols
    figheight = figsize_per_subplot[1] * n_rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figwidth, figheight), 
                            squeeze=False, constrained_layout=True)
    
    print(f"\n{'='*80}")
    print(f"Creating {n_rows} × {n_cols} subplot grid...")
    print(f"{'='*80}\n")
    
    # Plot each row (labeling rule)
    for row_idx, rule in enumerate(labeling_rules):
        print(f"Plotting row {row_idx+1}/{n_rows}: {rule}")
        
        # Column 0: Label plot
        plot_label_subplot(
            axes[row_idx, 0],
            hashtag=hashtag,
            timeseries_df=timeseries_df,
            labels_df=labels_data[rule],
            all_windows_df=all_windows_df,
            timeseries_type=timeseries_type,
            labeling_rule=rule
        )
        
        # Remaining columns: Prediction plots
        for col_idx, model in enumerate(models, start=1):
            plot_prediction_subplot(
                axes[row_idx, col_idx],
                hashtag=hashtag,
                timeseries_df=timeseries_df,
                predictions_df=predictions_data[model],
                all_windows_df=all_windows_df,
                timeseries_type=timeseries_type,
                model_name=model,
                use_relaxed=use_relaxed  # PASS FLAG
            )
    
    # # Overall title
    # fig.suptitle(f'Hashtag: #{hashtag} | Timeseries: {timeseries_type.capitalize()}',
    #             fontsize=14, weight='bold', y=0.995)
    
    # Save
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"\n✓ Saved plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    print(f"\n{'='*80}")
    print("✓ Comparison plot complete!")
    print(f"{'='*80}\n")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize and compare labeling rules and prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Exact predictions
  python plot_labeling_predictions.py --hashtag 4thofjuly --timeseries mentions
  
  # Relaxed predictions
  python plot_labeling_predictions.py \\
    --hashtag 4thofjuly \\
    --timeseries mentions \\
    --relaxed
  
  # Specific models with relaxed criteria
  python plot_labeling_predictions.py \\
    --hashtag inmyfeelings \\
    --models exp_040_random_forest_h550_0.5rel \\
    --relaxed
        """
    )
    
    parser.add_argument('--hashtag', required=True, help='Hashtag to plot')
    parser.add_argument('--timeseries', default='mentions', 
                       choices=['mentions', 'comments'],
                       help='Which timeseries to plot')
    parser.add_argument('--rules', nargs='+', 
                       help='Labeling rules to compare (auto-discover if not specified)')
    parser.add_argument('--models', nargs='+',
                       help='Model experiment IDs to compare (labels only if not specified)')
    parser.add_argument('--relaxed', action='store_true',
                       help='Use relaxed predictions (test_predictions_relaxed.csv)')  # NEW FLAG
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--figsize', type=int, nargs=2, default=[12, 3],
                       help='Figure size per subplot (width height)')
    parser.add_argument('--dpi', type=int, default=150,
                       help='Output resolution')
    
    args = parser.parse_args()
    
    plot_labeling_comparison(
        hashtag=args.hashtag,
        timeseries_type=args.timeseries,
        labeling_rules=args.rules,
        models=args.models,
        output_path=args.output,
        figsize_per_subplot=tuple(args.figsize),
        dpi=args.dpi,
        use_relaxed=args.relaxed  # PASS FLAG
    )

if __name__ == "__main__":
    main()