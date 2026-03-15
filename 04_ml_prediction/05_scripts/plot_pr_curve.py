#!/usr/bin/env python3
"""
Plot Precision-Recall curve from threshold data.

Creates a professional PR curve plot with optimal threshold marked.

Usage:
  python 04_ml_prediction/05_scripts/plot_pr_curve.py --input 04_ml_prediction/04_models/trained/exp_040_random_forest_h550_cpinside/threshold_curve.json --title "Random Forest"
  python plot_pr_curve.py --input threshold_curve.json --title "My Model"
  python plot_pr_curve.py --input threshold_curve.json --no-f1
"""

import argparse
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def load_threshold_data(json_path):
    """Load threshold curve data from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Validate required fields
    required = ['precision', 'recall', 'thresholds']
    missing = [field for field in required if field not in data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    
    return data

def plot_pr_curve(data, output_path, title=None, show_f1=True, figsize=(10, 8), dpi=300):
    """
    Plot precision-recall curve.
    
    Args:
        data: Dictionary with 'precision', 'recall', 'thresholds', and optionally 'optimal_threshold'
        output_path: Path to save the plot
        title: Custom title (defaults to "Precision-Recall Curve")
        show_f1: Whether to show F1 score lines
        figsize: Figure size tuple
        dpi: Output resolution
    """
    precision = np.array(data['precision'])
    recall = np.array(data['recall'])
    thresholds = np.array(data['thresholds'])
    optimal_threshold = data.get('optimal_threshold', None)
    
    # Sort by recall (ascending) for proper PR curve and AUC calculation
    sort_idx = np.argsort(recall)
    recall = recall[sort_idx]
    precision = precision[sort_idx]
    thresholds = thresholds[sort_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot PR curve
    ax.plot(recall, precision, 'b-', linewidth=2.5, label='PR Curve')
    
    # Plot F1 iso-lines (optional)
    if show_f1:
        f1_scores = [0.2, 0.4, 0.6, 0.8]
        for f1 in f1_scores:
            x = np.linspace(0.01, 1, 100)
            with np.errstate(divide='ignore', invalid='ignore'):
                y = f1 * x / (2 * x - f1)
            y = np.where((y >= 0) & (y <= 1), y, np.nan)
            ax.plot(x, y, '--', color='gray', alpha=0.3, linewidth=1)
            # Label position
            label_x = 0.9
            with np.errstate(divide='ignore', invalid='ignore'):
                label_y = f1 * label_x / (2 * label_x - f1)
            if 0 <= label_y <= 1:
                ax.text(label_x, label_y, f'F1={f1:.1f}', 
                       fontsize=8, color='gray', alpha=0.5)
    
    # Mark optimal threshold if provided
    if optimal_threshold is not None:
        # Find closest threshold
        idx = np.argmin(np.abs(thresholds - optimal_threshold))
        opt_recall = recall[idx]
        opt_precision = precision[idx]
        
        ax.plot(opt_recall, opt_precision, 'r*', markersize=20, 
               markeredgecolor='darkred', markeredgewidth=1.5,
               label=f'Optimal (t={optimal_threshold:.2f})')
        
        # Add annotation
        ax.annotate(f'P={opt_precision:.2f}\nR={opt_recall:.2f}',
                   xy=(opt_recall, opt_precision),
                   xytext=(opt_recall - 0.15, opt_precision + 0.1),
                   fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                                 color='red', lw=1.5))
    
    # Calculate AUC (area under PR curve using trapezoidal rule)
    # Use numpy's trapezoid (trapz is deprecated)
    try:
        auc = np.trapezoid(precision, recall)
    except AttributeError:
        # Fallback for older numpy versions
        auc = np.trapz(precision, recall)
    
    # Styling
    ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
    
    if title is None:
        title = 'Precision-Recall Curve'
    ax.set_title(f'{title}\n(AUC = {auc:.3f})', fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    # Add info box
    info_text = f'Thresholds: {len(thresholds)}\n'
    if optimal_threshold is not None:
        f1_opt = 2 * opt_precision * opt_recall / (opt_precision + opt_recall) if (opt_precision + opt_recall) > 0 else 0
        info_text += f'F1 at optimal: {f1_opt:.3f}'
    
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved plot to {output_path}")
    print(f"  AUC: {auc:.3f}")
    if optimal_threshold is not None:
        print(f"  Optimal threshold: {optimal_threshold:.3f}")
        print(f"  At optimal: Precision={opt_precision:.3f}, Recall={opt_recall:.3f}, F1={f1_opt:.3f}")

def main():
    parser = argparse.ArgumentParser(
        description="Plot Precision-Recall curve from threshold data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python plot_pr_curve.py --input threshold_curve.json
  
  # Custom title
  python plot_pr_curve.py --input threshold_curve.json --title "Random Forest Model"
  
  # Without F1 iso-lines
  python plot_pr_curve.py --input threshold_curve.json --no-f1
  
  # Custom output filename
  python plot_pr_curve.py --input threshold_curve.json --output my_pr_curve.png
  
  # Higher resolution
  python plot_pr_curve.py --input threshold_curve.json --dpi 600
        """
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Path to JSON file with threshold curve data'
    )
    
    parser.add_argument(
        '--output',
        default=None,
        help='Output filename (default: input_name_pr_curve.png in same directory)'
    )
    
    parser.add_argument(
        '--title',
        default=None,
        help='Custom plot title (default: "Precision-Recall Curve")'
    )
    
    parser.add_argument(
        '--no-f1',
        action='store_true',
        help='Hide F1 iso-lines'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Output resolution in DPI (default: 300)'
    )
    
    parser.add_argument(
        '--figsize',
        nargs=2,
        type=float,
        default=[10, 8],
        help='Figure size as width height (default: 10 8)'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    
    # Determine output path
    if args.output is None:
        output_path = input_path.parent / f"{input_path.stem}_pr_curve.png"
    else:
        output_path = Path(args.output)
        # If no directory specified, save in same directory as input
        if not output_path.is_absolute() and output_path.parent == Path('.'):
            output_path = input_path.parent / output_path
    
    print("="*80)
    print("Precision-Recall Curve Plot")
    print("="*80)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("="*80 + "\n")
    
    # Load data
    try:
        data = load_threshold_data(input_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Create plot
    try:
        plot_pr_curve(
            data,
            output_path,
            title=args.title,
            show_f1=not args.no_f1,
            figsize=tuple(args.figsize),
            dpi=args.dpi
        )
    except Exception as e:
        print(f"Error creating plot: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*80)
    print("Plot complete!")
    print("="*80)


if __name__ == "__main__":
    main()