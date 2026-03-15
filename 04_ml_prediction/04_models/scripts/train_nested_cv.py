#!/usr/bin/env python3
"""
Nested Cross-Validation Training Script

Implements rigorous nested CV with:
- Outer loop (K=5): Generalization performance estimation
- Inner loop (J=3): Hyperparameter tuning per outer fold
- Hashtag-level grouped stratified splitting

Usage:
    # Run nested CV with default config
    python 04_ml_prediction/04_models/scripts/train_nested_cv.py \
        --config 04_ml_prediction/04_models/configs/baseline.yaml \
        --model random_baseline \
        --dataset-version  h550_0.5rel_ts_net_attr_nocat

    python 04_ml_prediction/04_models/scripts/train_nested_cv.py \
        --config 04_ml_prediction/04_models/configs/logistic.yaml \
        --model logistic_default \
        --dataset-version  h550_0.5rel_ts_net_attr_nocat

    python 04_ml_prediction/04_models/scripts/train_nested_cv.py \
        --config 04_ml_prediction/04_models/configs/random_forest.yaml \
        --model random_forest_default \
        --dataset-version  h550_0.5rel_ts_net_attr_nocat

    python 04_ml_prediction/04_models/scripts/train_nested_cv.py \
        --config 04_ml_prediction/04_models/configs/xgboost.yaml \
        --model xgboost_default \
        --dataset-version  h550_0.5rel_ts_net_attr_nocat
    
    python 04_ml_prediction/04_models/scripts/train_nested_cv.py \
        --config 04_ml_prediction/04_models/configs/lightgbm.yaml \
        --model lightgbm_deep \
        --dataset-version  h550_cpinside_ts_net_infattr_nocat
    
    
    python 04_ml_prediction/04_models/scripts/train_nested_cv.py \
        --config 04_ml_prediction/04_models/configs/catboost.yaml \
        --model catboost_default \
        --dataset-version  h550_0.5rel_ts_net_attr_nocat
    
    # Custom CV folds
    python train_nested_cv.py \\
        --config configs/random_forest.yaml \\
        --model random_forest_deep \\
        --dataset-version h550_0.5rel \\
        --outer-folds 5 \\
        --inner-folds 3
    
    # Dry run to check splits
    python train_nested_cv.py \\
        --config configs/xgboost.yaml \\
        --model xgboost_default \\
        --dataset-version h550_0.5rel \\
        --dry-run
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import json
from datetime import datetime
from typing import Dict, Any
import time

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

# Add scripts directory
scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(scripts_dir))

# Add models directory for utils
models_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(models_dir))

from model_registry import get_model
from utils.cv_splitters import StratifiedHashtagKFold
from utils.threshold_optimization import find_optimal_threshold, evaluate_threshold_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)


# ============================================================================
# Data Loading (reuse from train.py)
# ============================================================================

def load_data(dataset_version: str, datasets_dir: Path):
    """Load full dataset (will be split by CV)."""
    print(f"\nLoading data (version: {dataset_version})...")
    
    # Try loading with seed suffix first
    full_path = datasets_dir / f"samples_full_{dataset_version}.parquet"
    
    if not full_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {full_path}\n"
            f"Run split_data_consistent.py first."
        )
    
    df = pd.read_parquet(full_path)
    
    print(f"✓ Loaded {len(df)} samples")
    print(f"  Hashtags: {df['hashtag'].nunique()}")
    print(f"  Burst rate: {df['label'].mean():.1%}")
    
    return df


def prepare_features(df: pd.DataFrame):
    """Extract features and labels."""
    # Feature columns
    feature_cols = [c for c in df.columns 
                    if c.startswith('ts_') or c.startswith('net_') or 
                       c.startswith('gae_') or c.startswith('emb_') or 
                       c.startswith('attr_')]
    
    # Remove non-numeric columns
    numeric_cols = [col for col in feature_cols 
                    if df[col].dtype in ['int64', 'float64', 'bool']]
    
    print(f"\n✓ Using {len(numeric_cols)} features")
    
    # Count feature types
    n_ts = len([c for c in numeric_cols if c.startswith('ts_')])
    n_net = len([c for c in numeric_cols if c.startswith('net_')])
    n_gae = len([c for c in numeric_cols if c.startswith('gae_')])
    n_emb = len([c for c in numeric_cols if c.startswith('emb_')])
    n_attr = len([c for c in numeric_cols if c.startswith('attr_')])
    
    print(f"  - Timeseries: {n_ts}")
    print(f"  - Network: {n_net}")
    print(f"  - GAE: {n_gae}")
    print(f"  - Embeddings: {n_emb}")
    print(f"  - Influencer Attrs: {n_attr}")
    
    # Extract
    X = df[numeric_cols].copy()
    y = df['label'].copy()
    
    # Handle NaN/inf
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    
    return X, y


# ============================================================================
# Nested CV Logic
# ============================================================================

def run_inner_cv_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str,
    config: Dict[str, Any],
    inner_cv: StratifiedHashtagKFold,
    samples_train_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Inner CV loop: Hyperparameter tuning.
    
    For now, uses FIXED hyperparameters from config.
    In Phase 2, add GridSearchCV here.
    
    Returns:
        best_params: Best hyperparameters found
    """
    print(f"\n  [Inner CV] Using fixed hyperparameters from config")
    
    # For now, just return config params (no tuning)
    # TODO Phase 2: Add grid search
    best_params = {k: v for k, v in config.items() 
                   if k not in ['model_type', 'description', 'threshold_optimization']}
    
    return best_params


def optimize_threshold_with_inner_cv(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    samples_train_df: pd.DataFrame,
    inner_cv: StratifiedHashtagKFold,
    threshold_config: Dict[str, Any]
) -> float:
    """
    Optimize threshold using inner CV on outer_train data.
    
    Averages optimal threshold across inner folds.
    """
    print(f"\n  [Inner CV] Optimizing threshold...")
    
    optimal_thresholds = []
    
    # Run inner CV
    for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
        inner_cv.split(samples_train_df)
    ):
        # Get inner validation data
        X_inner_val = X_train.iloc[inner_val_idx]
        y_inner_val = y_train.iloc[inner_val_idx]
        
        # Predict on inner val
        y_inner_val_proba = model.predict_proba(X_inner_val)[:, 1]
        
        # Find optimal threshold for this inner fold
        optimal_threshold, _ = find_optimal_threshold(
            y_inner_val.values,
            y_inner_val_proba,
            metric=threshold_config.get('metric', 'f1'),
            search_range=tuple(threshold_config.get('search_range', [0.1, 0.9])),
            search_step=threshold_config.get('search_step', 0.01)
        )
        
        optimal_thresholds.append(optimal_threshold)
    
    # Average across inner folds
    mean_threshold = np.mean(optimal_thresholds)
    std_threshold = np.std(optimal_thresholds)
    
    print(f"    Optimal thresholds: {optimal_thresholds}")
    print(f"    Mean: {mean_threshold:.3f} ± {std_threshold:.3f}")
    
    return mean_threshold


def evaluate_fold(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float
) -> Dict[str, float]:
    """Evaluate model on test fold."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    return {
        'auc': float(roc_auc_score(y_test, y_pred_proba)),
        'pr_auc': float(average_precision_score(y_test, y_pred_proba)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0))
    }


def run_nested_cv(
    samples_df: pd.DataFrame,
    model_type: str,
    config: Dict[str, Any],
    dataset_version: str,
    outer_cv: StratifiedHashtagKFold,
    inner_cv: StratifiedHashtagKFold,
    results_dir: Path
) -> pd.DataFrame:
    """
    Run full nested cross-validation.
    
    Returns:
        DataFrame with per-fold results
    """
    print(f"\n{'='*80}")
    print(f"NESTED CROSS-VALIDATION")
    print(f"Model: {config.get('description', model_type)}")
    print(f"Outer folds: {outer_cv.n_splits}, Inner folds: {inner_cv.n_splits}")
    print(f"{'='*80}")
    
    # Prepare features once
    X, y = prepare_features(samples_df)
    
    # Track results
    fold_results = []
    start_time = time.time()
    
    # OUTER LOOP
    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(
        outer_cv.split(samples_df)
    ):
        print(f"\n{'─'*80}")
        print(f"OUTER FOLD {outer_fold + 1}/{outer_cv.n_splits}")
        print(f"{'─'*80}")
        
        # Split data
        X_train = X.iloc[outer_train_idx]
        y_train = y.iloc[outer_train_idx]
        X_test = X.iloc[outer_test_idx]
        y_test = y.iloc[outer_test_idx]
        
        samples_train_df = samples_df.iloc[outer_train_idx]
        samples_test_df = samples_df.iloc[outer_test_idx]
        
        train_hashtags = samples_train_df['hashtag'].unique()
        test_hashtags = samples_test_df['hashtag'].unique()
        
        print(f"\n  Train: {len(X_train)} samples, {len(train_hashtags)} hashtags, "
              f"{y_train.mean():.1%} burst rate")
        print(f"  Test:  {len(X_test)} samples, {len(test_hashtags)} hashtags, "
              f"{y_test.mean():.1%} burst rate")
        
        # INNER CV: Hyperparameter tuning
        best_params = run_inner_cv_tuning(
            X_train, y_train, model_type, config, inner_cv, samples_train_df
        )
        
        # Train final model on full outer_train with best params
        print(f"\n  [Outer] Training final model on full train set...")
        final_config = config.copy()
        final_config.update(best_params)
        
        model = get_model(model_type, final_config)
        model.train(X_train, y_train)
        
        # Optimize threshold using inner CV (if enabled)
        threshold_config = config.get('threshold_optimization', {})
        if threshold_config.get('enabled', False):
            optimal_threshold = optimize_threshold_with_inner_cv(
                model, X_train, y_train, samples_train_df, 
                inner_cv, threshold_config
            )
            threshold_method = f"inner_cv_{threshold_config.get('metric', 'f1')}"
        else:
            optimal_threshold = 0.5
            threshold_method = 'default_0.5'
        
        print(f"  [Outer] Using threshold: {optimal_threshold:.3f}")
        
        # Evaluate on outer test (NEVER SEEN DURING TUNING!)
        test_metrics = evaluate_fold(model, X_test, y_test, optimal_threshold)
        
        print(f"\n  [Outer] Test Results:")
        print(f"    AUC:       {test_metrics['auc']:.4f}")
        print(f"    PR-AUC:    {test_metrics['pr_auc']:.4f}")
        print(f"    Precision: {test_metrics['precision']:.4f}")
        print(f"    Recall:    {test_metrics['recall']:.4f}")
        print(f"    F1:        {test_metrics['f1']:.4f}")
        
        # Store fold results
        fold_results.append({
            'outer_fold': outer_fold,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'n_train_hashtags': len(train_hashtags),
            'n_test_hashtags': len(test_hashtags),
            'train_burst_rate': float(y_train.mean()),
            'test_burst_rate': float(y_test.mean()),
            'threshold_used': optimal_threshold,
            'threshold_method': threshold_method,
            'test_auc': test_metrics['auc'],
            'test_pr_auc': test_metrics['pr_auc'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1']
        })
    
    total_time = time.time() - start_time
    
    # Convert to DataFrame
    results_df = pd.DataFrame(fold_results)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"NESTED CV COMPLETE")
    print(f"{'='*80}")
    print(f"\nPer-Fold Results:")
    print(results_df[['outer_fold', 'test_auc', 'test_pr_auc', 'test_f1']].to_string(index=False))
    
    print(f"\nAggregated Results:")
    print(f"  AUC:       {results_df['test_auc'].mean():.4f} ± {results_df['test_auc'].std():.4f}")
    print(f"  PR-AUC:    {results_df['test_pr_auc'].mean():.4f} ± {results_df['test_pr_auc'].std():.4f}")
    print(f"  Precision: {results_df['test_precision'].mean():.4f} ± {results_df['test_precision'].std():.4f}")
    print(f"  Recall:    {results_df['test_recall'].mean():.4f} ± {results_df['test_recall'].std():.4f}")
    print(f"  F1:        {results_df['test_f1'].mean():.4f} ± {results_df['test_f1'].std():.4f}")
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"{'='*80}\n")
    
    return results_df, total_time


# ============================================================================
# Saving Results
# ============================================================================

def save_nested_cv_results(
    fold_results_df: pd.DataFrame,
    model_type: str,
    config: Dict[str, Any],
    dataset_version: str,
    total_time: float,
    outer_folds: int,
    inner_folds: int,
    seed_id: str,  # ← NEW
    random_state: int,  # ← NEW
    results_dir: Path
):
    """Save nested CV results to CSV files."""
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate experiment ID
    experiment_id = get_next_nested_cv_id(results_dir)
    
    timestamp = datetime.now().isoformat()
    
    # === Per-Fold Results ===
    fold_results_csv = results_dir / "nested_cv_results.csv"
    
    fold_results_df['experiment_id'] = experiment_id
    fold_results_df['timestamp'] = timestamp
    fold_results_df['model_type'] = model_type
    fold_results_df['dataset_version'] = dataset_version
    fold_results_df['config_name'] = config.get('description', '')
    fold_results_df['seed_id'] = seed_id  # ← NEW
    fold_results_df['random_state'] = random_state  # ← NEW
    
    # Append to CSV
    if fold_results_csv.exists():
        existing_df = pd.read_csv(fold_results_csv)
        combined_df = pd.concat([existing_df, fold_results_df], ignore_index=True)
    else:
        combined_df = fold_results_df
    
    combined_df.to_csv(fold_results_csv, index=False)
    print(f"✓ Saved per-fold results to {fold_results_csv}")
    
    # === Aggregated Summary ===
    summary_csv = results_dir / "nested_cv_summary.csv"
    
    summary_row = {
        'experiment_id': experiment_id,
        'timestamp': timestamp,
        'model_type': model_type,
        'dataset_version': dataset_version,
        'config_name': config.get('description', ''),
        'seed_id': seed_id,  # ← NEW
        'random_state': random_state,  # ← NEW
        'n_outer_folds': outer_folds,
        'n_inner_folds': inner_folds,
        
        # Mean ± Std
        'mean_auc': fold_results_df['test_auc'].mean(),
        'std_auc': fold_results_df['test_auc'].std(),
        'mean_pr_auc': fold_results_df['test_pr_auc'].mean(),
        'std_pr_auc': fold_results_df['test_pr_auc'].std(),
        'mean_precision': fold_results_df['test_precision'].mean(),
        'std_precision': fold_results_df['test_precision'].std(),
        'mean_recall': fold_results_df['test_recall'].mean(),
        'std_recall': fold_results_df['test_recall'].std(),
        'mean_f1': fold_results_df['test_f1'].mean(),
        'std_f1': fold_results_df['test_f1'].std(),
        
        'total_runtime_sec': total_time,
        'total_runtime_min': total_time / 60
    }
    
    if summary_csv.exists():
        summary_df = pd.read_csv(summary_csv)
        summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        summary_df = pd.DataFrame([summary_row])
    
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ Saved summary to {summary_csv}")


def get_next_nested_cv_id(results_dir: Path) -> str:
    """Get next nested CV experiment ID."""
    results_csv = results_dir / "nested_cv_summary.csv"
    
    if not results_csv.exists():
        return "ncv_001"
    
    df = pd.read_csv(results_csv)
    if len(df) == 0:
        return "ncv_001"
    
    last_id = df['experiment_id'].iloc[-1]
    last_num = int(last_id.split('_')[1])
    return f"ncv_{last_num + 1:03d}"


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Nested Cross-Validation for hashtag burst prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default seed
    python train_nested_cv.py --config configs/lightgbm.yaml --model lightgbm_deep --dataset-version h550_0.5rel
    
    # Try different seeds for sensitivity analysis
    python train_nested_cv.py --config configs/lightgbm.yaml --model lightgbm_deep --dataset-version h550_0.5rel --seed-id seed1
    python train_nested_cv.py --config configs/lightgbm.yaml --model lightgbm_deep --dataset-version h550_0.5rel --seed-id seed2
    python train_nested_cv.py --config configs/lightgbm.yaml --model lightgbm_deep --dataset-version h550_0.5rel --seed-id seed10
    
    # Use actual integer if you prefer
    python train_nested_cv.py --config configs/lightgbm.yaml --model lightgbm_deep --dataset-version h550_0.5rel --seed-id 999
        """
    )
    
    parser.add_argument('--config', required=True, help='Path to model config YAML')
    parser.add_argument('--model', required=True, help='Model name from config')
    parser.add_argument('--dataset-version', required=True, help='Dataset version')
    
    parser.add_argument('--outer-folds', type=int, default=5, 
                        help='Number of outer CV folds (default: 5)')
    parser.add_argument('--inner-folds', type=int, default=3,
                        help='Number of inner CV folds (default: 3)')
    
    # NEW: Flexible seed management
    parser.add_argument('--seed-id', type=str, default='seed42',
                        help='Seed identifier for reproducibility (e.g., seed1, seed10, or any integer). Default: seed42')
    
    parser.add_argument('--dry-run', action='store_true',
                        help='Print CV splits without training')
    
    args = parser.parse_args()
    
    # Parse seed_id to integer
    random_state = parse_seed_id(args.seed_id)
    
    # Setup paths
    datasets_dir = Path("04_ml_prediction/03_datasets/outputs")
    results_dir = Path("04_ml_prediction/04_models/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    print(f"\nLoading config from {args.config}...")
    with open(args.config) as f:
        all_configs = yaml.safe_load(f)
    
    if args.model not in all_configs:
        available = ', '.join(all_configs.keys())
        raise ValueError(f"Model '{args.model}' not found. Available: {available}")
    
    config = all_configs[args.model]
    model_type = config['model_type']
    
    # Print experiment info
    print(f"\n{'='*80}")
    print(f"NESTED CV EXPERIMENT")
    print(f"{'='*80}")
    print(f"Model: {config.get('description', args.model)}")
    print(f"Dataset: {args.dataset_version}")
    print(f"Seed ID: {args.seed_id} (random_state={random_state})")  # ← NEW
    print(f"Outer folds: {args.outer_folds}, Inner folds: {args.inner_folds}")
    print(f"{'='*80}")
    
    # Load data
    samples_df = load_data(args.dataset_version, datasets_dir)
    
    # Setup CV splitters with parsed random_state
    outer_cv = StratifiedHashtagKFold(
        n_splits=args.outer_folds,
        shuffle=True,
        random_state=random_state  # ← Uses parsed seed
    )
    
    inner_cv = StratifiedHashtagKFold(
        n_splits=args.inner_folds,
        shuffle=True,
        random_state=random_state + 1  # ← Offset for inner loop
    )
    
    if args.dry_run:
        print(f"\n{'='*80}")
        print("DRY RUN: Showing CV splits without training")
        print(f"{'='*80}")
        
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(samples_df)):
            train_df = samples_df.iloc[train_idx]
            test_df = samples_df.iloc[test_idx]
            
            print(f"\nOuter Fold {fold + 1}:")
            print(f"  Train: {len(train_df)} samples, {train_df['hashtag'].nunique()} hashtags")
            print(f"  Test:  {len(test_df)} samples, {test_df['hashtag'].nunique()} hashtags")
            print(f"  Train burst rate: {train_df['label'].mean():.2%}")
            print(f"  Test burst rate:  {test_df['label'].mean():.2%}")
        
        print(f"\n✓ Dry run complete")
        return
    
    # Run nested CV
    fold_results_df, total_time = run_nested_cv(
        samples_df, model_type, config, args.dataset_version,
        outer_cv, inner_cv, results_dir
    )
    
    # Save results (with seed_id tracking)
    save_nested_cv_results(
        fold_results_df, model_type, config, args.dataset_version,
        total_time, args.outer_folds, args.inner_folds, 
        args.seed_id, random_state, results_dir  # ← Pass seed info
    )
    
    print(f"\n✓ Nested CV experiment complete!")
    print(f"  Seed ID: {args.seed_id}")
    print(f"  Results: {results_dir / 'nested_cv_results.csv'}")
    print(f"  Summary: {results_dir / 'nested_cv_summary.csv'}")


def parse_seed_id(seed_id: str) -> int:
    """
    Parse seed_id to integer random_state.
    
    Supports:
    - 'seed1', 'seed10', 'seed42' → extracts number
    - '999', '123' → uses directly
    - 'default' → uses 42
    
    Examples:
        'seed10' → 10
        'seed42' → 42
        '999' → 999
        'default' → 42
    """
    if seed_id == 'default':
        return 42
    
    # Try to extract number from 'seedXX' format
    if seed_id.startswith('seed'):
        try:
            return int(seed_id[4:])
        except ValueError:
            raise ValueError(f"Invalid seed_id format: '{seed_id}'. Use 'seedXX' or integer.")
    
    # Try to parse as direct integer
    try:
        return int(seed_id)
    except ValueError:
        raise ValueError(f"Invalid seed_id: '{seed_id}'. Use 'seedXX', integer, or 'default'.")


if __name__ == "__main__":
    main()