#!/usr/bin/env python3
"""
Model Training Script with Experiment Tracking

Trains models, tracks experiments, and saves results systematically.

Usage:
    # Train baseline models
    python train.py --config configs/baseline.yaml --model majority_baseline
    python train.py --config configs/baseline.yaml --model random_baseline
    
    # Train logistic regression
    python train.py --config 04_ml_prediction/04_models/configs/logistic.yaml --model logistic_default
    
    # Train with custom dataset version
    python 04_ml_prediction/04_models/scripts/train.py --config 04_ml_prediction/04_models/configs/logistic.yaml --model logistic_weak_reg --dataset-version h550_cpinside
    python 04_ml_prediction/04_models/scripts/train.py --config 04_ml_prediction/04_models/configs/random_forest.yaml --model random_forest_default --dataset-version h550_cpinside
    python 04_ml_prediction/04_models/scripts/train.py --config 04_ml_prediction/04_models/configs/xgboost.yaml --model xgboost_default --dataset-version h550_cpinside
    
    # Run all baselines
    python 04_ml_prediction/04_models/scripts/train.py --run-all-baselines --dataset-version h550_cpinside
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import json
from datetime import datetime
from typing import Dict, Any, Tuple

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from model_registry import get_model, MODEL_REGISTRY


# ============================================================================
# Experiment Tracking
# ============================================================================

def get_next_experiment_id(results_dir: Path) -> str:
    """Get next experiment ID (exp_001, exp_002, etc.)."""
    results_csv = results_dir / "experiment_results.csv"
    
    if not results_csv.exists():
        return "exp_001"
    
    df = pd.read_csv(results_csv)
    if len(df) == 0:
        return "exp_001"
    
    # Extract number from last experiment_id
    last_id = df['experiment_id'].iloc[-1]
    last_num = int(last_id.split('_')[1])
    return f"exp_{last_num + 1:03d}"


def log_experiment(
    experiment_id: str,
    model_type: str,
    config: Dict[str, Any],
    dataset_version: str,
    results: Dict[str, Any],
    results_dir: Path
):
    """
    Log experiment results to CSV.
    
    Args:
        experiment_id: Unique experiment ID
        model_type: Type of model
        config: Model configuration
        dataset_version: Dataset version used
        results: Training and evaluation results
        results_dir: Directory containing experiment_results.csv
    """
    results_csv = results_dir / "experiment_results.csv"
    
    # Prepare row
    row = {
        'experiment_id': experiment_id,
        'timestamp': datetime.now().isoformat(),
        'model_type': model_type,
        'dataset_version': dataset_version,
        'config_name': config.get('description', ''),
        
        # Dataset info
        'n_train': results['n_train'],
        'n_val': results['n_val'],
        'n_test': results['n_test'],
        'n_bursts_train': results['n_bursts_train'],
        'n_bursts_val': results['n_bursts_val'],
        'n_bursts_test': results['n_bursts_test'],
        
        # Training metrics
        'train_auc': results['train_auc'],
        'train_pr_auc': results['train_pr_auc'],
        
        # Validation metrics
        'val_auc': results['val_auc'],
        'val_pr_auc': results['val_pr_auc'],
        'val_precision': results['val_precision'],
        'val_recall': results['val_recall'],
        'val_f1': results['val_f1'],
        
        # Test metrics
        'test_auc': results['test_auc'],
        'test_pr_auc': results['test_pr_auc'],
        'test_precision': results['test_precision'],
        'test_recall': results['test_recall'],
        'test_f1': results['test_f1'],
        
        # Paths
        'model_path': str(results['model_path']),
        'config_path': str(results['config_path']),
    }
    
    # Append to CSV
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    
    df.to_csv(results_csv, index=False)
    print(f"✓ Logged experiment to {results_csv}")


# ============================================================================
# Data Loading
# ============================================================================

def load_data(dataset_version: str, datasets_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train/val/test splits.
    
    Returns:
        train_df, val_df, test_df
    """
    print(f"\nLoading data (version: {dataset_version})...")
    
    train_path = datasets_dir / f"samples_train_{dataset_version}.parquet"
    val_path = datasets_dir / f"samples_val_{dataset_version}.parquet"
    test_path = datasets_dir / f"samples_test_{dataset_version}.parquet"
    
    if not all(p.exists() for p in [train_path, val_path, test_path]):
        raise FileNotFoundError(
            f"Missing data files for version {dataset_version}. "
            f"Run split_data.py first."
        )
    
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    
    print(f"✓ Loaded data:")
    print(f"  Train: {len(train_df)} samples, {train_df['label'].sum()} bursts ({train_df['label'].mean():.1%})")
    print(f"  Val:   {len(val_df)} samples, {val_df['label'].sum()} bursts ({val_df['label'].mean():.1%})")
    print(f"  Test:  {len(test_df)} samples, {test_df['label'].sum()} bursts ({test_df['label'].mean():.1%})")
    
    return train_df, val_df, test_df


def prepare_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Extract features and labels from dataframes.
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    # Feature columns: all ts_* and net_* columns
    feature_cols = [c for c in train_df.columns if c.startswith('ts_') or c.startswith('net_')]
    
    print(f"\n✓ Using {len(feature_cols)} features")
    
    # Extract features and labels
    X_train = train_df[feature_cols].copy()
    y_train = train_df['label'].copy()
    
    X_val = val_df[feature_cols].copy()
    y_val = val_df['label'].copy()
    
    X_test = test_df[feature_cols].copy()
    y_test = test_df['label'].copy()
    
    # Handle any NaN or inf values
    for X in [X_train, X_val, X_test]:
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================================
# Model Training
# ============================================================================

def train_model(
    model_type: str,
    config: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
):
    """Train a model with given configuration."""
    print(f"\n{'='*80}")
    print(f"Training: {config.get('description', model_type)}")
    print(f"{'='*80}")
    
    # Create model
    model = get_model(model_type, config)
    
    # Train
    model.train(X_train, y_train, X_val, y_val)
    
    return model


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation on all splits.
    
    Returns dictionary with all metrics.
    """
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        precision_score, recall_score, f1_score
    )
    
    print(f"\n{'='*80}")
    print("Evaluation")
    print(f"{'='*80}")
    
    results = {}
    
    # Dataset sizes
    results['n_train'] = len(X_train)
    results['n_val'] = len(X_val)
    results['n_test'] = len(X_test)
    results['n_bursts_train'] = int(y_train.sum())
    results['n_bursts_val'] = int(y_val.sum())
    results['n_bursts_test'] = int(y_test.sum())
    
    # Evaluate on each split
    for split_name, X, y in [('train', X_train, y_train), 
                              ('val', X_val, y_val), 
                              ('test', X_test, y_test)]:
        
        # Get predictions
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
        
        # Compute metrics
        auc = roc_auc_score(y, y_pred_proba)
        pr_auc = average_precision_score(y, y_pred_proba)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        
        # Store results
        results[f'{split_name}_auc'] = float(auc)
        results[f'{split_name}_pr_auc'] = float(pr_auc)
        results[f'{split_name}_precision'] = float(precision)
        results[f'{split_name}_recall'] = float(recall)
        results[f'{split_name}_f1'] = float(f1)
        
        # Print
        print(f"\n{split_name.upper()} SET:")
        print(f"  AUC-ROC:     {auc:.4f}")
        print(f"  PR-AUC:      {pr_auc:.4f}")
        print(f"  Precision:   {precision:.4f}")
        print(f"  Recall:      {recall:.4f}")
        print(f"  F1:          {f1:.4f}")
    
    print(f"{'='*80}")
    
    return results


# ============================================================================
# Main Training Pipeline
# ============================================================================

def run_training(
    config_path: Path,
    model_name: str,
    dataset_version: str,
    datasets_dir: Path,
    results_dir: Path,
    trained_dir: Path
):
    """Run complete training pipeline for one model."""
    
    # Load configuration
    print(f"\nLoading config from {config_path}...")
    with open(config_path) as f:
        all_configs = yaml.safe_load(f)
    
    if model_name not in all_configs:
        available = ', '.join(all_configs.keys())
        raise ValueError(f"Model '{model_name}' not found in config. Available: {available}")
    
    config = all_configs[model_name]
    model_type = config['model_type']
    
    # Get experiment ID
    experiment_id = get_next_experiment_id(results_dir)
    print(f"\n{'='*80}")
    print(f"Experiment: {experiment_id}")
    print(f"Model: {model_name} ({model_type})")
    print(f"Dataset: {dataset_version}")
    print(f"{'='*80}")
    
    # Load data
    train_df, val_df, test_df = load_data(dataset_version, datasets_dir)
    
    # Prepare features
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_features(
        train_df, val_df, test_df
    )
    
    # Train model
    model = train_model(model_type, config, X_train, y_train, X_val, y_val)
    
    # Evaluate
    results = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Save model
    model_dir = trained_dir / f"{experiment_id}_{model_type}_{dataset_version}"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir)
    
    # Save feature importance if available
    feature_importance = model.get_feature_importance()
    if feature_importance is not None:
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in sorted(feature_importance.items(), key=lambda x: -x[1])
        ])
        importance_path = model_dir / "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        print(f"✓ Saved feature importance to {importance_path}")
    
    # Add paths to results
    results['model_path'] = model_dir
    results['config_path'] = config_path
    
    # Log experiment
    log_experiment(experiment_id, model_type, config, dataset_version, results, results_dir)
    
    print(f"\n{'='*80}")
    print(f"✓ Experiment {experiment_id} complete!")
    print(f"  Model saved to: {model_dir}")
    print(f"  Val AUC: {results['val_auc']:.4f}")
    print(f"  Test AUC: {results['test_auc']:.4f}")
    print(f"{'='*80}\n")
    
    return results


def run_all_baselines(dataset_version: str, datasets_dir: Path, results_dir: Path, trained_dir: Path):
    """Run all baseline models for quick sanity check."""
    baseline_config = Path("04_ml_prediction/04_models/configs/baseline.yaml")
    
    print("\n" + "="*80)
    print("Running All Baseline Models")
    print("="*80)
    
    baselines = ['majority_baseline', 'random_baseline']
    results_summary = []
    
    for model_name in baselines:
        try:
            results = run_training(
                baseline_config, model_name, dataset_version,
                datasets_dir, results_dir, trained_dir
            )
            results_summary.append({
                'model': model_name,
                'val_auc': results['val_auc'],
                'test_auc': results['test_auc']
            })
        except Exception as e:
            print(f"\n❌ Error running {model_name}: {e}\n")
    
    # Print summary
    print("\n" + "="*80)
    print("Baseline Results Summary")
    print("="*80)
    print(f"{'Model':<25} {'Val AUC':<15} {'Test AUC':<15}")
    print("-"*80)
    for r in results_summary:
        print(f"{r['model']:<25} {r['val_auc']:<15.4f} {r['test_auc']:<15.4f}")
    print("="*80)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train prediction models with experiment tracking")
    parser.add_argument(
        '--config',
        type=str,
        help='Path to model config YAML file'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Model name from config file'
    )
    parser.add_argument(
        '--dataset-version',
        type=str,
        default='v1',
        help='Dataset version to use (default: v1)'
    )
    parser.add_argument(
        '--run-all-baselines',
        action='store_true',
        help='Run all baseline models'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    datasets_dir = Path("04_ml_prediction/03_datasets/outputs")
    results_dir = Path("04_ml_prediction/04_models/results")
    trained_dir = Path("04_ml_prediction/04_models/trained")
    
    results_dir.mkdir(parents=True, exist_ok=True)
    trained_dir.mkdir(parents=True, exist_ok=True)
    
    # Run baselines or single model
    if args.run_all_baselines:
        run_all_baselines(args.dataset_version, datasets_dir, results_dir, trained_dir)
    else:
        if not args.config or not args.model:
            parser.error("--config and --model are required (or use --run-all-baselines)")
        
        run_training(
            Path(args.config),
            args.model,
            args.dataset_version,
            datasets_dir,
            results_dir,
            trained_dir
        )


if __name__ == "__main__":
    main()