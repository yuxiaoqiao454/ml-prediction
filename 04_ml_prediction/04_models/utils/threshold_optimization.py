#!/usr/bin/env python3
"""
Threshold Optimization Utilities

Find optimal classification thresholds using validation set.
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Dict, Tuple


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = 'f1',
    search_range: Tuple[float, float] = (0.1, 0.9),
    search_step: float = 0.01
) -> Tuple[float, Dict[float, float]]:
    """
    Find optimal classification threshold by grid search on validation set.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_proba : array-like
        Predicted probabilities for positive class
    metric : str
        Metric to optimize. Options:
        - 'f1': F1 score (harmonic mean of precision and recall)
        - 'precision': Precision score
        - 'recall': Recall score
        - 'youden': Youden's J statistic (sensitivity + specificity - 1)
        - 'balanced_accuracy': Balanced accuracy
    search_range : tuple
        (min_threshold, max_threshold) to search
    search_step : float
        Step size for grid search
    
    Returns:
    --------
    best_threshold : float
        Optimal threshold value
    threshold_scores : dict
        Mapping of threshold -> metric score
    """
    
    # Generate candidate thresholds
    thresholds = np.arange(search_range[0], search_range[1] + search_step, search_step)
    threshold_scores = {}
    
    for threshold in thresholds:
        # Convert probabilities to predictions
        y_pred = (y_proba >= threshold).astype(int)
        
        # Compute metric
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'youden':
            # Youden's J = sensitivity + specificity - 1
            tn = ((y_true == 0) & (y_pred == 0)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sensitivity + specificity - 1
        elif metric == 'balanced_accuracy':
            tn = ((y_true == 0) & (y_pred == 0)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = (sensitivity + specificity) / 2
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        threshold_scores[float(threshold)] = float(score)
    
    # Find best threshold
    best_threshold = max(threshold_scores.items(), key=lambda x: x[1])[0]
    
    return best_threshold, threshold_scores


def evaluate_threshold_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray = None
) -> Dict[str, np.ndarray]:
    """
    Compute precision, recall, and F1 scores across threshold range.
    
    Useful for plotting threshold selection curves.
    
    Returns:
    --------
    curves : dict
        Dictionary with keys 'thresholds', 'precision', 'recall', 'f1'
    """
    if thresholds is None:
        thresholds = np.arange(0.0, 1.01, 0.01)
    
    precisions = []
    recalls = []
    f1s = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
    
    return {
        'thresholds': thresholds,
        'precision': np.array(precisions),
        'recall': np.array(recalls),
        'f1': np.array(f1s)
    }

def find_optimal_threshold_constrained(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    objective: str = 'recall',
    constraint_metric: str = 'precision',
    constraint_min: float = 0.7,
    search_range: Tuple[float, float] = (0.1, 0.9),
    search_step: float = 0.01
) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal threshold with constraints.
    
    Example: Maximize recall subject to precision >= 0.7
    
    Parameters:
    -----------
    objective : str
        Metric to maximize: 'recall', 'precision', 'f1'
    constraint_metric : str
        Metric that must satisfy constraint: 'precision', 'recall'
    constraint_min : float
        Minimum value for constraint metric
    
    Returns:
    --------
    best_threshold : float
        Optimal threshold (or None if no feasible threshold found)
    metrics : dict
        {threshold: {objective: value, constraint: value}}
    """
    
    thresholds = np.arange(search_range[0], search_range[1] + search_step, search_step)
    
    feasible_thresholds = {}
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        # Compute both metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        metrics_dict = {'precision': precision, 'recall': recall, 'f1': f1}
        
        # Check if constraint is satisfied
        constraint_value = metrics_dict[constraint_metric]
        if constraint_value >= constraint_min:
            # Feasible! Store objective value
            objective_value = metrics_dict[objective]
            feasible_thresholds[threshold] = {
                'objective_value': objective_value,
                'constraint_value': constraint_value,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
    
    if not feasible_thresholds:
        # No feasible threshold found
        print(f"Warning: No threshold satisfies {constraint_metric} >= {constraint_min}")
        return None, {}
    
    # Find threshold with best objective value
    best_threshold = max(feasible_thresholds.items(), 
                        key=lambda x: x[1]['objective_value'])[0]
    
    return best_threshold, feasible_thresholds