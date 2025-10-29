#!/usr/bin/env python3
"""
Feature Registry for Time Series Features

Registry pattern allows modular feature extraction - easy to add/remove feature families.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

# ============================================================================
# Registry
# ============================================================================

TIMESERIES_FEATURE_REGISTRY = {}

def register_feature_family(name):
    """Decorator to register a feature extraction function."""
    def decorator(fn):
        TIMESERIES_FEATURE_REGISTRY[name] = fn
        return fn
    return decorator


# ============================================================================
# Helper Functions
# ============================================================================

def safe_divide(a, b, default=0.0):
    """Safe division returning default if b is zero."""
    return a / b if b != 0 else default


def get_window_slice(dates, counts, start_day, end_day):
    """
    Extract slice of data between start_day and end_day (1-indexed).
    
    Parameters:
    -----------
    dates : list of date strings
    counts : list of numeric values
    start_day : int (1 to 30)
    end_day : int (1 to 30)
    
    Returns:
    --------
    np.array of counts in the specified range
    """
    if len(counts) == 0:
        return np.array([])
    
    # Convert to 0-indexed
    start_idx = max(0, start_day - 1)
    end_idx = min(len(counts), end_day)
    
    return np.array(counts[start_idx:end_idx])


# ============================================================================
# Feature Families
# ============================================================================

@register_feature_family("level_dispersion")
def extract_level_dispersion(dates, counts, config, metric_name):
    """
    Basic statistical features: mean, std, median, CV.
    
    Returns dict with keys like: ts_{metric}_mean_raw, ts_{metric}_std_raw, etc.
    """
    if len(counts) == 0:
        return {}
    
    counts_arr = np.array(counts)
    mean_val = np.mean(counts_arr)
    std_val = np.std(counts_arr)
    median_val = np.median(counts_arr)
    cv_val = safe_divide(std_val, mean_val)
    
    features = {
        f'ts_{metric_name}_mean_raw': float(mean_val),
        f'ts_{metric_name}_std_raw': float(std_val),
        f'ts_{metric_name}_median_raw': float(median_val),
        f'ts_{metric_name}_cv_raw': float(cv_val),
    }
    
    # Add log-transformed versions if enabled
    if config.get('transformations', {}).get('log', False):
        features[f'ts_{metric_name}_mean_log'] = float(np.log1p(mean_val))
        features[f'ts_{metric_name}_std_log'] = float(np.log1p(std_val))
    
    return features


@register_feature_family("growth_acceleration")
def extract_growth_acceleration(dates, counts, config, metric_name):
    """
    Growth rate between recent and previous periods.
    
    Formula: (mean_last7 - mean_prev7) / mean_prev7
    """
    if len(counts) < 14:
        return {}
    
    windows = config.get('comparison_windows', [[24, 30], [17, 23]])
    last_window = windows[0]
    prev_window = windows[1]
    
    last_data = get_window_slice(dates, counts, last_window[0], last_window[1])
    prev_data = get_window_slice(dates, counts, prev_window[0], prev_window[1])
    
    mean_last = np.mean(last_data) if len(last_data) > 0 else 0
    mean_prev = np.mean(prev_data) if len(prev_data) > 0 else 0
    
    growth = safe_divide(mean_last - mean_prev, mean_prev, default=0.0)
    
    return {
        f'ts_{metric_name}_growth_7_14': float(growth),
        f'ts_{metric_name}_mean_last7': float(mean_last),
        f'ts_{metric_name}_mean_prev7': float(mean_prev),
    }


@register_feature_family("trend")
def extract_trend(dates, counts, config, metric_name):
    """
    Linear regression slope over the full window.
    
    Positive slope = upward trend
    """
    if len(counts) < 2:
        return {}
    
    x = np.arange(len(counts))
    y = np.array(counts)
    
    # Linear regression
    if len(x) > 1 and np.std(y) > 0:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    else:
        slope = 0.0
        r_value = 0.0
    
    return {
        f'ts_{metric_name}_slope_30': float(slope),
        f'ts_{metric_name}_trend_r2': float(r_value**2),
    }


@register_feature_family("peak_detection")
def extract_peak_detection(dates, counts, config, metric_name):
    """
    Peak-related features: max/mean ratio, number of peaks, peak timing.
    """
    if len(counts) == 0:
        return {}
    
    counts_arr = np.array(counts)
    mean_val = np.mean(counts_arr)
    max_val = np.max(counts_arr)
    
    peak_ratio = safe_divide(max_val, mean_val, default=1.0)
    
    # Find peaks using scipy
    min_prominence = config.get('min_prominence', 1.5)
    peaks, properties = find_peaks(counts_arr, prominence=mean_val * (min_prominence - 1))
    n_peaks = len(peaks)
    
    # Position of maximum (early/mid/late in window)
    max_idx = np.argmax(counts_arr)
    peak_position = (max_idx + 1) / len(counts_arr)  # Normalized 0-1
    
    return {
        f'ts_{metric_name}_peak_ratio': float(peak_ratio),
        f'ts_{metric_name}_n_peaks': int(n_peaks),
        f'ts_{metric_name}_peak_position': float(peak_position),
        f'ts_{metric_name}_max_raw': float(max_val),
    }


@register_feature_family("percentiles")
def extract_percentiles(dates, counts, config, metric_name):
    """
    Distribution percentiles: p25, p50, p75, p90, IQR.
    """
    if len(counts) == 0:
        return {}
    
    counts_arr = np.array(counts)
    quantiles = config.get('quantiles', [0.25, 0.50, 0.75, 0.90])
    
    features = {}
    for q in quantiles:
        pct = np.percentile(counts_arr, q * 100)
        features[f'ts_{metric_name}_p{int(q*100)}'] = float(pct)
    
    # IQR
    p25 = np.percentile(counts_arr, 25)
    p75 = np.percentile(counts_arr, 75)
    features[f'ts_{metric_name}_iqr'] = float(p75 - p25)
    
    return features


@register_feature_family("recency")
def extract_recency(dates, counts, config, metric_name):
    """
    Compare recent vs early period in window.
    
    Feature: mean_last7 / mean_first7 (recency ratio)
    """
    if len(counts) < 14:
        return {}
    
    periods = config.get('compare_periods', {
        'first_7': [1, 7],
        'last_7': [24, 30]
    })
    
    first_data = get_window_slice(dates, counts, *periods['first_7'])
    last_data = get_window_slice(dates, counts, *periods['last_7'])
    
    mean_first = np.mean(first_data) if len(first_data) > 0 else 0
    mean_last = np.mean(last_data) if len(last_data) > 0 else 0
    
    recency_ratio = safe_divide(mean_last, mean_first, default=1.0)
    
    return {
        f'ts_{metric_name}_recency_ratio': float(recency_ratio),
        f'ts_{metric_name}_mean_first7': float(mean_first),
    }


@register_feature_family("sparsity")
def extract_sparsity(dates, counts, config, metric_name):
    """
    Sparsity metrics: % of zero days, longest zero streak.
    """
    if len(counts) == 0:
        return {}
    
    counts_arr = np.array(counts)
    zero_threshold = config.get('zero_threshold', 0.01)
    
    is_zero = counts_arr < zero_threshold
    n_zeros = np.sum(is_zero)
    pct_zero = n_zeros / len(counts_arr)
    
    # Longest consecutive zeros
    max_zero_streak = 0
    current_streak = 0
    for val in is_zero:
        if val:
            current_streak += 1
            max_zero_streak = max(max_zero_streak, current_streak)
        else:
            current_streak = 0
    
    return {
        f'ts_{metric_name}_pct_zero': float(pct_zero),
        f'ts_{metric_name}_n_nonzero_days': int(len(counts_arr) - n_zeros),
        f'ts_{metric_name}_max_zero_streak': int(max_zero_streak),
    }


@register_feature_family("volatility")
def extract_volatility(dates, counts, config, metric_name):
    """
    Volatility of daily changes (std of first differences).
    """
    if len(counts) < 2:
        return {}
    
    counts_arr = np.array(counts)
    window_days = config.get('window_days', 7)
    
    # Full window volatility
    diffs = np.diff(counts_arr)
    vol_full = np.std(diffs)
    
    # Recent volatility (last N days)
    if len(counts_arr) >= window_days:
        recent_diffs = np.diff(counts_arr[-window_days:])
        vol_recent = np.std(recent_diffs)
    else:
        vol_recent = vol_full
    
    return {
        f'ts_{metric_name}_volatility_diff': float(vol_full),
        f'ts_{metric_name}_volatility_diff_recent7': float(vol_recent),
    }


@register_feature_family("autocorrelation")
def extract_autocorrelation(dates, counts, config, metric_name):
    """
    Autocorrelation at specified lags (momentum detection).
    """
    if len(counts) < 8:
        return {}
    
    lags = config.get('lags', [1, 3, 7])
    counts_arr = np.array(counts)
    
    features = {}
    for lag in lags:
        if len(counts_arr) > lag:
            # Pearson correlation between t and t-lag
            x = counts_arr[lag:]
            y = counts_arr[:-lag]
            if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
                acf = np.corrcoef(x, y)[0, 1]
            else:
                acf = 0.0
            features[f'ts_{metric_name}_acf{lag}'] = float(acf)
        else:
            features[f'ts_{metric_name}_acf{lag}'] = 0.0
    
    return features


@register_feature_family("changepoint")
def extract_changepoint(dates, counts, config, metric_name, label_row=None):
    """
    Change-point features from PELT results (loaded from labels).
    
    Requires label_row with: has_cp_near_boundary, cp_position, jump_absolute
    """
    if label_row is None:
        return {}
    
    # Extract from label row (column names depend on metric)
    cp_col = f'has_cp_near_boundary_{metric_name}'
    jump_col = f'jump_absolute_{metric_name}'
    
    has_cp = label_row.get(cp_col, False)
    jump = label_row.get(jump_col, 0.0)
    
    return {
        f'ts_{metric_name}_recent_cp_flag': int(has_cp) if has_cp is not None else 0,
        f'ts_{metric_name}_last_cp_jump': float(jump) if jump is not None else 0.0,
    }