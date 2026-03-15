#!/usr/bin/env python3
"""
Influencer Attribute Feature Registry

All feature computation functions for influencer attributes.
Follows the exact specification from the briefing.

Feature prefix: attr_*
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy


# ============================================================================
# Registry
# ============================================================================

INFLUENCER_ATTR_REGISTRY = {}

def register_influencer_attr_feature(name):
    """Decorator to register an influencer attribute feature function."""
    def decorator(fn):
        INFLUENCER_ATTR_REGISTRY[name] = fn
        return fn
    return decorator


# ============================================================================
# Helper Functions
# ============================================================================

def safe_log(x):
    """Safe log transformation: log(x + 1)"""
    return np.log(x + 1)


def safe_divide(a, b, default=0.0):
    """Safe division with default for zero denominator."""
    return float(a / b) if b != 0 else default


def compute_herfindahl(values):
    """
    Compute Herfindahl-Hirschman Index (concentration measure).
    H = sum(p_i^2) where p_i = value_i / sum(values)
    
    Returns value between 0 (perfect equality) and 1 (perfect concentration).
    """
    if len(values) == 0 or np.sum(values) == 0:
        return 0.0
    
    total = np.sum(values)
    proportions = values / total
    return float(np.sum(proportions ** 2))


def compute_entropy(counts):
    """
    Compute Shannon entropy from counts.
    H = -sum(p_i * log(p_i)) where p_i = count_i / sum(counts)
    """
    if len(counts) == 0 or np.sum(counts) == 0:
        return 0.0
    
    total = np.sum(counts)
    proportions = counts / total
    # Add epsilon to avoid log(0)
    return float(-np.sum(proportions * np.log(proportions + 1e-10)))


# ============================================================================
# Level 1: Aggregated Attribute Features
# ============================================================================

@register_influencer_attr_feature("level1_aggregates")
def extract_level1_aggregates(influencer_attrs_df, config):
    """
    Level 1: Basic aggregated statistics on log-transformed attributes.
    
    Parameters:
    -----------
    influencer_attrs_df : DataFrame
        Filtered to influencers in this window with valid attributes.
        Columns: username, category, followers, followees, posts
    
    Returns:
    --------
    dict of features with attr_* prefix
    """
    if len(influencer_attrs_df) == 0:
        return {}
    
    features = {}
    
    # Log-transform attributes
    f_log = safe_log(influencer_attrs_df['followers'].values)
    w_log = safe_log(influencer_attrs_df['followees'].values)
    p_log = safe_log(influencer_attrs_df['posts'].values)
    
    # Follower-followee ratio (log)
    r_log = np.log((influencer_attrs_df['followers'].values + 1) / 
                   (influencer_attrs_df['followees'].values + 1))
    
    # Followers-per-post (log)
    g_log = np.log((influencer_attrs_df['followers'].values + 1) / 
                   (influencer_attrs_df['posts'].values + 1))
    
    # 3.1 Followers (log-transformed)
    features['attr_followers_log_mean'] = float(np.mean(f_log))
    features['attr_followers_log_std'] = float(np.std(f_log))
    features['attr_followers_log_median'] = float(np.median(f_log))
    features['attr_followers_log_p90'] = float(np.percentile(f_log, 90))
    features['attr_followers_log_max'] = float(np.max(f_log))
    
    # 3.2 Followees (log-transformed)
    features['attr_followees_log_mean'] = float(np.mean(w_log))
    features['attr_followees_log_std'] = float(np.std(w_log))
    features['attr_followees_log_median'] = float(np.median(w_log))
    features['attr_followees_log_p90'] = float(np.percentile(w_log, 90))
    features['attr_followees_log_max'] = float(np.max(w_log))
    
    # 3.3 Posts (log-transformed)
    features['attr_posts_log_mean'] = float(np.mean(p_log))
    features['attr_posts_log_std'] = float(np.std(p_log))
    features['attr_posts_log_median'] = float(np.median(p_log))
    features['attr_posts_log_p90'] = float(np.percentile(p_log, 90))
    features['attr_posts_log_max'] = float(np.max(p_log))
    
    # 3.4 Follower-Followee Ratio
    features['attr_ff_ratio_log_mean'] = float(np.mean(r_log))
    features['attr_ff_ratio_log_std'] = float(np.std(r_log))
    
    # 3.5 Followers-per-Post
    features['attr_fpp_log_mean'] = float(np.mean(g_log))
    features['attr_fpp_log_std'] = float(np.std(g_log))
    
    return features


# ============================================================================
# Level 2: Micro-Macro Tier Composition
# ============================================================================

@register_influencer_attr_feature("level2_tiers")
def extract_level2_tiers(influencer_attrs_df, config):
    """
    Level 2: Influencer tier composition (micro/mid/macro/mega).
    
    Tiers based on raw follower counts:
    - Micro: < 10,000
    - Mid: 10,000 - 100,000
    - Macro: 100,000 - 1,000,000
    - Mega: > 1,000,000
    """
    if len(influencer_attrs_df) == 0:
        return {}
    
    followers = influencer_attrs_df['followers'].values
    N = len(followers)
    
    # Count each tier
    n_micro = np.sum(followers < 10000)
    n_mid = np.sum((followers >= 10000) & (followers < 100000))
    n_macro = np.sum((followers >= 100000) & (followers < 1000000))
    n_mega = np.sum(followers >= 1000000)
    
    return {
        'attr_frac_micro': float(n_micro / N),
        'attr_frac_mid': float(n_mid / N),
        'attr_frac_macro': float(n_macro / N),
        'attr_frac_mega': float(n_mega / N),
    }


# ============================================================================
# Level 3: Tier Concentration Features
# ============================================================================

@register_influencer_attr_feature("level3_concentration")
def extract_level3_concentration(influencer_attrs_df, config):
    """
    Level 3: Follower concentration measures.
    
    - Top-1 and Top-5 follower share
    - Herfindahl concentration index
    """
    if len(influencer_attrs_df) == 0:
        return {}
    
    followers = influencer_attrs_df['followers'].values
    total_followers = np.sum(followers)
    
    if total_followers == 0:
        return {
            'attr_top1_follower_share': 0.0,
            'attr_top5_follower_share': 0.0,
            'attr_follower_concentration_H': 0.0,
        }
    
    # Sort followers descending
    sorted_followers = np.sort(followers)[::-1]
    
    # Top-1 share
    top1_share = sorted_followers[0] / total_followers
    
    # Top-5 share
    top5 = sorted_followers[:min(5, len(sorted_followers))]
    top5_share = np.sum(top5) / total_followers
    
    # Herfindahl concentration
    H = compute_herfindahl(followers)
    
    return {
        'attr_top1_follower_share': float(top1_share),
        'attr_top5_follower_share': float(top5_share),
        'attr_follower_concentration_H': float(H),
    }


# ============================================================================
# Level 4: Category Composition Features
# ============================================================================

@register_influencer_attr_feature("level4_categories")
def extract_level4_categories(influencer_attrs_df, config):
    """
    Level 4: Category diversity and composition.
    
    - Number of distinct categories
    - Fraction of largest category
    - Category entropy (diversity measure)
    """
    if len(influencer_attrs_df) == 0:
        return {}
    
    categories = influencer_attrs_df['category'].values
    
    # Count categories
    unique_cats, counts = np.unique(categories, return_counts=True)
    n_categories = len(unique_cats)
    N = len(categories)
    
    # Largest category fraction
    max_count = np.max(counts)
    frac_largest = max_count / N
    
    # Category entropy
    cat_entropy = compute_entropy(counts)
    
    return {
        'attr_cat_num_categories': int(n_categories),
        'attr_cat_frac_largest': float(frac_largest),
        'attr_cat_entropy': float(cat_entropy),
    }


# ============================================================================
# Level 5: Attribute-Network Interaction (Optional)
# ============================================================================

@register_influencer_attr_feature("level5_network_interaction")
def extract_level5_network_interaction(influencer_attrs_df, degree_data, config):
    """
    Level 5: Correlation between attributes and network position.
    
    Parameters:
    -----------
    influencer_attrs_df : DataFrame with username, followers, posts
    degree_data : dict mapping username -> degree (from network)
    
    Returns:
    --------
    Pearson correlations:
    - attr_corr_followers_degree
    - attr_corr_posts_degree
    """
    if len(influencer_attrs_df) < 4:  # Need at least 4 points for correlation
        return {
            'attr_corr_followers_degree': np.nan,
            'attr_corr_posts_degree': np.nan,
        }
    
    # Match influencers to their degrees
    matched_data = []
    for _, row in influencer_attrs_df.iterrows():
        username = row['username']
        if username in degree_data:
            matched_data.append({
                'followers': row['followers'],
                'posts': row['posts'],
                'degree': degree_data[username]
            })
    
    if len(matched_data) < 4:
        return {
            'attr_corr_followers_degree': np.nan,
            'attr_corr_posts_degree': np.nan,
        }
    
    df_matched = pd.DataFrame(matched_data)
    
    # Compute correlations
    try:
        corr_followers = float(df_matched['followers'].corr(df_matched['degree']))
        corr_posts = float(df_matched['posts'].corr(df_matched['degree']))
    except Exception:
        corr_followers = np.nan
        corr_posts = np.nan
    
    return {
        'attr_corr_followers_degree': corr_followers,
        'attr_corr_posts_degree': corr_posts,
    }


# ============================================================================
# Temporal Delta Features
# ============================================================================

def compute_deltas(curr_features, prev_features):
    """
    Compute delta features between current and previous window.
    
    Only for scalar numeric features (not flags, not metadata).
    Returns dict with attr_delta_* prefix.
    """
    if prev_features is None:
        return {}
    
    delta_features = {}
    
    # List of features to compute deltas for
    delta_keys = [
        # Level 1
        'attr_followers_log_mean', 'attr_followers_log_std', 'attr_followers_log_median',
        'attr_followees_log_mean', 'attr_followees_log_std',
        'attr_posts_log_mean', 'attr_posts_log_std',
        'attr_ff_ratio_log_mean', 'attr_ff_ratio_log_std',
        'attr_fpp_log_mean', 'attr_fpp_log_std',
        # Level 2
        'attr_frac_micro', 'attr_frac_mid', 'attr_frac_macro', 'attr_frac_mega',
        # Level 3
        'attr_top1_follower_share', 'attr_top5_follower_share', 'attr_follower_concentration_H',
        # Level 4
        'attr_cat_num_categories', 'attr_cat_frac_largest', 'attr_cat_entropy',
        # Level 5
        'attr_corr_followers_degree', 'attr_corr_posts_degree',
    ]
    
    for key in delta_keys:
        curr_val = curr_features.get(key)
        prev_val = prev_features.get(key)
        
        # Only compute delta if both values are present and not NaN
        if curr_val is not None and prev_val is not None:
            if not (np.isnan(curr_val) or np.isnan(prev_val)):
                delta_key = key.replace('attr_', 'attr_delta_')
                delta_features[delta_key] = float(curr_val - prev_val)
    
    return delta_features