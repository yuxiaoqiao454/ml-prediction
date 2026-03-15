#!/usr/bin/env python3
"""
Cross-validation splitters that respect hashtag grouping and stratification.

Key principle: All samples from the same hashtag MUST stay together in the same fold.
This prevents leakage from overlapping time windows within a hashtag.

Usage:
    from cv_splitters import StratifiedHashtagKFold
    
    cv = StratifiedHashtagKFold(n_splits=5, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(samples_df)):
        train_data = samples_df.iloc[train_idx]
        test_data = samples_df.iloc[test_idx]
"""

import numpy as np
import pandas as pd
from typing import Iterator, Tuple
from sklearn.model_selection import StratifiedKFold


class StratifiedHashtagKFold:
    """
    K-Fold CV that keeps all samples from the same hashtag together,
    while stratifying by burst rate.
    
    This is critical because:
    1. Consecutive windows from the same hashtag have overlapping data
    2. Splitting a hashtag across folds would cause temporal leakage
    3. We want to test generalization to NEW hashtags
    
    Strategy:
    1. Group samples by hashtag
    2. Compute burst rate per hashtag (for stratification)
    3. Bin hashtags by burst rate quartiles
    4. Use StratifiedKFold on hashtags (not samples)
    5. Map hashtag assignments back to sample indices
    """
    
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        """
        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle before splitting
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        
    def split(self, samples_df: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for each fold.
        
        Args:
            samples_df: DataFrame with columns ['hashtag', 'label', ...]
            
        Yields:
            (train_indices, test_indices) for each fold
        """
        # Step 1: Compute per-hashtag burst rates
        hashtag_stats = samples_df.groupby('hashtag').agg({
            'label': ['sum', 'count', 'mean']
        }).reset_index()
        hashtag_stats.columns = ['hashtag', 'n_bursts', 'n_samples', 'burst_rate']
        
        # Step 2: Create stratification bins
        hashtag_stats = self._create_burst_bins(hashtag_stats)
        
        # Step 3: Split hashtags using StratifiedKFold
        hashtags = hashtag_stats['hashtag'].values
        strata = hashtag_stats['burst_bin'].values
        
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
        
        # Step 4: For each fold, map hashtags → sample indices
        for hashtag_train_idx, hashtag_test_idx in skf.split(hashtags, strata):
            train_hashtags = set(hashtags[hashtag_train_idx])
            test_hashtags = set(hashtags[hashtag_test_idx])
            
            # Get sample indices for these hashtags
            train_mask = samples_df['hashtag'].isin(train_hashtags)
            test_mask = samples_df['hashtag'].isin(test_hashtags)
            
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            yield train_indices, test_indices
    
    def _create_burst_bins(self, hashtag_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Bin hashtags by burst rate for stratification.
        
        Uses quartiles if enough unique values, otherwise fewer bins.
        Merges bins with <2 hashtags to ensure StratifiedKFold works.
        """
        n_unique = hashtag_stats['burst_rate'].nunique()
        
        if n_unique < 2:
            # All hashtags have same burst rate
            hashtag_stats['burst_bin'] = 'all'
        elif n_unique < 4:
            # Use equal-width bins
            hashtag_stats['burst_bin'] = pd.cut(
                hashtag_stats['burst_rate'],
                bins=n_unique,
                duplicates='drop'
            )
        else:
            # Use quartiles
            try:
                hashtag_stats['burst_bin'] = pd.qcut(
                    hashtag_stats['burst_rate'],
                    q=4,
                    labels=['very_low', 'low', 'medium', 'high'],
                    duplicates='drop'
                )
            except ValueError:
                # Fall back to 3 bins if quartiles fail
                hashtag_stats['burst_bin'] = pd.cut(
                    hashtag_stats['burst_rate'],
                    bins=3,
                    duplicates='drop'
                )
        
        # Merge rare bins (bins with <2 hashtags can't be stratified)
        hashtag_stats = self._merge_rare_bins(hashtag_stats)
        
        return hashtag_stats
    
    def _merge_rare_bins(self, hashtag_stats: pd.DataFrame) -> pd.DataFrame:
        """Merge bins with <2 hashtags into nearest larger bin."""
        bin_counts = hashtag_stats['burst_bin'].value_counts()
        
        if (bin_counts < 2).any():
            rare_bins = bin_counts[bin_counts < 2].index
            non_rare_bins = bin_counts[bin_counts >= 2].index
            
            if len(non_rare_bins) == 0:
                # All bins are rare, just use 'all'
                hashtag_stats['burst_bin'] = 'all'
            else:
                # Merge each rare bin into nearest non-rare bin
                for rare_bin in rare_bins:
                    rare_mask = hashtag_stats['burst_bin'] == rare_bin
                    if not rare_mask.any():
                        continue
                    
                    # Find nearest bin by median burst rate
                    rare_rate = hashtag_stats.loc[rare_mask, 'burst_rate'].median()
                    bin_medians = hashtag_stats[
                        hashtag_stats['burst_bin'].isin(non_rare_bins)
                    ].groupby('burst_bin', observed=True)['burst_rate'].median()
                    
                    closest_bin = bin_medians.index[
                        (bin_medians - rare_rate).abs().argmin()
                    ]
                    
                    hashtag_stats.loc[rare_mask, 'burst_bin'] = closest_bin
        
        return hashtag_stats
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splits (sklearn compatibility)."""
        return self.n_splits