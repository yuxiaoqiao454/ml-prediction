#!/usr/bin/env python3
"""
Recover experiment_results.csv from corrupted backup.

The backup has duplicate column headers. This script cleans it up.
"""

import pandas as pd
import numpy as np

# Read the corrupted CSV
print("Reading corrupted backup CSV...")
df = pd.read_csv('04_ml_prediction/04_models/results/experiment_results_backup.csv')

print(f"Original shape: {df.shape}")
print(f"Original columns: {len(df.columns)}")

# The issue: columns are duplicated
# First 25 columns are original (with spaces)
# Then duplicate columns start at position 25

# Define correct column names (without spaces)
correct_columns = [
    'experiment_id',
    'timestamp',
    'model_type',
    'dataset_version',
    'config_name',
    'n_train',
    'n_val',
    'n_test',
    'n_bursts_train',
    'n_bursts_val',
    'n_bursts_test',
    'train_auc',
    'train_pr_auc',
    'val_auc',
    'val_pr_auc',
    'val_precision',
    'val_recall',
    'val_f1',
    'test_auc',
    'test_pr_auc',
    'test_precision',
    'test_recall',
    'test_f1',
    'model_path',
    'config_path',
]

# Get actual column names (they have spaces)
actual_cols = df.columns.tolist()
print(f"\nFirst 30 columns:")
for i, col in enumerate(actual_cols[:30]):
    print(f"  {i}: '{col}'")

# Strategy: The first 25 columns contain the real data
# The remaining columns are from exp_024 only

# Keep only first 25 columns for rows 0-22 (exp_001 to exp_023)
df_old = df.iloc[:23, :25].copy()
df_old.columns = correct_columns

# For exp_024 (row 23), extract from the duplicated columns (positions 25+)
df_new_row = df.iloc[23:24].copy()

# Extract exp_024 data from the second set of columns (starting at position 26)
exp_024 = {
    'experiment_id': 'exp_024',
    'timestamp': df_new_row.iloc[0, 26] if len(df.columns) > 26 else '2025-11-04T17:09:49.116858',
    'model_type': df_new_row.iloc[0, 27] if len(df.columns) > 27 else 'random_forest',
    'dataset_version': df_new_row.iloc[0, 28] if len(df.columns) > 28 else 'h550_cpinside',
    'config_name': df_new_row.iloc[0, 29] if len(df.columns) > 29 else 'Shallow Random Forest (less overfitting)',
    'n_train': df_new_row.iloc[0, 30] if len(df.columns) > 30 else 8329.0,
    'n_val': df_new_row.iloc[0, 31] if len(df.columns) > 31 else 1836.0,
    'n_test': df_new_row.iloc[0, 32] if len(df.columns) > 32 else 1957.0,
    'n_bursts_train': df_new_row.iloc[0, 33] if len(df.columns) > 33 else 691.0,
    'n_bursts_val': df_new_row.iloc[0, 34] if len(df.columns) > 34 else 139.0,
    'n_bursts_test': df_new_row.iloc[0, 35] if len(df.columns) > 35 else 166.0,
    'train_auc': df_new_row.iloc[0, 36] if len(df.columns) > 36 else 0.8846346377640322,
    'train_pr_auc': df_new_row.iloc[0, 37] if len(df.columns) > 37 else 0.5636653855364958,
    'val_auc': df_new_row.iloc[0, 38] if len(df.columns) > 38 else 0.7667869240258942,
    'val_pr_auc': df_new_row.iloc[0, 39] if len(df.columns) > 39 else 0.3273023504863284,
    'val_precision': df_new_row.iloc[0, 40] if len(df.columns) > 40 else 0.2478632478632478,
    'val_recall': df_new_row.iloc[0, 41] if len(df.columns) > 41 else 0.4172661870503597,
    'val_f1': df_new_row.iloc[0, 42] if len(df.columns) > 42 else 0.3109919571045576,
    'test_auc': df_new_row.iloc[0, 43] if len(df.columns) > 43 else 0.8408979300787741,
    'test_pr_auc': df_new_row.iloc[0, 44] if len(df.columns) > 44 else 0.4712145462221874,
    'test_precision': df_new_row.iloc[0, 45] if len(df.columns) > 45 else 0.3405017921146953,
    'test_recall': df_new_row.iloc[0, 46] if len(df.columns) > 46 else 0.572289156626506,
    'test_f1': df_new_row.iloc[0, 47] if len(df.columns) > 47 else 0.4269662921348314,
    'model_path': df_new_row.iloc[0, 48] if len(df.columns) > 48 else '04_ml_prediction/04_models/trained/exp_024_random_forest_h550_cpinside',
    'config_path': df_new_row.iloc[0, 49] if len(df.columns) > 49 else '04_ml_prediction/04_models/configs/random_forest.yaml',
}

# Create exp_024 DataFrame
df_024 = pd.DataFrame([exp_024])

# Combine all experiments
df_recovered = pd.concat([df_old, df_024], ignore_index=True)

# Add threshold columns (all used default 0.5)
df_recovered['threshold_used'] = 0.5
df_recovered['threshold_method'] = 'default_0.5'
df_recovered['threshold_val_score'] = None
df_recovered['threshold_constraint'] = None

# Clean up any whitespace in string columns
for col in df_recovered.columns:
    if df_recovered[col].dtype == 'object':
        df_recovered[col] = df_recovered[col].astype(str).str.strip()

# Convert numeric columns
numeric_cols = [
    'n_train', 'n_val', 'n_test', 
    'n_bursts_train', 'n_bursts_val', 'n_bursts_test',
    'train_auc', 'train_pr_auc',
    'val_auc', 'val_pr_auc', 'val_precision', 'val_recall', 'val_f1',
    'test_auc', 'test_pr_auc', 'test_precision', 'test_recall', 'test_f1',
    'threshold_used'
]

for col in numeric_cols:
    df_recovered[col] = pd.to_numeric(df_recovered[col], errors='coerce')

print(f"\n{'='*80}")
print("RECOVERED DATA")
print(f"{'='*80}")
print(f"Shape: {df_recovered.shape}")
print(f"Experiments: {len(df_recovered)}")
print(f"\nFirst few rows:")
print(df_recovered[['experiment_id', 'model_type', 'dataset_version', 'test_auc']].head(10))
print(f"\nLast few rows:")
print(df_recovered[['experiment_id', 'model_type', 'dataset_version', 'test_auc']].tail(5))

# Check for any missing data
print(f"\n{'='*80}")
print("DATA QUALITY CHECK")
print(f"{'='*80}")
missing_counts = df_recovered.isnull().sum()
if missing_counts.sum() == 0:
    print("✓ No missing data!")
else:
    print("Missing values per column:")
    print(missing_counts[missing_counts > 0])

# Save recovered CSV
output_path = '04_ml_prediction/04_models/results/experiment_results_RECOVERED.csv'
df_recovered.to_csv(output_path, index=False)
print(f"\n✓ Saved recovered data to: {output_path}")
print(f"\n{'='*80}")
print("NEXT STEPS:")
print(f"{'='*80}")
print("1. Review the recovered CSV to verify all data is correct")
print("2. Backup your original: mv experiment_results.csv experiment_results_BROKEN.csv")
print("3. Replace with recovered: mv experiment_results_RECOVERED.csv experiment_results.csv")
print(f"{'='*80}")