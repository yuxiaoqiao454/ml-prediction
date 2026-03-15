#!/usr/bin/env python3
"""Fix corrupted experiment_results.csv"""

import pandas as pd
from pathlib import Path

csv_path = Path("04_ml_prediction/04_models/results/experiment_results_legacy.csv")

# Read the corrupted CSV
df = pd.read_csv(csv_path)

print(f"Current columns: {list(df.columns)}")
print(f"Current shape: {df.shape}")

# Define correct column order (original columns + new threshold columns)
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
    # New columns
    'threshold_used',
    'threshold_method',
    'threshold_val_score',
    'threshold_constraint',
]

# Add missing columns with defaults
for col in correct_columns:
    if col not in df.columns:
        if col == 'threshold_used':
            df[col] = 0.5
        elif col == 'threshold_method':
            df[col] = 'default_0.5'
        else:
            df[col] = None

# Reorder columns
df = df[correct_columns]

# Save backup first
backup_path = csv_path.parent / "experiment_results_backup.csv"
df_original = pd.read_csv(csv_path)
df_original.to_csv(backup_path, index=False)
print(f"✓ Backup saved to {backup_path}")

# Save fixed version
df.to_csv(csv_path, index=False)
print(f"✓ Fixed CSV saved to {csv_path}")
print(f"New shape: {df.shape}")
print(f"Columns: {list(df.columns)}")