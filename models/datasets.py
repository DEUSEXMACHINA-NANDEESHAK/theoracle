# ===== TheOracle: Dataset Splitter =====
# ===== CELL 1: Imports =====

import os
import numpy as np
import pandas as pd

# ===== CELL 2: Temporal Split =====

def create_temporal_splits(features_path="data/features/match_features.parquet",
                           warmup_end=20041231,
                           train_end=20231231,
                           val_end=20241231):
    """
    Create temporal train/validation/test splits.
    
    CRITICAL: We split by TIME, never randomly.
    Random splits cause data leakage because future matches
    would leak into training data.
    
    Split structure:
    - Warmup: 2000-2004 (EXCLUDED from training — only used to warm up ELO)
    - Train:  2005-2023 (model training)
    - Val:    2024      (hyperparameter tuning)
    - Test:   2025+     (final evaluation, used ONCE)
    
    Returns:
        dict with 'train', 'val', 'test' DataFrames + metadata
    """
    print("📂 Loading feature store...")
    df = pd.read_parquet(features_path)
    
    # Filter out warmup period (2000-2004)
    # These matches were used to warm up ELO but should NOT be in training
    df = df[df['tourney_date'] > warmup_end].copy()
    
    # Split by time
    train = df[df['tourney_date'] <= train_end].copy()
    val = df[(df['tourney_date'] > train_end) & (df['tourney_date'] <= val_end)].copy()
    test = df[df['tourney_date'] > val_end].copy()
    
    print(f"\n📊 Dataset splits:")
    print(f"   Train: {len(train):,} matches (2005-2023)")
    print(f"   Val:   {len(val):,} matches (2024)")
    print(f"   Test:  {len(test):,} matches (2025+)")
    print(f"\n   Surface distribution (Train):")
    print(f"   {dict(train['surface'].value_counts())}")
    print(f"   Target balance: {train['winner_is_a'].mean():.3f} (should be ~0.50)")
    
    return {
        'train': train,
        'val': val,
        'test': test,
        'full': df,
    }

# ===== CELL 3: Feature Selection =====

# Columns to EXCLUDE from model features
EXCLUDE_COLS = [
    'match_id', 'tourney_date', 'surface', 'winner_is_a',  # Meta / target
]

# Columns that should only be used as identifiers
ID_COLS = ['match_id', 'tourney_date']

def get_feature_columns(df):
    """Get list of feature columns (exclude target, IDs, and non-numeric)."""
    feature_cols = []
    for col in df.columns:
        if col in EXCLUDE_COLS:
            continue
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
            feature_cols.append(col)
    return feature_cols

def prepare_xy(df, feature_cols=None):
    """
    Prepare feature matrix X and target vector y.
    
    Handles:
    - Feature selection
    - Missing value imputation (median for numeric)
    - Inf value replacement
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)
    
    X = df[feature_cols].copy()
    y = df['winner_is_a'].values
    
    # Replace inf with NaN, then impute
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with median (per column)
    medians = X.median()
    X = X.fillna(medians)
    
    # Safety: any remaining NaN → 0
    X = X.fillna(0)
    
    return X, y, feature_cols, medians

# ===== CELL 4: Per-Surface Split =====

def split_by_surface(splits):
    """
    Create per-surface datasets for surface-specific models.
    
    Returns dict: surface → {'train': (X, y), 'val': (X, y), 'test': (X, y)}
    """
    feature_cols = get_feature_columns(splits['train'])
    
    surface_data = {}
    for surface in ['Clay', 'Grass', 'Hard']:
        surface_splits = {}
        for split_name in ['train', 'val', 'test']:
            df_split = splits[split_name]
            df_surface = df_split[df_split['surface'] == surface].copy()
            
            if len(df_surface) > 0:
                X, y, _, medians = prepare_xy(df_surface, feature_cols)
                surface_splits[split_name] = {
                    'X': X, 'y': y, 'df': df_surface, 'medians': medians,
                }
            else:
                surface_splits[split_name] = None
        
        surface_data[surface] = surface_splits
        
        train_count = len(surface_splits['train']['y']) if surface_splits['train'] else 0
        val_count = len(surface_splits['val']['y']) if surface_splits['val'] else 0
        test_count = len(surface_splits['test']['y']) if surface_splits['test'] else 0
        print(f"   {surface}: Train={train_count:,} Val={val_count:,} Test={test_count:,}")
    
    return surface_data, feature_cols
