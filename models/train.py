# ===== TheOracle: Model Training Pipeline =====
# ===== CELL 1: Imports =====

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from models.datasets import create_temporal_splits, split_by_surface, prepare_xy, get_feature_columns
from models.baseline_elo import EloBaseline
from models.xgboost_model import TennisXGBoost

# ===== CELL 2: Train All Models =====

def train_all_models(features_path="data/features/match_features.parquet",
                     gpu=False):
    """
    Train the complete model suite:
    1. ELO-only baseline (benchmark)
    2. XGBoost per-surface models (Clay, Grass, Hard)
    
    Returns dict of trained models and their evaluation results.
    """
    print("=" * 60)
    print("🎾 TheOracle: Model Training Pipeline")
    print("=" * 60)
    
    # ---- Step 1: Create temporal splits ----
    print("\n📊 STEP 1: Creating temporal splits...")
    splits = create_temporal_splits(features_path)
    
    # ---- Step 2: Baseline evaluation ----
    print("\n📊 STEP 2: ELO Baseline evaluation...")
    feature_cols = get_feature_columns(splits['train'])
    
    baseline = EloBaseline(elo_col='elo_diff_surface')
    
    X_val_all, y_val_all, _, _ = prepare_xy(splits['val'], feature_cols)
    baseline_results = baseline.evaluate(X_val_all, y_val_all, "ELO Baseline (Val 2024)")
    
    if len(splits['test']) > 0:
        X_test_all, y_test_all, _, _ = prepare_xy(splits['test'], feature_cols)
        baseline.evaluate(X_test_all, y_test_all, "ELO Baseline (Test 2025+)")
    
    # ---- Step 3: Per-surface XGBoost ----
    print("\n🔧 STEP 3: Training per-surface XGBoost models...")
    surface_data, feature_cols = split_by_surface(splits)
    
    models = {}
    results = {'baseline': baseline_results}
    
    for surface in ['Clay', 'Grass', 'Hard']:
        print(f"\n{'─'*40}")
        print(f"🎯 Training {surface} model...")
        
        sd = surface_data[surface]
        
        if sd['train'] is None or len(sd['train']['y']) < 100:
            print(f"  ⚠️  Insufficient {surface} data. Skipping.")
            continue
        
        model = TennisXGBoost(surface)
        
        # Train with validation early stopping
        val_X = sd['val']['X'] if sd['val'] else None
        val_y = sd['val']['y'] if sd['val'] else None
        
        model.train(
            sd['train']['X'], sd['train']['y'],
            val_X, val_y,
            feature_cols=feature_cols,
            medians=sd['train']['medians'],
        )
        
        # Evaluate on validation
        if sd['val'] is not None:
            val_results = model.evaluate(sd['val']['X'], sd['val']['y'], f"XGBoost {surface} (Val)")
            results[f'xgb_{surface.lower()}_val'] = val_results
        
        # Evaluate on test
        if sd['test'] is not None and len(sd['test']['y']) > 0:
            test_results = model.evaluate(sd['test']['X'], sd['test']['y'], f"XGBoost {surface} (Test)")
            results[f'xgb_{surface.lower()}_test'] = test_results
        
        # Feature importance
        print(f"\n  📋 Top 15 features ({surface}):")
        fi = model.feature_importance(15)
        for _, row in fi.iterrows():
            bar = '█' * int(row['importance'] * 100)
            print(f"    {row['feature']:35s} {bar} {row['importance']:.4f}")
        
        # Save model
        model.save()
        models[surface] = model
    
    # ---- Step 4: Combined evaluation ----
    print(f"\n{'='*60}")
    print("📊 STEP 4: Combined evaluation across all surfaces...")
    
    # Evaluate combined accuracy (using appropriate surface model for each match)
    for split_name, split_label in [('val', 'Validation 2024'), ('test', 'Test 2025+')]:
        split_df = splits[split_name]
        if len(split_df) == 0:
            continue
        
        correct = 0
        total = 0
        
        for surface, model in models.items():
            surface_df = split_df[split_df['surface'] == surface]
            if len(surface_df) == 0:
                continue
            
            X, y, _, _ = prepare_xy(surface_df, feature_cols)
            preds = model.predict(X)
            correct += (preds == y).sum()
            total += len(y)
        
        if total > 0:
            combined_acc = correct / total
            print(f"  🎯 Combined {split_label}: {combined_acc:.4f} ({combined_acc*100:.1f}%) [{total} matches]")
    
    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETE")
    print(f"   Models saved to: models/artifacts/")
    print(f"{'='*60}")
    
    return models, results

# ===== CELL 3: Entrypoint =====

if __name__ == "__main__":
    models, results = train_all_models()
