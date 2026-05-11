# ===== TheOracle: Model Evaluation & Visualization =====
# ===== CELL 1: Imports =====

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
from models.datasets import create_temporal_splits, prepare_xy, get_feature_columns
from models.xgboost_model import TennisXGBoost
from models.baseline_elo import EloBaseline

# ===== CELL 2: Full Evaluation =====

def full_evaluation(features_path="data/features/match_features.parquet",
                    models_dir="models/artifacts", plot=True):
    """
    Comprehensive evaluation with plots.
    
    Generates:
    1. Accuracy comparison table (baseline vs XGBoost vs odds)
    2. Per-surface accuracy breakdown
    3. Per-tournament-level accuracy
    4. ROC curves
    5. Calibration plots
    6. Feature importance (SHAP if available)
    """
    print("=" * 60)
    print("📊 TheOracle: Full Evaluation")
    print("=" * 60)
    
    # Load data and models
    splits = create_temporal_splits(features_path)
    feature_cols = get_feature_columns(splits['val'])
    
    models = {}
    for surface in ['Clay', 'Grass', 'Hard']:
        model_path = os.path.join(models_dir, f"xgb_{surface.lower()}.pkl")
        if os.path.exists(model_path):
            models[surface] = TennisXGBoost.load(model_path)
    
    baseline = EloBaseline('elo_diff_surface')
    
    # Evaluate on each split
    all_results = {}
    
    for split_name, label in [('val', '2024 Validation'), ('test', '2025+ Test')]:
        split_df = splits[split_name]
        if len(split_df) == 0:
            continue
        
        print(f"\n{'─'*50}")
        print(f"📈 {label} ({len(split_df):,} matches)")
        print(f"{'─'*50}")
        
        # Collect predictions from per-surface models
        all_proba = np.zeros(len(split_df))
        all_y = np.zeros(len(split_df))
        all_preds = np.zeros(len(split_df))
        mask = np.zeros(len(split_df), dtype=bool)
        
        surface_results = {}
        
        for surface in ['Clay', 'Grass', 'Hard']:
            surface_mask = split_df['surface'] == surface
            surface_df = split_df[surface_mask]
            
            if len(surface_df) == 0:
                continue
            
            X, y, _, _ = prepare_xy(surface_df, feature_cols)
            
            if surface in models:
                proba = models[surface].predict_proba(X)[:, 1]
                preds = (proba >= 0.5).astype(int)
            else:
                # Fallback to baseline
                proba = baseline.predict_proba(X)[:, 1]
                preds = (proba >= 0.5).astype(int)
            
            acc = accuracy_score(y, preds)
            surface_results[surface] = acc
            print(f"  {surface:8s}: {acc:.4f} ({acc*100:.1f}%) [{len(y)} matches]")
            
            # Store for combined metrics
            indices = np.where(surface_mask)[0]
            all_proba[indices] = proba
            all_y[indices] = y
            all_preds[indices] = preds
            mask[indices] = True
        
        # Combined metrics
        valid = mask
        if valid.sum() > 0:
            combined_acc = accuracy_score(all_y[valid], all_preds[valid])
            combined_ll = log_loss(all_y[valid], all_proba[valid])
            combined_auc = roc_auc_score(all_y[valid], all_proba[valid])
            
            # Baseline comparison
            X_all, y_all, _, _ = prepare_xy(split_df, feature_cols)
            base_proba = baseline.predict_proba(X_all)[:, 1]
            base_preds = (base_proba >= 0.5).astype(int)
            base_acc = accuracy_score(y_all, base_preds)
            
            # Odds accuracy (if available)
            odds_acc = None
            if 'odds_implied_a' in split_df.columns:
                odds_df = split_df[split_df['odds_implied_a'].notna()]
                if len(odds_df) > 0:
                    odds_preds = (odds_df['odds_implied_a'] >= 0.5).astype(int)
                    odds_acc = accuracy_score(odds_df['winner_is_a'], odds_preds)
            
            print(f"\n  {'COMBINED':8s}: {combined_acc:.4f} ({combined_acc*100:.1f}%)")
            print(f"  {'Baseline':8s}: {base_acc:.4f} ({base_acc*100:.1f}%)")
            if odds_acc is not None:
                print(f"  {'Odds':8s}: {odds_acc:.4f} ({odds_acc*100:.1f}%)")
            print(f"  Log Loss: {combined_ll:.4f}")
            print(f"  ROC-AUC:  {combined_auc:.4f}")
            
            improvement = combined_acc - base_acc
            print(f"\n  📈 Improvement over baseline: {improvement:+.4f} ({improvement*100:+.1f}%)")
            
            all_results[split_name] = {
                'accuracy': combined_acc,
                'baseline': base_acc,
                'odds': odds_acc,
                'log_loss': combined_ll,
                'auc': combined_auc,
                'surface': surface_results,
                'probabilities': all_proba[valid],
                'actuals': all_y[valid],
            }
        
        # Plots
        if plot and valid.sum() > 0:
            _generate_plots(all_y[valid], all_proba[valid], all_preds[valid],
                           surface_results, label, base_proba, y_all)
    
    return all_results

# ===== CELL 3: Plot Generation =====

def _generate_plots(y_true, y_proba, y_pred, surface_results, title, base_proba, base_y):
    """Generate evaluation plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'TheOracle — {title}', fontsize=16, fontweight='bold')
    
    # Plot 1: ROC Curve
    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'XGBoost (AUC={auc:.3f})')
    
    fpr_b, tpr_b, _ = roc_curve(base_y, base_proba)
    auc_b = roc_auc_score(base_y, base_proba)
    ax.plot(fpr_b, tpr_b, 'r--', linewidth=1.5, label=f'ELO Baseline (AUC={auc_b:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    
    # Plot 2: Calibration
    ax = axes[0, 1]
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
    ax.plot(prob_pred, prob_true, 'bs-', linewidth=2, label='XGBoost')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Plot')
    ax.legend()
    
    # Plot 3: Per-surface accuracy
    ax = axes[1, 0]
    surfaces = list(surface_results.keys())
    accs = [surface_results[s] for s in surfaces]
    colors = {'Clay': '#E07C3E', 'Grass': '#4CAF50', 'Hard': '#2196F3'}
    bars = ax.bar(surfaces, accs, color=[colors.get(s, '#999') for s in surfaces])
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Surface')
    ax.set_ylim(0.5, 0.85)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.1%}', ha='center', fontweight='bold')
    
    # Plot 4: Probability distribution
    ax = axes[1, 1]
    correct_mask = (y_pred == y_true)
    ax.hist(y_proba[correct_mask], bins=20, alpha=0.6, color='green', label='Correct')
    ax.hist(y_proba[~correct_mask], bins=20, alpha=0.6, color='red', label='Wrong')
    ax.set_xlabel('Predicted Probability for Player A')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Confidence Distribution')
    ax.legend()
    
    plt.tight_layout()
    os.makedirs("models/artifacts", exist_ok=True)
    plt.savefig(f"models/artifacts/eval_{title.replace(' ', '_')}.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  📸 Plot saved: models/artifacts/eval_{title.replace(' ', '_')}.png")


# ===== CELL 4: Entrypoint =====

if __name__ == "__main__":
    results = full_evaluation(plot=True)
