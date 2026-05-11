# ===== TheOracle: ELO-Only Baseline Model =====
# ===== CELL 1: Imports =====

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

# ===== CELL 2: Baseline Predictor =====

class EloBaseline:
    """
    Simple baseline that predicts match winner using only ELO difference.
    
    This is our "beat this" benchmark. If the XGBoost model can't beat
    this, our additional features aren't adding value.
    
    Uses logistic function: P(A wins) = 1 / (1 + 10^((elo_b - elo_a) / 400))
    """
    
    def __init__(self, elo_col='elo_diff_surface'):
        self.elo_col = elo_col
    
    def predict_proba(self, X):
        """Predict win probability from ELO difference."""
        if self.elo_col in X.columns:
            elo_diff = X[self.elo_col].values
        else:
            # Fallback to overall ELO diff
            elo_diff = X.get('elo_diff_overall', np.zeros(len(X))).values
        
        # Logistic function
        prob_a = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        prob_a = np.clip(prob_a, 0.01, 0.99)
        return np.column_stack([1 - prob_a, prob_a])
    
    def predict(self, X):
        """Predict match winner (0 or 1)."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
    
    def evaluate(self, X, y, label="ELO Baseline"):
        """Evaluate baseline on a dataset."""
        proba = self.predict_proba(X)
        preds = self.predict(X)
        
        acc = accuracy_score(y, preds)
        ll = log_loss(y, proba[:, 1])
        auc = roc_auc_score(y, proba[:, 1])
        
        print(f"\n📊 {label}:")
        print(f"   Accuracy: {acc:.4f} ({acc*100:.1f}%)")
        print(f"   Log Loss: {ll:.4f}")
        print(f"   ROC-AUC:  {auc:.4f}")
        
        return {'accuracy': acc, 'log_loss': ll, 'roc_auc': auc}
