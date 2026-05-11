# ===== TheOracle: XGBoost Per-Surface Model =====
# ===== CELL 1: Imports =====

import os
import yaml
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss

# ===== CELL 2: XGBoost Model Wrapper =====

class TennisXGBoost:
    """
    XGBoost classifier wrapper for tennis match prediction.
    
    One model per surface (Clay, Grass, Hard).
    Config-driven hyperparameters from YAML files.
    """
    
    def __init__(self, surface, config_path=None):
        self.surface = surface
        self.model = None
        self.feature_cols = None
        self.medians = None  # For imputation at inference time
        
        # Load config
        if config_path is None:
            config_path = f"configs/model_xgb_{surface.lower()}.yaml"
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.params = config.get('hyperparameters', {})
        else:
            # Default params
            self.params = {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_lambda': 1.0,
                'reg_alpha': 0.1,
                'min_child_weight': 3,
                'eval_metric': 'logloss',
                'early_stopping_rounds': 50,
                'tree_method': 'auto',
                'n_jobs': -1,  # Use all available CPU cores
                'verbosity': 0,
            }
    
    def train(self, X_train, y_train, X_val=None, y_val=None, feature_cols=None, medians=None):
        """
        Train XGBoost model with early stopping on validation set.
        """
        self.feature_cols = feature_cols
        self.medians = medians
        
        # Separate early stopping params
        early_stopping = self.params.pop('early_stopping_rounds', 50)
        
        # Check GPU availability
        tree_method = self.params.get('tree_method', 'auto')
        try:
            if tree_method == 'auto':
                # Try GPU first
                test_model = xgb.XGBClassifier(tree_method='gpu_hist', n_estimators=1, verbosity=0)
                test_model.fit(X_train.iloc[:10], y_train[:10])
                self.params['tree_method'] = 'gpu_hist'
                print(f"  🚀 GPU detected! Using gpu_hist for {self.surface}")
        except Exception:
            self.params['tree_method'] = 'hist'
            print(f"  💻 Using CPU (hist) for {self.surface}")
        
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            **self.params,
        )
        
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            # Manually check early stopping
            print(f"  ✅ {self.surface} model trained ({self.model.n_estimators} trees)")
        else:
            self.model.fit(X_train, y_train, verbose=False)
            print(f"  ✅ {self.surface} model trained ({self.params.get('n_estimators', 500)} trees)")
        
        # Restore early stopping param
        self.params['early_stopping_rounds'] = early_stopping
    
    def predict_proba(self, X):
        """Predict win probabilities."""
        if self.model is None:
            raise RuntimeError("Model not trained yet!")
        X_clean = self._prepare_input(X)
        return self.model.predict_proba(X_clean)
    
    def predict(self, X):
        """Predict match winner."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
    
    def _prepare_input(self, X):
        """Prepare input features (handle missing columns, impute)."""
        if self.feature_cols is not None:
            # Ensure all expected columns present
            missing = set(self.feature_cols) - set(X.columns)
            X_out = X.copy()
            for col in missing:
                X_out[col] = 0
            X_out = X_out[self.feature_cols]
        else:
            X_out = X.copy()
        
        X_out = X_out.replace([np.inf, -np.inf], np.nan)
        if self.medians is not None:
            X_out = X_out.fillna(self.medians)
        X_out = X_out.fillna(0)
        return X_out
    
    def evaluate(self, X, y, label=None):
        """Evaluate model on a dataset."""
        if label is None:
            label = f"XGBoost {self.surface}"
        
        proba = self.predict_proba(X)
        preds = self.predict(X)
        
        acc = accuracy_score(y, preds)
        ll = log_loss(y, proba[:, 1])
        auc = roc_auc_score(y, proba[:, 1])
        brier = brier_score_loss(y, proba[:, 1])
        
        print(f"\n📊 {label}:")
        print(f"   Accuracy:    {acc:.4f} ({acc*100:.1f}%)")
        print(f"   Log Loss:    {ll:.4f}")
        print(f"   ROC-AUC:     {auc:.4f}")
        print(f"   Brier Score: {brier:.4f}")
        
        return {'accuracy': acc, 'log_loss': ll, 'roc_auc': auc, 'brier': brier}
    
    def feature_importance(self, top_n=20):
        """Get top N most important features."""
        if self.model is None or self.feature_cols is None:
            return pd.DataFrame()
        
        importance = self.model.feature_importances_
        fi = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importance,
        }).sort_values('importance', ascending=False)
        
        return fi.head(top_n)
    
    def save(self, path=None):
        """Save model to disk."""
        if path is None:
            os.makedirs("models/artifacts", exist_ok=True)
            path = f"models/artifacts/xgb_{self.surface.lower()}.pkl"
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_cols': self.feature_cols,
                'medians': self.medians,
                'surface': self.surface,
                'params': self.params,
            }, f)
        print(f"  💾 Saved {self.surface} model → {path}")
    
    @classmethod
    def load(cls, path):
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(data['surface'])
        model.model = data['model']
        model.feature_cols = data['feature_cols']
        model.medians = data['medians']
        model.params = data['params']
        return model
