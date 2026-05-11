# ===== TheOracle: Data Leakage Tests =====
# CRITICAL: These tests verify that NO future data leaks into features.
#
# ===== CELL 1: Imports =====

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import unittest

# ===== CELL 2: Leakage Tests =====

class TestNoDataLeakage(unittest.TestCase):
    """
    Verify that the feature store has no data leakage.
    
    These tests are CRITICAL for model integrity.
    Green Code's original 85% accuracy was inflated by leakage;
    fixing it dropped to a realistic ~66-67%.
    """
    
    @classmethod
    def setUpClass(cls):
        """Load feature store and match data."""
        features_path = "data/features/match_features.parquet"
        matches_path = "data/clean/matches.parquet"
        
        if not os.path.exists(features_path):
            raise unittest.SkipTest("Feature store not built yet")
        
        cls.features = pd.read_parquet(features_path)
        if os.path.exists(matches_path):
            cls.matches = pd.read_parquet(matches_path)
        else:
            cls.matches = None
    
    def test_target_balance(self):
        """Target should be roughly 50/50 (deterministic ordering)."""
        mean = self.features['winner_is_a'].mean()
        # Should be close to 0.5 since we order players by ID
        self.assertGreater(mean, 0.35, f"Target too imbalanced: {mean:.3f}")
        self.assertLess(mean, 0.65, f"Target too imbalanced: {mean:.3f}")
    
    def test_no_perfect_features(self):
        """No single feature should perfectly predict the target."""
        y = self.features['winner_is_a']
        
        for col in self.features.columns:
            if col in ['match_id', 'tourney_date', 'surface', 'winner_is_a']:
                continue
            
            vals = self.features[col].dropna()
            if len(vals) == 0:
                continue
            
            # Check correlation with target
            if vals.dtype in [np.float64, np.float32, np.int64, np.int32]:
                corr = abs(self.features[[col, 'winner_is_a']].corr().iloc[0, 1])
                self.assertLess(
                    corr, 0.95,
                    f"Feature '{col}' has suspiciously high correlation ({corr:.3f}) — possible leakage!"
                )
    
    def test_elo_not_post_match(self):
        """
        ELO features should show realistic distribution.
        If ELO diff perfectly separates winners, it's likely post-match ELO.
        """
        if 'elo_diff_overall' not in self.features.columns:
            self.skipTest("ELO features not found")
        
        elo_diff = self.features['elo_diff_overall']
        y = self.features['winner_is_a']
        
        # When elo_diff > 0, player A should win more often but NOT always
        positive_mask = elo_diff > 0
        if positive_mask.sum() > 100:
            win_rate = y[positive_mask].mean()
            # Should be above 0.5 but well below 1.0
            self.assertGreater(win_rate, 0.45, "ELO diff doesn't predict at all")
            self.assertLess(win_rate, 0.90, "ELO diff too predictive — likely post-match!")
    
    def test_chronological_order(self):
        """Features should be in chronological order."""
        dates = self.features['tourney_date'].values
        # Allow some ties (same tournament date) but overall increasing
        diffs = np.diff(dates.astype(float))
        negative_jumps = (diffs < -10000).sum()  # More than a year backward
        self.assertEqual(
            negative_jumps, 0,
            f"Found {negative_jumps} large backward time jumps — ordering is wrong!"
        )
    
    def test_feature_count_reasonable(self):
        """Should have a reasonable number of features (not too few, not absurd)."""
        n_features = len([c for c in self.features.columns 
                         if c not in ['match_id', 'tourney_date', 'surface', 'winner_is_a']])
        self.assertGreater(n_features, 20, "Too few features")
        self.assertLess(n_features, 500, "Suspiciously many features")
    
    def test_no_future_h2h(self):
        """H2H counts should be reasonable (not inflated by future matches)."""
        if 'h2h_total' not in self.features.columns:
            self.skipTest("H2H features not found")
        
        # Max H2H should be reasonable (even Djokovic-Nadal have ~60 meetings)
        max_h2h = self.features['h2h_total'].max()
        self.assertLess(max_h2h, 100, f"Max H2H={max_h2h} — suspiciously high")


# ===== CELL 3: ELO Sanity Tests =====

class TestEloSanity(unittest.TestCase):
    """Test that ELO ratings make tennis sense."""
    
    @classmethod
    def setUpClass(cls):
        features_path = "data/features/match_features.parquet"
        if not os.path.exists(features_path):
            raise unittest.SkipTest("Feature store not built yet")
        cls.features = pd.read_parquet(features_path)
    
    def test_elo_range(self):
        """ELO ratings should be in a reasonable range."""
        for col in ['elo_overall_a', 'elo_overall_b']:
            if col in self.features.columns:
                vals = self.features[col].dropna()
                self.assertGreater(vals.min(), 800, f"{col} min too low")
                self.assertLess(vals.max(), 3000, f"{col} max too high")
    
    def test_surface_elo_exists(self):
        """Surface ELO features should exist."""
        expected = ['elo_surface_a', 'elo_surface_b', 'elo_diff_surface']
        for col in expected:
            self.assertIn(col, self.features.columns, f"Missing: {col}")
    
    def test_elo_diff_predictive(self):
        """ELO difference should be at least somewhat predictive."""
        if 'elo_diff_surface' not in self.features.columns:
            self.skipTest("ELO diff not found")
        
        elo_diff = self.features['elo_diff_surface']
        y = self.features['winner_is_a']
        
        # Top quintile of ELO diff should have higher win rate
        threshold = elo_diff.quantile(0.8)
        high_diff = y[elo_diff > threshold].mean()
        low_diff = y[elo_diff < elo_diff.quantile(0.2)].mean()
        
        self.assertGreater(
            high_diff, low_diff,
            "High ELO diff doesn't predict better — ELO is broken!"
        )


# ===== CELL 4: Run Tests =====

if __name__ == "__main__":
    unittest.main(verbosity=2)
