# ===== TheOracle: ELO Engine Tests =====
# ===== CELL 1: Imports =====

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest
from features.elo_engine import (
    PlayerEloState, compute_k_factor, compute_expected_score,
    age_adjustment, load_elo_config
)

# ===== CELL 2: K-Factor Tests =====

class TestKFactor(unittest.TestCase):
    """Test tennis-specific K-factor computation."""
    
    def setUp(self):
        self.config = load_elo_config()
    
    def test_k_decreases_with_experience(self):
        """K should decrease as player plays more matches."""
        k_new = compute_k_factor(10, 'A', None, 3, 3, self.config)
        k_exp = compute_k_factor(200, 'A', None, 3, 3, self.config)
        self.assertGreater(k_new, k_exp, "K should decrease with experience")
    
    def test_k_higher_for_slams(self):
        """Grand Slam matches should have higher K than 250s."""
        k_slam = compute_k_factor(50, 'G', None, 3, 5, self.config)
        k_250 = compute_k_factor(50, 'A', None, 2, 3, self.config)
        self.assertGreater(k_slam, k_250)
    
    def test_inactivity_boosts_k(self):
        """Long absence should increase K (more uncertainty)."""
        k_active = compute_k_factor(50, 'A', 7, 3, 3, self.config)
        k_inactive = compute_k_factor(50, 'A', 180, 3, 3, self.config)
        self.assertGreater(k_inactive, k_active)
    
    def test_straight_sets_higher_k(self):
        """Straight sets win should give higher K (more informative)."""
        k_straight = compute_k_factor(50, 'A', None, 2, 3, self.config)  # 2 of 3 sets
        k_full = compute_k_factor(50, 'A', None, 3, 3, self.config)      # 3 of 3 sets
        self.assertGreater(k_straight, k_full)

# ===== CELL 3: Expected Score Tests =====

class TestExpectedScore(unittest.TestCase):
    """Test expected score with tennis adjustments."""
    
    def setUp(self):
        self.config = load_elo_config()
    
    def test_equal_elo_gives_50_50(self):
        """Equal ELO should give ~50% expected score."""
        expected = compute_expected_score(1500, 1500, config=self.config)
        self.assertAlmostEqual(expected, 0.5, places=1)
    
    def test_higher_elo_favored(self):
        """Higher ELO player should have >50% expected score."""
        expected = compute_expected_score(1800, 1500, config=self.config)
        self.assertGreater(expected, 0.5)
    
    def test_h2h_adjusts_expected(self):
        """H2H record should nudge the expected score."""
        no_h2h = compute_expected_score(1700, 1700, 0, 0, config=self.config)
        with_h2h = compute_expected_score(1700, 1700, 10, 2, config=self.config)
        self.assertGreater(with_h2h, no_h2h, "H2H advantage should increase expected")
    
    def test_bo5_amplifies_favorite(self):
        """BO5 should increase the favorite's expected score."""
        bo3 = compute_expected_score(1800, 1500, best_of=3, config=self.config)
        bo5 = compute_expected_score(1800, 1500, best_of=5, config=self.config)
        self.assertGreater(bo5, bo3, "BO5 should favor the better player more")
    
    def test_expected_bounded(self):
        """Expected score should be between 0.01 and 0.99."""
        expected = compute_expected_score(2500, 1000, config=self.config)
        self.assertGreater(expected, 0.0)
        self.assertLess(expected, 1.0)

# ===== CELL 4: Age Adjustment Tests =====

class TestAgeAdjustment(unittest.TestCase):
    
    def setUp(self):
        self.config = load_elo_config()
    
    def test_young_player_no_drag(self):
        """Players under 32 should have no age drag."""
        elo = age_adjustment(1800, 25, self.config)
        self.assertEqual(elo, 1800)
    
    def test_old_player_gets_drag(self):
        """Players over 32 should have reduced ELO."""
        elo = age_adjustment(1800, 35, self.config)
        self.assertLess(elo, 1800)

# ===== CELL 5: Player State Tests =====

class TestPlayerState(unittest.TestCase):
    
    def test_initial_state(self):
        """New player should have default ratings."""
        p = PlayerEloState(1, initial_rating=1500)
        self.assertEqual(p.elo_overall, 1500)
        self.assertEqual(p.elo_surface['Clay'], 1500)
        self.assertEqual(p.matches_played, 0)
    
    def test_surface_elo_independent(self):
        """Surface ELOs should be independently modifiable."""
        p = PlayerEloState(1)
        p.elo_surface['Clay'] = 1700
        p.elo_surface['Grass'] = 1400
        self.assertNotEqual(p.elo_surface['Clay'], p.elo_surface['Grass'])


# ===== CELL 6: Run Tests =====

if __name__ == "__main__":
    unittest.main(verbosity=2)
