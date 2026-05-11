# ===== TheOracle: Tennis-Specific ELO Engine =====
# The CROWN JEWEL of the feature system.
# Goes far beyond chess ELO with tennis-specific adjustments.
#
# ===== CELL 1: Imports and Configuration =====

import numpy as np
import pandas as pd
import yaml
from collections import defaultdict
from tqdm import tqdm

def load_elo_config(config_path="configs/elo_config.yaml"):
    """Load ELO engine configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# ===== CELL 2: Player ELO State =====

class PlayerEloState:
    """
    Maintains all 7 ELO tracks for a single player.
    
    Tracks:
    1. elo_overall       — All matches
    2. elo_clay           — Clay matches only
    3. elo_grass          — Grass matches only
    4. elo_hard           — Hard court matches only
    5. elo_recent         — Rolling window of last N matches (all surfaces)
    6. elo_recent_surface — Rolling window of last N matches on current surface
    7. elo_weighted       — Blended overall + surface by court speed
    """
    
    def __init__(self, player_id, initial_rating=1500):
        self.player_id = player_id
        self.initial_rating = initial_rating
        
        # Track 1-4: Overall and surface-specific ratings
        self.elo_overall = initial_rating
        self.elo_surface = {
            'Clay': initial_rating,
            'Grass': initial_rating,
            'Hard': initial_rating,
        }
        
        # Match counters
        self.matches_played = 0
        self.surface_matches = {'Clay': 0, 'Grass': 0, 'Hard': 0}
        
        # Last match date (for inactivity calculation)
        self.last_match_date = None
        self.last_surface_match_date = {'Clay': None, 'Grass': None, 'Hard': None}
        
        # Recent results for Track 5-6 (rolling ELO)
        self.recent_results = []          # List of (date, opponent_elo, won, surface)
        self.recent_surface_results = {
            'Clay': [], 'Grass': [], 'Hard': [],
        }
        
        # Age tracking
        self.birth_date = None
    
    def get_surface_elo(self, surface):
        """Get ELO for a specific surface."""
        return self.elo_surface.get(surface, self.initial_rating)
    
    def get_recent_elo(self, window=20):
        """Compute rolling ELO from recent results."""
        if len(self.recent_results) == 0:
            return self.initial_rating
        recent = self.recent_results[-window:]
        rating = self.initial_rating
        for _, opp_elo, won, _ in recent:
            expected = 1 / (1 + 10 ** ((opp_elo - rating) / 400))
            rating += 32 * (won - expected)  # Fixed K=32 for recent window
        return rating
    
    def get_recent_surface_elo(self, surface, window=10):
        """Compute rolling ELO from recent results on specific surface."""
        results = self.recent_surface_results.get(surface, [])
        if len(results) == 0:
            return self.get_surface_elo(surface)
        recent = results[-window:]
        rating = self.initial_rating
        for _, opp_elo, won in recent:
            expected = 1 / (1 + 10 ** ((opp_elo - rating) / 400))
            rating += 32 * (won - expected)
        return rating
    
    def get_weighted_elo(self, surface, alpha=0.6):
        """Blend overall and surface ELO."""
        return alpha * self.get_surface_elo(surface) + (1 - alpha) * self.elo_overall

# ===== CELL 3: Tennis-Specific K-Factor =====

def compute_k_factor(matches_played, tourney_level, days_inactive,
                     sets_played, total_possible_sets, config):
    """
    Compute tennis-specific K-factor.
    
    Enhancements beyond chess:
    1. Experience-based decay (more matches → lower K)
    2. Tournament level weighting (Slams > 250s)
    3. Inactivity boost (long absence → higher K uncertainty)
    4. Margin of victory (straight sets → more informative)
    """
    kf = config['k_factor']
    
    # Base K from experience: K = 250 / ((matches + 5) ^ 0.4)
    base_k = kf['base_k_numerator'] / ((matches_played + kf['base_k_offset']) ** kf['base_k_exponent'])
    
    # Tournament level multiplier
    level_mult = kf['level_multipliers'].get(tourney_level, 1.0)
    
    # Inactivity boost
    inactivity_boost = 1.0
    if days_inactive is not None and days_inactive > kf['inactivity_threshold_days']:
        inactivity_boost = 1.0 + min(
            (days_inactive - kf['inactivity_threshold_days']) / 365,
            kf['inactivity_max_boost']
        )
    
    # Margin of victory factor
    margin_factor = 1.0
    if sets_played is not None and total_possible_sets is not None and total_possible_sets > 0:
        # Straight sets win = sets_played / total = 2/3 or 3/5
        # Factor = 1 + bonus * (1 - ratio)
        ratio = sets_played / total_possible_sets
        margin_factor = 1.0 + kf['margin_of_victory_bonus'] * (1 - ratio)
    
    return base_k * level_mult * inactivity_boost * margin_factor

# ===== CELL 4: H2H-Adjusted Expected Score =====

def compute_expected_score(elo_a, elo_b, h2h_a_wins=0, h2h_b_wins=0,
                           best_of=3, config=None):
    """
    Compute expected win probability with tennis-specific adjustments.
    
    1. Standard ELO expected score
    2. H2H adjustment (up to 15% influence with 20+ meetings)
    3. Best-of-5 advantage factor (favorites win more in BO5)
    """
    # Base ELO expected score
    base_expected = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))
    
    # H2H adjustment
    if config is not None:
        h2h_cfg = config['h2h']
        total_h2h = h2h_a_wins + h2h_b_wins
        if total_h2h >= h2h_cfg['min_matches']:
            h2h_ratio = h2h_a_wins / total_h2h
            weight = min(total_h2h / h2h_cfg['weight_denominator'], h2h_cfg['max_weight'])
            base_expected = (1 - weight) * base_expected + weight * h2h_ratio
    
    # Best-of-5 advantage: favorites win more often in longer formats
    if config is not None and best_of == 5:
        bo5_factor = config.get('bo5_advantage_factor', 1.05)
        if base_expected > 0.5:
            # Push probability further from 0.5 for the favorite
            base_expected = 0.5 + (base_expected - 0.5) * bo5_factor
        elif base_expected < 0.5:
            base_expected = 0.5 - (0.5 - base_expected) * bo5_factor
    
    return np.clip(base_expected, 0.01, 0.99)

# ===== CELL 5: Age Curve Adjustment =====

def age_adjustment(elo, age, config):
    """
    Apply aging curve drag after onset age.
    Tennis players decline physically after ~32.
    """
    if age is None or config is None:
        return elo
    
    aging_cfg = config.get('aging', {})
    onset = aging_cfg.get('onset_age', 32)
    drag = aging_cfg.get('drag_per_year', 5)
    
    if age > onset:
        years_past = age - onset
        return elo - drag * years_past
    return elo

# ===== CELL 6: Main ELO Processor =====

class TennisEloEngine:
    """
    Process all matches chronologically and compute ELO features.
    
    CRITICAL: Features are computed BEFORE the match is processed.
    The match result is used to UPDATE the ratings AFTER feature extraction.
    This prevents data leakage.
    """
    
    def __init__(self, config_path="configs/elo_config.yaml"):
        self.config = load_elo_config(config_path)
        self.players = {}  # player_id → PlayerEloState
        self.h2h = defaultdict(lambda: [0, 0])  # (p1, p2) → [p1_wins, p2_wins]
    
    def get_or_create_player(self, player_id):
        """Get existing player state or create new one."""
        if player_id not in self.players:
            self.players[player_id] = PlayerEloState(
                player_id, self.config['initial_rating']
            )
        return self.players[player_id]
    
    def compute_features_for_match(self, row):
        """
        Compute all ELO-derived features for a match BEFORE processing it.
        
        Returns dict of feature values. All use PRE-MATCH state only.
        """
        p_a = self.get_or_create_player(int(row['player_a_id']))
        p_b = self.get_or_create_player(int(row['player_b_id']))
        surface = row['surface']
        best_of = int(row.get('best_of', 3))
        
        # Age calculation
        age_a = row.get('a_age', None)
        age_b = row.get('b_age', None)
        
        # H2H record
        h2h_key = (min(p_a.player_id, p_b.player_id), max(p_a.player_id, p_b.player_id))
        h2h_record = self.h2h[h2h_key]
        if p_a.player_id == h2h_key[0]:
            h2h_a, h2h_b = h2h_record[0], h2h_record[1]
        else:
            h2h_a, h2h_b = h2h_record[1], h2h_record[0]
        
        # Get pre-match ratings
        elo_overall_a = age_adjustment(p_a.elo_overall, age_a, self.config)
        elo_overall_b = age_adjustment(p_b.elo_overall, age_b, self.config)
        elo_surface_a = p_a.get_surface_elo(surface)
        elo_surface_b = p_b.get_surface_elo(surface)
        elo_recent_a = p_a.get_recent_elo(self.config['recent_overall_window'])
        elo_recent_b = p_b.get_recent_elo(self.config['recent_overall_window'])
        elo_recent_surf_a = p_a.get_recent_surface_elo(surface, self.config['recent_surface_window'])
        elo_recent_surf_b = p_b.get_recent_surface_elo(surface, self.config['recent_surface_window'])
        alpha = self.config['surface_blend_alpha']
        elo_weighted_a = p_a.get_weighted_elo(surface, alpha)
        elo_weighted_b = p_b.get_weighted_elo(surface, alpha)
        
        # Expected score (with H2H and BO5 adjustment)
        expected_a = compute_expected_score(
            elo_surface_a, elo_surface_b,
            h2h_a, h2h_b, best_of, self.config
        )
        
        features = {
            # Raw ratings
            'elo_overall_a': elo_overall_a,
            'elo_overall_b': elo_overall_b,
            'elo_surface_a': elo_surface_a,
            'elo_surface_b': elo_surface_b,
            'elo_recent_a': elo_recent_a,
            'elo_recent_b': elo_recent_b,
            'elo_recent_surf_a': elo_recent_surf_a,
            'elo_recent_surf_b': elo_recent_surf_b,
            'elo_weighted_a': elo_weighted_a,
            'elo_weighted_b': elo_weighted_b,
            
            # Differences (key features for model)
            'elo_diff_overall': elo_overall_a - elo_overall_b,
            'elo_diff_surface': elo_surface_a - elo_surface_b,
            'elo_diff_recent': elo_recent_a - elo_recent_b,
            'elo_diff_recent_surf': elo_recent_surf_a - elo_recent_surf_b,
            'elo_diff_weighted': elo_weighted_a - elo_weighted_b,
            
            # Surface specialization signals
            'elo_surface_delta_a': elo_surface_a - p_a.elo_overall,
            'elo_surface_delta_b': elo_surface_b - p_b.elo_overall,
            'surface_specialist_diff': (elo_surface_a - p_a.elo_overall) - (elo_surface_b - p_b.elo_overall),
            
            # Surface specialist index (high stdev = specialist, low = all-rounder)
            'specialist_index_a': np.std([p_a.elo_surface[s] for s in ['Clay', 'Grass', 'Hard']]),
            'specialist_index_b': np.std([p_b.elo_surface[s] for s in ['Clay', 'Grass', 'Hard']]),
            
            # Experience
            'matches_played_a': p_a.matches_played,
            'matches_played_b': p_b.matches_played,
            'surface_matches_a': p_a.surface_matches.get(surface, 0),
            'surface_matches_b': p_b.surface_matches.get(surface, 0),
            
            # H2H
            'h2h_a': h2h_a,
            'h2h_b': h2h_b,
            'h2h_total': h2h_a + h2h_b,
            
            # Expected win probability
            'elo_expected_a': expected_a,
        }
        
        return features
    
    def update_after_match(self, row, winner_is_a):
        """
        Update all ELO ratings AFTER a match is played.
        Called AFTER features are computed for this match.
        """
        p_a = self.get_or_create_player(int(row['player_a_id']))
        p_b = self.get_or_create_player(int(row['player_b_id']))
        surface = row['surface']
        tourney_level = row.get('tourney_level', 'A')
        tourney_date = int(row['tourney_date'])
        total_sets = row.get('total_sets', None)
        best_of = int(row.get('best_of', 3))
        
        # Compute days since last match for each player
        days_inactive_a = None
        days_inactive_b = None
        if p_a.last_match_date is not None:
            try:
                d1 = pd.to_datetime(str(p_a.last_match_date), format='%Y%m%d')
                d2 = pd.to_datetime(str(tourney_date), format='%Y%m%d')
                days_inactive_a = (d2 - d1).days
            except (ValueError, TypeError):
                pass
        if p_b.last_match_date is not None:
            try:
                d1 = pd.to_datetime(str(p_b.last_match_date), format='%Y%m%d')
                d2 = pd.to_datetime(str(tourney_date), format='%Y%m%d')
                days_inactive_b = (d2 - d1).days
            except (ValueError, TypeError):
                pass
        
        # Compute K-factors
        k_a = compute_k_factor(
            p_a.matches_played, tourney_level, days_inactive_a,
            total_sets, best_of, self.config
        )
        k_b = compute_k_factor(
            p_b.matches_played, tourney_level, days_inactive_b,
            total_sets, best_of, self.config
        )
        
        # Results
        result_a = 1.0 if winner_is_a else 0.0
        result_b = 1.0 - result_a
        
        # ---- Update Track 1: Overall ELO ----
        expected_a = 1.0 / (1.0 + 10.0 ** ((p_b.elo_overall - p_a.elo_overall) / 400.0))
        expected_b = 1.0 - expected_a
        p_a.elo_overall += k_a * (result_a - expected_a)
        p_b.elo_overall += k_b * (result_b - expected_b)
        
        # ---- Update Track 2-4: Surface ELO ----
        exp_surf_a = 1.0 / (1.0 + 10.0 ** ((p_b.elo_surface[surface] - p_a.elo_surface[surface]) / 400.0))
        exp_surf_b = 1.0 - exp_surf_a
        p_a.elo_surface[surface] += k_a * (result_a - exp_surf_a)
        p_b.elo_surface[surface] += k_b * (result_b - exp_surf_b)
        
        # ---- Update Track 5-6: Recent results ----
        opp_elo_for_a = p_b.elo_overall
        opp_elo_for_b = p_a.elo_overall
        p_a.recent_results.append((tourney_date, opp_elo_for_a, result_a, surface))
        p_b.recent_results.append((tourney_date, opp_elo_for_b, result_b, surface))
        p_a.recent_surface_results[surface].append((tourney_date, opp_elo_for_a, result_a))
        p_b.recent_surface_results[surface].append((tourney_date, opp_elo_for_b, result_b))
        
        # Trim recent results to prevent memory bloat
        max_recent = 100
        if len(p_a.recent_results) > max_recent:
            p_a.recent_results = p_a.recent_results[-max_recent:]
        if len(p_b.recent_results) > max_recent:
            p_b.recent_results = p_b.recent_results[-max_recent:]
        for s in ['Clay', 'Grass', 'Hard']:
            if len(p_a.recent_surface_results[s]) > max_recent:
                p_a.recent_surface_results[s] = p_a.recent_surface_results[s][-max_recent:]
            if len(p_b.recent_surface_results[s]) > max_recent:
                p_b.recent_surface_results[s] = p_b.recent_surface_results[s][-max_recent:]
        
        # ---- Update counters ----
        p_a.matches_played += 1
        p_b.matches_played += 1
        p_a.surface_matches[surface] = p_a.surface_matches.get(surface, 0) + 1
        p_b.surface_matches[surface] = p_b.surface_matches.get(surface, 0) + 1
        p_a.last_match_date = tourney_date
        p_b.last_match_date = tourney_date
        p_a.last_surface_match_date[surface] = tourney_date
        p_b.last_surface_match_date[surface] = tourney_date
        
        # ---- Update H2H ----
        h2h_key = (min(p_a.player_id, p_b.player_id), max(p_a.player_id, p_b.player_id))
        if p_a.player_id == h2h_key[0]:
            self.h2h[h2h_key][0 if winner_is_a else 1] += 1
        else:
            self.h2h[h2h_key][1 if winner_is_a else 0] += 1
    
    def process_all_matches(self, matches_df):
        """
        Process all matches chronologically and compute ELO features.
        
        CRITICAL ORDER: For each match:
        1. Compute features (using PRE-MATCH state)
        2. Store features
        3. Update ratings (using match result)
        
        This is the LEAKAGE PREVENTION protocol.
        """
        # Ensure chronological order
        matches_df = matches_df.sort_values('tourney_date').reset_index(drop=True)
        
        print(f"⚡ Processing {len(matches_df):,} matches through ELO engine...")
        
        all_features = []
        
        for idx, row in tqdm(matches_df.iterrows(), total=len(matches_df), desc="ELO Engine"):
            # Step 1: Compute features BEFORE the match
            features = self.compute_features_for_match(row)
            features['match_id'] = row.get('match_id', idx)
            all_features.append(features)
            
            # Step 2: Update ratings AFTER the match
            winner_is_a = bool(row['winner_is_a'])
            self.update_after_match(row, winner_is_a)
        
        elo_features_df = pd.DataFrame(all_features)
        print(f"✅ ELO features computed: {elo_features_df.shape[1]} features per match")
        
        return elo_features_df
    
    def get_player_ratings(self, top_n=50):
        """Get current top players by overall ELO."""
        ratings = []
        for pid, state in self.players.items():
            if state.matches_played >= 20:
                ratings.append({
                    'player_id': pid,
                    'elo_overall': state.elo_overall,
                    'elo_clay': state.elo_surface['Clay'],
                    'elo_grass': state.elo_surface['Grass'],
                    'elo_hard': state.elo_surface['Hard'],
                    'matches': state.matches_played,
                })
        df = pd.DataFrame(ratings).sort_values('elo_overall', ascending=False)
        return df.head(top_n)
