# ===== TheOracle: Tennis-Specific ELO Engine =====
# The CROWN JEWEL of the feature system.
# Bulletproof version — defensive coding throughout.
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
    Maintains all ELO tracks for a single player.

    Tracks:
    1. elo_overall       — All matches
    2. elo_surface       — Per-surface (Clay / Grass / Hard)
    3. recent_results    — Rolling window for trend detection
    """

    def __init__(self, player_id, initial_rating=1500):
        self.player_id = player_id
        self.initial_rating = initial_rating

        self.elo_overall = float(initial_rating)
        self.elo_surface = {
            'Clay': float(initial_rating),
            'Grass': float(initial_rating),
            'Hard': float(initial_rating),
        }

        self.matches_played = 0
        self.surface_matches = {'Clay': 0, 'Grass': 0, 'Hard': 0}

        self.last_match_date = None
        self.recent_results = []          # (date, opp_elo, won, surface)
        self.recent_surface_results = {'Clay': [], 'Grass': [], 'Hard': []}

    def get_surface_elo(self, surface):
        return self.elo_surface.get(surface, self.initial_rating)

    def get_recent_elo(self, window=20):
        if not self.recent_results:
            return self.initial_rating
        rating = float(self.initial_rating)
        for _, opp_elo, won, _ in self.recent_results[-window:]:
            try:
                exp = 1.0 / (1.0 + 10.0 ** ((float(opp_elo) - rating) / 400.0))
                rating += 32.0 * (float(won) - exp)
            except (TypeError, ValueError, ZeroDivisionError, FloatingPointError):
                continue
        return rating if np.isfinite(rating) else self.initial_rating

    def get_recent_surface_elo(self, surface, window=10):
        results = self.recent_surface_results.get(surface, [])
        if not results:
            return self.get_surface_elo(surface)
        rating = float(self.initial_rating)
        for _, opp_elo, won in results[-window:]:
            try:
                exp = 1.0 / (1.0 + 10.0 ** ((float(opp_elo) - rating) / 400.0))
                rating += 32.0 * (float(won) - exp)
            except (TypeError, ValueError, ZeroDivisionError, FloatingPointError):
                continue
        return rating if np.isfinite(rating) else self.get_surface_elo(surface)

    def get_weighted_elo(self, surface, alpha=0.6):
        return alpha * self.get_surface_elo(surface) + (1.0 - alpha) * self.elo_overall


# ===== CELL 3: Tennis-Specific K-Factor =====

def compute_k_factor(matches_played, tourney_level, days_inactive,
                     sets_played, total_possible_sets, config):
    """
    Compute tennis-specific K-factor.
    Fully defensive — never returns NaN or infinity.
    """
    try:
        kf = config['k_factor']

        # Base K: experience decay
        offset = max(kf.get('base_k_offset', 5), 1)
        exponent = kf.get('base_k_exponent', 0.4)
        numerator = kf.get('base_k_numerator', 250)
        base_k = numerator / ((max(matches_played, 0) + offset) ** exponent)

        # Tournament level multiplier
        level_mult = kf.get('level_multipliers', {}).get(str(tourney_level), 1.0)
        if not np.isfinite(level_mult):
            level_mult = 1.0

        # Inactivity boost
        inactivity_boost = 1.0
        if days_inactive is not None and np.isfinite(days_inactive):
            threshold = kf.get('inactivity_threshold_days', 60)
            max_boost = kf.get('inactivity_max_boost', 0.5)
            if days_inactive > threshold:
                inactivity_boost = 1.0 + min((days_inactive - threshold) / 365.0, max_boost)

        # Margin of victory — SAFE division
        margin_factor = 1.0
        if (sets_played is not None and total_possible_sets is not None and
                np.isfinite(float(sets_played)) and float(total_possible_sets) > 0):
            ratio = min(float(sets_played) / float(total_possible_sets), 1.0)
            margin_factor = 1.0 + kf.get('margin_of_victory_bonus', 0.15) * (1.0 - ratio)

        result = base_k * level_mult * inactivity_boost * margin_factor
        return result if np.isfinite(result) and result > 0 else 32.0

    except Exception:
        return 32.0  # Safe fallback K-factor


# ===== CELL 4: H2H-Adjusted Expected Score =====

def compute_expected_score(elo_a, elo_b, h2h_a_wins=0, h2h_b_wins=0,
                           best_of=3, config=None):
    """
    Expected win probability — fully defensive.
    Never returns NaN or values outside [0.01, 0.99].
    """
    try:
        # Guard inputs
        elo_a = float(elo_a) if np.isfinite(float(elo_a)) else 1500.0
        elo_b = float(elo_b) if np.isfinite(float(elo_b)) else 1500.0

        base_expected = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))

        # H2H adjustment
        if config is not None:
            h2h_cfg = config.get('h2h', {})
            total_h2h = int(h2h_a_wins) + int(h2h_b_wins)
            min_matches = h2h_cfg.get('min_matches', 5)
            if total_h2h >= min_matches:
                h2h_ratio = h2h_a_wins / total_h2h
                weight = min(total_h2h / max(h2h_cfg.get('weight_denominator', 20), 1),
                             h2h_cfg.get('max_weight', 0.15))
                base_expected = (1.0 - weight) * base_expected + weight * h2h_ratio

        # Best-of-5 squeeze
        if config is not None and int(best_of) == 5:
            bo5_factor = config.get('bo5_advantage_factor', 1.05)
            if base_expected > 0.5:
                base_expected = 0.5 + (base_expected - 0.5) * bo5_factor
            else:
                base_expected = 0.5 - (0.5 - base_expected) * bo5_factor

        return float(np.clip(base_expected, 0.01, 0.99))

    except Exception:
        return 0.5  # Neutral fallback


# ===== CELL 5: Age Curve Adjustment =====

def age_adjustment(elo, age, config):
    """Apply aging curve drag after onset age."""
    try:
        if age is None or not np.isfinite(float(age)):
            return elo
        aging_cfg = config.get('aging', {})
        onset = aging_cfg.get('onset_age', 32)
        drag = aging_cfg.get('drag_per_year', 5)
        if float(age) > onset:
            result = float(elo) - drag * (float(age) - onset)
            return result if np.isfinite(result) else elo
    except Exception:
        pass
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
        self.players = {}
        self.h2h = defaultdict(lambda: [0, 0])

    def get_or_create_player(self, player_id):
        pid = int(player_id)
        if pid not in self.players:
            self.players[pid] = PlayerEloState(pid, self.config['initial_rating'])
        return self.players[pid]

    def compute_features_for_match(self, row):
        """
        Compute all ELO-derived features BEFORE the match.
        Returns a dict of floats — never NaN-poisoned.
        """
        try:
            p_a = self.get_or_create_player(int(row['player_a_id']))
            p_b = self.get_or_create_player(int(row['player_b_id']))
            surface = str(row.get('surface', 'Hard'))
            best_of = int(row.get('best_of', 3) or 3)
            if best_of not in (3, 5):
                best_of = 3

            age_a = row.get('a_age', None)
            age_b = row.get('b_age', None)

            # H2H record
            h2h_key = (min(p_a.player_id, p_b.player_id), max(p_a.player_id, p_b.player_id))
            h2h_rec = self.h2h[h2h_key]
            if p_a.player_id == h2h_key[0]:
                h2h_a, h2h_b = h2h_rec[0], h2h_rec[1]
            else:
                h2h_a, h2h_b = h2h_rec[1], h2h_rec[0]

            # Pre-match ratings
            elo_overall_a = age_adjustment(p_a.elo_overall, age_a, self.config)
            elo_overall_b = age_adjustment(p_b.elo_overall, age_b, self.config)
            elo_surface_a = p_a.get_surface_elo(surface)
            elo_surface_b = p_b.get_surface_elo(surface)
            elo_recent_a = p_a.get_recent_elo(self.config.get('recent_overall_window', 20))
            elo_recent_b = p_b.get_recent_elo(self.config.get('recent_overall_window', 20))
            elo_recent_surf_a = p_a.get_recent_surface_elo(surface, self.config.get('recent_surface_window', 10))
            elo_recent_surf_b = p_b.get_recent_surface_elo(surface, self.config.get('recent_surface_window', 10))
            alpha = self.config.get('surface_blend_alpha', 0.6)
            elo_weighted_a = p_a.get_weighted_elo(surface, alpha)
            elo_weighted_b = p_b.get_weighted_elo(surface, alpha)

            expected_a = compute_expected_score(
                elo_surface_a, elo_surface_b,
                h2h_a, h2h_b, best_of, self.config
            )

            features = {
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
                'elo_diff_overall': elo_overall_a - elo_overall_b,
                'elo_diff_surface': elo_surface_a - elo_surface_b,
                'elo_diff_recent': elo_recent_a - elo_recent_b,
                'elo_diff_recent_surf': elo_recent_surf_a - elo_recent_surf_b,
                'elo_diff_weighted': elo_weighted_a - elo_weighted_b,
                'elo_surface_delta_a': elo_surface_a - p_a.elo_overall,
                'elo_surface_delta_b': elo_surface_b - p_b.elo_overall,
                'surface_specialist_diff': (elo_surface_a - p_a.elo_overall) - (elo_surface_b - p_b.elo_overall),
                'specialist_index_a': float(np.std([p_a.elo_surface[s] for s in ['Clay', 'Grass', 'Hard']])),
                'specialist_index_b': float(np.std([p_b.elo_surface[s] for s in ['Clay', 'Grass', 'Hard']])),
                'matches_played_a': p_a.matches_played,
                'matches_played_b': p_b.matches_played,
                'surface_matches_a': p_a.surface_matches.get(surface, 0),
                'surface_matches_b': p_b.surface_matches.get(surface, 0),
                'h2h_a': h2h_a,
                'h2h_b': h2h_b,
                'h2h_total': h2h_a + h2h_b,
                'elo_expected_a': expected_a,
            }

            return features

        except Exception as e:
            # Return safe defaults rather than crashing the whole pipeline
            return {
                'elo_overall_a': 1500.0, 'elo_overall_b': 1500.0,
                'elo_surface_a': 1500.0, 'elo_surface_b': 1500.0,
                'elo_recent_a': 1500.0, 'elo_recent_b': 1500.0,
                'elo_recent_surf_a': 1500.0, 'elo_recent_surf_b': 1500.0,
                'elo_weighted_a': 1500.0, 'elo_weighted_b': 1500.0,
                'elo_diff_overall': 0.0, 'elo_diff_surface': 0.0,
                'elo_diff_recent': 0.0, 'elo_diff_recent_surf': 0.0,
                'elo_diff_weighted': 0.0,
                'elo_surface_delta_a': 0.0, 'elo_surface_delta_b': 0.0,
                'surface_specialist_diff': 0.0,
                'specialist_index_a': 0.0, 'specialist_index_b': 0.0,
                'matches_played_a': 0, 'matches_played_b': 0,
                'surface_matches_a': 0, 'surface_matches_b': 0,
                'h2h_a': 0, 'h2h_b': 0, 'h2h_total': 0,
                'elo_expected_a': 0.5,
            }

    def update_after_match(self, row, winner_is_a):
        """
        Update all ELO ratings AFTER a match is played.
        Fully defensive — a single bad row cannot corrupt the engine.
        """
        try:
            p_a = self.get_or_create_player(int(row['player_a_id']))
            p_b = self.get_or_create_player(int(row['player_b_id']))
            surface = str(row.get('surface', 'Hard'))
            tourney_level = str(row.get('tourney_level', 'A'))
            tourney_date = int(row['tourney_date'])
            total_sets = row.get('total_sets', None)
            best_of = int(row.get('best_of', 3) or 3)
            if best_of not in (3, 5):
                best_of = 3

            # Days inactive
            days_a, days_b = None, None
            try:
                if p_a.last_match_date is not None:
                    d1 = pd.to_datetime(str(p_a.last_match_date), format='%Y%m%d')
                    d2 = pd.to_datetime(str(tourney_date), format='%Y%m%d')
                    days_a = max((d2 - d1).days, 0)
            except Exception:
                pass
            try:
                if p_b.last_match_date is not None:
                    d1 = pd.to_datetime(str(p_b.last_match_date), format='%Y%m%d')
                    d2 = pd.to_datetime(str(tourney_date), format='%Y%m%d')
                    days_b = max((d2 - d1).days, 0)
            except Exception:
                pass

            k_a = compute_k_factor(p_a.matches_played, tourney_level, days_a,
                                   total_sets, best_of, self.config)
            k_b = compute_k_factor(p_b.matches_played, tourney_level, days_b,
                                   total_sets, best_of, self.config)

            res_a = 1.0 if winner_is_a else 0.0
            res_b = 1.0 - res_a

            # Update overall ELO
            exp_overall_a = 1.0 / (1.0 + 10.0 ** ((p_b.elo_overall - p_a.elo_overall) / 400.0))
            new_a = p_a.elo_overall + k_a * (res_a - exp_overall_a)
            new_b = p_b.elo_overall + k_b * (res_b - (1.0 - exp_overall_a))
            if np.isfinite(new_a):
                p_a.elo_overall = new_a
            if np.isfinite(new_b):
                p_b.elo_overall = new_b

            # Update surface ELO
            exp_surf_a = 1.0 / (1.0 + 10.0 ** ((p_b.elo_surface[surface] - p_a.elo_surface[surface]) / 400.0))
            new_sa = p_a.elo_surface[surface] + k_a * (res_a - exp_surf_a)
            new_sb = p_b.elo_surface[surface] + k_b * (res_b - (1.0 - exp_surf_a))
            if np.isfinite(new_sa):
                p_a.elo_surface[surface] = new_sa
            if np.isfinite(new_sb):
                p_b.elo_surface[surface] = new_sb

            # Recent results
            opp_elo_for_a = p_b.elo_overall
            opp_elo_for_b = p_a.elo_overall
            p_a.recent_results.append((tourney_date, opp_elo_for_a, res_a, surface))
            p_b.recent_results.append((tourney_date, opp_elo_for_b, res_b, surface))
            p_a.recent_surface_results[surface].append((tourney_date, opp_elo_for_a, res_a))
            p_b.recent_surface_results[surface].append((tourney_date, opp_elo_for_b, res_b))

            # Trim memory
            max_recent = 100
            p_a.recent_results = p_a.recent_results[-max_recent:]
            p_b.recent_results = p_b.recent_results[-max_recent:]
            for s in ['Clay', 'Grass', 'Hard']:
                p_a.recent_surface_results[s] = p_a.recent_surface_results[s][-max_recent:]
                p_b.recent_surface_results[s] = p_b.recent_surface_results[s][-max_recent:]

            # Update counters
            p_a.matches_played += 1
            p_b.matches_played += 1
            p_a.surface_matches[surface] = p_a.surface_matches.get(surface, 0) + 1
            p_b.surface_matches[surface] = p_b.surface_matches.get(surface, 0) + 1
            p_a.last_match_date = tourney_date
            p_b.last_match_date = tourney_date

            # H2H
            h2h_key = (min(p_a.player_id, p_b.player_id), max(p_a.player_id, p_b.player_id))
            if p_a.player_id == h2h_key[0]:
                self.h2h[h2h_key][0 if winner_is_a else 1] += 1
            else:
                self.h2h[h2h_key][1 if winner_is_a else 0] += 1

        except Exception:
            pass  # Never let a single bad row crash the engine

    def process_all_matches(self, matches_df):
        """
        Process all matches chronologically and compute ELO features.
        LEAKAGE-FREE: compute → store → update.
        """
        matches_df = matches_df.sort_values('tourney_date').reset_index(drop=True)
        print(f"⚡ Processing {len(matches_df):,} matches through ELO engine...")

        all_features = []
        for idx, row in tqdm(matches_df.iterrows(), total=len(matches_df), desc="ELO Engine"):
            features = self.compute_features_for_match(row)
            features['match_id'] = row.get('match_id', idx)
            all_features.append(features)
            self.update_after_match(row, bool(row['winner_is_a']))

        elo_features_df = pd.DataFrame(all_features)
        print(f"✅ ELO features computed: {elo_features_df.shape[1]} features per match")
        return elo_features_df

    def get_player_ratings(self, clean_dir="data/clean", top_n=20):
        """Get current top players sorted by overall ELO, with player names."""
        data = []

        # Try to load player names
        names = {}
        try:
            players_df = pd.read_parquet(f"{clean_dir}/players.parquet")
            if 'name_last' in players_df.columns and 'name_first' in players_df.columns:
                names = dict(zip(
                    players_df['player_id'].astype(int),
                    players_df['name_last'] + ", " + players_df['name_first']
                ))
        except Exception:
            pass

        for pid, state in self.players.items():
            if state.matches_played < 20:
                continue
            elo = state.elo_overall
            if not np.isfinite(elo):
                continue
            data.append({
                'name': names.get(int(pid), f"ID:{pid}"),
                'elo_overall': round(elo, 1),
                'elo_clay': round(state.elo_surface.get('Clay', 1500), 1),
                'elo_grass': round(state.elo_surface.get('Grass', 1500), 1),
                'elo_hard': round(state.elo_surface.get('Hard', 1500), 1),
                'matches': state.matches_played,
            })

        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        return df.sort_values('elo_overall', ascending=False).head(top_n).reset_index(drop=True)


# ===== CELL 7: Entrypoint =====

if __name__ == "__main__":
    engine = TennisEloEngine()
    print("ELO engine initialized successfully.")
