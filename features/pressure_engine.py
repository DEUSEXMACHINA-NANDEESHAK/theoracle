# ===== TheOracle: Pressure & Clutch Engine =====
# ===== CELL 1: Imports =====

import numpy as np
import pandas as pd
from collections import defaultdict

# ===== CELL 2: Pressure Feature Computer =====

class PressureEngine:
    """
    Computes mental toughness / clutch performance features.
    
    Features:
    - bp_saved_rate: Career break points saved %
    - bp_saved_rate_surface: Surface-specific BP saved %
    - bp_conversion_rate: Break point conversion %
    - tiebreak_win_rate: Overall and surface-specific
    - deciding_set_win_rate: Win % in 3rd/5th sets
    - comeback_rate: Win after losing set 1
    - pressure_index: Composite clutch score
    """
    
    def __init__(self):
        # player_id → cumulative stats
        self.stats = defaultdict(lambda: {
            'bp_faced': 0, 'bp_saved': 0,
            'bp_chances': 0, 'bp_converted': 0,
            'bp_faced_surface': defaultdict(int),
            'bp_saved_surface': defaultdict(int),
            'tiebreaks_played': 0, 'tiebreaks_won': 0,
            'tb_surface': defaultdict(lambda: [0, 0]),  # [played, won]
            'deciding_sets': 0, 'deciding_sets_won': 0,
            'matches_lost_set1': 0, 'comebacks': 0,
        })
    
    def compute_features_for_match(self, row):
        """Compute pressure features BEFORE match using only past data."""
        features = {}
        
        for side in ['a', 'b']:
            pid = int(row[f'player_{side}_id'])
            s = self.stats[pid]
            surface = row['surface']
            
            # Break point saved rate
            bp_saved_rate = s['bp_saved'] / max(s['bp_faced'], 1)
            bp_saved_surface = s['bp_saved_surface'][surface] / max(s['bp_faced_surface'][surface], 1)
            
            # Break point conversion rate
            bp_conv_rate = s['bp_converted'] / max(s['bp_chances'], 1)
            
            # Tiebreak win rate
            tb_win_rate = s['tiebreaks_won'] / max(s['tiebreaks_played'], 1)
            tb_data = s['tb_surface'][surface]
            tb_win_surface = tb_data[1] / max(tb_data[0], 1)
            
            # Deciding set win rate
            dec_win_rate = s['deciding_sets_won'] / max(s['deciding_sets'], 1)
            
            # Comeback rate
            comeback_rate = s['comebacks'] / max(s['matches_lost_set1'], 1)
            
            # Composite pressure index (weighted average of clutch stats)
            pressure_idx = (
                0.30 * bp_saved_rate +
                0.20 * bp_conv_rate +
                0.20 * tb_win_rate +
                0.15 * dec_win_rate +
                0.15 * comeback_rate
            )
            
            features[f'bp_saved_rate_{side}'] = bp_saved_rate
            features[f'bp_saved_surface_{side}'] = bp_saved_surface
            features[f'bp_conv_rate_{side}'] = bp_conv_rate
            features[f'tb_win_rate_{side}'] = tb_win_rate
            features[f'tb_win_surface_{side}'] = tb_win_surface
            features[f'deciding_set_rate_{side}'] = dec_win_rate
            features[f'comeback_rate_{side}'] = comeback_rate
            features[f'pressure_index_{side}'] = pressure_idx
            features[f'bp_data_points_{side}'] = s['bp_faced']
        
        # Differences
        features['pressure_diff'] = features['pressure_index_a'] - features['pressure_index_b']
        features['bp_saved_diff'] = features['bp_saved_rate_a'] - features['bp_saved_rate_b']
        features['tb_win_diff'] = features['tb_win_rate_a'] - features['tb_win_rate_b']
        
        return features
    
    def update_after_match(self, row, winner_is_a):
        """Update pressure stats after a match."""
        surface = row['surface']
        
        for side, is_winner in [('a', winner_is_a), ('b', not winner_is_a)]:
            pid = int(row[f'player_{side}_id'])
            opp_side = 'b' if side == 'a' else 'a'
            s = self.stats[pid]
            
            # Break points (from match stats if available)
            bp_faced = row.get(f'{side}_bpFaced', None)
            bp_saved = row.get(f'{side}_bpSaved', None)
            opp_bp_faced = row.get(f'{opp_side}_bpFaced', None)
            opp_bp_saved = row.get(f'{opp_side}_bpSaved', None)
            
            if bp_faced is not None and not pd.isna(bp_faced):
                bp_faced = int(bp_faced)
                bp_saved_count = int(bp_saved) if bp_saved is not None and not pd.isna(bp_saved) else 0
                s['bp_faced'] += bp_faced
                s['bp_saved'] += bp_saved_count
                s['bp_faced_surface'][surface] += bp_faced
                s['bp_saved_surface'][surface] += bp_saved_count
            
            # Break point conversion = opponent's bp_faced - bp_saved
            if opp_bp_faced is not None and not pd.isna(opp_bp_faced):
                opp_bp_faced = int(opp_bp_faced)
                opp_bp_saved_count = int(opp_bp_saved) if opp_bp_saved is not None and not pd.isna(opp_bp_saved) else 0
                bp_converted = opp_bp_faced - opp_bp_saved_count
                s['bp_chances'] += opp_bp_faced
                s['bp_converted'] += bp_converted
            
            # Tiebreaks (parse from score)
            score = str(row.get('score', ''))
            tb_count = score.count('(')  # Each tiebreak has parentheses
            if tb_count > 0:
                s['tiebreaks_played'] += tb_count
                s['tb_surface'][surface][0] += tb_count
                # Rough estimate: count tiebreaks won by looking at set scores
                # More accurate with point-level data
                sets_won = row.get(f'sets_{side}', 0)
                tb_won_est = min(tb_count, sets_won if sets_won else 0)
                s['tiebreaks_won'] += tb_won_est
                s['tb_surface'][surface][1] += tb_won_est
            
            # Deciding set
            total_sets = row.get('total_sets', None)
            best_of = int(row.get('best_of', 3))
            if total_sets is not None and total_sets == best_of:
                s['deciding_sets'] += 1
                if is_winner:
                    s['deciding_sets_won'] += 1
            
            # Comeback (lost first set but won match)
            sets_a = row.get('sets_a', None)
            sets_b = row.get('sets_b', None)
            if sets_a is not None and sets_b is not None:
                # Determine if this player lost the first set
                # Heuristic: if winner won 3-1 or 3-2 (BO5) or 2-1 (BO3), there was a lost set
                winner_sets = max(sets_a, sets_b) if (sets_a is not None) else None
                loser_sets = min(sets_a, sets_b) if (sets_b is not None) else None
                if loser_sets and loser_sets > 0 and is_winner:
                    s['matches_lost_set1'] += 1
                    s['comebacks'] += 1
                elif loser_sets and loser_sets > 0 and not is_winner:
                    s['matches_lost_set1'] += 1
