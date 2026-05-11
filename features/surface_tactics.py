# ===== TheOracle: Surface-Specific Tactics Engine =====
# ===== CELL 1: Imports =====

import numpy as np
import pandas as pd
from collections import defaultdict

# ===== CELL 2: Surface Tactics Computer =====

class SurfaceTacticsEngine:
    """
    Computes surface-specific playing style profiles.
    
    Clay: Return game, second serve, grinding ability matter most
    Grass: Serve dominance, aces, first-strike tennis
    Hard: Balanced — movement, versatility, forehand power
    
    All stats are rolling averages from the player's history on each surface.
    """
    
    def __init__(self, min_matches=5):
        self.min_matches = min_matches
        # player_id → surface → list of stat dicts
        self.stats_history = defaultdict(lambda: defaultdict(list))
    
    def compute_features_for_match(self, row):
        """Compute surface-specific tactical features BEFORE match."""
        features = {}
        surface = row['surface']
        
        for side in ['a', 'b']:
            pid = int(row[f'player_{side}_id'])
            surf_stats = self.stats_history[pid][surface]
            all_stats = []
            for s in ['Clay', 'Grass', 'Hard']:
                all_stats.extend(self.stats_history[pid][s])
            
            # Compute averages from history on this surface
            if len(surf_stats) >= self.min_matches:
                stats = self._average_stats(surf_stats[-30:])  # Last 30 on surface
            elif len(all_stats) >= self.min_matches:
                stats = self._average_stats(all_stats[-30:])   # Fallback to all surfaces
            else:
                stats = self._default_stats()
            
            # Also compute overall stats for comparison
            overall_stats = self._average_stats(all_stats[-50:]) if len(all_stats) >= self.min_matches else self._default_stats()
            
            features[f'ace_rate_{side}'] = stats['ace_rate']
            features[f'df_rate_{side}'] = stats['df_rate']
            features[f'first_serve_pct_{side}'] = stats['first_serve_pct']
            features[f'first_serve_won_{side}'] = stats['first_serve_won']
            features[f'second_serve_won_{side}'] = stats['second_serve_won']
            features[f'return_pts_won_{side}'] = stats['return_pts_won']
            features[f'bp_per_game_{side}'] = stats['bp_per_game']
            
            # Composite indices
            features[f'serve_dominance_{side}'] = (
                stats['first_serve_won'] * stats['first_serve_pct'] +
                stats['second_serve_won'] * (1 - stats['first_serve_pct'])
            )
            
            # Surface-specific composites
            if surface == 'Clay':
                # Clay grind = return ability + second serve resilience
                features[f'clay_grind_{side}'] = (
                    0.5 * stats['return_pts_won'] +
                    0.3 * stats['second_serve_won'] +
                    0.2 * (1 - stats['df_rate'])
                )
            elif surface == 'Grass':
                # Grass serve = ace rate + first serve won
                features[f'grass_serve_{side}'] = (
                    0.4 * stats['ace_rate'] +
                    0.4 * stats['first_serve_won'] +
                    0.2 * stats['first_serve_pct']
                )
            else:  # Hard
                # Hard versatility = balance of serve and return
                features[f'hard_versatility_{side}'] = (
                    0.5 * features[f'serve_dominance_{side}'] +
                    0.5 * stats['return_pts_won']
                )
            
            # Surface adaptation: how different are stats on this surface vs overall?
            features[f'surface_serve_adapt_{side}'] = stats['first_serve_won'] - overall_stats['first_serve_won']
            features[f'surface_return_adapt_{side}'] = stats['return_pts_won'] - overall_stats['return_pts_won']
            features[f'surface_stat_count_{side}'] = len(surf_stats)
        
        # Differences
        features['serve_dominance_diff'] = features.get('serve_dominance_a', 0.5) - features.get('serve_dominance_b', 0.5)
        features['return_pts_diff'] = features.get('return_pts_won_a', 0.4) - features.get('return_pts_won_b', 0.4)
        features['ace_rate_diff'] = features.get('ace_rate_a', 0) - features.get('ace_rate_b', 0)
        
        return features
    
    def _average_stats(self, stat_list):
        """Average a list of stat dicts."""
        if not stat_list:
            return self._default_stats()
        keys = stat_list[0].keys()
        return {k: np.mean([s[k] for s in stat_list if k in s]) for k in keys}
    
    def _default_stats(self):
        """Default stats for players with insufficient data."""
        return {
            'ace_rate': 0.07,
            'df_rate': 0.03,
            'first_serve_pct': 0.60,
            'first_serve_won': 0.70,
            'second_serve_won': 0.50,
            'return_pts_won': 0.38,
            'bp_per_game': 0.5,
        }
    
    def update_after_match(self, row):
        """Extract and store match stats after match."""
        surface = row['surface']
        
        for side in ['a', 'b']:
            pid = int(row[f'player_{side}_id'])
            
            # Extract raw stats from match data
            svpt = pd.to_numeric(row.get(f'{side}_svpt', None), errors='coerce')
            first_in = pd.to_numeric(row.get(f'{side}_1stIn', None), errors='coerce')
            first_won = pd.to_numeric(row.get(f'{side}_1stWon', None), errors='coerce')
            second_won = pd.to_numeric(row.get(f'{side}_2ndWon', None), errors='coerce')
            aces = pd.to_numeric(row.get(f'{side}_ace', None), errors='coerce')
            dfs = pd.to_numeric(row.get(f'{side}_df', None), errors='coerce')
            bp_faced = pd.to_numeric(row.get(f'{side}_bpFaced', None), errors='coerce')
            
            if pd.isna(svpt) or svpt == 0:
                continue  # No stats available for this match
            
            second_in = svpt - first_in if not pd.isna(first_in) else 0
            
            stat_dict = {
                'ace_rate': aces / svpt if not pd.isna(aces) else 0.07,
                'df_rate': dfs / svpt if not pd.isna(dfs) else 0.03,
                'first_serve_pct': first_in / svpt if not pd.isna(first_in) else 0.60,
                'first_serve_won': first_won / max(first_in, 1) if not pd.isna(first_won) and not pd.isna(first_in) else 0.70,
                'second_serve_won': second_won / max(second_in, 1) if not pd.isna(second_won) and second_in > 0 else 0.50,
                'return_pts_won': 0.38,  # Will compute from opponent's serve stats
                'bp_per_game': bp_faced / max(svpt / 4, 1) if not pd.isna(bp_faced) else 0.5,
            }
            
            # Compute return points won from opponent's serve stats
            opp = 'b' if side == 'a' else 'a'
            opp_svpt = pd.to_numeric(row.get(f'{opp}_svpt', None), errors='coerce')
            opp_1st_won = pd.to_numeric(row.get(f'{opp}_1stWon', None), errors='coerce')
            opp_2nd_won = pd.to_numeric(row.get(f'{opp}_2ndWon', None), errors='coerce')
            
            if not pd.isna(opp_svpt) and opp_svpt > 0:
                opp_pts_won = (opp_1st_won if not pd.isna(opp_1st_won) else 0) + \
                              (opp_2nd_won if not pd.isna(opp_2nd_won) else 0)
                return_pts_won = 1 - (opp_pts_won / opp_svpt)
                stat_dict['return_pts_won'] = np.clip(return_pts_won, 0, 1)
            
            self.stats_history[pid][surface].append(stat_dict)
            if len(self.stats_history[pid][surface]) > 100:
                self.stats_history[pid][surface] = self.stats_history[pid][surface][-100:]
