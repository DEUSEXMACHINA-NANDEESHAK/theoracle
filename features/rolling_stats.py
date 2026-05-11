# ===== TheOracle: Rolling Stats Engine =====
# ===== CELL 1: Imports =====

import numpy as np
import pandas as pd
from collections import defaultdict

# ===== CELL 2: Rolling Statistics Computer =====

class RollingStatsEngine:
    """
    Computes windowed aggregate statistics for each player.
    
    Provides additional statistical features beyond what the
    specialized engines compute. Focuses on raw stat averages
    over recent windows.
    """
    
    def __init__(self):
        # player_id → list of (date, stat_dict)
        self.history = defaultdict(list)
    
    def compute_features_for_match(self, row):
        """Compute rolling stat features BEFORE match."""
        features = {}
        
        for side in ['a', 'b']:
            pid = int(row[f'player_{side}_id'])
            hist = self.history[pid]
            
            if len(hist) < 3:
                features[f'avg_aces_10_{side}'] = np.nan
                features[f'avg_df_10_{side}'] = np.nan
                features[f'avg_svpt_10_{side}'] = np.nan
                features[f'avg_1st_pct_10_{side}'] = np.nan
                features[f'avg_minutes_10_{side}'] = np.nan
                features[f'win_rate_20_{side}'] = 0.5
                features[f'surface_frac_{side}'] = 0.0
                continue
            
            last_10 = hist[-10:] if len(hist) >= 10 else hist
            last_20 = hist[-20:] if len(hist) >= 20 else hist
            last_50 = hist[-50:] if len(hist) >= 50 else hist
            
            # Average stats over last 10 matches
            features[f'avg_aces_10_{side}'] = np.nanmean([h[1].get('aces', np.nan) for h in last_10])
            features[f'avg_df_10_{side}'] = np.nanmean([h[1].get('dfs', np.nan) for h in last_10])
            features[f'avg_svpt_10_{side}'] = np.nanmean([h[1].get('svpt', np.nan) for h in last_10])
            features[f'avg_1st_pct_10_{side}'] = np.nanmean([h[1].get('first_pct', np.nan) for h in last_10])
            features[f'avg_minutes_10_{side}'] = np.nanmean([h[1].get('minutes', np.nan) for h in last_10])
            
            # Win rate over last 20
            features[f'win_rate_20_{side}'] = np.mean([h[1].get('won', 0) for h in last_20])
            
            # Fraction of last 50 matches on this surface
            surface = row['surface']
            surface_count = sum(1 for h in last_50 if h[1].get('surface') == surface)
            features[f'surface_frac_{side}'] = surface_count / len(last_50)
        
        return features
    
    def update_after_match(self, row, winner_is_a):
        """Store match stats for rolling computation."""
        for side, won in [('a', winner_is_a), ('b', not winner_is_a)]:
            pid = int(row[f'player_{side}_id'])
            date = int(row['tourney_date'])
            
            stats = {
                'won': int(won),
                'surface': row['surface'],
                'aces': pd.to_numeric(row.get(f'{side}_ace', np.nan), errors='coerce'),
                'dfs': pd.to_numeric(row.get(f'{side}_df', np.nan), errors='coerce'),
                'svpt': pd.to_numeric(row.get(f'{side}_svpt', np.nan), errors='coerce'),
                'minutes': pd.to_numeric(row.get('minutes', np.nan), errors='coerce'),
            }
            
            svpt = stats['svpt']
            first_in = pd.to_numeric(row.get(f'{side}_1stIn', np.nan), errors='coerce')
            stats['first_pct'] = first_in / svpt if not pd.isna(svpt) and svpt > 0 and not pd.isna(first_in) else np.nan
            
            self.history[pid].append((date, stats))
            if len(self.history[pid]) > 200:
                self.history[pid] = self.history[pid][-200:]
