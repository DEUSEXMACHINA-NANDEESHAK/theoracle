# ===== TheOracle: Momentum & Confidence Engine =====
# ===== CELL 1: Imports =====

import numpy as np
import pandas as pd
from collections import defaultdict

# ===== CELL 2: Momentum Feature Computer =====

class MomentumEngine:
    """
    Tracks player form, streaks, and confidence trajectory.
    
    Uses EWMA (Exponentially Weighted Moving Average) to create a
    smooth confidence signal that decays old results naturally.
    """
    
    def __init__(self, ewma_span=10):
        self.ewma_span = ewma_span
        # player_id → list of (date, won, surface, opponent_elo)
        self.results = defaultdict(list)
    
    def compute_features_for_match(self, row):
        """Compute momentum features BEFORE match using only past data."""
        features = {}
        
        for side in ['a', 'b']:
            pid = int(row[f'player_{side}_id'])
            surface = row['surface']
            history = self.results[pid]
            
            if len(history) == 0:
                features[f'win_streak_{side}'] = 0
                features[f'loss_streak_{side}'] = 0
                features[f'form_last_5_{side}'] = 0.5
                features[f'form_last_10_{side}'] = 0.5
                features[f'form_last_20_{side}'] = 0.5
                features[f'form_surface_10_{side}'] = 0.5
                features[f'elo_trend_30d_{side}'] = 0.0
                features[f'avg_opp_elo_5_{side}'] = 1500.0
                features[f'confidence_idx_{side}'] = 0.5
                features[f'tourney_momentum_{side}'] = 0
                continue
            
            # Current streaks
            win_streak = 0
            loss_streak = 0
            for _, won, _, _ in reversed(history):
                if won == 1:
                    win_streak += 1
                    if loss_streak == 0:
                        pass  # Still counting win streak
                    else:
                        break
                else:
                    loss_streak += 1
                    if win_streak == 0:
                        pass
                    else:
                        break
            # Only keep the active streak
            if history[-1][1] == 1:
                loss_streak = 0
                win_streak = sum(1 for _, w, _, _ in reversed(history) if w == 1)
                # Stop at first loss
                count = 0
                for _, w, _, _ in reversed(history):
                    if w == 1:
                        count += 1
                    else:
                        break
                win_streak = count
            else:
                win_streak = 0
                count = 0
                for _, w, _, _ in reversed(history):
                    if w == 0:
                        count += 1
                    else:
                        break
                loss_streak = count
            
            # Win rate over last N matches
            last_5 = history[-5:] if len(history) >= 5 else history
            last_10 = history[-10:] if len(history) >= 10 else history
            last_20 = history[-20:] if len(history) >= 20 else history
            
            form_5 = np.mean([h[1] for h in last_5])
            form_10 = np.mean([h[1] for h in last_10])
            form_20 = np.mean([h[1] for h in last_20])
            
            # Surface-specific form (last 10 on this surface)
            surface_history = [h for h in history if h[2] == surface]
            surf_10 = surface_history[-10:] if len(surface_history) >= 10 else surface_history
            form_surface = np.mean([h[1] for h in surf_10]) if surf_10 else 0.5
            
            # ELO trend (slope of opponent-adjusted results over last 30 days)
            try:
                match_date = pd.to_datetime(str(int(row['tourney_date'])), format='%Y%m%d')
                recent_30d = []
                for h_date, h_won, _, _ in reversed(history):
                    try:
                        h_dt = pd.to_datetime(str(h_date), format='%Y%m%d')
                        if (match_date - h_dt).days <= 30:
                            recent_30d.append(h_won)
                        else:
                            break
                    except (ValueError, TypeError):
                        continue
                
                if len(recent_30d) >= 3:
                    # Simple linear trend: positive = improving
                    x = np.arange(len(recent_30d))
                    slope = np.polyfit(x, recent_30d, 1)[0]
                    elo_trend = slope
                else:
                    elo_trend = 0.0
            except (ValueError, TypeError):
                elo_trend = 0.0
            
            # Average opponent ELO in last 5 matches
            opp_elos = [h[3] for h in last_5 if h[3] is not None]
            avg_opp_elo = np.mean(opp_elos) if opp_elos else 1500.0
            
            # EWMA confidence index (exponentially weighted win rate)
            results_binary = [h[1] for h in history[-30:]]
            if len(results_binary) >= 2:
                ewma = pd.Series(results_binary).ewm(span=self.ewma_span, min_periods=1).mean()
                confidence = ewma.iloc[-1]
            else:
                confidence = form_5
            
            # Tournament momentum (wins in current tournament)
            current_tourney = row.get('tourney_id', None)
            tourney_momentum = 0
            if current_tourney is not None:
                # Count recent wins at this tournament (same tourney_id implies same event)
                round_num = row.get('round_number', 0)
                tourney_momentum = max(0, round_num - 1)  # Approximate: round implies prior wins
            
            features[f'win_streak_{side}'] = win_streak
            features[f'loss_streak_{side}'] = loss_streak
            features[f'form_last_5_{side}'] = form_5
            features[f'form_last_10_{side}'] = form_10
            features[f'form_last_20_{side}'] = form_20
            features[f'form_surface_10_{side}'] = form_surface
            features[f'elo_trend_30d_{side}'] = elo_trend
            features[f'avg_opp_elo_5_{side}'] = avg_opp_elo
            features[f'confidence_idx_{side}'] = confidence
            features[f'tourney_momentum_{side}'] = tourney_momentum
        
        # Differences
        features['form_10_diff'] = features['form_last_10_a'] - features['form_last_10_b']
        features['form_surface_diff'] = features['form_surface_10_a'] - features['form_surface_10_b']
        features['confidence_diff'] = features['confidence_idx_a'] - features['confidence_idx_b']
        features['streak_diff'] = features['win_streak_a'] - features['win_streak_b']
        
        return features
    
    def update_after_match(self, row, winner_is_a, opp_elo_a=None, opp_elo_b=None):
        """Update momentum tracking after match."""
        match_date = int(row['tourney_date'])
        surface = row['surface']
        
        for side, won in [('a', winner_is_a), ('b', not winner_is_a)]:
            pid = int(row[f'player_{side}_id'])
            opp_elo = opp_elo_a if side == 'b' else opp_elo_b
            self.results[pid].append((match_date, int(won), surface, opp_elo))
            if len(self.results[pid]) > 200:
                self.results[pid] = self.results[pid][-200:]
