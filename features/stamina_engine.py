# ===== TheOracle: Stamina & Fatigue Engine =====
# ===== CELL 1: Imports =====

import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# ===== CELL 2: Stamina Feature Computer =====

class StaminaEngine:
    """
    Computes fatigue and recovery features for each player.
    
    Features:
    - days_rest: Days since last match
    - matches_last_7d/14d: Match density (workload)
    - sets_played_last_3: Recent physical load from score parsing
    - five_set_count_30d: Grueling match count
    - fatigue_index: Composite workload / (rest × age_factor)
    """
    
    def __init__(self):
        # player_id → list of (date, total_sets, best_of)
        self.match_history = defaultdict(list)
    
    def compute_features_for_match(self, row):
        """Compute stamina features BEFORE the match using only past data."""
        features = {}
        
        for side in ['a', 'b']:
            pid = int(row[f'player_{side}_id'])
            match_date = int(row['tourney_date'])
            history = self.match_history[pid]
            
            # Days since last match
            if history:
                last_date = history[-1][0]
                try:
                    d1 = pd.to_datetime(str(last_date), format='%Y%m%d')
                    d2 = pd.to_datetime(str(match_date), format='%Y%m%d')
                    days_rest = max((d2 - d1).days, 0)
                except (ValueError, TypeError):
                    days_rest = 30  # Default if date parsing fails
            else:
                days_rest = 30  # First match — assume well rested
            
            # Matches in last N days
            matches_7d = 0
            matches_14d = 0
            sets_last_3 = 0
            five_set_count_30d = 0
            total_minutes_last_5 = 0
            match_count_for_minutes = 0
            
            try:
                current_dt = pd.to_datetime(str(match_date), format='%Y%m%d')
            except (ValueError, TypeError):
                current_dt = None
            
            if current_dt is not None:
                for h_date, h_sets, h_best_of, h_minutes in reversed(history):
                    try:
                        h_dt = pd.to_datetime(str(h_date), format='%Y%m%d')
                        days_ago = (current_dt - h_dt).days
                    except (ValueError, TypeError):
                        continue
                    
                    if days_ago <= 7:
                        matches_7d += 1
                    if days_ago <= 14:
                        matches_14d += 1
                    if days_ago <= 30 and h_best_of == 5 and h_sets is not None and h_sets >= 5:
                        five_set_count_30d += 1
                    if days_ago > 30:
                        break  # History is sorted, no need to check further
                
                # Sets in last 3 matches
                recent_3 = history[-3:] if len(history) >= 3 else history
                sets_last_3 = sum(h[1] for h in recent_3 if h[1] is not None)
                
                # Average match duration last 5
                recent_5 = history[-5:] if len(history) >= 5 else history
                for h in recent_5:
                    if h[3] is not None and h[3] > 0:
                        total_minutes_last_5 += h[3]
                        match_count_for_minutes += 1
            
            avg_minutes = total_minutes_last_5 / match_count_for_minutes if match_count_for_minutes > 0 else 90
            
            # Fatigue index: higher = more fatigued
            # Composite of workload over rest period, adjusted by how grueling recent matches were
            workload = matches_14d * 1.0 + five_set_count_30d * 2.0 + sets_last_3 * 0.3
            rest_factor = max(days_rest, 1)  # Avoid division by zero
            fatigue_index = workload / rest_factor
            
            features[f'days_rest_{side}'] = days_rest
            features[f'matches_7d_{side}'] = matches_7d
            features[f'matches_14d_{side}'] = matches_14d
            features[f'sets_last_3_{side}'] = sets_last_3
            features[f'five_set_30d_{side}'] = five_set_count_30d
            features[f'avg_match_minutes_{side}'] = avg_minutes
            features[f'fatigue_index_{side}'] = fatigue_index
        
        # Differences
        features['days_rest_diff'] = features['days_rest_a'] - features['days_rest_b']
        features['fatigue_diff'] = features['fatigue_index_a'] - features['fatigue_index_b']
        
        return features
    
    def update_after_match(self, row):
        """Update match history after a match is played."""
        match_date = int(row['tourney_date'])
        total_sets = row.get('total_sets', None)
        best_of = int(row.get('best_of', 3))
        minutes = row.get('minutes', None)
        if pd.isna(minutes):
            minutes = None
        
        for side in ['a', 'b']:
            pid = int(row[f'player_{side}_id'])
            self.match_history[pid].append((match_date, total_sets, best_of, minutes))
            # Keep only last 100 matches to save memory
            if len(self.match_history[pid]) > 100:
                self.match_history[pid] = self.match_history[pid][-100:]
