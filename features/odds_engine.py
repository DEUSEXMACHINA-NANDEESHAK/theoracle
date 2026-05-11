# ===== TheOracle: Pre-Match Odds Engine =====
# ===== CELL 1: Imports =====

import numpy as np
import pandas as pd

# ===== CELL 2: Odds Feature Computer =====

class OddsEngine:
    """
    Integrates pre-match betting odds as a "Bollinger Band" reference.
    
    The odds are NOT used as a direct predictor. Instead:
    1. Implied probabilities show what the market thinks
    2. Model-vs-odds divergence shows where our model disagrees
    3. Odds spread shows market confidence level
    
    This lets the XGBoost learn when to trust the market and when
    our proprietary features (ELO, stamina, momentum) reveal an edge.
    """
    
    def __init__(self, odds_df=None):
        self.odds_lookup = {}
        if odds_df is not None and len(odds_df) > 0:
            self._build_lookup(odds_df)
    
    def _build_lookup(self, odds_df):
        """Build a lookup table from odds data keyed by player names + date."""
        for _, row in odds_df.iterrows():
            try:
                w_name = str(row.get('winner_name', '')).strip().lower()
                l_name = str(row.get('loser_name', '')).strip().lower()
                date = str(row.get('match_date', ''))
                
                if not w_name or not l_name:
                    continue
                
                key = self._make_key(w_name, l_name, date)
                self.odds_lookup[key] = {
                    'implied_prob_w': row.get('implied_prob_w', np.nan),
                    'implied_prob_l': row.get('implied_prob_l', np.nan),
                    'winner_name': w_name,
                    'loser_name': l_name,
                }
            except Exception:
                continue
        
        print(f"📊 Odds lookup built: {len(self.odds_lookup):,} matches")
    
    def _make_key(self, name1, name2, date):
        """Create lookup key from player names and date."""
        # Sort names for consistency
        sorted_names = tuple(sorted([name1, name2]))
        # Use date prefix for matching (YYYY-MM-DD or YYYYMMDD)
        date_key = str(date).replace('-', '')[:8]
        return (sorted_names[0], sorted_names[1], date_key)
    
    def compute_features_for_match(self, row, elo_expected_a=None):
        """
        Compute odds-based features for a match.
        
        Args:
            row: Match row from normalized DataFrame
            elo_expected_a: Our model's expected probability for player A
                           (from ELO engine, to compute divergence)
        """
        features = {
            'odds_implied_a': np.nan,
            'odds_implied_b': np.nan,
            'odds_spread': np.nan,
            'odds_available': 0,
            'elo_vs_odds_divergence': np.nan,
        }
        
        if not self.odds_lookup:
            return features
        
        # Try to find matching odds record
        a_name = str(row.get('player_a_name', '')).strip().lower()
        b_name = str(row.get('player_b_name', '')).strip().lower()
        date = str(row.get('tourney_date', ''))
        
        key = self._make_key(a_name, b_name, date)
        odds = self.odds_lookup.get(key, None)
        
        if odds is None:
            # Try fuzzy matching on last name only
            a_last = a_name.split()[-1] if a_name else ''
            b_last = b_name.split()[-1] if b_name else ''
            key2 = self._make_key(a_last, b_last, date)
            odds = self.odds_lookup.get(key2, None)
        
        if odds is not None:
            # Determine which implied prob maps to player A
            if odds['winner_name'] in a_name or a_name in odds['winner_name']:
                features['odds_implied_a'] = odds['implied_prob_w']
                features['odds_implied_b'] = odds['implied_prob_l']
            else:
                features['odds_implied_a'] = odds['implied_prob_l']
                features['odds_implied_b'] = odds['implied_prob_w']
            
            features['odds_available'] = 1
            
            # Odds spread (market confidence: higher = more certain outcome)
            if not np.isnan(features['odds_implied_a']):
                features['odds_spread'] = abs(features['odds_implied_a'] - features['odds_implied_b'])
            
            # ELO vs Odds divergence
            # Positive = our model favors A more than the market does
            if elo_expected_a is not None and not np.isnan(features['odds_implied_a']):
                features['elo_vs_odds_divergence'] = elo_expected_a - features['odds_implied_a']
        
        return features
