# ===== TheOracle: Environment & Context Engine =====
# ===== CELL 1: Imports =====

import numpy as np
import pandas as pd
import os

# ===== CELL 2: Environment Feature Computer =====

class EnvironmentEngine:
    """
    Computes match context and environmental features.
    
    - Tournament level encoding (Grand Slam > Masters > 500 > 250)
    - Round progression
    - Best-of-3 vs Best-of-5
    - Indoor/outdoor
    - Home advantage
    - Weather (temperature, humidity, wind) when available
    """
    
    # Tournament level encoding
    LEVEL_MAP = {
        'G': 4,   # Grand Slam
        'M': 3,   # Masters 1000
        'A': 2,   # ATP 500/250
        'D': 1,   # Davis Cup
        'C': 0,   # Challenger
        'S': 0,   # Satellite
    }
    
    def __init__(self, players_df=None, weather_dir="data/raw/weather"):
        self.weather_cache = {}
        self.weather_dir = weather_dir
        
        # Build player country lookup
        self.player_country = {}
        if players_df is not None:
            for _, p in players_df.iterrows():
                pid = p.get('player_id')
                country = p.get('country_code', '')
                if pid is not None:
                    self.player_country[int(pid)] = str(country)
        
        # Tournament country mapping (major events)
        self.tourney_country = {
            'australian open': 'AUS', 'melbourne': 'AUS',
            'roland garros': 'FRA', 'paris': 'FRA',
            'wimbledon': 'GBR', 'london': 'GBR', "queen's": 'GBR',
            'us open': 'USA', 'new york': 'USA', 'indian wells': 'USA',
            'miami': 'USA', 'cincinnati': 'USA', 'washington': 'USA',
            'monte carlo': 'MON', 'madrid': 'ESP', 'barcelona': 'ESP',
            'rome': 'ITA', 'hamburg': 'GER', 'halle': 'GER',
            'shanghai': 'CHN', 'beijing': 'CHN', 'tokyo': 'JPN',
            'dubai': 'UAE', 'doha': 'QAT', 'rio': 'BRA',
            'buenos aires': 'ARG', 'acapulco': 'MEX',
            'montreal': 'CAN', 'toronto': 'CAN',
            'rotterdam': 'NED', 'basel': 'SUI', 'vienna': 'AUT',
        }
        
        self._load_weather_data()
    
    def _load_weather_data(self):
        """Load cached weather data if available."""
        weather_path = os.path.join(self.weather_dir, "all_weather.parquet")
        if os.path.exists(weather_path):
            try:
                self.weather_df = pd.read_parquet(weather_path)
                self.weather_df['date'] = pd.to_datetime(self.weather_df['date'])
            except Exception:
                self.weather_df = pd.DataFrame()
        else:
            self.weather_df = pd.DataFrame()
    
    def compute_features_for_match(self, row):
        """Compute environment/context features for a match."""
        features = {}
        
        # Tournament level
        level = row.get('tourney_level', 'A')
        features['tourney_level_enc'] = self.LEVEL_MAP.get(level, 1)
        
        # Round number
        features['round_number'] = int(row.get('round_number', 0))
        
        # Best of
        features['best_of'] = int(row.get('best_of', 3))
        features['is_bo5'] = 1 if features['best_of'] == 5 else 0
        
        # Indoor
        features['is_indoor'] = int(row.get('is_indoor', 0))
        
        # Home advantage
        tourney_name = str(row.get('tourney_name', '')).lower()
        tourney_country = None
        for key, country in self.tourney_country.items():
            if key in tourney_name:
                tourney_country = country
                break
        
        for side in ['a', 'b']:
            pid = int(row[f'player_{side}_id'])
            player_country = self.player_country.get(pid, '')
            features[f'home_advantage_{side}'] = 1 if (
                tourney_country and player_country and
                player_country == tourney_country
            ) else 0
        
        features['home_adv_diff'] = features['home_advantage_a'] - features['home_advantage_b']
        
        # Ranking features
        rank_a = row.get('a_rank', None)
        rank_b = row.get('b_rank', None)
        if rank_a is not None and rank_b is not None:
            rank_a = pd.to_numeric(rank_a, errors='coerce')
            rank_b = pd.to_numeric(rank_b, errors='coerce')
            if not pd.isna(rank_a) and not pd.isna(rank_b):
                features['rank_diff'] = rank_b - rank_a  # Positive = a is better ranked
                features['rank_ratio'] = min(rank_a, rank_b) / max(rank_a, rank_b, 1)
                features['log_rank_diff'] = np.log1p(rank_b) - np.log1p(rank_a)
            else:
                features['rank_diff'] = 0
                features['rank_ratio'] = 1.0
                features['log_rank_diff'] = 0
        else:
            features['rank_diff'] = 0
            features['rank_ratio'] = 1.0
            features['log_rank_diff'] = 0
        
        # Weather features (if available)
        features['temperature_c'] = np.nan
        features['humidity_pct'] = np.nan
        features['wind_speed_kmh'] = np.nan
        
        if len(self.weather_df) > 0:
            try:
                match_date = pd.to_datetime(str(int(row['tourney_date'])), format='%Y%m%d')
                weather_match = self.weather_df[
                    (self.weather_df['tourney_name'].str.lower().str.contains(tourney_name[:10])) &
                    (self.weather_df['date'] == match_date)
                ]
                if len(weather_match) > 0:
                    features['temperature_c'] = weather_match.iloc[0]['temperature_c']
                    features['humidity_pct'] = weather_match.iloc[0]['humidity_pct']
                    features['wind_speed_kmh'] = weather_match.iloc[0]['wind_speed_kmh']
            except (ValueError, TypeError):
                pass
        
        # Age features
        for side in ['a', 'b']:
            age_col = f'{side}_age'
            if age_col in row and not pd.isna(row.get(age_col)):
                features[f'age_{side}'] = float(row[age_col])
            else:
                features[f'age_{side}'] = 27.0  # Default average
        
        features['age_diff'] = features['age_a'] - features['age_b']
        
        # Height features
        for side in ['a', 'b']:
            ht_col = f'{side}_ht'
            if ht_col in row and not pd.isna(row.get(ht_col)):
                features[f'height_{side}'] = float(row[ht_col])
            else:
                features[f'height_{side}'] = 185.0  # Default average
        
        features['height_diff'] = features['height_a'] - features['height_b']
        
        return features
