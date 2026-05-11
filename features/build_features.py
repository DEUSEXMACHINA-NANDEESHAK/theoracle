# ===== TheOracle: Master Feature Builder =====
# Orchestrates all 7 feature engines to build the feature store.
#
# ===== CELL 1: Imports =====

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from tqdm import tqdm

from features.elo_engine import TennisEloEngine
from features.stamina_engine import StaminaEngine
from features.pressure_engine import PressureEngine
from features.momentum_engine import MomentumEngine
from features.surface_tactics import SurfaceTacticsEngine
from features.environment_engine import EnvironmentEngine
from features.odds_engine import OddsEngine
from features.rolling_stats import RollingStatsEngine

# ===== CELL 2: Feature Store Builder =====

def build_feature_store(clean_dir="data/clean", features_dir="data/features",
                        elo_config="configs/elo_config.yaml"):
    """
    Build the complete feature store by processing all matches chronologically.
    
    For EACH match (in chronological order):
    1. Compute features from all 7 engines (using PRE-MATCH state)
    2. Store feature row
    3. Update all engine states (using match result — POST-MATCH)
    
    This strict ordering is the LEAKAGE PREVENTION protocol.
    
    Output: features/match_features.parquet
    """
    os.makedirs(features_dir, exist_ok=True)
    
    # Load clean data
    print("📂 Loading clean data...")
    matches = pd.read_parquet(os.path.join(clean_dir, "matches.parquet"))
    
    players = None
    players_path = os.path.join(clean_dir, "players.parquet")
    if os.path.exists(players_path):
        players = pd.read_parquet(players_path)
    
    odds_df = None
    odds_path = os.path.join(clean_dir, "odds.parquet")
    if os.path.exists(odds_path):
        odds_df = pd.read_parquet(odds_path)
    
    # Sort chronologically (CRITICAL)
    matches = matches.sort_values('tourney_date').reset_index(drop=True)
    
    print(f"📊 Processing {len(matches):,} matches through 7 feature engines...")
    
    # Initialize all engines
    elo_engine = TennisEloEngine(elo_config)
    stamina_engine = StaminaEngine()
    pressure_engine = PressureEngine()
    momentum_engine = MomentumEngine()
    surface_engine = SurfaceTacticsEngine()
    env_engine = EnvironmentEngine(players_df=players)
    odds_engine = OddsEngine(odds_df)
    rolling_engine = RollingStatsEngine()
    
    # Process each match
    all_features = []
    
    for idx, row in tqdm(matches.iterrows(), total=len(matches), desc="🔨 Building Features"):
        # ============================================
        # STEP 1: COMPUTE FEATURES (PRE-MATCH STATE)
        # ============================================
        
        # Engine 1: ELO features
        elo_feats = elo_engine.compute_features_for_match(row)
        
        # Engine 2: Stamina features
        stamina_feats = stamina_engine.compute_features_for_match(row)
        
        # Engine 3: Pressure features
        pressure_feats = pressure_engine.compute_features_for_match(row)
        
        # Engine 4: Momentum features
        momentum_feats = momentum_engine.compute_features_for_match(row)
        
        # Engine 5: Surface tactics features
        surface_feats = surface_engine.compute_features_for_match(row)
        
        # Engine 6: Environment features
        env_feats = env_engine.compute_features_for_match(row)
        
        # Engine 7: Odds features (needs ELO expected prob for divergence)
        elo_expected_a = elo_feats.get('elo_expected_a', None)
        odds_feats = odds_engine.compute_features_for_match(row, elo_expected_a)
        
        # Rolling stats
        rolling_feats = rolling_engine.compute_features_for_match(row)
        
        # Merge all features
        match_features = {
            'match_id': row.get('match_id', idx),
            'tourney_date': row['tourney_date'],
            'surface': row['surface'],
            'winner_is_a': row['winner_is_a'],  # TARGET variable
        }
        match_features.update(elo_feats)
        match_features.update(stamina_feats)
        match_features.update(pressure_feats)
        match_features.update(momentum_feats)
        match_features.update(surface_feats)
        match_features.update(env_feats)
        match_features.update(odds_feats)
        match_features.update(rolling_feats)
        
        all_features.append(match_features)
        
        # ============================================
        # STEP 2: UPDATE ENGINE STATES (POST-MATCH)
        # ============================================
        
        winner_is_a = bool(row['winner_is_a'])
        
        elo_engine.update_after_match(row, winner_is_a)
        stamina_engine.update_after_match(row)
        pressure_engine.update_after_match(row, winner_is_a)
        momentum_engine.update_after_match(
            row, winner_is_a,
            opp_elo_a=elo_feats.get('elo_overall_b'),
            opp_elo_b=elo_feats.get('elo_overall_a'),
        )
        surface_engine.update_after_match(row)
        rolling_engine.update_after_match(row, winner_is_a)
    
    # Build DataFrame
    features_df = pd.DataFrame(all_features)
    
    # Save
    output_path = os.path.join(features_dir, "match_features.parquet")
    features_df.to_parquet(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"✅ FEATURE STORE BUILT")
    print(f"   Matches: {len(features_df):,}")
    print(f"   Features: {features_df.shape[1]} columns")
    print(f"   Saved to: {output_path}")
    print(f"   Feature groups:")
    print(f"     ELO:        {sum(1 for c in features_df.columns if 'elo' in c.lower())} features")
    print(f"     Stamina:    {sum(1 for c in features_df.columns if any(x in c for x in ['rest', 'fatigue', 'matches_7d', 'matches_14d', 'sets_last']))} features")
    print(f"     Pressure:   {sum(1 for c in features_df.columns if any(x in c for x in ['bp_', 'tb_', 'pressure', 'comeback', 'deciding']))} features")
    print(f"     Momentum:   {sum(1 for c in features_df.columns if any(x in c for x in ['streak', 'form_', 'confidence', 'momentum', 'trend']))} features")
    print(f"     Surface:    {sum(1 for c in features_df.columns if any(x in c for x in ['serve_dom', 'ace_rate', 'df_rate', 'first_serve', 'second_serve', 'return_pts', 'grind', 'grass_serve', 'versatil']))} features")
    print(f"     Environ:    {sum(1 for c in features_df.columns if any(x in c for x in ['temp', 'humid', 'wind', 'indoor', 'home', 'rank_', 'age_', 'height', 'level', 'round_', 'best_of']))} features")
    print(f"     Odds:       {sum(1 for c in features_df.columns if 'odds' in c.lower() or 'divergence' in c)} features")
    print(f"{'='*60}")
    
    # Print top ELO rankings as a sanity check
    print("\n🏆 Current Top 20 by ELO:")
    top = elo_engine.get_player_ratings(20)
    print(top.to_string(index=False))
    
    return features_df

# ===== CELL 3: Entrypoint =====

if __name__ == "__main__":
    features = build_feature_store()
