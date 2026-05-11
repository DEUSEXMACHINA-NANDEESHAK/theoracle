# ===== TheOracle: Data Normalizer =====
# ===== CELL 1: Imports =====

import os
import pandas as pd
import numpy as np
from ingestion.fetch_sackmann import load_config

# ===== CELL 2: Schema Normalization =====

def normalize_matches(atp_df, challenger_df=None, config=None):
    """
    Normalize raw Sackmann data into canonical schema.
    
    Canonical schema ensures:
    - Consistent column names
    - Surface normalization (Carpet → Hard)
    - Proper date parsing
    - player_a / player_b ordering (alphabetical by ID for consistency)
    - 'winner_is_a' binary target column
    
    CRITICAL: We assign player_a and player_b in a DETERMINISTIC way
    (lower player_id = player_a) so the model doesn't learn that
    player_a always wins (which would be data leakage).
    """
    if config is None:
        config = load_config()
    
    # Process ATP matches
    print("🔧 Normalizing ATP match data...")
    atp_clean = _normalize_single_source(atp_df, source='atp')
    
    # Process challenger matches if provided
    if challenger_df is not None and len(challenger_df) > 0:
        print("🔧 Normalizing challenger data...")
        chall_clean = _normalize_single_source(challenger_df, source='challenger')
        combined = pd.concat([atp_clean, chall_clean], ignore_index=True)
    else:
        combined = atp_clean
    
    # Sort chronologically
    combined = combined.sort_values('tourney_date').reset_index(drop=True)
    
    # Assign unique match_id
    combined['match_id'] = range(len(combined))
    
    print(f"✅ Normalized {len(combined):,} total matches")
    print(f"   ATP: {len(atp_clean):,} | Challenger: {len(combined) - len(atp_clean):,}")
    print(f"   Date range: {combined['tourney_date'].min()} → {combined['tourney_date'].max()}")
    print(f"   Surfaces: {dict(combined['surface'].value_counts())}")
    
    return combined

# ===== CELL 3: Single Source Normalizer =====

def _normalize_single_source(df, source='atp'):
    """Normalize a single data source to canonical schema."""
    df = df.copy()
    
    # ---- Surface normalization ----
    df['surface'] = df['surface'].fillna('Hard')
    df['surface'] = df['surface'].replace({
        'Carpet': 'Hard',
        'None': 'Hard',
    })
    # Keep only Clay, Grass, Hard
    valid_surfaces = ['Clay', 'Grass', 'Hard']
    df = df[df['surface'].isin(valid_surfaces)].copy()
    
    # ---- Date parsing ----
    df['tourney_date'] = pd.to_numeric(df['tourney_date'], errors='coerce')
    df = df.dropna(subset=['tourney_date'])
    df['tourney_date'] = df['tourney_date'].astype(int)
    
    # ---- Ensure required columns exist ----
    required = ['winner_id', 'loser_id', 'winner_name', 'loser_name', 'surface']
    for col in required:
        if col not in df.columns:
            print(f"  [ERROR] Missing required column: {col}")
            return pd.DataFrame()
    
    # ---- Deterministic player_a / player_b assignment ----
    # Lower player_id = player_a. This prevents the model from learning
    # that one position always wins.
    df['player_a_id'] = df[['winner_id', 'loser_id']].min(axis=1).astype(int)
    df['player_b_id'] = df[['winner_id', 'loser_id']].max(axis=1).astype(int)
    
    # Who won? 1 if player_a won, 0 if player_b won
    df['winner_is_a'] = (df['winner_id'] == df['player_a_id']).astype(int)
    
    # Map names to a/b
    df['player_a_name'] = np.where(
        df['winner_id'] == df['player_a_id'],
        df['winner_name'], df['loser_name']
    )
    df['player_b_name'] = np.where(
        df['winner_id'] == df['player_a_id'],
        df['loser_name'], df['winner_name']
    )
    
    # ---- Map match stats to player_a / player_b ----
    stat_cols_w = [c for c in df.columns if c.startswith('w_')]
    stat_cols_l = [c for c in df.columns if c.startswith('l_')]
    
    for w_col in stat_cols_w:
        stat_name = w_col[2:]  # Remove 'w_' prefix
        l_col = f'l_{stat_name}'
        if l_col in df.columns:
            a_col = f'a_{stat_name}'
            b_col = f'b_{stat_name}'
            df[a_col] = np.where(
                df['winner_is_a'] == 1,
                pd.to_numeric(df[w_col], errors='coerce'),
                pd.to_numeric(df[l_col], errors='coerce')
            )
            df[b_col] = np.where(
                df['winner_is_a'] == 1,
                pd.to_numeric(df[l_col], errors='coerce'),
                pd.to_numeric(df[w_col], errors='coerce')
            )
    
    # ---- Map ranking/seed to a/b ----
    for prefix in ['rank', 'rank_points', 'seed']:
        w_col = f'winner_{prefix}'
        l_col = f'loser_{prefix}'
        if w_col in df.columns and l_col in df.columns:
            df[f'a_{prefix}'] = np.where(
                df['winner_is_a'] == 1,
                pd.to_numeric(df[w_col], errors='coerce'),
                pd.to_numeric(df[l_col], errors='coerce')
            )
            df[f'b_{prefix}'] = np.where(
                df['winner_is_a'] == 1,
                pd.to_numeric(df[l_col], errors='coerce'),
                pd.to_numeric(df[w_col], errors='coerce')
            )
    
    # ---- Tournament metadata ----
    df['tourney_level'] = df.get('tourney_level', pd.Series('A', index=df.index))
    df['best_of'] = pd.to_numeric(df.get('best_of', pd.Series(3, index=df.index)), errors='coerce').fillna(3).astype(int)
    
    # Parse round
    round_order = {'F': 7, 'SF': 6, 'QF': 5, 'R16': 4, 'R32': 3, 'R64': 2, 'R128': 1, 'RR': 3}
    df['round_number'] = df['round'].map(round_order).fillna(0).astype(int)
    
    # Indoor flag (from tourney_name heuristics or if available)
    if 'indoor_outdoor' in df.columns:
        df['is_indoor'] = (df['indoor_outdoor'] == 'Indoor').astype(int)
    else:
        df['is_indoor'] = 0  # Default outdoor, will be refined later
    
    # Source tag
    df['source'] = source
    
    # ---- Select canonical columns ----
    keep_cols = [
        'match_id', 'tourney_id', 'tourney_name', 'tourney_date', 'tourney_level',
        'surface', 'best_of', 'round', 'round_number', 'is_indoor',
        'player_a_id', 'player_b_id', 'player_a_name', 'player_b_name',
        'winner_is_a', 'score', 'minutes', 'source',
    ]
    
    # Add all a_* and b_* stat columns
    stat_a_cols = [c for c in df.columns if c.startswith('a_')]
    stat_b_cols = [c for c in df.columns if c.startswith('b_')]
    keep_cols.extend(stat_a_cols)
    keep_cols.extend(stat_b_cols)
    
    # Only keep columns that exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    
    return df[keep_cols].copy()

# ===== CELL 4: Parse Score for Set Counts =====

def parse_score(score_str):
    """
    Parse a tennis score string to extract set information.
    
    Returns dict with:
    - sets_a, sets_b: sets won by each player
    - total_sets: total sets played
    - was_retirement: whether match ended in retirement/walkover
    """
    if not isinstance(score_str, str) or score_str.strip() == '':
        return {'sets_a': None, 'sets_b': None, 'total_sets': None, 'was_retirement': False}
    
    # Check for retirement/walkover
    was_ret = any(tag in score_str.upper() for tag in ['RET', 'W/O', 'DEF', 'ABN'])
    
    sets_a = 0
    sets_b = 0
    
    try:
        parts = score_str.replace('RET', '').replace('W/O', '').replace('DEF', '').strip().split()
        for part in parts:
            part = part.strip('()')
            if '-' in part:
                games = part.split('-')
                g_a = int(games[0].split('(')[0])
                g_b = int(games[1].split('(')[0])
                if g_a > g_b:
                    sets_a += 1
                elif g_b > g_a:
                    sets_b += 1
    except (ValueError, IndexError):
        pass
    
    total = sets_a + sets_b
    return {
        'sets_a': sets_a if total > 0 else None,
        'sets_b': sets_b if total > 0 else None,
        'total_sets': total if total > 0 else None,
        'was_retirement': was_ret,
    }

def add_score_features(df):
    """Add parsed score features to match DataFrame."""
    print("📊 Parsing scores...")
    score_data = df['score'].apply(parse_score)
    score_df = pd.DataFrame(score_data.tolist())
    
    for col in score_df.columns:
        df[col] = score_df[col].values
    
    # Winner sets are always the max of sets_a/sets_b
    # But we need to align with winner_is_a
    df['winner_sets'] = np.where(df['winner_is_a'] == 1, df['sets_a'], df['sets_b'])
    df['loser_sets'] = np.where(df['winner_is_a'] == 1, df['sets_b'], df['sets_a'])
    
    return df

# ===== CELL 5: Save Clean Data =====

def save_clean_data(matches_df, players_df=None, clean_dir="data/clean"):
    """Save normalized data as parquet files."""
    os.makedirs(clean_dir, exist_ok=True)
    
    matches_path = os.path.join(clean_dir, "matches.parquet")
    matches_df.to_parquet(matches_path, index=False)
    print(f"💾 Saved {len(matches_df):,} matches → {matches_path}")
    
    if players_df is not None:
        players_path = os.path.join(clean_dir, "players.parquet")
        players_df.to_parquet(players_path, index=False)
        print(f"💾 Saved {len(players_df):,} players → {players_path}")

def load_clean_data(clean_dir="data/clean"):
    """Load previously cleaned data."""
    matches = pd.read_parquet(os.path.join(clean_dir, "matches.parquet"))
    players = None
    players_path = os.path.join(clean_dir, "players.parquet")
    if os.path.exists(players_path):
        players = pd.read_parquet(players_path)
    return matches, players
