# ===== TheOracle: Ingestion Pipeline Orchestrator =====
# ===== CELL 1: Imports =====

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.fetch_sackmann import load_config, fetch_atp_matches, load_atp_matches, load_players
from ingestion.fetch_challengers import fetch_challenger_matches, load_challenger_matches
from ingestion.fetch_odds import fetch_odds_data, load_odds_data
from ingestion.fetch_weather import fetch_weather_for_matches
from ingestion.normalize import normalize_matches, add_score_features, save_clean_data

# ===== CELL 2: Full Ingestion Pipeline =====

def run_full_ingestion(config_path="configs/data_sources.yaml", skip_weather=False):
    """
    Run the complete data ingestion pipeline.
    
    Steps:
    1. Download ATP match data from JeffSackmann/tennis_atp
    2. Download Challenger/Qualifier data
    3. Download betting odds from tennis-data.co.uk
    4. Normalize all data to canonical schema
    5. Parse scores for set counts
    6. Optionally fetch weather data
    7. Save clean data as parquet
    
    Args:
        config_path: Path to data_sources.yaml
        skip_weather: If True, skip weather fetching (saves time)
    
    Returns:
        Tuple of (matches_df, players_df, odds_df)
    """
    config = load_config(config_path)
    
    print("=" * 60)
    print("🎾 TheOracle: Data Ingestion Pipeline")
    print("=" * 60)
    
    # ---- Step 1: Download ATP matches ----
    print("\n📥 STEP 1: Downloading ATP match data...")
    fetch_atp_matches(config)
    atp_df = load_atp_matches()
    
    # ---- Step 2: Download Challenger data ----
    print("\n📥 STEP 2: Downloading Challenger/Qualifier data...")
    fetch_challenger_matches(config)
    chall_df = load_challenger_matches()
    
    # ---- Step 3: Download betting odds ----
    print("\n📥 STEP 3: Downloading betting odds...")
    try:
        fetch_odds_data(config)
        odds_df = load_odds_data()
    except Exception as e:
        print(f"  [WARN] Odds download failed: {e}. Continuing without odds.")
        odds_df = None
    
    # ---- Step 4: Load player data ----
    print("\n📥 STEP 4: Loading player biographical data...")
    players_df = load_players()
    
    # ---- Step 5: Normalize ----
    print("\n🔧 STEP 5: Normalizing data...")
    matches_df = normalize_matches(atp_df, chall_df, config)
    
    # ---- Step 6: Parse scores ----
    print("\n📊 STEP 6: Parsing scores...")
    matches_df = add_score_features(matches_df)
    
    # ---- Step 7: Weather (optional) ----
    if not skip_weather:
        print("\n🌤️  STEP 7: Fetching weather data...")
        try:
            weather_df = fetch_weather_for_matches(matches_df)
            if len(weather_df) > 0:
                weather_path = os.path.join(config['paths']['raw_dir'], "weather", "all_weather.parquet")
                os.makedirs(os.path.dirname(weather_path), exist_ok=True)
                weather_df.to_parquet(weather_path, index=False)
                print(f"  💾 Saved weather data")
        except Exception as e:
            print(f"  [WARN] Weather fetch failed: {e}. Continuing without weather.")
    else:
        print("\n⏭️  STEP 7: Skipping weather (skip_weather=True)")
    
    # ---- Step 8: Save clean data ----
    print("\n💾 STEP 8: Saving clean data...")
    save_clean_data(matches_df, players_df, config['paths']['clean_dir'])
    
    if odds_df is not None and len(odds_df) > 0:
        odds_path = os.path.join(config['paths']['clean_dir'], "odds.parquet")
        odds_df.to_parquet(odds_path, index=False)
        print(f"  💾 Saved {len(odds_df):,} odds records")
    
    # ---- Summary ----
    print("\n" + "=" * 60)
    print("✅ INGESTION COMPLETE")
    print(f"   Total matches: {len(matches_df):,}")
    print(f"   Date range: {matches_df['tourney_date'].min()} → {matches_df['tourney_date'].max()}")
    print(f"   Surfaces: {dict(matches_df['surface'].value_counts())}")
    print(f"   Sources: {dict(matches_df['source'].value_counts())}")
    print("=" * 60)
    
    return matches_df, players_df, odds_df

# ===== CELL 3: Entrypoint =====

if __name__ == "__main__":
    matches, players, odds = run_full_ingestion(skip_weather=True)
