# ===== TheOracle: Betting Odds Fetcher =====
# ===== CELL 1: Imports =====

import os
import sys
import pandas as pd
import requests
from tqdm import tqdm
import time
from ingestion.fetch_sackmann import load_config

# ===== CELL 2: Odds Download & Parsing =====

def fetch_odds_data(config=None, raw_dir="data/raw/odds"):
    """
    Download historical pre-match betting odds from tennis-data.co.uk.
    
    This site provides free XLS/CSV files with closing odds from multiple
    bookmakers (Bet365, Pinnacle, etc.) for ATP matches from 2000 onwards.
    
    We use pre-match closing odds as a "Bollinger Band" reference —
    a feature that tells the model what the market thinks, NOT a direct predictor.
    The model learns when to agree or disagree with the market.
    """
    if config is None:
        config = load_config()
    
    odds_cfg = config['odds']
    os.makedirs(raw_dir, exist_ok=True)
    
    # tennis-data.co.uk organizes files by year
    # URL pattern: http://www.tennis-data.co.uk/{year}/{year}.xlsx
    base_url = odds_cfg['base_url']
    
    print("💰 Downloading betting odds data...")
    downloaded = 0
    
    for year in tqdm(range(odds_cfg['years_start'], odds_cfg['years_end'] + 1), desc="Odds"):
        # Try multiple URL patterns (site changed format over years)
        urls_to_try = [
            f"{base_url}/{year}/{year}.xlsx",
            f"{base_url}/{year}/{year}.xls",
            f"{base_url}/{year}/{year}.csv",
        ]
        
        dest_path = os.path.join(raw_dir, f"odds_{year}.xlsx")
        
        for url in urls_to_try:
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=30)
                if response.status_code == 200 and len(response.content) > 1000:
                    ext = url.split('.')[-1]
                    dest_path = os.path.join(raw_dir, f"odds_{year}.{ext}")
                    with open(dest_path, 'wb') as f:
                        f.write(response.content)
                    downloaded += 1
                    break
            except requests.RequestException:
                continue
        
        time.sleep(0.5)  # Be respectful to the server
    
    print(f"✅ Odds data: {downloaded} files downloaded")
    return raw_dir

# ===== CELL 3: Load and Parse Odds =====

def load_odds_data(raw_dir="data/raw/odds", year_start=2005, year_end=2026):
    """
    Load downloaded odds files and extract pre-match closing odds.
    
    Returns DataFrame with columns:
    - tourney_date, winner_name, loser_name (for matching)
    - odds_w_pinnacle, odds_l_pinnacle (Pinnacle closing odds — most efficient market)
    - odds_w_b365, odds_l_b365 (Bet365 closing odds — fallback)
    - odds_w_avg, odds_l_avg (market average)
    - implied_prob_w, implied_prob_l (from Pinnacle, overround removed)
    """
    all_dfs = []
    
    for year in range(year_start, year_end + 1):
        # Try different extensions
        for ext in ['xlsx', 'xls', 'csv']:
            fpath = os.path.join(raw_dir, f"odds_{year}.{ext}")
            if os.path.exists(fpath):
                try:
                    if ext == 'csv':
                        df = pd.read_csv(fpath, low_memory=False)
                    elif ext == 'xls':
                        # Legacy Excel format
                        try:
                            df = pd.read_excel(fpath, engine='xlrd')
                        except ImportError:
                            print("  [INFO] Installing xlrd for legacy Excel support...")
                            import subprocess
                            subprocess.check_call([sys.executable, "-m", "pip", "install", "xlrd"])
                            df = pd.read_excel(fpath, engine='xlrd')
                    else:
                        # Modern Excel format
                        df = pd.read_excel(fpath, engine='openpyxl')
                    
                    df['source_year'] = year
                    all_dfs.append(df)
                    break
                except Exception as e:
                    print(f"  [WARN] Error reading odds_{year}.{ext}: {e}")
    
    if not all_dfs:
        print("⚠️  No odds files found. Odds features will be NaN.")
        return pd.DataFrame()
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Standardize column names (tennis-data.co.uk uses various conventions)
    col_map = {
        'Winner': 'winner_name',
        'Loser': 'loser_name',
        'Date': 'match_date',
        'Surface': 'surface',
        'Tournament': 'tourney_name',
        'Round': 'round',
        'PSW': 'odds_w_pinnacle',   # Pinnacle winner odds
        'PSL': 'odds_l_pinnacle',   # Pinnacle loser odds
        'B365W': 'odds_w_b365',     # Bet365 winner odds
        'B365L': 'odds_l_b365',
        'AvgW': 'odds_w_avg',       # Market average winner odds
        'AvgL': 'odds_l_avg',
    }
    
    # Rename columns that exist
    rename_dict = {k: v for k, v in col_map.items() if k in combined.columns}
    combined = combined.rename(columns=rename_dict)
    
    # Compute implied probabilities (remove overround using Pinnacle)
    combined = _compute_implied_probabilities(combined)
    
    print(f"📦 Loaded {len(combined):,} odds records ({year_start}-{year_end})")
    return combined

# ===== CELL 4: Implied Probability Calculation =====

def _compute_implied_probabilities(df):
    """
    Convert decimal odds to fair implied probabilities.
    
    Decimal odds include the bookmaker's overround (margin).
    We remove it to get "fair" probabilities that sum to 1.0.
    
    Uses Pinnacle odds (most efficient) with B365/Avg as fallback.
    """
    # Pick best available odds source
    for w_col, l_col, label in [
        ('odds_w_pinnacle', 'odds_l_pinnacle', 'Pinnacle'),
        ('odds_w_b365', 'odds_l_b365', 'Bet365'),
        ('odds_w_avg', 'odds_l_avg', 'Average'),
    ]:
        if w_col in df.columns and l_col in df.columns:
            # Raw implied probabilities (sum > 1 due to overround)
            raw_prob_w = 1.0 / pd.to_numeric(df[w_col], errors='coerce')
            raw_prob_l = 1.0 / pd.to_numeric(df[l_col], errors='coerce')
            
            # Remove overround: normalize so probabilities sum to 1
            overround = raw_prob_w + raw_prob_l
            df['implied_prob_w'] = raw_prob_w / overround
            df['implied_prob_l'] = raw_prob_l / overround
            df['odds_source'] = label
            df['overround'] = overround
            
            valid = df['implied_prob_w'].notna().sum()
            print(f"  Using {label} odds ({valid:,} valid records)")
            break
    else:
        df['implied_prob_w'] = float('nan')
        df['implied_prob_l'] = float('nan')
        df['odds_source'] = 'none'
    
    return df


# ===== CELL 5: Entrypoint =====

if __name__ == "__main__":
    fetch_odds_data()
    df = load_odds_data()
    if len(df) > 0:
        print(f"\nOdds source: {df['odds_source'].iloc[0]}")
        print(f"Avg implied prob winner: {df['implied_prob_w'].mean():.3f}")
        print(f"Avg overround: {df['overround'].mean():.3f}")
