# ===== TheOracle: Jeff Sackmann ATP Data Fetcher =====
# ===== CELL 1: Imports and Configuration =====

import os
import hashlib
import requests
import pandas as pd
from tqdm import tqdm
import yaml
import time

def load_config(config_path="configs/data_sources.yaml"):
    """Load data source configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# ===== CELL 2: Download Utilities =====

def download_file(url, dest_path, retries=3, timeout=30):
    """
    Download a file from URL to local path with retry logic.
    Returns True if file was downloaded/updated, False if unchanged.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    # Check if file exists and compute checksum
    existing_hash = None
    if os.path.exists(dest_path):
        with open(dest_path, 'rb') as f:
            existing_hash = hashlib.md5(f.read()).hexdigest()
    
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 404:
                print(f"  [SKIP] Not found: {os.path.basename(dest_path)}")
                return False
            response.raise_for_status()
            
            # Check if content changed
            new_hash = hashlib.md5(response.content).hexdigest()
            if new_hash == existing_hash:
                return False  # No change
            
            with open(dest_path, 'wb') as f:
                f.write(response.content)
            return True
            
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"  [ERROR] Failed to download {url}: {e}")
                return False

# ===== CELL 3: Fetch ATP Match Data =====

def fetch_atp_matches(config=None, raw_dir="data/raw/sackmann"):
    """
    Download all ATP match CSVs from Jeff Sackmann's GitHub repository.
    
    Downloads:
    - atp_matches_YYYY.csv (2000-2026) — main tour matches
    - atp_players.csv — player biographical data
    - atp_rankings_current.csv — latest rankings
    
    Years 2000-2004 are for ELO warm-up only (not used in training).
    Years 2005+ are the actual data for training and evaluation.
    """
    if config is None:
        config = load_config()
    
    sackmann = config['sackmann']
    base_url = sackmann['base_url']
    os.makedirs(raw_dir, exist_ok=True)
    
    # ---- Download player biographical data ----
    print("📋 Downloading player data...")
    players_url = f"{base_url}/{sackmann['players_file']}"
    download_file(players_url, os.path.join(raw_dir, sackmann['players_file']))
    
    # ---- Download rankings ----
    print("📊 Downloading rankings...")
    rankings_url = f"{base_url}/{sackmann['rankings_current']}"
    download_file(rankings_url, os.path.join(raw_dir, sackmann['rankings_current']))
    
    # Also download decade rankings for historical data
    for decade in ['00s', '10s', '20s']:
        fname = f"atp_rankings_{decade}.csv"
        url = f"{base_url}/{fname}"
        download_file(url, os.path.join(raw_dir, fname))
    
    # ---- Download match data (2000-2026) ----
    warmup_start = min(sackmann['warmup_years'])
    data_end = sackmann['data_years_end']
    
    print(f"🎾 Downloading ATP match data ({warmup_start}-{data_end})...")
    downloaded = 0
    skipped = 0
    
    for year in tqdm(range(warmup_start, data_end + 1), desc="ATP Matches"):
        fname = sackmann['matches_pattern'].format(year=year)
        url = f"{base_url}/{fname}"
        dest = os.path.join(raw_dir, fname)
        
        if download_file(url, dest):
            downloaded += 1
        else:
            skipped += 1
    
    print(f"✅ ATP matches: {downloaded} downloaded, {skipped} skipped/unchanged")
    return raw_dir

# ===== CELL 4: Load Downloaded Data =====

def load_atp_matches(raw_dir="data/raw/sackmann", year_start=2000, year_end=2026):
    """
    Load and concatenate all downloaded ATP match CSVs into a single DataFrame.
    
    Returns:
        pd.DataFrame with all matches from year_start to year_end
    """
    all_dfs = []
    
    for year in range(year_start, year_end + 1):
        fpath = os.path.join(raw_dir, f"atp_matches_{year}.csv")
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath, low_memory=False)
                df['source'] = 'atp'
                df['source_year'] = year
                all_dfs.append(df)
            except Exception as e:
                print(f"  [WARN] Error reading {fpath}: {e}")
    
    if not all_dfs:
        raise FileNotFoundError(f"No ATP match files found in {raw_dir}")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"📦 Loaded {len(combined):,} ATP matches ({year_start}-{year_end})")
    return combined

def load_players(raw_dir="data/raw/sackmann"):
    """Load player biographical data."""
    fpath = os.path.join(raw_dir, "atp_players.csv")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Player file not found: {fpath}")
    
    players = pd.read_csv(fpath, low_memory=False)
    
    # Handle different column name variations for birth date
    for col in ['birthdate', 'dob', 'birth_date']:
        if col in players.columns:
            players = players.rename(columns={col: 'birth_date'})
            break
    
    if 'birth_date' in players.columns:
        players['birth_date'] = pd.to_numeric(players['birth_date'], errors='coerce')
    
    print(f"📦 Loaded {len(players):,} players")
    return players


# ===== CELL 5: Entrypoint =====

if __name__ == "__main__":
    config = load_config()
    fetch_atp_matches(config)
    df = load_atp_matches()
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSurface distribution:\n{df['surface'].value_counts()}")
    print(f"\nTourney level distribution:\n{df['tourney_level'].value_counts()}")
