# ===== TheOracle: Challenger & Qualifier Data Fetcher =====
# ===== CELL 1: Imports =====

import os
import pandas as pd
from tqdm import tqdm
from ingestion.fetch_sackmann import load_config, download_file

# ===== CELL 2: Fetch Challenger/Qualifier Match Data =====

def fetch_challenger_matches(config=None, raw_dir="data/raw/sackmann"):
    """
    Download ATP Challenger and Qualifying match CSVs.
    
    These are ESSENTIAL for:
    - Building profiles on new/upcoming players who haven't played many ATP matches
    - Giving the model knowledge about qualifiers entering main draws
    - Green Code specifically improved his model by adding this data
    
    Stats available from 2008 onwards for challengers.
    """
    if config is None:
        config = load_config()
    
    sackmann = config['sackmann']
    base_url = sackmann['base_url']
    os.makedirs(raw_dir, exist_ok=True)
    
    warmup_start = min(sackmann['warmup_years'])
    data_end = sackmann['data_years_end']
    
    print(f"🏋️ Downloading Challenger/Qualifier data ({warmup_start}-{data_end})...")
    downloaded = 0
    
    for year in tqdm(range(warmup_start, data_end + 1), desc="Challengers"):
        fname = sackmann['qual_chall_pattern'].format(year=year)
        url = f"{base_url}/{fname}"
        dest = os.path.join(raw_dir, fname)
        
        if download_file(url, dest):
            downloaded += 1
    
    print(f"✅ Challenger data: {downloaded} files downloaded")
    return raw_dir

# ===== CELL 3: Load Challenger Data =====

def load_challenger_matches(raw_dir="data/raw/sackmann", year_start=2000, year_end=2026):
    """
    Load all challenger/qualifier match CSVs.
    
    Returns:
        pd.DataFrame with challenger matches, tagged with source='challenger'
    """
    all_dfs = []
    
    for year in range(year_start, year_end + 1):
        fpath = os.path.join(raw_dir, f"atp_matches_qual_chall_{year}.csv")
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath, low_memory=False)
                df['source'] = 'challenger'
                df['source_year'] = year
                all_dfs.append(df)
            except Exception as e:
                print(f"  [WARN] Error reading {fpath}: {e}")
    
    if not all_dfs:
        print("⚠️  No challenger files found. This is okay for initial setup.")
        return pd.DataFrame()
    
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"📦 Loaded {len(combined):,} challenger/qualifier matches ({year_start}-{year_end})")
    return combined


# ===== CELL 4: Entrypoint =====

if __name__ == "__main__":
    fetch_challenger_matches()
    df = load_challenger_matches()
    if len(df) > 0:
        print(f"\nSurface distribution:\n{df['surface'].value_counts()}")
        print(f"\nTourney level distribution:\n{df['tourney_level'].value_counts()}")
