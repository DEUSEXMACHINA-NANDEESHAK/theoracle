# ===== TheOracle: Weather Data Fetcher =====
# ===== CELL 1: Imports and Venue Mapping =====

import os, json, time
import pandas as pd
import requests
from tqdm import tqdm
from ingestion.fetch_sackmann import load_config

TOURNAMENT_VENUES = {
    'Australian Open': (-37.82, 144.98, 'Melbourne'),
    'Roland Garros': (48.85, 2.25, 'Paris'),
    'Wimbledon': (51.43, -0.21, 'London'),
    'US Open': (40.75, -73.85, 'New York'),
    'Indian Wells': (33.72, -116.31, 'Indian Wells'),
    'Miami': (25.71, -80.16, 'Miami'),
    'Monte Carlo': (43.75, 7.44, 'Monte Carlo'),
    'Madrid': (40.37, -3.69, 'Madrid'),
    'Rome': (41.93, 12.46, 'Rome'),
    'Montreal': (45.53, -73.64, 'Montreal'),
    'Cincinnati': (39.30, -84.32, 'Cincinnati'),
    'Shanghai': (31.04, 121.50, 'Shanghai'),
    'Dubai': (25.23, 55.32, 'Dubai'),
    'Barcelona': (41.39, 2.12, 'Barcelona'),
    'Hamburg': (53.57, 10.03, 'Hamburg'),
    'Tokyo': (35.70, 139.74, 'Tokyo'),
    'Basel': (47.54, 7.62, 'Basel'),
    'Vienna': (48.21, 16.36, 'Vienna'),
    'Halle': (52.06, 8.36, 'Halle'),
}

# ===== CELL 2: Weather Fetching =====

def fetch_weather_for_tournament(tourney_name, start_date, end_date, raw_dir="data/raw/weather"):
    """Fetch historical weather via OpenMeteo free API."""
    venue = _match_venue(tourney_name)
    if venue is None:
        return None
    lat, lon, city = venue
    cache_key = f"{city}_{start_date}_{end_date}".replace(" ", "_")
    cache_path = os.path.join(raw_dir, f"weather_{cache_key}.json")
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return pd.DataFrame(json.load(f))
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        'latitude': lat, 'longitude': lon,
        'start_date': start_date, 'end_date': end_date,
        'daily': 'temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_max',
        'timezone': 'auto',
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if 'daily' in data:
            weather_df = pd.DataFrame({
                'date': data['daily']['time'],
                'temperature_c': data['daily']['temperature_2m_mean'],
                'humidity_pct': data['daily'].get('relative_humidity_2m_mean'),
                'wind_speed_kmh': data['daily']['wind_speed_10m_max'],
                'city': city, 'tourney_name': tourney_name,
            })
            os.makedirs(raw_dir, exist_ok=True)
            weather_df.to_json(cache_path, orient='records')
            return weather_df
    except Exception as e:
        print(f"  [WARN] Weather fetch failed for {tourney_name}: {e}")
    return None

def _match_venue(tourney_name):
    """Match tournament name to venue coordinates."""
    name_lower = str(tourney_name).lower()
    for key, value in TOURNAMENT_VENUES.items():
        if key.lower() in name_lower or name_lower in key.lower():
            return value
    for key, value in TOURNAMENT_VENUES.items():
        if set(key.lower().split()) & set(name_lower.split()):
            return value
    return None

# ===== CELL 3: Bulk Weather Fetch =====

def fetch_weather_for_matches(matches_df, raw_dir="data/raw/weather"):
    """Fetch weather for all unique tournaments in match dataset."""
    os.makedirs(raw_dir, exist_ok=True)
    if 'tourney_date' not in matches_df.columns:
        print("⚠️  No tourney_date column. Skipping weather.")
        return pd.DataFrame()
    
    matches_df = matches_df.copy()
    matches_df['tourney_date_str'] = matches_df['tourney_date'].astype(str)
    tourneys = matches_df.groupby(['tourney_name', 'source_year']).agg(
        min_date=('tourney_date_str', 'min'),
    ).reset_index()
    
    print(f"🌤️  Fetching weather for {len(tourneys)} tournaments...")
    all_weather = []
    for _, row in tqdm(tourneys.iterrows(), total=len(tourneys), desc="Weather"):
        try:
            min_d = str(int(float(row['min_date'])))
            start = f"{min_d[:4]}-{min_d[4:6]}-{min_d[6:8]}"
            end = (pd.to_datetime(start) + pd.Timedelta(days=14)).strftime('%Y-%m-%d')
            w = fetch_weather_for_tournament(row['tourney_name'], start, end, raw_dir)
            if w is not None:
                all_weather.append(w)
        except Exception:
            continue
        time.sleep(0.1)
    
    if all_weather:
        return pd.concat(all_weather, ignore_index=True)
    return pd.DataFrame()
