import requests
from bs4 import BeautifulSoup
import json
import os
import re
from datetime import datetime

def search_tournament_url(tournament_name):
    """
    Searches TennisExplorer for a tournament by name and returns its URL.
    """
    search_query = tournament_name.replace(" ", "+")
    # TennisExplorer's internal search often requires specific cookies, 
    # so we'll use a direct name-matching approach on their list page or 
    # try to guess the URL pattern which is very consistent.
    
    year = datetime.now().year
    slug = tournament_name.lower().replace(" ", "-").replace("masters", "").strip("-")
    
    # Common variations
    candidates = [
        f"https://www.tennisexplorer.com/{slug}-masters/{year}/atp-men/",
        f"https://www.tennisexplorer.com/{slug}/{year}/atp-men/",
        f"https://www.tennisexplorer.com/{slug}-open/{year}/atp-men/",
    ]
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    print(f"🔍 Searching for '{tournament_name}'...")
    
    for url in candidates:
        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                print(f"✨ Found tournament URL: {url}")
                return url
        except:
            continue
            
    # If guessing fails, try to scrape the "This week's tournaments" section
    try:
        home_url = "https://www.tennisexplorer.com/"
        res = requests.get(home_url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        links = soup.select('div.box.shadow.box-gray a') # Matches in the sidebar
        for link in links:
            if slug in link.get_text().lower():
                found_url = "https://www.tennisexplorer.com" + link['href']
                print(f"✨ Found in live sidebar: {found_url}")
                return found_url
    except:
        pass

    return None

def fetch_and_build_draw(tournament_url, output_path=None):
    """
    Existing logic to scrape the draw from a URL.
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(tournament_url, headers=headers, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ HTTP Error: {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    
    title = soup.find('h1')
    tourney_name = title.get_text().strip() if title else "Tournament"
    # 2. Find the Main Results Table (Ignore sidebars/news)
    players_found = {}
    
    # Target only the results table or draw container
    main_container = soup.select_one('div#tournament-tab-draw, table.result, div.box-main')
    if not main_container:
        main_container = soup # Fallback to whole page
    
    # News and other tournament keywords to ignore
    blacklist = ['challenger', 'itf', 'utr', 'wta', 'masters', 'open', 'today', 'rankings', 'results', 'news', 'story']
    
    # Specific selectors for player links
    player_links = main_container.select('td.rtxt a, td.ltxt a')
    
    for link in player_links:
        name = link.get_text().strip()
        
        # 1. Basic length check
        if not name or len(name) < 4: continue
        
        # 2. Filter out other tournaments and headlines (very aggressive)
        lower_name = name.lower()
        if any(word in lower_name for word in blacklist): continue
        if len(name.split()) > 2: continue # Most player names are "Lastname F." or "First Last"
        
        # 3. Skip names that contain numbers (dates/years)
        if any(char.isdigit() for char in name): continue
        
        if name not in players_found:
            players_found[name] = {"name": name, "seed": None, "status": "IN"}

    if not players_found:
        print("⚠️  Warning: No players found in main container. Check site structure.")

    # Update logic (simplified for speed)
    content = soup.get_text()
    for name in players_found:
        last_name = name.split()[-1]
        if f"{last_name} lost" in content or "defeated by" in content.lower():
            players_found[name]['status'] = 'OUT'

    data = {
        "tournament": tourney_name,
        "surface": "Clay" if "rome" in tournament_url.lower() or "garros" in tournament_url.lower() else "Hard",
        "last_updated": str(datetime.now()),
        "players": list(players_found.values())
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return data

if __name__ == "__main__":
    url = search_tournament_url("Rome Masters")
    if url:
        fetch_and_build_draw(url)
