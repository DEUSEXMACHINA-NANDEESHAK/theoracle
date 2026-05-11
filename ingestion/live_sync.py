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
    # 2. Find the Draw Table
    players_found = {} # name -> {seed, status}
    
    # News keyword blacklist to avoid headlines in the player list
    blacklist = ['says', 'defeat', 'victory', 'set up', 'showdown', 'game', 'win', 'highlights', 'passes away', 'news', 'story']
    
    # Try multiple selectors to find player links
    selectors = ['td.rtxt a', 'td.ltxt a', 'div.draw-table a']
    
    for selector in selectors:
        player_links = soup.select(selector)
        for link in player_links:
            name = link.get_text().strip()
            
            # CLEANING & FILTERING
            # 1. Skip tiny strings
            if not name or len(name) < 4: continue
            # 2. Skip long headlines (more than 3 words)
            if len(name.split()) > 3: continue
            # 3. Skip blacklist items
            if any(word in name.lower() for word in blacklist): continue
            # 4. Skip generic tournament names
            if any(x in name.lower() for x in ['masters', 'open', 'atp', 'wta']): continue
            
            # Format usually "Lastname F." or "Firstname Lastname"
            if name not in players_found:
                players_found[name] = {"name": name, "seed": None, "status": "IN"}

    if not players_found:
        print("⚠️  Warning: Scraper found 0 players. Check site structure.")

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
