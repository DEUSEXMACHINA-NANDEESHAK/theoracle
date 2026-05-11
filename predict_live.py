import json
import os
import argparse
import pandas as pd
from models.tournament_sim import TournamentSimulator
from ingestion.live_sync import fetch_and_build_draw

def run_live_prediction(url, name=None, simulations=5000):
    """
    Orchestrates the fetch -> sync -> predict flow for ANY tournament.
    """
    # 1. Generate filename from URL if name not provided
    if not name:
        name = url.split('/')[-3] if url.endswith('/') else url.split('/')[-2]
    
    draw_path = f"draws/{name.replace('-', '_')}.json"
    
    # 2. DOWNLOAD LATEST DRAW & STATUS
    print(f"\n📡 SYNCING: {name}...")
    fetch_and_build_draw(url, draw_path)
    
    # 3. Load updated data
    if not os.path.exists(draw_path):
        print(f"❌ Error: Could not create draw file at {draw_path}")
        return

    with open(draw_path, 'r') as f:
        draw_data = json.load(f)

    print("\n" + "="*70)
    print(f"🔮 THE ORACLE: {draw_data['tournament'].upper()}")
    print(f"   Surface: {draw_data['surface']} | Status: {draw_data.get('last_updated', 'Live')}")
    print("="*70)

    # 4. Run Simulation
    sim = TournamentSimulator()
    results = sim.simulate_tournament(
        draw_data['players'], 
        surface=draw_data['surface'], 
        n_simulations=simulations
    )
    
    print(f"\n✅ Prediction complete for {draw_data['tournament']}")
    print(f"📂 Data saved to: {draw_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TheOracle: Universal Tournament Predictor")
    parser.add_argument("--url", help="TennisExplorer tournament URL")
    parser.add_argument("--query", help="Name of the tournament to search for (e.g. 'Rome Masters')")
    parser.add_argument("--name", help="Optional name for the tournament file")
    parser.add_argument("--sims", type=int, default=5000, help="Number of simulations")
    
    args = parser.parse_args()
    
    target_url = args.url
    
    # If query is provided, search for the URL
    if args.query:
        from ingestion.live_sync import search_tournament_url
        target_url = search_tournament_url(args.query)
        if not target_url:
            print(f"❌ Could not find a live tournament matching '{args.query}'")
            print("💡 Try providing the direct --url instead.")
            exit(1)
            
    if not target_url:
        print("❌ Error: You must provide either --url or --query")
        parser.print_help()
        exit(1)
        
    run_live_prediction(target_url, args.name, args.sims)
