import json
import os
import pandas as pd
from models.tournament_sim import TournamentSimulator
from ingestion.live_sync import sync_rome_draw

def run_live_prediction(draw_path="draws/rome_2026.json"):
    # 1. Sync with real world
    sync_rome_draw(draw_path)
    
    # 2. Load updated data
    if not os.path.exists(draw_path):
        print(f"❌ Draw file not found: {draw_path}")
        return

    with open(draw_path, 'r') as f:
        draw_data = json.load(f)

    print("\n" + "="*70)
    print(f"🔮 THE ORACLE: LIVE PREDICTION - {draw_data['tournament']}")
    print(f"   Status: {draw_data['current_round']}")
    print("="*70)

    # Filter out knocked out players
    active_players = [p for p in draw_data['players'] if p['status'] == 'IN']
    out_players = [p for p in draw_data['players'] if p['status'] == 'OUT']

    print(f"✅ Active Players: {len(active_players)}")
    print(f"❌ Knocked Out:   {len(out_players)}")
    if out_players:
        print(f"   ({', '.join([p['name'] for p in out_players])})")

    # Run Simulation
    sim = TournamentSimulator()
    
    # We pass the full draw but the simulator now handles status='OUT' internally
    # as per my recent update to models/tournament_sim.py
    results = sim.simulate_tournament(
        draw_data['players'], 
        surface=draw_data['surface'], 
        n_simulations=5000
    )

    print("\n💡 Tip: Update the 'status' in " + draw_path + " to 'OUT' as players lose!")

if __name__ == "__main__":
    run_live_prediction()
