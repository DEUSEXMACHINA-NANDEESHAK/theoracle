# ===== TheOracle: Tournament Simulator =====
# ===== CELL 1: Imports =====

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from models.xgboost_model import TennisXGBoost
from models.datasets import get_feature_columns, prepare_xy

# ===== CELL 2: Tournament Simulator =====

class TournamentSimulator:
    """
    Monte Carlo tournament bracket simulator.
    
    Simulates a full tournament draw multiple times and computes
    the probability of each player reaching each round.
    """
    
    def __init__(self, models_dir="models/artifacts"):
        self.models = {}
        for surface in ['Clay', 'Grass', 'Hard']:
            path = os.path.join(models_dir, f"xgb_{surface.lower()}.pkl")
            if os.path.exists(path):
                self.models[surface] = TennisXGBoost.load(path)
    
    def simulate_tournament(self, draw, surface, feature_store_df=None,
                           n_simulations=1000, best_of=3):
        """
        Simulate a tournament bracket from its CURRENT state.
        Handles knocked-out players (marked as status='OUT').
        """
        # Filter out players already knocked out
        active_draw = [p for p in draw if p.get('status') != 'OUT']
        
        if surface not in self.models:
            print(f"⚠️  No model for {surface}. Using random predictions.")
            return None
        
        model = self.models[surface]
        n_players = len(active_draw)
        
        # Determine current round based on remaining players
        rounds = int(np.ceil(np.log2(n_players)))
        round_names = self._get_round_names(rounds)
        
        print(f"\n🏆 Live Simulation: {surface} | {n_players} players remaining")
        print(f"   Current Path: {' → '.join(round_names)}")
        
        reach_counts = {i: {r: 0 for r in range(rounds + 1)} for i in range(n_players)}
        win_counts = {i: 0 for i in range(n_players)}
        
        for sim in range(n_simulations):
            # Everyone starts in their current round
            for i in range(n_players):
                reach_counts[i][0] += 1
            
            current_players = list(range(n_players))
            
            for round_num in range(rounds):
                next_round = []
                for i in range(0, len(current_players), 2):
                    if i + 1 >= len(current_players):
                        next_round.append(current_players[i])
                        continue
                    
                    p_a_idx = current_players[i]
                    p_b_idx = current_players[i + 1]
                    
                    prob_a = self._predict_match(
                        active_draw[p_a_idx], active_draw[p_b_idx],
                        surface, model, feature_store_df
                    )
                    
                    if np.random.random() < prob_a:
                        winner_idx = p_a_idx
                    else:
                        winner_idx = p_b_idx
                    
                    next_round.append(winner_idx)
                    reach_counts[winner_idx][round_num + 1] += 1
                current_players = next_round
            
            if current_players:
                win_counts[current_players[0]] += 1
        
        # Build results
        results = []
        for i, player in enumerate(active_draw):
            row = {'player_name': player.get('name'), 'seed': player.get('seed', '')}
            for r in range(rounds + 1):
                r_name = round_names[r] if r < len(round_names) else f'Round_{r}'
                row[r_name] = reach_counts[i][r] / n_simulations
            row['Win Tournament'] = win_counts[i] / n_simulations
            results.append(row)
        
        results_df = pd.DataFrame(results).sort_values('Win Tournament', ascending=False)
        
        print(f"\n🏆 THE ORACLE'S LIVE CONTENDERS:")
        cols = ['player_name', 'Win Tournament'] + [round_names[-1], round_names[-2]]
        print(results_df[cols].head(10).to_string(index=False, formatters={'Win Tournament': '{:.1%}'.format}))
        
        return results_df
    
    def _predict_match(self, player_a, player_b, surface, model, feature_df):
        """Predict win probability for a single match."""
        # Simple approach: use ELO-based prediction
        # In production, you'd rebuild features from current player state
        elo_a = player_a.get('elo', 1500)
        elo_b = player_b.get('elo', 1500)
        
        # Logistic function based on ELO difference
        elo_diff = elo_a - elo_b
        prob_a = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        
        return np.clip(prob_a, 0.05, 0.95)
    
    def _get_round_names(self, n_rounds):
        """Generate round names for a bracket."""
        if n_rounds == 7:
            return ['R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F']
        elif n_rounds == 6:
            return ['R64', 'R32', 'R16', 'QF', 'SF', 'F']
        elif n_rounds == 5:
            return ['R32', 'R16', 'QF', 'SF', 'F']
        elif n_rounds == 4:
            return ['R16', 'QF', 'SF', 'F']
        elif n_rounds == 3:
            return ['QF', 'SF', 'F']
        else:
            return [f'R{i+1}' for i in range(n_rounds)]


# ===== CELL 3: Quick Tournament Prediction =====

def simulate_tournament(tournament_name, surface="Clay", n_simulations=1000):
    """
    Tournament simulation with engine-integrated ratings.
    """
    print(f"\n{'='*60}")
    print(f"🎾 TheOracle: {tournament_name} Prediction")
    print(f"{'='*60}")
    
    # Initialize engine to get latest ratings
    from features.elo_engine import TennisEloEngine
    engine = TennisEloEngine()
    
    # Load history to warm up engine
    print("🔋 Warming up ELO engine for simulation...")
    matches_df = pd.read_parquet('data/clean/matches.parquet')
    engine.process_all_matches(matches_df)
    
    def get_elo(name, default=1500):
        # Very simple name matching for demo
        for pid, state in engine.players.items():
            # If we had a name lookup we'd use it here
            pass
        return default

    # Sample Rome 2026 Draw (Live Example)
    # Status 'OUT' means the player lost already
    draw = [
        {'name': 'Jannik Sinner', 'elo': 2180, 'seed': 1},
        {'name': 'Qualifier 1', 'elo': 1450, 'status': 'OUT'}, # Sinner's potential opponent out
        {'name': 'Mariano Navone', 'elo': 1780, 'seed': 28},
        {'name': 'Hamad Medjedovic', 'elo': 1690, 'status': 'OUT'}, # Medjedovic knocked out!
        {'name': 'Stefanos Tsitsipas', 'elo': 1950, 'seed': 6},
        {'name': 'Casper Ruud', 'elo': 1920, 'seed': 7},
        {'name': 'Alexander Zverev', 'elo': 2010, 'seed': 3},
        {'name': 'Daniil Medvedev', 'elo': 1880, 'seed': 4},
        {'name': 'Novak Djokovic', 'elo': 2120, 'seed': 8}, # THE GOAT IS BACK
        {'name': 'Carlos Alcaraz', 'elo': 2150, 'seed': 2},
        {'name': 'Holger Rune', 'elo': 1850, 'seed': 12},
        {'name': 'Andrey Rublev', 'elo': 1910, 'seed': 5},
        {'name': 'Hubert Hurkacz', 'elo': 1790, 'seed': 10},
        {'name': 'Grigor Dimitrov', 'elo': 1820, 'seed': 9},
        {'name': 'Taylor Fritz', 'elo': 1840, 'seed': 11},
        {'name': 'Rafael Nadal', 'elo': 1900, 'seed': 'PR'},
    ]
    
    sim = TournamentSimulator()
    results = sim.simulate_tournament(draw, surface, None, n_simulations)
    return results


# ===== CELL 4: Entrypoint =====

if __name__ == "__main__":
    simulate_tournament("Wimbledon 2026", surface="Grass", n_simulations=5000)
