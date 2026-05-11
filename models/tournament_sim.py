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
    
    def simulate_tournament(self, draw, surface, feature_store_df,
                           n_simulations=1000, best_of=3):
        """
        Simulate a tournament bracket.
        
        Args:
            draw: List of player dicts with at least 'player_id' and 'name'
                  Order matters — adjacent players are paired.
            surface: 'Clay', 'Grass', or 'Hard'
            feature_store_df: Feature store DataFrame for building features
            n_simulations: Number of Monte Carlo simulations
            best_of: 3 or 5 (Grand Slams = 5)
        
        Returns:
            DataFrame with player probabilities of reaching each round
        """
        if surface not in self.models:
            print(f"⚠️  No model for {surface}. Using random predictions.")
            return None
        
        model = self.models[surface]
        n_players = len(draw)
        rounds = int(np.log2(n_players))
        round_names = self._get_round_names(rounds)
        
        print(f"\n🏆 Simulating tournament ({surface}, {n_players} players, {n_simulations} sims)")
        print(f"   Rounds: {' → '.join(round_names)}")
        
        # Track how often each player reaches each round
        # player_index → round → count
        reach_counts = {i: {r: 0 for r in range(rounds + 1)} for i in range(n_players)}
        win_counts = {i: 0 for i in range(n_players)}
        
        for sim in range(n_simulations):
            # Everyone starts in R1
            for i in range(n_players):
                reach_counts[i][0] += 1
            
            # Simulate bracket
            current_players = list(range(n_players))
            
            for round_num in range(rounds):
                next_round = []
                
                for i in range(0, len(current_players), 2):
                    if i + 1 >= len(current_players):
                        next_round.append(current_players[i])
                        continue
                    
                    p_a_idx = current_players[i]
                    p_b_idx = current_players[i + 1]
                    
                    # Get win probability
                    prob_a = self._predict_match(
                        draw[p_a_idx], draw[p_b_idx],
                        surface, model, feature_store_df
                    )
                    
                    # Simulate outcome
                    if np.random.random() < prob_a:
                        winner_idx = p_a_idx
                    else:
                        winner_idx = p_b_idx
                    
                    next_round.append(winner_idx)
                    reach_counts[winner_idx][round_num + 1] += 1
                
                current_players = next_round
            
            # Tournament winner
            if current_players:
                win_counts[current_players[0]] += 1
        
        # Build results DataFrame
        results = []
        for i, player in enumerate(draw):
            row = {
                'player_name': player.get('name', f'Player_{i}'),
                'seed': player.get('seed', ''),
            }
            for r in range(rounds + 1):
                round_name = round_names[r] if r < len(round_names) else f'R{r}'
                row[round_name] = reach_counts[i][r] / n_simulations
            
            row['Win Tournament'] = win_counts[i] / n_simulations
            results.append(row)
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Win Tournament', ascending=False)
        
        # Print top contenders
        print(f"\n🏆 Top 10 Predicted Winners:")
        print(f"{'Player':25s} {'Win %':>8s} {'Final':>8s} {'SF':>8s} {'QF':>8s}")
        print(f"{'─'*60}")
        for _, row in results_df.head(10).iterrows():
            name = row['player_name'][:24]
            win = row.get('Win Tournament', 0)
            final = row.get('F', row.get('Final', 0))
            sf = row.get('SF', 0)
            qf = row.get('QF', 0)
            print(f"{name:25s} {win:7.1%} {final:7.1%} {sf:7.1%} {qf:7.1%}")
        
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

def simulate_tournament(tournament_name, surface="Grass", n_simulations=1000):
    """
    Quick tournament simulation with sample draw.
    
    For demo purposes — in production you'd feed the actual draw.
    """
    print(f"\n{'='*60}")
    print(f"🎾 TheOracle: {tournament_name} Prediction")
    print(f"{'='*60}")
    
    # Sample top players (you would replace with actual draw)
    draw = [
        {'name': 'Jannik Sinner', 'elo': 2150, 'seed': 1},
        {'name': 'Qualifier 1', 'elo': 1400, 'seed': ''},
        {'name': 'Hubert Hurkacz', 'elo': 1800, 'seed': 16},
        {'name': 'Felix Auger-Aliassime', 'elo': 1750, 'seed': 21},
        {'name': 'Stefanos Tsitsipas', 'elo': 1850, 'seed': 11},
        {'name': 'Frances Tiafoe', 'elo': 1700, 'seed': 23},
        {'name': 'Ben Shelton', 'elo': 1780, 'seed': 14},
        {'name': 'Alexander Zverev', 'elo': 2050, 'seed': 4},
        {'name': 'Andrey Rublev', 'elo': 1900, 'seed': 6},
        {'name': 'Tommy Paul', 'elo': 1770, 'seed': 12},
        {'name': 'Holger Rune', 'elo': 1800, 'seed': 15},
        {'name': 'Casper Ruud', 'elo': 1820, 'seed': 7},
        {'name': 'Taylor Fritz', 'elo': 1850, 'seed': 10},
        {'name': 'Grigor Dimitrov', 'elo': 1780, 'seed': 9},
        {'name': 'Daniil Medvedev', 'elo': 2000, 'seed': 5},
        {'name': 'Carlos Alcaraz', 'elo': 2130, 'seed': 2},
    ]
    
    # Pad to power of 2 if needed
    while len(draw) & (len(draw) - 1) != 0:
        draw.append({'name': f'Bye {len(draw)}', 'elo': 1200, 'seed': ''})
    
    sim = TournamentSimulator()
    results = sim.simulate_tournament(draw, surface, None, n_simulations)
    return results


# ===== CELL 4: Entrypoint =====

if __name__ == "__main__":
    simulate_tournament("Wimbledon 2026", surface="Grass", n_simulations=5000)
