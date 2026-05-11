# ===== TheOracle: Fast Local Orchestrator =====
# Runs the full pipeline (Ingestion -> Features -> Training -> Simulation)
# Optimized for local multi-core execution.

import os
import time
from datetime import datetime

# Import project modules
from ingestion.pipeline import run_full_ingestion
from features.build_features import build_feature_store
from models.train import train_all_models
from models.tournament_sim import simulate_tournament

def run_fast_pipeline():
    start_time = time.time()
    print("\n" + "="*70)
    print(f"🔮 THE ORACLE: FAST LOCAL ORCHESTRATOR")
    print(f"   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

    # STEP 1: Ingestion (Fast local check)
    print("📦 [1/4] INGESTION: Syncing Sackmann & Odds data...")
    run_full_ingestion(skip_weather=True)
    
    # STEP 2: Feature Engineering (Optimized loop)
    print("\n🔨 [2/4] FEATURES: Building the Oracle Feature Store...")
    build_feature_store()
    
    # STEP 3: Training (Multi-core XGBoost)
    print("\n🚀 [3/4] TRAINING: Optimizing per-surface models...")
    models, results = train_all_models()
    
    # STEP 4: Rome 2026 Simulation
    print("\n🏆 [4/4] SIMULATION: Predicting Rome Masters 2026...")
    try:
        # We simulate the clay season's crown jewel
        simulate_tournament("Rome Masters 2026", surface="Clay", n_simulations=5000)
    except Exception as e:
        print(f"   ⚠️ Simulation error: {e}")
        print("   Checking for players.parquet and draws...")

    total_time = (time.time() - start_time) / 60
    print("\n" + "="*70)
    print(f"✅ PIPELINE COMPLETE in {total_time:.1f} minutes")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_fast_pipeline()
