# 🎾 TheOracle — ATP Tennis Match Prediction System

> Industry-grade ML system for predicting ATP tennis match outcomes using surface-aware ELO, stamina/momentum/pressure modeling, and per-surface XGBoost ensembles.

Inspired by [Green Code's YouTube series](https://www.youtube.com/watch?v=LkJpNLIaeVk) on tennis prediction with Random Forest and XGBoost, extended with sophisticated surface partitioning, mental/physical state modeling, and environmental context features.

## 🏗️ Architecture

```
[5 Data Sources] → [Data Platform] → [7 Feature Engines] → [3 XGBoost Models] → [Predictions]
```

- **Data**: JeffSackmann ATP + Challengers, tennis-data.co.uk odds, OpenMeteo weather
- **Features**: 7 specialized engines (ELO, Stamina, Pressure, Momentum, Surface Tactics, Environment, Odds)
- **Models**: Per-surface XGBoost (Clay, Grass, Hard) + ELO baseline benchmark
- **Output**: Match predictions, tournament simulations, SHAP explanations

## 🚀 Quick Start (Google Colab)

### Step 1: Setup

```python
# Install dependencies
!pip install -q xgboost scikit-learn pandas numpy matplotlib seaborn shap pyyaml requests beautifulsoup4 pyarrow tqdm openpyxl

# Clone repo
!git clone https://github.com/DEUSEXMACHINA-NANDEESHAK/theoracle.git

# Change directory
%cd theoracle
```

### Step 2: Download Data (~5 min)

```python
from ingestion.pipeline import run_full_ingestion
matches, players, odds = run_full_ingestion(skip_weather=True)
```

### Step 3: Build Features (~10 min)

```python
from features.build_features import build_feature_store
features = build_feature_store()
```

### Step 4: Train Models (~5 min with GPU)

```python
from models.train import train_all_models
models, results = train_all_models()
```

### Step 5: Evaluate

```python
from models.evaluate import full_evaluation
full_evaluation(plot=True)
```

### Step 6: Predict Tournament

```python
from models.tournament_sim import simulate_tournament
simulate_tournament("Wimbledon 2026", surface="Grass", n_simulations=5000)
```

## 📁 Project Structure

```
theoracle/
├── configs/           # YAML configurations (ELO params, XGBoost hyperparams)
├── ingestion/         # Data download and normalization
├── features/          # 7 feature engines + orchestrator
├── models/            # Training, evaluation, tournament simulation
├── tests/             # Leakage detection + ELO sanity tests
└── requirements.txt   # Python dependencies
```

## 🎯 Feature Engines

| Engine          | Features                                       | Key Signal            |
| --------------- | ---------------------------------------------- | --------------------- |
| **ELO**         | 7-track ratings, H2H, aging, margin-of-victory | Core skill estimation |
| **Stamina**     | Rest days, match density, fatigue index        | Physical state        |
| **Pressure**    | Break points, tiebreaks, comebacks             | Mental toughness      |
| **Momentum**    | Streaks, EWMA confidence, ELO trend            | Current form          |
| **Surface**     | Serve/return profiles, clay grind, grass serve | Play style fit        |
| **Environment** | Weather, altitude, home advantage, rankings    | Match context         |
| **Odds**        | Implied probabilities, market divergence       | Reference band        |

## ⚠️ Leakage Prevention

Every feature is computed using ONLY pre-match data. The strict protocol:

1. **Compute** features (pre-match state)
2. **Store** feature row
3. **Update** engine states (post-match result)

This is verified by `tests/test_leakage.py`.

## 📊 Data Sources

- [JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp) — Core match data (CC BY-NC-SA 4.0)
- [tennis-data.co.uk](http://www.tennis-data.co.uk) — Historical betting odds
- [OpenMeteo](https://open-meteo.com) — Weather data (free API)

## 📜 License

This project uses data from Jeff Sackmann's tennis_atp repository, licensed under CC BY-NC-SA 4.0. **Non-commercial use only.**
