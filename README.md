# NBA Game Win Probability Prediction
A machine learning project that predicts NBA game outcomes using team stats and recent performance.

## About this project
I started this mainly because I’ve always wanted to try building a sports forecasting model. As a basketball fan, this is first and foremost a passion project of mine.

The model looks at things like:
- How teams have been playing recently (last 10 games)
- Team efficiency stats (offensive/defensive ratings)
- Basic box score stats (shooting percentages, rebounds, assists)
- Game context (home court, rest days, back-to-backs)

**Training Data:** 2021-22, 2022-23 and 2023-24 NBA seasons
**Test Data:** 2024-25 season. Possibly several games of 2025-26 season up for implementation.

**Goal:** Achieve 65-70% accuracy with proper probability calibration (Brier Score < 0.24)

## How to start

### Prerequisites
- Python 3.8+
- pip

### How to install
```bash
git clone https://github.com/YOUR_USERNAME/nba-game-prediction.git
cd nba-game-prediction
pip install -r requirements.txt
```

### Run Demo
```bash
python demo.py
```

**By the end you should see something like:**
```
 Model Comparison
                     Accuracy     Brier
Logistic Regression  0.693061  0.200723
Random Forest        0.691429  0.204916
XGBoost              0.652245  0.224619
```

## How It Works

### Data Collection
Training: 2021-22 through 2023-24 seasons (~3,700 games)

Testing: 2024-25 season games that have already been played (~1,200 games)

I used real games that already happened so we can properly test the model's accuracy.

### What the Model Looks At

**Rolling Statistics (10-game window):**
For each team, calculate rolling averages of:
- Points (PTS)
- Field Goal % (FG_PCT)
- 3-Point % (FG3_PCT)
- Free Throw % (FT_PCT)
- Rebounds (REB)
- Assists (AST)
- Steals (STL), Blocks (BLK), Turnovers (TOV)

**Contextual Features:**
- Home court indicator (IS_HOME)
- Rest days since last game (HOME_REST_DAYS, AWAY_REST_DAYS)
- Back-to-back game indicator (BACK_TO_BACK)

**Differential Features:**
- DIFF_PTS_L10 = HOME_PTS_L10 - AWAY_PTS_L10
- DIFF_FG_PCT_L10, DIFF_REB_L10, etc.
- Captures relative team strength

### Models

I decided to use three following models for comparison:

- Logistic Regression - Simple but effective baseline

- Random Forest - Handles complex patterns well

- XGBoost - Usually the best performer for this kind of data

Turns out they all perform similarly (around 65-70% accuracy), which is actually pretty good.

### Evaluation
**Primary Metric: Brier Score**
- Measures probability calibration
- Formula: Mean((predicted_prob - actual_outcome)²)
- Lower is better (0 = perfect, 0.25 = random)
- **Target:** < 0.24

**Secondary Metric: Accuracy**
- Percentage of games predicted correctly
- **Target:** > 65%

**Why Brier Score?**
- We predict probabilities, not just win/loss
- Penalizes overconfident wrong predictions
- Standard metric in probabilistic forecasting

### Key Insights
- Home court advantage: ~3-4 percentage points
- Rolling averages (10 games) optimal window
- Differential features most predictive
- Back-to-back games: ~5% performance drop

## For Developers
If you want to run the full pipeline:
```bash
# Collect fresh data from NBA API
python src/data_collection.py

# Create features from raw data
python src/feature_engineering.py

# Train all models
python src/train.py

# Compare model performance
python src/evaluate.py
```

**Note:** Pre-trained models and processed data are included in the repository. You can skip directly to evaluation with `python demo.py`.

## Data Sources

**NBA Official API** (via `nba_api` Python package)
- Game results and box scores
- Team statistics  
- Player participation

**Data Availability:**
- All data is publicly available
- No authentication required
- API rate limits: ~1-2 requests per second (respected in code)

**Date Ranges:**
- Training: October 2021 - April 2024
- Test: October 2024 - January 2025

## About the Code
This was built for my Intro to AI class. The code is intentionally kept simple and readable rather than overly optimized. I used:
- pandas for data processing
- scikit-learn for machine learning
- nba_api to get game data
- Standard Python libraries for everything else

## Limitations & Future Work
Right now the model is pretty basic. Some things I'd like to improve:
- Add player-level data (injuries, star players)
- Include betting lines as features
- Try neural networks
- Make predictions for future games
- Try to also include playoffs?
- Be able to run a playoff simulations

## License
This is a student project - feel free to use the code for learning purposes!

## Author
**Aziz Umarbaev**
- Course: COM-214 - Introduction to Artificial Intelligence
- Github: github.com/Choppy314
