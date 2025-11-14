# NBA Game Win Probability Prediction

**Machine Learning project predicting NBA game outcomes using team statistics and player impact metrics.**

## Project Overview
This is first of all a passion project of mine as I have always wanted to try doing something like this. This project uses machine learning to predict NBA game winners based on:
- Team efficiency metrics (Offensive/Defensive Rating, Pace, etc...)
- Rolling averages (last 10 games performance)
- Advanced statistics (Field Goal %, 3-Point %, Rebounds, Assists, etc.)
- Contextual factors (home court advantage, rest days, back-to-back games)
- Differential features (home team stats minus away team stats)

**Training Data:** 2022-23 and 2023-24 NBA seasons (~2,460 complete games)  
**Test Data:** 2024-25 season (October 2024 - January 2025, ~400-500 games already played). Possibly several games of 2025-26 season.

**Goal:** Achieve 65-70% accuracy with proper probability calibration (Brier Score < 0.24)

## How to start

### Prerequisites
- Python 3.8+
- pip

### How to install
```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/nba-game-prediction.git
cd nba-game-prediction

# 2. Install dependencies
pip install -r requirements.txt
```

### Run Demo
```bash
# Quick evaluation using pre-trained model
python demo.py
```

**Expected Output:**
```
NBA GAME PREDICTION - EVALUATION 
Loading test data...
✓ Loaded 412 test games from 2024-25 season

Loading pre-trained model...
✓ Model loaded 

Evaluating on test set...

RESULTS: 
Accuracy:     67.23%
Brier Score:  0.2289

Test Period: October 2024 - January 2025
Evaluation complete!
```

## How It Works

### Data Collection
**Training Seasons (Complete):**
- 2022-23 season: 1,230 games
- 2023-24 season: 1,230 games
- **Total training data:** ~2,460 games

**Test Season (Games Already Played):**
- 2024-25 season: October 2024 - January 2025
- **Test data:** ~400-500 games (varies by collection date)

**Why this split?**
- Works in a similar real-world prediction scenario

### Feature Engineering

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

**Total Features:** ~30

### Models

I decided to use three following models for comparison:

1. **Logistic Regression** (Baseline)
   - Simple, interpretable
   - Good for understanding feature importance
   
2. **Random Forest**
   - Handles non-linear relationships
   - Provides feature importance rankings

3. **XGBoost** (Primary Model)
   - Best performance
   - State-of-the-art for tabular data
   - Used for final predictions

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

## Reproducing Results
### Full Pipeline (Optional - Data Already Provided)
If you were to try to optimize my model or just choose different train and test datasets, here is what you would have to modify and the run.
```bash
# Step 1: Collect data (~30 minutes)
python src/data_collection.py

# Step 2: Create features (~10 minutes)  
python src/feature_engineering.py

# Step 3: Train models (~5 minutes)
python src/train.py

# Step 4: Evaluate
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
- Training: October 2022 - April 2024
- Test: October 2024 - January 2025

## How to test the model
TODO

## Code Attribution
TODO

## Limitations & Future Work
TODO

## Dependencies
TODO

## Author
**Aziz Umarbaev**
- Course: COM-214 - Introduction to Artificial Intelligence
- Github: github.com/Choppy314
