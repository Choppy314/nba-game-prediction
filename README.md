# NBA Game Win Probability Prediction
A machine learning project that predicts NBA game outcomes using team stats and recent performance.

## About this project
I started this mainly because I’ve always wanted to try building a sports forecasting model. As a basketball fan, this is first and foremost a passion project of mine.

The model looks at things like:
- How teams have been playing recently (last 10 games)
- Team efficiency stats (offensive/defensive ratings)
- Basic box score stats (shooting percentages, rebounds, assists)
- Game context (home court, rest days, back-to-backs)

The model was trained on three NBA seasons (2021-22, 2022-23, 2023-24) and tested on the 2024-25 season, achieving approximately 68-69% prediction accuracy

## How to start

### Prerequisites
- Python 3.8+
- pip

### How to install
Download or clone this project and then navigate to the project directory in your terminal

```bash
cd nba-game-prediction
```

Then, install all required packages

```bash
pip install -r requirements.txt
```

### How to run
#### Option 1: Automated pipeline

```bash
python demo.py
```

#### Option 2: Manual pipeline
Run each script individually:

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
Here are the main insights I was able to outline:

1. Model Performance Comparison

It seems that feature engineering matters more than model complexity itself. There is not much of a substantial gap between the simplest (Logistric Regression) and most complex (XGBoost) models. Given that, it suggests that the well-constructed features already capture most of the patterns. Projects with poorly engineered features tend to see 10-15% gaps between simple and complex models.
Additionally, all models significantly outperform baselines. For example, random guessing (50%) or always picking a home team (around 54-59%) are outperformed by the used models

2. Most Predictive Features

- Net Rating Differential is probably the single strongest predictor. This metric (offensive rating - defensive rating per 100 posessions) aims to capture overall team quality. To demonstrate, a team with +8 Net Rating playing a team with -3 Net Rating creates an 11-point differential, which in turn heavily influences outcomes.
- Recent performance, namely rolling 10-game averages, outshine the regular season-long statistics. A team averaging 118 PPG over their last 10 games better reflects current form of the team, rather than their season average of 111 PPG, which probably shows the early-season struggles. Recent statistics manage to capture the momentum and the current team state.
- Back-to-back games create a win probability disadvantage. Physical fatigue affects every aspect of the game, including the mental part of it. This effect appears consistently across all teams regardless of roster depth.
- Rest differential matters when gaps are significant. To demonstrate, a team with 3 days facing a back-to-back opponent has a 2-day advantage, which plays a role in their win probability.

3. Home Court Advantage

- Home teams usually win approximately 58% of games consistently across seasons. After considering the expected team statistics and averages, home court also influences the win probability. The advantage is consistent across teams, suggesting the common factors rather than specific arena characteristics:

  - Travel fatigue and disrupted routines
  - Home crowd support and subtle referee bias
  -  Maintaining normal sleep and meal schedules

4. Limitations & Future Work
Right now the model is pretty basic. Some things I'd like to improve:
- Add player-level data (injuries, star players, trades): Star player injuries dramatically affect the team performance. In the same fashion, sudden trades create this lag periods where predictions use non-relevant and outdated roster constructions. These are only a few examples.
- No situational context: The model does not take into account the rivalries, possible playoff stakes and coaching changes, for example.
- Include betting lines as features
- Inherent randomness: There is always a possibility of a 40% three-point shooter shooting way below or above the average in any single game. Additionally, last-second shots, referee calls and hot/cold streaks present a volatility that no model can predict.
- Try neural networks
- Make predictions for future games
- Try to also include playoffs?
- Be able to run a playoff simulations
   
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

## License
This is a student project - feel free to use the code for learning purposes!

## Author
**Aziz Umarbaev**
- Course: COM-214 - Introduction to Artificial Intelligence
- Github: github.com/Choppy314
