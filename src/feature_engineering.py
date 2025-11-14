"""
Feature Engineering Module
Creates features from raw game data and team advanced stats
"""
# TODO: maybe add playoff games later?

import pandas as pd
import numpy as np
from datetime import timedelta

def fix_season_column(df):
    # Had to convert SEASON to string - kept getting merge errors
    # Also for some reason, the SEASON in games.csv is looking really weird like shown in comments below. I tried to fix them
    # Convert "22023" -> "2022-23" and etc...

    dates = pd.to_datetime(df['GAME_DATE'])
    season_year = dates.dt.year - (dates.dt.month < 10).astype(int)
    df['SEASON'] = season_year.astype(str) + '-' + ((season_year + 1) % 100).astype(str).str.zfill(2)
    return df

def merge_team_advanced_stats(games_df, team_stats_df):
    # Merge team advanced stats with games data
    advanced_cols = [
        'TEAM_ID', 'SEASON', 'OFF_RATING', 'DEF_RATING', 'NET_RATING',
        'EFG_PCT', 'TS_PCT', 'PACE', 'AST_PCT', 'AST_TO', 'AST_RATIO',
        'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT'
    ]

    available_cols = [col for col in advanced_cols if col in team_stats_df.columns]
    team_stats_clean = team_stats_df[available_cols].copy()    

    # Trying here to ensure similar datatypes
    games_df['SEASON'] = games_df['SEASON'].astype(str)
    team_stats_clean['SEASON'] = team_stats_clean['SEASON'].astype(str)
    games_df['HOME_TEAM_ID'] = games_df['HOME_TEAM_ID'].astype(int)
    games_df['AWAY_TEAM_ID'] = games_df['AWAY_TEAM_ID'].astype(int)
    team_stats_clean['TEAM_ID'] = team_stats_clean['TEAM_ID'].astype(int)    
    
    # Merge home and away stats
    merged = games_df.merge(team_stats_clean, left_on=['HOME_TEAM_ID','SEASON'], right_on=['TEAM_ID','SEASON'], how='left')
    merged.rename(columns={c:f'HOME_{c}' for c in available_cols if c not in ['TEAM_ID','SEASON']}, inplace=True)
    merged.drop(columns=['TEAM_ID'], inplace=True)
    
    merged = merged.merge(team_stats_clean, left_on=['AWAY_TEAM_ID','SEASON'], right_on=['TEAM_ID','SEASON'], how='left')
    merged.rename(columns={c:f'AWAY_{c}' for c in available_cols if c not in ['TEAM_ID','SEASON']}, inplace=True)
    merged.drop(columns=['TEAM_ID'], inplace=True)

    return merged
 
def calculate_rolling_averages(games_df, window=10):
    # Calculate rolling averages for each team, by default 10, because it is actually calculated like that during the season as far as I am aware. 
    # You could change it though
    games_df = games_df.sort_values('GAME_DATE').reset_index(drop=True)
    stats = ['PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'STL', 'BLK', 'TOV']

    team_rolling_stats = {}
    all_teams = pd.unique(games_df[['HOME_TEAM', 'AWAY_TEAM']].values.ravel())

    for team in all_teams:
        # Combine home and away games for this team
        home_games = games_df[games_df['HOME_TEAM'] == team].copy()
        away_games = games_df[games_df['AWAY_TEAM'] == team].copy()
        
        home_renamed = home_games.rename(columns={f'HOME_{stat}': stat for stat in stats})
        away_renamed = away_games.rename(columns={f'AWAY_{stat}': stat for stat in stats})

        home_renamed['TEAM'] = team
        away_renamed['TEAM'] = team        
        
        team_games = pd.concat([
            home_renamed[['GAME_DATE', 'TEAM'] + stats],
            away_renamed[['GAME_DATE', 'TEAM'] + stats]
        ]).sort_values('GAME_DATE').reset_index(drop=True)

        for stat in stats:
            team_games[f'{stat}_rolling'] = team_games[stat].rolling(
                window=window, min_periods=1
            ).mean()     

        team_rolling_stats[team] = team_games

    games_with_features = games_df.copy()

    for stat in stats:
        games_with_features[f'HOME_{stat}_L{window}'] = np.nan
        games_with_features[f'AWAY_{stat}_L{window}'] = np.nan

    for idx, game in games_with_features.iterrows():
        game_date = game['GAME_DATE']
        home_team, away_team = game['HOME_TEAM'], game['AWAY_TEAM']

        home_stats = team_rolling_stats[home_team]
        home_prev_games = home_stats[home_stats['GAME_DATE'] < game_date]    

        if len(home_prev_games) > 0:
            latest_home = home_prev_games.iloc[-1]
            for stat in stats:
                games_with_features.at[idx, f'HOME_{stat}_L{window}'] = latest_home[f'{stat}_rolling']
        
        away_stats = team_rolling_stats[away_team]
        away_prev_games = away_stats[away_stats['GAME_DATE'] < game_date]
        
        if len(away_prev_games) > 0:
            latest_away = away_prev_games.iloc[-1]
            for stat in stats:
                games_with_features.at[idx, f'AWAY_{stat}_L{window}'] = latest_away[f'{stat}_rolling']
    
    
    return games_with_features

def add_rest_days(games_df):
    # Calculate rest days for each team
    games_df = games_df.sort_values('GAME_DATE').reset_index(drop=True)
    games_df['HOME_REST_DAYS'] = 3 # Just putting by default just in case
    games_df['AWAY_REST_DAYS'] = 3
     
    team_last_game = {}

    for idx, game in games_df.iterrows():
        game_date = game['GAME_DATE']
        home_team = game['HOME_TEAM']
        away_team = game['AWAY_TEAM']

        if home_team in team_last_game:
            rest_days = (game_date - team_last_game[home_team]).days
            games_df.at[idx, 'HOME_REST_DAYS'] = rest_days
        
        if away_team in team_last_game:
            rest_days = (game_date - team_last_game[away_team]).days
            games_df.at[idx, 'AWAY_REST_DAYS'] = rest_days
        
        team_last_game[home_team] = game_date
        team_last_game[away_team] = game_date
    
    # Adding back-to-back indicators
    games_df['HOME_BACK_TO_BACK'] = (games_df['HOME_REST_DAYS'] == 1).astype(int)
    games_df['AWAY_BACK_TO_BACK'] = (games_df['AWAY_REST_DAYS'] == 1).astype(int)
    
    return games_df

def create_differential_features(games_df):
    # Create differential features
    stats = [
        'PTS_L10', 'FG_PCT_L10', 'FG3_PCT_L10', 'FT_PCT_L10', 
        'REB_L10', 'AST_L10', 'STL_L10', 'BLK_L10', 'TOV_L10',
        'REST_DAYS', 'OFF_RATING', 'DEF_RATING', 'NET_RATING',
        'TS_PCT', 'EFG_PCT', 'PACE', 'AST_PCT', 'AST_TO', 
        'AST_RATIO', 'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT'
    ]

    diff_count = 0
    for stat in stats:
        home_col = f'HOME_{stat}'
        away_col = f'AWAY_{stat}'
        
        if home_col in games_df.columns and away_col in games_df.columns:
            games_df[f'DIFF_{stat}'] = games_df[home_col] - games_df[away_col]
            diff_count += 1
    
    print(f"\nCreated {diff_count} differential features")
    
    return games_df    

def create_features(games_df, team_stats_df=None, window=10):
    # Create all features for model training
    print("\nFeature engineering")

    if team_stats_df is not None:
        df = merge_team_advanced_stats(games_df, team_stats_df)
    else:
        df = games_df.copy()
        print("\nSkipping advanced stats (not provided)")

    df = calculate_rolling_averages(df, window=window)
    df = add_rest_days(df)
    df = create_differential_features(df)
    df['IS_HOME'] = 1

    # Cleaning data just in case
    print("\nCleaning data.")
    df = df.dropna(subset=[f'HOME_PTS_L{window}', f'AWAY_PTS_L{window}'])

    print("\nFeature engineering done.")
    return df    

def split_train_test(df, test_season='2024'):
    # Split data into training and test sets based on season
    print(f"\nTest season is {test_season}'")
    
    train_df = df[~df['SEASON'].str.contains(test_season)].copy() # Excluding 2024-25 season (out test season)
    test_df = df[df['SEASON'].str.contains(test_season)].copy()
    
    print(f"Training: {len(train_df):,} games, \nTest: {len(test_df):,} games")
    return train_df, test_df


def prepare_features_and_target(df):
    # Separate features (X) and target (y)
    feature_cols = [col for col in df.columns if any([
        '_L10' in col,
        'DIFF_' in col,
        col in ['IS_HOME', 'HOME_REST_DAYS', 'AWAY_REST_DAYS', 
                'HOME_BACK_TO_BACK', 'AWAY_BACK_TO_BACK',
                'HOME_OFF_RATING', 'AWAY_OFF_RATING',
                'HOME_DEF_RATING', 'AWAY_DEF_RATING',
                'HOME_NET_RATING', 'AWAY_NET_RATING',
                'HOME_PACE', 'AWAY_PACE',
                'HOME_TS_PCT', 'AWAY_TS_PCT',
                'HOME_EFG_PCT', 'AWAY_EFG_PCT']
    ])]
    
    X = df[feature_cols].copy()
    y = df['HOME_WIN'].copy()
    
    print(f"\nPrepared {len(feature_cols)} features")
    
    return X, y, feature_cols

def main():
    print("Loading raw data")

    games = pd.read_csv('data/raw/games.csv')
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])

    games = fix_season_column(games) # Here I am fixing the SEASON column problem

    team_stats = pd.read_csv('data/raw/team_stats.csv')
    
    print(f"\nLoaded {len(games)} games")
    print(f"\nLoaded {len(team_stats)} team-season records")

    df_with_features = create_features(games, team_stats, window=10)

    train_df, test_df = split_train_test(df_with_features, test_season='2024')

    X_train, y_train, feature_names = prepare_features_and_target(train_df)
    X_test, y_test, _ = prepare_features_and_target(test_df)

    X_train.to_csv('data/processed/X_train.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False, header=['home_win'])
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False, header=['home_win'])           

    with open('data/processed/feature_names.txt', 'w') as f:
        for feat in feature_names:
            f.write(f"{feat}\n")   

    print("\nAll processed data saved to data/processed/")
    print("\nCheck feature_names.txt to see the features")

if __name__ == "__main__":
    main()