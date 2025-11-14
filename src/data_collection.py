"""
Data Collection Module
Collects NBA game data using NBA API

Code Attribution:
- I used nba_api package by Swar Patel (https://github.com/swar/nba_api)
- Data processing and aggregation is original implementation
"""

import pandas as pd
import time
from nba_api.stats.endpoints import leaguegamefinder, leaguedashteamstats

def collect_season_games(season):
    # Collect all games for a specific season
    print(f"\nCollecting games for {season} season.")

    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            league_id_nullable='00', # I am pretty sure we have to specify it, because we can accidentally get data from WNBA and G-League
            season_type_nullable="Regular Season"
        )

        games = gamefinder.get_data_frames()[0]
        print(f"\nCollected {len(games)} game records.")

        return games
    
    except Exception as e:
        print(f"\nError collecting game records: {e}")
        return None
    
def collect_team_advanced_stats(season):
    # Collect advanced team statistics
    print(f"\nCollecting team stats for {season} season.")

    try:
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            league_id_nullable="00",
            measure_type_detailed_defense='Advanced',
            per_mode_detailed='PerGame'
        )

        stats = team_stats.get_data_frames()[0]  
        print(f"\nCollected stats for {len(stats)} teams")

        return stats          
    
    except Exception as e:
        print(f"\nError collecting team stats: {e}")
        return None
    
def process_games_to_single_row(games_df):
    # Convert games from team perspective (2 rows per game) to game perspective (1 row per game)
    # Each game appears twice in raw data (once for each team). This combines them into single row with home/away structure.
    print("\nProcessing games to single-row format.")

    games_df = games_df.sort_values(['GAME_ID', 'MATCHUP']).reset_index(drop=True)
    games_df['IS_HOME'] = games_df['MATCHUP'].str.contains('vs.')

    home_games = games_df[games_df['IS_HOME'] == True].copy()
    away_games = games_df[games_df['IS_HOME'] == False].copy()

    home_games = home_games.reset_index(drop=True)
    away_games = away_games.reset_index(drop=True)

    combined = pd.merge(
        home_games,
        away_games,
        on='GAME_ID',
        suffixes=('_home', '_away')
    )

    game_data = pd.DataFrame({
        'GAME_ID': combined['GAME_ID'],
        'GAME_DATE': combined['GAME_DATE_home'],
        'SEASON': combined['SEASON_ID_home'],
        
        # Home team
        'HOME_TEAM_ID': combined['TEAM_ID_home'],
        'HOME_TEAM': combined['TEAM_ABBREVIATION_home'],
        'HOME_PTS': combined['PTS_home'],
        'HOME_FGM': combined['FGM_home'],
        'HOME_FGA': combined['FGA_home'],
        'HOME_FG_PCT': combined['FG_PCT_home'],
        'HOME_FG3M': combined['FG3M_home'],
        'HOME_FG3A': combined['FG3A_home'],
        'HOME_FG3_PCT': combined['FG3_PCT_home'],
        'HOME_FTM': combined['FTM_home'],
        'HOME_FTA': combined['FTA_home'],
        'HOME_FT_PCT': combined['FT_PCT_home'],
        'HOME_OREB': combined['OREB_home'],
        'HOME_DREB': combined['DREB_home'],
        'HOME_REB': combined['REB_home'],
        'HOME_AST': combined['AST_home'],
        'HOME_STL': combined['STL_home'],
        'HOME_BLK': combined['BLK_home'],
        'HOME_TOV': combined['TOV_home'],
        'HOME_PF': combined['PF_home'],
        'HOME_PLUS_MINUS': combined['PLUS_MINUS_home'],
        
        # Away team
        'AWAY_TEAM_ID': combined['TEAM_ID_away'],
        'AWAY_TEAM': combined['TEAM_ABBREVIATION_away'],
        'AWAY_PTS': combined['PTS_away'],
        'AWAY_FGM': combined['FGM_away'],
        'AWAY_FGA': combined['FGA_away'],
        'AWAY_FG_PCT': combined['FG_PCT_away'],
        'AWAY_FG3M': combined['FG3M_away'],
        'AWAY_FG3A': combined['FG3A_away'],
        'AWAY_FG3_PCT': combined['FG3_PCT_away'],
        'AWAY_FTM': combined['FTM_away'],
        'AWAY_FTA': combined['FTA_away'],
        'AWAY_FT_PCT': combined['FT_PCT_away'],
        'AWAY_OREB': combined['OREB_away'],
        'AWAY_DREB': combined['DREB_away'],
        'AWAY_REB': combined['REB_away'],
        'AWAY_AST': combined['AST_away'],
        'AWAY_STL': combined['STL_away'],
        'AWAY_BLK': combined['BLK_away'],
        'AWAY_TOV': combined['TOV_away'],
        'AWAY_PF': combined['PF_away'],
        'AWAY_PLUS_MINUS': combined['PLUS_MINUS_away'],
        
        # Outcome (target variable)
        'HOME_WIN': (combined['WL_home'] == 'W').astype(int)
    })

    game_data['GAME_DATE'] = pd.to_datetime(game_data['GAME_DATE'])

    print(f"\nProcessed {len(game_data)} games")
    
    return game_data

def collect_all_data(training_seasons=['2021-22', '2022-23', '2023-24'], test_season='2024-25'):
    # Collect all necessary data for training and testing
    print("\nNBA DATA COLLECTION")
    print(f"\nTraining seasons: {training_seasons}")
    print(f"\nTest season: {test_season}")

    all_games = []
    all_team_stats = []

    for season in training_seasons:
        print(f"\nProcessing {season}")

        games = collect_season_games(season)
        if games is not None:
            all_games.append(games)

        time.sleep(1)

        team_stats = collect_team_advanced_stats(season)
        if team_stats is not None:
            team_stats['SEASON'] = season
            all_team_stats.append(team_stats)

        time.sleep(1)

    print("\nCollecting test data")
    print(f"\nProcessing {test_season}")

    test_games = collect_season_games(test_season)
    if test_games is not None:
        # Filtering to games that have already been played, so we can check the model accuracy
        test_games = test_games[test_games['PTS'].notna()].copy()
        all_games.append(test_games)
        print(f"\nNote: Using {len(test_games)//2} games from {test_season} (games already played)")

    time.sleep(1)

    test_team_stats = collect_team_advanced_stats(test_season)
    if test_team_stats is not None:
        test_team_stats['SEASON'] = test_season
        all_team_stats.append(test_team_stats)

    print("\nCombining data")

    combined_games = pd.concat(all_games, ignore_index=True)
    combined_team_stats = pd.concat(all_team_stats, ignore_index=True)

    processed_games = process_games_to_single_row(combined_games)

    print("\nCollection done")
    print(f"\nTotal games {len(processed_games)}")

    for season in training_seasons + [test_season]:
        season_games = processed_games[processed_games['SEASON'].str.contains(season.split('-')[0])]
        print(f"\n{season}: {len(season_games)} games")

    print(f"\nDate range: {processed_games['GAME_DATE'].min()} to {processed_games['GAME_DATE'].max()}")

    return {
        'games': processed_games,
        'team_stats': combined_team_stats
    }

def save_data(data, output_dir='data/raw'):
    # Save collected data to CSV files

    print(f"\nSaving data to {output_dir}")

    games_path = f"{output_dir}/games.csv"
    data['games'].to_csv(games_path, index=False)

    team_stats_path = f"{output_dir}/team_stats.csv"
    data['team_stats'].to_csv(team_stats_path, index = False)

    print("\nData saved")

def main():
    data = collect_all_data(
        training_seasons=['2021-22', '2022-23', '2023-24'],
        test_season='2024-25'
    )

    save_data(data)

    return data

if __name__ == "__main__":
    data = main()