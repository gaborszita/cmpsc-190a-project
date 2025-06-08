from nba_api.stats.endpoints import playergamelog, teamgamelog
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def get_or_load_csv(filename, fetch_func):
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        # datetime
        if 'GAME_DATE' in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        return df
    else:
        df = fetch_func()
        df.to_csv(filename, index=False)
        return df

def prepare_player_data(player_id, team_id, season):
    player_csv = f"player_{player_id}_{season}.csv"
    team_csv = f"team_{team_id}_{season}.csv"

    def fetch_player():
        return playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
    def fetch_team():
        return teamgamelog.TeamGameLog(team_id=team_id, season=season).get_data_frames()[0]

    player_df = get_or_load_csv(player_csv, fetch_player)
    team_df = get_or_load_csv(team_csv, fetch_team)

    player_df['GAME_DATE'] = pd.to_datetime(player_df['GAME_DATE'])
    team_df['GAME_DATE'] = pd.to_datetime(team_df['GAME_DATE'])

    # create a complete dataset with all team games
    df = team_df[['GAME_DATE', 'MATCHUP', 'WL']].copy()

    # merge player stats where available
    df = df.merge(
        player_df[['GAME_DATE', 'MIN']], 
        on='GAME_DATE', 
        how='left'
    )

    # if MIN is missing, it's a DNP, so set to 0
    df['MIN'] = df['MIN'].fillna(0)

    # sort by date
    df = df.sort_values('GAME_DATE')

    return df

def predict_will_play(game_data):
    predictions = game_data.copy()
    predictions['prediction'] = 1  # 1 means will play
    return predictions

if __name__ == "__main__":
    # test on multiple players not in training set
    TEST_PLAYERS = {
        'Aaron Gordon': {'id': '203932', 'team_id': '1610612743'},  # nuggets
        'Cade Cunningham': {'id': '1630595', 'team_id': '1610612765'},  # pistons
        'Paolo Banchero': {'id': '1631094', 'team_id': '1610612753'},  # magic
        'Jabari Smith Jr.': {'id': '1631095', 'team_id': '1610612745'},  # rockets
        'Bennedict Mathurin': {'id': '1631097', 'team_id': '1610612754'},  # pacers
        'Jaden Ivey': {'id': '1631093', 'team_id': '1610612765'},  # pistons
        'Keegan Murray': {'id': '1631099', 'team_id': '1610612758'},  # kings
        'Jalen Williams': {'id': '1631114', 'team_id': '1610612760'},  # thunder
        'Walker Kessler': {'id': '1631117', 'team_id': '1610612762'}  # jazz
    }

    print("\nTesting baseline model (always predict will play)...")
    seasons = ['2024']
    overall_results = []
    
    for player_name, player_info in TEST_PLAYERS.items():
        print(f"\nTesting on {player_name}...")
        
        player_results_across_seasons = []
        for test_season in seasons:
            try:
                test_df = prepare_player_data(player_info['id'], player_info['team_id'], test_season)
                predictions_df = predict_will_play(test_df)

                # calculate metrics
                total_games = len(predictions_df)
                actual_dnps = len(predictions_df[predictions_df['MIN'] == 0])
                accuracy = (predictions_df['prediction'] == predictions_df['MIN'].apply(lambda x: 0 if x == 0 else 1)).mean()
                dnp_accuracy = len(predictions_df[(predictions_df['prediction'] == 0) & (predictions_df['MIN'] == 0)]) / actual_dnps if actual_dnps > 0 else 0

                # store results
                player_results = {
                    'Player': player_name,
                    'Season': test_season,
                    'Total Games': total_games,
                    'Actual DNPs': actual_dnps,
                    'Overall Accuracy': accuracy,
                    'DNP Accuracy': dnp_accuracy
                }
                player_results_across_seasons.append(player_results)

                # # print confusion matrix
                # cm = confusion_matrix(
                #     predictions_df['MIN'].apply(lambda x: 1 if x == 0 else 0),
                #     predictions_df['prediction'].apply(lambda x: 0 if x == 1 else 1)  # Convert to DNP prediction
                # )
                # print(f"\nConfusion Matrix for {player_name} - Season {test_season}:")
                # print(cm)

            except Exception as e:
                print(f"Error processing {player_name} for season {test_season}: {str(e)}")
                continue
        
        if player_results_across_seasons:
            seasons_df = pd.DataFrame(player_results_across_seasons)
            avg_results = {
                'Player': player_name,
                'Total Games': seasons_df['Total Games'].sum(),
                'Actual DNPs': seasons_df['Actual DNPs'].sum(),
                'Overall Accuracy': seasons_df['Overall Accuracy'].mean(),
                'DNP Accuracy': seasons_df['DNP Accuracy'].mean()
            }
            overall_results.append(avg_results)

    # print aggregate results
    print("\n=== Aggregate Results===")
    results_df = pd.DataFrame(overall_results)
    
    if not results_df.empty:
        print("\nAverage Metrics Across All Players:")
        print(f"Average Overall Accuracy: {results_df['Overall Accuracy'].mean():.2%}")
        print(f"Average DNP Accuracy: {results_df['DNP Accuracy'].mean():.2%}")
        print(f"Total Games Tested: {results_df['Total Games'].sum()}")
        print(f"Total Actual DNPs: {results_df['Actual DNPs'].sum()}")

        # print detailed results table
        print("\nDetailed Results by Player (Averaged Across Seasons):")
        print(results_df.to_string(index=False))

        # graph the results
        plt.figure(figsize=(10, 6))
        plt.bar(results_df['Player'], results_df['Overall Accuracy'])
        plt.xlabel('Player')
        plt.ylabel('Overall Accuracy')
        plt.title('Average Overall Accuracy by Player Across Seasons')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
