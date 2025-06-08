from nba_api.stats.endpoints import playergamelog, teamgamelog
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

PLAYERS_TO_TRAIN = {
    'LeBron James': {'id': '2544', 'team_id': '1610612747'},  # lakers
    'Stephen Curry': {'id': '201939', 'team_id': '1610612744'},  # warriors
    'Kevin Durant': {'id': '201142', 'team_id': '1610612756'},  # suns
    'Giannis Antetokounmpo': {'id': '203507', 'team_id': '1610612749'},  # bucks
    'Nikola Jokic': {'id': '203999', 'team_id': '1610612743'},  # nuggets
    'Joel Embiid': {'id': '203954', 'team_id': '1610612755'},  # 76ers
    'Jayson Tatum': {'id': '1628369', 'team_id': '1610612738'},  # celtics
    'Damian Lillard': {'id': '203081', 'team_id': '1610612749'},  # bucks
    'Devin Booker': {'id': '1626164', 'team_id': '1610612756'},  # suns
    'Kawhi Leonard': {'id': '202695', 'team_id': '1610612746'},  # clippers
    'Paul George': {'id': '202331', 'team_id': '1610612746'},  # clippers
    'Donovan Mitchell': {'id': '1628378', 'team_id': '1610612739'},  # cavs
    'Ja Morant': {'id': '1629630', 'team_id': '1610612763'},  # grizzlies
    'Zion Williamson': {'id': '1629627', 'team_id': '1610612740'},  # pelicans
    'Trae Young': {'id': '1629027', 'team_id': '1610612737'},  # hawks
    'Shai Gilgeous-Alexander': {'id': '1628983', 'team_id': '1610612760'},  # thunder
    'Tyrese Haliburton': {'id': '1630169', 'team_id': '1610612754'},  # pacers
    'Jalen Brunson': {'id': '1628973', 'team_id': '1610612752'},  # knicks
    'Bam Adebayo': {'id': '1628389', 'team_id': '1610612748'},  # heat
    'Pascal Siakam': {'id': '1627783', 'team_id': '1610612754'},  # pacers
    'Jaylen Brown': {'id': '1627759', 'team_id': '1610612738'},  # celtics
    'Domantas Sabonis': {'id': '1627734', 'team_id': '1610612758'},  # kings
    'Rudy Gobert': {'id': '203497', 'team_id': '1610612750'},  # timberwolves
    'Karl-Anthony Towns': {'id': '1626157', 'team_id': '1610612750'},  # timberwolves
    'Anthony Edwards': {'id': '1630162', 'team_id': '1610612750'},  # timberwolves
    'Victor Wembanyama': {'id': '1641705', 'team_id': '1610612759'}  # spurs
}

def prepare_player_data(player_id, team_id, season):
    # use static csvs if available, otherwise fetch and save
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

    # if min is 0, then its a dnp
    df['DNP'] = df['MIN'].apply(lambda x: 1 if x == 0 else 0)

    # calculate features using previous game data
    df['PREV_MIN'] = df['MIN'].shift(1)  # minutes played from previous game
    df['PREV_2_MIN'] = df['MIN'].shift(2)  # minutes played from 2 games ago
    df['PREV_3_MIN'] = df['MIN'].shift(3)  # minutes played from 3 games ago
    
    # calculate rolling averages using previous games only
    df['AVG_MIN_LAST_3'] = df['MIN'].shift(1).rolling(window=3, min_periods=1).mean()
    
    # rest days and back-to-back
    df['REST_DAYS'] = df['GAME_DATE'].diff().dt.days
    df['BACK_TO_BACK'] = df['REST_DAYS'].apply(lambda x: 1 if x == 1 else 0)
    
    # game number and season progression
    df['GAME_NUMBER'] = range(1, len(df) + 1)
    df['SEASON_PROGRESS'] = df['GAME_NUMBER'] / 82  # season progress as percentage
    
    # home/away indicator
    df['IS_HOME'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    
    # previous dnp pattern
    df['PREV_DNP'] = df['DNP'].shift(1)

    # dnp count in last 5 games
    df['DNPs_LAST_5'] = df['DNP'].shift(1).rolling(window=5, min_periods=1).sum()

    return df

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


def train_model(df):
    # features using only past data
    features = [
        'PREV_MIN', 'PREV_2_MIN', 'PREV_3_MIN',
        'AVG_MIN_LAST_3', 'BACK_TO_BACK', 'REST_DAYS',
        'GAME_NUMBER', 'SEASON_PROGRESS', 'IS_HOME', 'DNPs_LAST_5'
    ]
    
    X = df[features].fillna(0)
    y = df['DNP']

    # split the data 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # normalize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler, features

def predict_dnp(model, scaler, features, df):
    X = df[features].fillna(0)
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    # add predictions to dataframe
    df['PREDICTED_DNP'] = predictions
    df['DNP_PROBABILITY'] = probabilities[:, 1]  # probability of dnp
    
    return df

def get_multiple_players_data(seasons):
    all_players_data = []
    
    for season in seasons:
        print(f"\nProcessing season {season}...")
        for player_name, player_info in PLAYERS_TO_TRAIN.items():
            print(f"Getting data for {player_name}...")
            try:
                player_df = prepare_player_data(player_info['id'], player_info['team_id'], season)
                player_df['PLAYER_NAME'] = player_name
                player_df['SEASON'] = season
                all_players_data.append(player_df)
            except Exception as e:
                print(f"Error processing {player_name} for season {season}: {str(e)}")
                continue
    
    # combine all players' data
    combined_df = pd.concat(all_players_data, ignore_index=True)
    return combined_df

# train on 2024
print("Training model on 2024 data...")
seasons = ['2024'] 
combined_df = get_multiple_players_data(seasons)
model, scaler, features = train_model(combined_df)

# print training results
print("\nTraining Results (Combined Dataset):")
print(f"Total games: {len(combined_df)}")
print(f"Games played: {len(combined_df[combined_df['MIN'] > 0])}")
print(f"DNPs: {len(combined_df[combined_df['MIN'] == 0])}")
print(f"Number of players: {len(combined_df['PLAYER_NAME'].unique())}")
print(f"Number of seasons: {len(combined_df['SEASON'].unique())}")

# print feature importance
print(f"Feature importance: {features}")
print(f"Feature importance: {model.feature_importances_}")


# test on random player not in training set
# Test the model on a player not in training set
# test_player_id = '203939'  # Aaron Gordon (Nuggets)
# test_team_id = '1610612743'  # Nuggets
# test_season = '2024'

# print("\nTesting model on player not in training set (Aaron Gordon)...")
# test_df = prepare_player_data(test_player_id, test_team_id, test_season)
# test_df = predict_dnp(model, scaler, features, test_df)

# # Print test results
# print("\nTest Results:")
# print(f"Total games predicted: {len(test_df)}")
# print(f"Predicted DNPs: {len(test_df[test_df['PREDICTED_DNP'] == 1])}")
# print(f"Actual DNPs: {len(test_df[test_df['MIN'] == 0])}")

# # print games where aaron dnp
# print(test_df[test_df['MIN'] == 0])

# # print games where predicted dnp is 1
# print(test_df[test_df['PREDICTED_DNP'] == 1])


# test on multiple players not in training set
TEST_PLAYERS = {
    'Aaron Gordon': {'id': '203932', 'team_id': '1610612743'},  # Nuggets
    'Cade Cunningham': {'id': '1630595', 'team_id': '1610612765'},  # Pistons
    'Paolo Banchero': {'id': '1631094', 'team_id': '1610612753'},  # Magic
    'Jabari Smith Jr.': {'id': '1631095', 'team_id': '1610612745'},  # Rockets
    'Bennedict Mathurin': {'id': '1631097', 'team_id': '1610612754'},  # Pacers
    'Jaden Ivey': {'id': '1631093', 'team_id': '1610612765'},  # Pistons
    'Keegan Murray': {'id': '1631099', 'team_id': '1610612758'},  # Kings
    'Jalen Williams': {'id': '1631114', 'team_id': '1610612760'},  # Thunder
    'Walker Kessler': {'id': '1631117', 'team_id': '1610612762'}  # Jazz
}

test_seasons = ['2024']
overall_results = []

print("\nTesting ensemble model on multiple players not in training set...")
for player_name, player_info in TEST_PLAYERS.items():
    print(f"\nTesting on {player_name}...")
    
    player_results_across_seasons = []
    for test_season in test_seasons:
        test_df = prepare_player_data(player_info['id'], player_info['team_id'], test_season)
        test_df = predict_dnp(model, scaler, features, test_df)

        # calculate metrics
        total_games = len(test_df)
        predicted_dnps = len(test_df[test_df['PREDICTED_DNP'] == 1])
        actual_dnps = len(test_df[test_df['MIN'] == 0])
        accuracy = (test_df['PREDICTED_DNP'] == test_df['MIN'].apply(lambda x: 1 if x == 0 else 0)).mean()
        dnp_accuracy = len(test_df[(test_df['PREDICTED_DNP'] == 1) & (test_df['MIN'] == 0)]) / actual_dnps if actual_dnps > 0 else 0

        # store results
        player_results = {
            'Player': player_name,
            'Season': test_season,
            'Total Games': total_games,
            'Predicted DNPs': predicted_dnps,
            'Actual DNPs': actual_dnps,
            'Overall Accuracy': accuracy,
            'DNP Accuracy': dnp_accuracy
        }
        player_results_across_seasons.append(player_results)

        # print individual results
        # print(f"\nSeason {test_season}:")
        # print(f"Total games: {total_games}")
        # print(f"Predicted DNPs: {predicted_dnps}")
        # print(f"Actual DNPs: {actual_dnps}")
        # print(f"Overall Accuracy: {accuracy:.2%}")
        # print(f"DNP Prediction Accuracy: {dnp_accuracy:.2%}")

        # print confusion matrix
        # cm = confusion_matrix(
        #     test_df['MIN'].apply(lambda x: 1 if x == 0 else 0),
        #     test_df['PREDICTED_DNP']
        # )
        # print("\nConfusion Matrix:")
        # print(cm)


    if player_results_across_seasons:
        seasons_df = pd.DataFrame(player_results_across_seasons)
        avg_results = {
            'Player': player_name,
            'Total Games': seasons_df['Total Games'].sum(),
            'Predicted DNPs': seasons_df['Predicted DNPs'].sum(),
            'Actual DNPs': seasons_df['Actual DNPs'].sum(),
            'Overall Accuracy': seasons_df['Overall Accuracy'].mean(),
            'DNP Accuracy': seasons_df['DNP Accuracy'].mean()
        }
        overall_results.append(avg_results)

# print aggregate results
print("\n=== Aggregate Results ===")
results_df = pd.DataFrame(overall_results)
print("\nAverage Metrics Across All Players:")
print(f"Average Overall Accuracy: {results_df['Overall Accuracy'].mean():.2%}")
print(f"Average DNP Accuracy: {results_df['DNP Accuracy'].mean():.2%}")
print(f"Total Games Tested: {results_df['Total Games'].sum()}")
print(f"Total Predicted DNPs: {results_df['Predicted DNPs'].sum()}")
print(f"Total Actual DNPs: {results_df['Actual DNPs'].sum()}")

# print detailed results table
print("\nDetailed Results by Player:")
print(results_df.to_string(index=False))

# graph the accuracy
plt.figure(figsize=(10, 6))
plt.bar(results_df['Player'], results_df['Overall Accuracy'])
plt.xlabel('Player')
plt.ylabel('Overall Accuracy')
plt.title('Overall Accuracy by Player')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# graph recall
plt.figure(figsize=(10, 6))
plt.bar(results_df['Player'], results_df['DNP Accuracy'])
plt.xlabel('Player')
plt.ylabel('DNP Recall')
plt.title('DNP Recall by Player')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()