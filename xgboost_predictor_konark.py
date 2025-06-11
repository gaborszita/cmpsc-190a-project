import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# NBA API imports
from nba_api.stats.endpoints import leaguegamefinder, teamgamelog, boxscoretraditionalv2
from nba_api.stats.static import teams
from nba_api.stats.endpoints import teamestimatedmetrics
import time

class NBAMoneyLinePredictor:
    def __init__(self, seasons=['2023-24', '2022-23'], create_plots=False):
        """
        Initialize the NBA Money Line Predictor
        
        Args:
            seasons: List of seasons to fetch data for (format: 'YYYY-YY')
            create_plots: Whether to generate visualization plots
        """
        self.seasons = seasons
        self.team_dict = {team['abbreviation']: team['id'] for team in teams.get_teams()}
        self.scaler = StandardScaler()
        self.model = None
        self.create_plots = create_plots
        
    def fetch_game_data(self):
        """Fetch historical game data from NBA API"""
        print("Fetching game data...")
        all_games = []
        
        for season in self.seasons:
            print(f"Fetching {season} season...")
            # Get all games for the season
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable='Regular Season'
            )
            games = gamefinder.get_data_frames()[0]
            all_games.append(games)
            time.sleep(1)  # Rate limiting
            
        games_df = pd.concat(all_games, ignore_index=True)
        games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
        
        # Sort by date and team
        games_df = games_df.sort_values(['GAME_DATE', 'TEAM_ID'])
        
        return games_df
    
    def calculate_rolling_stats(self, games_df, window=10):
        """Calculate rolling statistics for each team"""
        print("Calculating rolling statistics...")
        
        rolling_stats = []
        
        for team_id in games_df['TEAM_ID'].unique():
            team_games = games_df[games_df['TEAM_ID'] == team_id].copy()
            team_games = team_games.sort_values('GAME_DATE')
            
            # Calculate rolling averages
            rolling_cols = ['PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'STL', 'BLK', 'TOV']
            
            for col in rolling_cols:
                if col in team_games.columns:
                    team_games[f'{col}_ROLL_{window}'] = team_games[col].rolling(
                        window=window, min_periods=1
                    ).mean().shift(1)  # Shift to avoid data leakage
            
            # Win percentage over last N games
            team_games['WIN_ROLL'] = (team_games['WL'] == 'W').rolling(
                window=window, min_periods=1
            ).mean().shift(1)
            
            # Rest days
            team_games['REST_DAYS'] = team_games['GAME_DATE'].diff().dt.days.fillna(3)
            
            rolling_stats.append(team_games)
        
        return pd.concat(rolling_stats, ignore_index=True)
    
    def create_matchup_features(self, games_with_rolling):
        """Create features for each matchup"""
        print("Creating matchup features...")
        
        matchups = []
        
        # Group by game ID to get matchups
        for game_id, game_group in games_with_rolling.groupby('GAME_ID'):
            if len(game_group) != 2:  # Skip if not exactly 2 teams
                continue
                
            game_group = game_group.sort_values('TEAM_ID')
            team1 = game_group.iloc[0]
            team2 = game_group.iloc[1]
            
            # Determine home team (MATCHUP contains @ for away games)
            team1_is_home = '@' not in team1['MATCHUP']
            
            if team1_is_home:
                home_team = team1
                away_team = team2
            else:
                home_team = team2
                away_team = team1
            
            # Create feature dictionary
            features = {
                'GAME_ID': game_id,
                'GAME_DATE': home_team['GAME_DATE'],
                'HOME_TEAM_ID': home_team['TEAM_ID'],
                'AWAY_TEAM_ID': away_team['TEAM_ID'],
                'HOME_WIN': 1 if home_team['WL'] == 'W' else 0,
                
                # Home team features
                'HOME_PTS_ROLL': home_team.get('PTS_ROLL_10', 0),
                'HOME_FG_PCT_ROLL': home_team.get('FG_PCT_ROLL_10', 0),
                'HOME_FG3_PCT_ROLL': home_team.get('FG3_PCT_ROLL_10', 0),
                'HOME_FT_PCT_ROLL': home_team.get('FT_PCT_ROLL_10', 0),
                'HOME_REB_ROLL': home_team.get('REB_ROLL_10', 0),
                'HOME_AST_ROLL': home_team.get('AST_ROLL_10', 0),
                'HOME_STL_ROLL': home_team.get('STL_ROLL_10', 0),
                'HOME_BLK_ROLL': home_team.get('BLK_ROLL_10', 0),
                'HOME_TOV_ROLL': home_team.get('TOV_ROLL_10', 0),
                'HOME_WIN_ROLL': home_team.get('WIN_ROLL', 0.5),
                'HOME_REST_DAYS': home_team.get('REST_DAYS', 2),
                
                # Away team features
                'AWAY_PTS_ROLL': away_team.get('PTS_ROLL_10', 0),
                'AWAY_FG_PCT_ROLL': away_team.get('FG_PCT_ROLL_10', 0),
                'AWAY_FG3_PCT_ROLL': away_team.get('FG3_PCT_ROLL_10', 0),
                'AWAY_FT_PCT_ROLL': away_team.get('FT_PCT_ROLL_10', 0),
                'AWAY_REB_ROLL': away_team.get('REB_ROLL_10', 0),
                'AWAY_AST_ROLL': away_team.get('AST_ROLL_10', 0),
                'AWAY_STL_ROLL': away_team.get('STL_ROLL_10', 0),
                'AWAY_BLK_ROLL': away_team.get('BLK_ROLL_10', 0),
                'AWAY_TOV_ROLL': away_team.get('TOV_ROLL_10', 0),
                'AWAY_WIN_ROLL': away_team.get('WIN_ROLL', 0.5),
                'AWAY_REST_DAYS': away_team.get('REST_DAYS', 2),
            }
            
            # Add differential features
            for stat in ['PTS', 'FG_PCT', 'FG3_PCT', 'REB', 'AST', 'WIN']:
                home_val = features.get(f'HOME_{stat}_ROLL', 0)
                away_val = features.get(f'AWAY_{stat}_ROLL', 0)
                features[f'{stat}_DIFF'] = home_val - away_val
            
            # Rest advantage
            features['REST_ADVANTAGE'] = features['HOME_REST_DAYS'] - features['AWAY_REST_DAYS']
            
            matchups.append(features)
        
        return pd.DataFrame(matchups)
    
    def prepare_features(self, matchups_df):
        """Prepare features for modeling"""
        print("Preparing features for modeling...")
        
        # Select feature columns
        feature_cols = [col for col in matchups_df.columns if col not in [
            'GAME_ID', 'GAME_DATE', 'HOME_TEAM_ID', 'AWAY_TEAM_ID', 'HOME_WIN'
        ]]
        
        # Handle missing values
        matchups_df[feature_cols] = matchups_df[feature_cols].fillna(matchups_df[feature_cols].mean())
        
        # Add month and day of week features
        matchups_df['MONTH'] = matchups_df['GAME_DATE'].dt.month
        matchups_df['DAY_OF_WEEK'] = matchups_df['GAME_DATE'].dt.dayofweek
        
        feature_cols.extend(['MONTH', 'DAY_OF_WEEK'])
        
        return matchups_df, feature_cols
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        print("Training XGBoost model...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # XGBoost parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.03,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 1,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'random_state': 42,
            'verbosity': 0  # Reduce output noise
        }
        
        # Train model
        self.model = xgb.XGBClassifier(**params)
        
        # Fit with early stopping (compatible with different XGBoost versions)
        try:
            # Try new API first (XGBoost 2.0+)
            self.model.set_params(early_stopping_rounds=20)
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=100
            )
        except (TypeError, AttributeError):
            # Fall back to older API
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=100
            )
        
        # Evaluate on validation set
        y_pred = self.model.predict(X_val_scaled)
        y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        
        print(f"\nValidation Accuracy: {accuracy_score(y_val, y_pred):.4f}")
        print(f"Validation AUC: {roc_auc_score(y_val, y_pred_proba):.4f}")
        print(f"Validation Log Loss: {log_loss(y_val, y_pred_proba):.4f}")
        
        return self.model
    
    def get_feature_importance(self, feature_names):
        """Get and display feature importance"""
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 Most Important Features:")
        print(feature_importance.head(15))
        
        return feature_importance
    
    def predict_game(self, home_team_abbr, away_team_abbr, game_date=None):
        """
        Predict the outcome of a specific game
        
        Args:
            home_team_abbr: Home team abbreviation (e.g., 'LAL')
            away_team_abbr: Away team abbreviation (e.g., 'BOS')
            game_date: Date of the game (default: today)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Run train() first.")
        
        if game_date is None:
            game_date = datetime.now()
        
        print(f"\nPredicting: {away_team_abbr} @ {home_team_abbr} on {game_date.strftime('%Y-%m-%d')}")
        
        try:
            # Get team IDs
            home_team_id = self.team_dict.get(home_team_abbr)
            away_team_id = self.team_dict.get(away_team_abbr)
            
            if not home_team_id or not away_team_id:
                raise ValueError(f"Invalid team abbreviation: {home_team_abbr} or {away_team_abbr}")
            
            # Fetch recent games for both teams
            home_games = self.fetch_recent_team_stats(home_team_id, game_date)
            away_games = self.fetch_recent_team_stats(away_team_id, game_date)
            
            # Create feature vector
            features = self.create_prediction_features(home_games, away_games)
            
            # Make prediction
            X = pd.DataFrame([features])
            X_scaled = self.scaler.transform(X[self.feature_cols])
            
            prob = self.model.predict_proba(X_scaled)[0]
            prediction = self.model.predict(X_scaled)[0]
            
            print(f"\n{'='*50}")
            print(f"PREDICTION RESULTS")
            print(f"{'='*50}")
            print(f"Home Win Probability ({home_team_abbr}): {prob[1]:.3f}")
            print(f"Away Win Probability ({away_team_abbr}): {prob[0]:.3f}")
            print(f"Predicted Winner: {home_team_abbr if prediction == 1 else away_team_abbr}")
            
            # Convert to American odds for betting context
            if prob[1] > 0.5:
                american_odds_home = -int(prob[1] / (1 - prob[1]) * 100)
                american_odds_away = int((1 - prob[0]) / prob[0] * 100)
            else:
                american_odds_home = int((1 - prob[1]) / prob[1] * 100)
                american_odds_away = -int(prob[0] / (1 - prob[0]) * 100)
            
            print(f"\nImplied American Odds:")
            print(f"{home_team_abbr}: {american_odds_home:+d}")
            print(f"{away_team_abbr}: {american_odds_away:+d}")
            
            return {
                'home_team': home_team_abbr,
                'away_team': away_team_abbr,
                'home_win_prob': prob[1],
                'away_win_prob': prob[0],
                'predicted_winner': home_team_abbr if prediction == 1 else away_team_abbr,
                'american_odds_home': american_odds_home,
                'american_odds_away': american_odds_away
            }
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def fetch_recent_team_stats(self, team_id, before_date, num_games=10):
        """Fetch recent team statistics"""
        # Get team's recent games
        gamelog = teamgamelog.TeamGameLog(
            team_id=team_id,
            season='2023-24',  # Would need to determine current season
            season_type_all_star='Regular Season'
        )
        games = gamelog.get_data_frames()[0]
        games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
        
        # Filter games before the prediction date
        games = games[games['GAME_DATE'] < before_date]
        games = games.head(num_games)
        
        return games
    
    def create_prediction_features(self, home_games, away_games):
        """Create feature vector for prediction"""
        features = {}
        
        # Calculate rolling stats for home team
        home_stats = {
            'HOME_PTS_ROLL': home_games['PTS'].mean() if len(home_games) > 0 else 100,
            'HOME_FG_PCT_ROLL': home_games['FG_PCT'].mean() if len(home_games) > 0 else 0.45,
            'HOME_FG3_PCT_ROLL': home_games['FG3_PCT'].mean() if len(home_games) > 0 else 0.35,
            'HOME_FT_PCT_ROLL': home_games['FT_PCT'].mean() if len(home_games) > 0 else 0.75,
            'HOME_REB_ROLL': home_games['REB'].mean() if len(home_games) > 0 else 45,
            'HOME_AST_ROLL': home_games['AST'].mean() if len(home_games) > 0 else 25,
            'HOME_STL_ROLL': home_games['STL'].mean() if len(home_games) > 0 else 7,
            'HOME_BLK_ROLL': home_games['BLK'].mean() if len(home_games) > 0 else 5,
            'HOME_TOV_ROLL': home_games['TOV'].mean() if len(home_games) > 0 else 15,
            'HOME_WIN_ROLL': (home_games['WL'] == 'W').mean() if len(home_games) > 0 else 0.5,
            'HOME_REST_DAYS': 2  # Default, would calculate actual rest days
        }
        
        # Calculate rolling stats for away team
        away_stats = {
            'AWAY_PTS_ROLL': away_games['PTS'].mean() if len(away_games) > 0 else 100,
            'AWAY_FG_PCT_ROLL': away_games['FG_PCT'].mean() if len(away_games) > 0 else 0.45,
            'AWAY_FG3_PCT_ROLL': away_games['FG3_PCT'].mean() if len(away_games) > 0 else 0.35,
            'AWAY_FT_PCT_ROLL': away_games['FT_PCT'].mean() if len(away_games) > 0 else 0.75,
            'AWAY_REB_ROLL': away_games['REB'].mean() if len(away_games) > 0 else 45,
            'AWAY_AST_ROLL': away_games['AST'].mean() if len(away_games) > 0 else 25,
            'AWAY_STL_ROLL': away_games['STL'].mean() if len(away_games) > 0 else 7,
            'AWAY_BLK_ROLL': away_games['BLK'].mean() if len(away_games) > 0 else 5,
            'AWAY_TOV_ROLL': away_games['TOV'].mean() if len(away_games) > 0 else 15,
            'AWAY_WIN_ROLL': (away_games['WL'] == 'W').mean() if len(away_games) > 0 else 0.5,
            'AWAY_REST_DAYS': 2  # Default
        }
        
        features.update(home_stats)
        features.update(away_stats)
        
        # Add differential features
        for stat in ['PTS', 'FG_PCT', 'FG3_PCT', 'REB', 'AST', 'WIN']:
            features[f'{stat}_DIFF'] = features[f'HOME_{stat}_ROLL'] - features[f'AWAY_{stat}_ROLL']
        
        features['REST_ADVANTAGE'] = features['HOME_REST_DAYS'] - features['AWAY_REST_DAYS']
        features['MONTH'] = datetime.now().month
        features['DAY_OF_WEEK'] = datetime.now().weekday()
        
        return features
    
    def save_feature_columns(self, feature_cols):
        """Save feature columns for later use in predictions"""
        self.feature_cols = feature_cols
        
    def run_pipeline(self):
        """Run the complete pipeline"""
        # Fetch data
        games_df = self.fetch_game_data()
        
        # Calculate rolling stats
        games_with_rolling = self.calculate_rolling_stats(games_df)
        
        # Create matchup features
        matchups_df = self.create_matchup_features(games_with_rolling)
        
        # Prepare features
        matchups_df, feature_cols = self.prepare_features(matchups_df)
        
        # Remove early season games with insufficient rolling data
        matchups_df = matchups_df[matchups_df['GAME_DATE'] > matchups_df['GAME_DATE'].min() + timedelta(days=30)]
        
        # Split data chronologically
        split_date = matchups_df['GAME_DATE'].quantile(0.8)
        train_df = matchups_df[matchups_df['GAME_DATE'] < split_date]
        test_df = matchups_df[matchups_df['GAME_DATE'] >= split_date]
        
        X_train = train_df[feature_cols]
        y_train = train_df['HOME_WIN']
        X_test = test_df[feature_cols]
        y_test = test_df['HOME_WIN']
        
        # Further split train into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Train model
        self.train_model(X_train, y_train, X_val, y_val)
        
        # Save feature columns for predictions
        self.save_feature_columns(feature_cols)
        
        # Final evaluation on test set
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print(f"\n{'='*50}")
        print("TEST SET PERFORMANCE")
        print(f"{'='*50}")
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Test AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        print(f"Test Log Loss: {log_loss(y_test, y_pred_proba):.4f}")
        
        # Feature importance
        feature_importance_df = self.get_feature_importance(feature_cols)
        
        # Calibration check
        print("\nProbability Calibration Check:")
        for i in range(0, 10, 2):
            lower = i/10
            upper = (i+2)/10
            mask = (y_pred_proba >= lower) & (y_pred_proba < upper)
            if mask.sum() > 0:
                actual_win_rate = y_test[mask].mean()
                predicted_win_rate = y_pred_proba[mask].mean()
                print(f"Predicted: {predicted_win_rate:.3f}, Actual: {actual_win_rate:.3f} (n={mask.sum()})")
        
        # Generate visualizations if requested
        if hasattr(self, 'create_plots') and self.create_plots:
            print("\nGenerating performance visualizations...")
            from nba_predictor_plots import visualize_model_performance
            visualize_model_performance(y_test, y_pred_proba, test_df, feature_importance_df)
        
        return self.model, test_df

# Example usage
if __name__ == "__main__":
    import argparse
    import pickle
    import os
    
    parser = argparse.ArgumentParser(description='NBA Money Line Predictor using XGBoost')
    parser.add_argument('command', choices=['train', 'predict', 'evaluate', 'fetch-today'],
                        help='Command to execute')
    
    # Training arguments
    parser.add_argument('--seasons', nargs='+', default=['2023-24'],
                        help='Seasons to use for training (e.g., 2023-24 2022-23)')
    parser.add_argument('--save-model', type=str, default='nba_predictor.pkl',
                        help='Path to save the trained model')
    parser.add_argument('--plots', action='store_true',
                        help='Generate visualization plots after training')
    
    # Prediction arguments
    parser.add_argument('--home', type=str, help='Home team abbreviation (e.g., LAL)')
    parser.add_argument('--away', type=str, help='Away team abbreviation (e.g., BOS)')
    parser.add_argument('--model-path', type=str, default='nba_predictor.pkl',
                        help='Path to load the model from')
    
    # Evaluation arguments
    parser.add_argument('--test-season', type=str, help='Season to evaluate on')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        print(f"Training model with seasons: {args.seasons}")
        predictor = NBAMoneyLinePredictor(seasons=args.seasons, create_plots=args.plots)
        model, test_results = predictor.run_pipeline()
        
        # Save the model
        with open(args.save_model, 'wb') as f:
            pickle.dump({
                'model': predictor.model,
                'scaler': predictor.scaler,
                'feature_cols': predictor.feature_cols if hasattr(predictor, 'feature_cols') else None
            }, f)
        print(f"\nModel saved to {args.save_model}")
        
    elif args.command == 'predict':
        if not args.home or not args.away:
            parser.error("predict requires --home and --away arguments")
        
        # Load saved model
        if not os.path.exists(args.model_path):
            parser.error(f"Model file {args.model_path} not found. Train a model first.")
        
        with open(args.model_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        predictor = NBAMoneyLinePredictor()
        predictor.model = saved_data['model']
        predictor.scaler = saved_data['scaler']
        predictor.feature_cols = saved_data.get('feature_cols', [])
        
        # Make prediction
        result = predictor.predict_game(args.home.upper(), args.away.upper())
        
    elif args.command == 'evaluate':
        # Load model and evaluate on specific season
        if not os.path.exists(args.model_path):
            parser.error(f"Model file {args.model_path} not found. Train a model first.")
        
        with open(args.model_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        print(f"Model evaluation on test set:")
        print("Feature importance and performance metrics were shown during training.")
        print("To evaluate on new data, implement evaluation on specific season.")
        
    elif args.command == 'fetch-today':
        print("Fetching today's NBA games...")
        from datetime import datetime
        today = datetime.now()
        
        # Fetch today's games from NBA API
        from nba_api.stats.endpoints import scoreboardv2
        
        try:
            scoreboard = scoreboardv2.ScoreboardV2(game_date=today.strftime('%Y-%m-%d'))
            games = scoreboard.get_data_frames()[0]
            
            if len(games) == 0:
                print("No games scheduled for today.")
            else:
                print(f"\nGames for {today.strftime('%A, %B %d, %Y')}:")
                print("-" * 50)
                
                # Load model if available
                model_loaded = False
                if os.path.exists(args.model_path):
                    with open(args.model_path, 'rb') as f:
                        saved_data = pickle.load(f)
                    predictor = NBAMoneyLinePredictor()
                    predictor.model = saved_data['model']
                    predictor.scaler = saved_data['scaler']
                    predictor.feature_cols = saved_data.get('feature_cols', [])
                    model_loaded = True
                
                # Group games by matchup
                for i in range(0, len(games), 2):
                    if i+1 < len(games):
                        game1 = games.iloc[i]
                        game2 = games.iloc[i+1]
                        
                        # Determine home/away
                        if '@' in game1['TEAM_ABBREVIATION']:
                            away_team = game1['TEAM_ABBREVIATION']
                            home_team = game2['TEAM_ABBREVIATION']
                        else:
                            home_team = game1['TEAM_ABBREVIATION']
                            away_team = game2['TEAM_ABBREVIATION']
                        
                        print(f"\n{away_team} @ {home_team}")
                        
                        if model_loaded:
                            # Make prediction
                            result = predictor.predict_game(home_team, away_team, today)
                            if result:
                                print(f"Model prediction: {result['predicted_winner']} "
                                      f"({result['home_win_prob']:.1%} home win probability)")
                
        except Exception as e:
            print(f"Error fetching today's games: {e}")