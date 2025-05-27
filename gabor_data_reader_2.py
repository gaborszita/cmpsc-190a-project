import sqlite3
import os
import csv
from itertools import islice
import pandas as pd
import time

conn = sqlite3.connect('NBA-Web-Scraper/data/NBA-Boxscore-Database.sqlite')

query = 'SELECT * from game_info'
game_info = pd.read_sql(query, con=conn)
# account for team name change following the 2013-2014 season
game_info['away_team'].replace('CHA', 'CHO', inplace=True)
game_info['home_team'].replace('CHA', 'CHO', inplace=True)

query = 'SELECT * from team_stats'
team_stats = pd.read_sql(query, con=conn)
# account for team name change following the 2013-2014 season
team_stats['team'].replace('CHA', 'CHO', inplace=True)

game_ids = list(game_info['game_id'].unique())

def create_gid_query(game_id, date, lim, team='away'):
    away_gid_query = f'''
    SELECT gi.game_id
    FROM game_info gi
    WHERE (gi.away_team = (SELECT away_team FROM game_info WHERE game_id = '{game_id}')
            AND gi.date < '{date}')
       OR (gi.home_team = (SELECT away_team FROM game_info WHERE game_id = '{game_id}')
            AND gi.date < '{date}')
    ORDER BY gi.date DESC
    LIMIT {lim};
    '''
    
    home_gid_query = f'''
    SELECT gi.game_id
    FROM game_info gi
    WHERE (gi.away_team = (SELECT home_team FROM game_info WHERE game_id = '{game_id}')
            AND gi.date < '{date}')
       OR (gi.home_team = (SELECT home_team FROM game_info WHERE game_id = '{game_id}')
            AND gi.date < '{date}')
    ORDER BY gi.date DESC
    LIMIT {lim};
    '''
    
    if team == 'away':
        return away_gid_query
    elif team == 'home':
        return home_gid_query

def create_4F_df(query, conn, team_stats, team='away'):
    
    gid_query_df = pd.read_sql(query, con=conn)
    stat_df = gid_query_df.merge(team_stats[['game_id', 'team', 'FG', 'FGA', 'FT', 'FTA', '3P', 'ORB%', 'TOV']])
    
    if team == 'away':
        stat_df = stat_df[stat_df['team'] == away_team].reset_index(drop=True)
    elif team =='home':
        stat_df = stat_df[stat_df['team'] == home_team].reset_index(drop=True)
    
    eFGp = (stat_df['FG'].mean() + (0.5 * stat_df['3P'].mean())) / stat_df['FGA'].mean()
    FTr = stat_df['FT'].mean() / stat_df['FGA'].mean()
    ORBp = stat_df['ORB%'].mean()*0.01
    TOVp = stat_df['TOV'].mean() / (stat_df['FGA'].mean() + (0.44*stat_df['FTA'].mean()) + stat_df['TOV'].mean())
    
    if team == 'away':
        comp_df = pd.DataFrame(data = [[eFGp, FTr, ORBp, TOVp]], columns=['a_eFGp', 'a_FTr', 'a_ORBp', 'a_TOVp'])
    elif team =='home':
        comp_df = pd.DataFrame(data = [[eFGp, FTr, ORBp, TOVp]], columns=['h_eFGp', 'h_FTr', 'h_ORBp', 'h_TOVp'])
    
    return comp_df

def create_4F_df_individual(query, conn, team_stats, team='away'):
    gid_query_df = pd.read_sql(query, con=conn)
    stat_df = gid_query_df.merge(team_stats[['game_id', 'team', 'FG', 'FGA', 'FT', 'FTA', '3P', 'ORB%', 'TOV']])
    
    if team == 'away':
        stat_df = stat_df[stat_df['team'] == away_team].reset_index(drop=True)
    elif team == 'home':
        stat_df = stat_df[stat_df['team'] == home_team].reset_index(drop=True)
    
    eFGp = (stat_df['FG'] + (0.5 * stat_df['3P'])) / stat_df['FGA']
    FTr = stat_df['FT'] / stat_df['FGA']
    ORBp = stat_df['ORB%'] * 0.01
    TOVp = stat_df['TOV'] / (stat_df['FGA'] + (0.44 * stat_df['FTA']) + stat_df['TOV'])
    
    # Create column names dynamically for each statistic
    columns = (
        [f'eFGp_{i+1}' for i in range(len(eFGp))] +
        [f'FTr_{i+1}' for i in range(len(FTr))] +
        [f'ORBp_{i+1}' for i in range(len(ORBp))] +
        [f'TOVp_{i+1}' for i in range(len(TOVp))]
    )
    
    # Combine all statistics into a single DataFrame
    comp_df = pd.DataFrame(
        data=[list(eFGp) + list(FTr) + list(ORBp) + list(TOVp)],
        columns=columns
    )
    
    # Prefix column names based on the team
    if team == 'away':
        comp_df.columns = [f'a_{col}' for col in comp_df.columns]
    elif team == 'home':
        comp_df.columns = [f'h_{col}' for col in comp_df.columns]
    
    return comp_df

team_factor_10_path = "cache_data_2/team_factor_10.csv"

if not os.path.exists(team_factor_10_path):
  four_factor_columns = list(game_info.columns) + ['a_eFGp', 'a_FTr', 'a_ORBp', 'a_TOVp', 'h_eFGp', 'h_FTr', 'h_ORBp', 'h_TOVp']
  team_factor_10 = pd.DataFrame(columns=four_factor_columns)

  for game_id in game_ids:

      date = game_info[game_info['game_id'] == game_id]['date'].values[0]
      away_team = game_info[game_info['game_id'] == game_id]['away_team'].values[0]
      home_team = game_info[game_info['game_id'] == game_id]['home_team'].values[0]

      away_gid_query = create_gid_query(game_id = game_id, date=date, lim=10, team='away')
      away_stats = create_4F_df(query=away_gid_query, conn=conn, team_stats=team_stats, team='away')

      home_gid_query = create_gid_query(game_id=game_id, date=date, lim=10, team='home')
      home_stats = create_4F_df(query=home_gid_query, conn=conn, team_stats=team_stats, team='home')
      
      agg_stats = pd.concat([away_stats, home_stats], axis=1)
      gid_info = game_info[game_info['game_id'] == game_id].reset_index(drop=True)
      
      stats_4F = pd.concat([gid_info, agg_stats], axis=1)
      team_factor_10 = pd.concat([team_factor_10, stats_4F], ignore_index=True)

team_factor_10_individual_path = "cache_data_2/team_factor_individual_10.csv"

if not os.path.exists(team_factor_10_individual_path):
  limit = 10
  stat_columns = (
        [f'a_eFGp_{i+1}' for i in range(limit)] +
        [f'a_FTr_{i+1}' for i in range(limit)] +
        [f'a_ORBp_{i+1}' for i in range(limit)] +
        [f'a_TOVp_{i+1}' for i in range(limit)] +
        [f'h_eFGp_{i+1}' for i in range(limit)] +
        [f'h_FTr_{i+1}' for i in range(limit)] +
        [f'h_ORBp_{i+1}' for i in range(limit)] +
        [f'h_TOVp_{i+1}' for i in range(limit)]
    )
  four_factor_columns = list(game_info.columns) + stat_columns
  team_factor_10 = pd.DataFrame(columns=four_factor_columns)

  counter = 0
  start_time = time.time()
  prev_time = start_time

  for game_id in game_ids:

      date = game_info[game_info['game_id'] == game_id]['date'].values[0]
      away_team = game_info[game_info['game_id'] == game_id]['away_team'].values[0]
      home_team = game_info[game_info['game_id'] == game_id]['home_team'].values[0]

      away_gid_query = create_gid_query(game_id = game_id, date=date, lim=limit, team='away')
      away_stats = create_4F_df_individual(query=away_gid_query, conn=conn, team_stats=team_stats, team='away')

      home_gid_query = create_gid_query(game_id=game_id, date=date, lim=limit, team='home')
      home_stats = create_4F_df_individual(query=home_gid_query, conn=conn, team_stats=team_stats, team='home')

      agg_stats = pd.concat([away_stats, home_stats], axis=1)
      gid_info = game_info[game_info['game_id'] == game_id].reset_index(drop=True)
      
      stats_4F = pd.concat([gid_info, agg_stats], axis=1)
      team_factor_10 = pd.concat([team_factor_10, stats_4F], ignore_index=True)

      counter += 1

      #print_freq = 1000
      #if counter % 1000 == 0:
      #  #avg_time_per_iteration = (time.time() - start_time) / counter
      #  new_time = time.time()
      #  avg_time_per_iteration = (new_time - prev_time)
      #  prev_time = new_time
      #  eta = (len(game_ids)-counter) * (avg_time_per_iteration/1000)
      #  print("Progress: " + str(counter/len(game_ids)) + " ETA: " + str(eta) + " s")
      
    
  team_factor_10.to_csv(team_factor_10_individual_path)

conn.close()