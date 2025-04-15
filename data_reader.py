# Data schema:
# The database contains 3 tables:
# game_info columns: ['game_id', 'season', 'date', 'away_team', 'away_score', 'home_team', 'home_score', 'result']
# team_stats columns: ['game_id', 'team', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', '+/-', 'TS%', 'eFG%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'ORtg', 'DRtg', 'BPM']
# player_stats columns: ['game_id', 'player', 'team', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', '+/-', 'TS%', 'eFG%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'ORtg', 'DRtg', 'BPM', 'PIE']

import sqlite3

con = sqlite3.connect('NBA-Web-Scraper/data/NBA-Boxscore-Database.sqlite')
cur = con.cursor()

"""
# print top 5 rows from game_info
print("GAME INFO HEAD")
cur.execute("SELECT * FROM game_info LIMIT 5")
print([description[0] for description in cur.description])
rows = cur.fetchall()
for row in rows:
  print(row)

# print top 5 rows from team_stats
print("TEAM STATS HEAD")
cur.execute("SELECT * FROM team_stats LIMIT 5")
print([description[0] for description in cur.description])
rows = cur.fetchall()
for row in rows:
  print(row)

# print top 2 rows from player_stats
print("PLAYER STATS HEAD")
cur.execute("SELECT * FROM player_stats LIMIT 2")
print([description[0] for description in cur.description])
rows = cur.fetchall()
for row in rows:
  print(row)
"""

cur.execute("SELECT * FROM game_info")
game_info = cur.fetchall()

cur.execute("SELECT * FROM team_stats")
team_stats = cur.fetchall()

cur.execute("SELECT * FROM player_stats")
player_stats = cur.fetchall()

train_test_date_cutoff = "2024-01-01";
res = cur.execute("SELECT * FROM game_info WHERE date < " + train_test_date_cutoff)
game_info_train = cur.fetchall()
res = cur.execute("SELECT * FROM game_info WHERE date >= " + train_test_date_cutoff)
game_info_test = cur.fetchall()


con.close()