# Data schema:
# The database contains 3 tables:
# game_info columns: ['game_id', 'season', 'date', 'away_team', 'away_score', 'home_team', 'home_score', 'result']
# team_stats columns: ['game_id', 'team', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', '+/-', 'TS%', 'eFG%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'ORtg', 'DRtg', 'BPM']
# player_stats columns: ['game_id', 'player', 'team', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', '+/-', 'TS%', 'eFG%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'ORtg', 'DRtg', 'BPM', 'PIE']

import sqlite3
import os
import csv

con = sqlite3.connect('NBA-Web-Scraper/data/NBA-Boxscore-Database.sqlite')
cur = con.cursor()

# game_info is the entire game_info table
cur.execute("SELECT * FROM game_info")
game_info = cur.fetchall()

# team_stats is the entire team_stats table
cur.execute("SELECT * FROM team_stats")
team_stats = cur.fetchall()

# player_stats is teh entire player_stats table
cur.execute("SELECT * FROM player_stats")
player_stats = cur.fetchall()

# game_info_detailed contains information about a game, like game_info,
# but with more information
# it contains the attributes result, away_team, home_team,
# ts_home.PTS, ts_home.`FG%`, ts_home.`3P%`, ts_home.`FT%`, ts_home.AST, ts_home.STL, ts_home.ORtg, ts_home.`TS%`, ts_home.STL, ts_home.BLK, ts_home.DRtg, ts_home.`DRB%`, ts_home.`eFG%`, ts_home.`TOV%`,
# ts_away.PTS, ts_away.`FG%`, ts_away.`3P%`, ts_away.`FT%`, ts_away.AST, ts_away.STL, ts_away.ORtg, ts_away.`TS%`, ts_away.STL, ts_away.BLK, ts_away.DRtg, ts_away.`DRB%`, ts_away.`eFG%`, ts_away.`TOV%`
#
# away_team and home_team are strings, everything else is an int

# It takes long to query this data from the database, so we only query it on the first run.
# On the first run we export it to a csv file, so in subsequent runs we can just read the csv file.
game_info_detailed_path = "cache_data/game_info_detailed.csv"

# only query from the database if the csv file does not exist
if not os.path.exists(game_info_detailed_path):
  # gave ChatGPT the list of fields in the team_stats table ans asked it
  # "which of these are most useful for basketball prediction?"
  # to select useful features
  res = cur.execute("SELECT result, away_team, home_team, " \
                    "ts_home.PTS, ts_home.`FG%`, ts_home.`3P%`, ts_home.`FT%`, ts_home.AST, ts_home.STL, ts_home.ORtg, ts_home.`TS%`, ts_home.STL, ts_home.BLK, ts_home.DRtg, ts_home.`DRB%`, ts_home.`eFG%`, ts_home.`TOV%`, " \
                    "ts_away.PTS, ts_away.`FG%`, ts_away.`3P%`, ts_away.`FT%`, ts_away.AST, ts_away.STL, ts_away.ORtg, ts_away.`TS%`, ts_away.STL, ts_away.BLK, ts_away.DRtg, ts_away.`DRB%`, ts_away.`eFG%`, ts_away.`TOV%` " \
                    "FROM game_info " \
                    "INNER JOIN team_stats ts_home ON " \
                    "ts_home.game_id = " \
                    "(SELECT game_id FROM game_info game_info_2 " \
                      "WHERE game_info_2.date < game_info.date AND (game_info_2.home_team = game_info.home_team OR  game_info_2.away_team = game_info.home_team)"\
                      "ORDER BY date DESC LIMIT 1) " \
                    "AND ts_home.team = game_info.home_team " \
                    "INNER JOIN team_stats ts_away ON " \
                    "ts_away.game_id = " \
                    "(SELECT game_id FROM game_info game_info_2 " \
                      "WHERE game_info_2.date < game_info.date AND (game_info_2.home_team = game_info.away_team OR  game_info_2.away_team = game_info.away_team)"\
                      "ORDER BY date DESC LIMIT 1) " \
                    "AND ts_away.team = game_info.away_team "
                    "ORDER BY date ")
  game_info_detailed = cur.fetchall()

  with open(game_info_detailed_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(game_info_detailed)

with open(game_info_detailed_path, 'r', newline='') as file:
  reader = csv.reader(file)
  game_info_detailed = [[int(row[0]), row[1], row[2]] + [float(elem) for elem in row[3:]] for row in reader]

con.close()