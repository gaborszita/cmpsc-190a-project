# Data schema:
# The database contains 3 tables:
# game_info columns: ['game_id', 'season', 'date', 'away_team', 'away_score', 'home_team', 'home_score', 'result']
# team_stats columns: ['game_id', 'team', 'MP', 'FG', 'FTr', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', '+/-', 'TS%', 'eFG%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'ORtg', 'DRtg', 'BPM']
# player_stats columns: ['game_id', 'player', 'team', 'MP', 'FG', 'FTr', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', '+/-', 'TS%', 'eFG%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'ORtg', 'DRtg', 'BPM', 'PIE']

import sqlite3
import os
import csv
from itertools import islice

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
  print("Fetching game_info_detailed...")
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

game_info_detailed_2_path = "cache_data/game_info_2_detailed.csv"

if not os.path.exists(game_info_detailed_2_path):
  print("Fetching game_info_detailed_2...")
  res = cur.execute("""
                    SELECT 
                        gi.result,
                        gi.away_team,
                        gi.home_team,

                        -- Home team four factors
                        (SELECT AVG(ts."eFG%") FROM team_stats ts 
                        JOIN game_info gi2 ON ts.game_id = gi2.game_id 
                        WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
                          AND ts.team = gi.home_team 
                          AND gi2.date < gi.date 
                        ORDER BY gi2.date DESC LIMIT 10) AS home_eFG,

                        (SELECT AVG(ts."TOV%") FROM team_stats ts 
                        JOIN game_info gi2 ON ts.game_id = gi2.game_id 
                        WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
                          AND ts.team = gi.home_team 
                          AND gi2.date < gi.date 
                        ORDER BY gi2.date DESC LIMIT 10) AS home_TOV,

                        (SELECT AVG(ts."ORB%") FROM team_stats ts 
                        JOIN game_info gi2 ON ts.game_id = gi2.game_id 
                        WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
                          AND ts.team = gi.home_team 
                          AND gi2.date < gi.date 
                        ORDER BY gi2.date DESC LIMIT 10) AS home_ORB,

                        (SELECT AVG(ts."FTr") FROM team_stats ts 
                        JOIN game_info gi2 ON ts.game_id = gi2.game_id 
                        WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
                          AND ts.team = gi.home_team 
                          AND gi2.date < gi.date 
                        ORDER BY gi2.date DESC LIMIT 10) AS home_FT_FTr,

                        -- Away team four factors
                        (SELECT AVG(ts."eFG%") FROM team_stats ts 
                        JOIN game_info gi2 ON ts.game_id = gi2.game_id 
                        WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
                          AND ts.team = gi.away_team 
                          AND gi2.date < gi.date 
                        ORDER BY gi2.date DESC LIMIT 10) AS away_eFG,

                        (SELECT AVG(ts."TOV%") FROM team_stats ts 
                        JOIN game_info gi2 ON ts.game_id = gi2.game_id 
                        WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
                          AND ts.team = gi.away_team 
                          AND gi2.date < gi.date 
                        ORDER BY gi2.date DESC LIMIT 10) AS away_TOV,

                        (SELECT AVG(ts."ORB%") FROM team_stats ts 
                        JOIN game_info gi2 ON ts.game_id = gi2.game_id 
                        WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
                          AND ts.team = gi.away_team 
                          AND gi2.date < gi.date 
                        ORDER BY gi2.date DESC LIMIT 10) AS away_ORB,

                        (SELECT AVG(ts."FTr") FROM team_stats ts 
                        JOIN game_info gi2 ON ts.game_id = gi2.game_id 
                        WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
                          AND ts.team = gi.away_team 
                          AND gi2.date < gi.date 
                        ORDER BY gi2.date DESC LIMIT 10) AS away_FT_FTr

                    FROM game_info gi
                    ORDER BY gi.date;
                    """)
  
  game_info_detailed_2 = cur.fetchall()

  with open(game_info_detailed_2_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(game_info_detailed_2)

game_info_detailed_2 = []

with open(game_info_detailed_2_path, 'r', newline='') as file:
  reader = csv.reader(file)
  counter = 0
  for row in reader:
    if counter > 50:
      try:
        game_info_detailed_2.append([int(row[0]), row[1], row[2]] + [float(elem) for elem in row[3:]])
      except:
        pass
    counter += 1
  #game_info_detailed_2 = [[int(row[0]), row[1], row[2]] + [float(elem) for elem in row[3:]] for row in islice(reader, 100, None)]

game_info_last_20_all_path = "cache_data/game_info_last_20_all_path.csv"

if not os.path.exists(game_info_last_20_all_path):
  print("Fetching game_info_last_20_all_path...")
  res = cur.execute("""
                    SELECT 
    gi.result,
    gi.away_team,
    gi.home_team,

    -- Home team eFG% for last 10 games
    (SELECT ts."eFG%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 0) AS home_eFG_1,
    (SELECT ts."eFG%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 1) AS home_eFG_2,
    (SELECT ts."eFG%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 2) AS home_eFG_3,
    (SELECT ts."eFG%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 3) AS home_eFG_4,
    (SELECT ts."eFG%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 4) AS home_eFG_5,
    (SELECT ts."eFG%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 5) AS home_eFG_6,
    (SELECT ts."eFG%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 6) AS home_eFG_7,
    (SELECT ts."eFG%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 7) AS home_eFG_8,
    (SELECT ts."eFG%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 8) AS home_eFG_9,
    (SELECT ts."eFG%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 9) AS home_eFG_10,

    -- Home team TOV% for last 10 games
    (SELECT ts."TOV%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 0) AS home_TOV_1,
    (SELECT ts."TOV%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 1) AS home_TOV_2,
    (SELECT ts."TOV%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 2) AS home_TOV_3,
    (SELECT ts."TOV%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 3) AS home_TOV_4,
    (SELECT ts."TOV%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 4) AS home_TOV_5,
    (SELECT ts."TOV%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 5) AS home_TOV_6,
    (SELECT ts."TOV%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 6) AS home_TOV_7,
    (SELECT ts."TOV%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 7) AS home_TOV_8,
    (SELECT ts."TOV%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 8) AS home_TOV_9,
    (SELECT ts."TOV%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 9) AS home_TOV_10,

    -- Home team ORB% for last 10 games
    (SELECT ts."ORB%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 0) AS home_ORB_1,
    (SELECT ts."ORB%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 1) AS home_ORB_2,
    (SELECT ts."ORB%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 2) AS home_ORB_3,
    (SELECT ts."ORB%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 3) AS home_ORB_4,
    (SELECT ts."ORB%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 4) AS home_ORB_5,
    (SELECT ts."ORB%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 5) AS home_ORB_6,
    (SELECT ts."ORB%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 6) AS home_ORB_7,
    (SELECT ts."ORB%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 7) AS home_ORB_8,
    (SELECT ts."ORB%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 8) AS home_ORB_9,
    (SELECT ts."ORB%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 9) AS home_ORB_10,

    -- Home team FT/FTr for last 10 games
    (SELECT ts."FTr" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 0) AS home_FT_FTr_1,
    (SELECT ts."FTr" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 1) AS home_FT_FTr_2,
    (SELECT ts."FTr" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 2) AS home_FT_FTr_3,
    (SELECT ts."FTr" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 3) AS home_FT_FTr_4,
    (SELECT ts."FTr" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 4) AS home_FT_FTr_5,
    (SELECT ts."FTr" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 5) AS home_FT_FTr_6,
    (SELECT ts."FTr" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 6) AS home_FT_FTr_7,
    (SELECT ts."FTr" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 7) AS home_FT_FTr_8,
    (SELECT ts."FTr" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 8) AS home_FT_FTr_9,
    (SELECT ts."FTr" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.home_team OR gi2.away_team = gi.home_team) 
       AND ts.team = gi.home_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 9) AS home_FT_FTr_10,

    -- Away team eFG% for last 10 games
    (SELECT ts."eFG%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 0) AS away_eFG_1,
    (SELECT ts."eFG%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 1) AS away_eFG_2,
    (SELECT ts."eFG%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 2) AS away_eFG_3,
    (SELECT ts."eFG%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 3) AS away_eFG_4,
    (SELECT ts."eFG%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 4) AS away_eFG_5,
    (SELECT ts."eFG%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 5) AS away_eFG_6,
    (SELECT ts."eFG%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 6) AS away_eFG_7,
    (SELECT ts."eFG%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 7) AS away_eFG_8,
    (SELECT ts."eFG%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 8) AS away_eFG_9,
    (SELECT ts."eFG%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 9) AS away_eFG_10,

    -- Away team TOV% for last 10 games
    (SELECT ts."TOV%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 0) AS away_TOV_1,
    (SELECT ts."TOV%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 1) AS away_TOV_2,
    (SELECT ts."TOV%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 2) AS away_TOV_3,
    (SELECT ts."TOV%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 3) AS away_TOV_4,
    (SELECT ts."TOV%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 4) AS away_TOV_5,
    (SELECT ts."TOV%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 5) AS away_TOV_6,
    (SELECT ts."TOV%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 6) AS away_TOV_7,
    (SELECT ts."TOV%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 7) AS away_TOV_8,
    (SELECT ts."TOV%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 8) AS away_TOV_9,
    (SELECT ts."TOV%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 9) AS away_TOV_10,

    -- Away team ORB% for last 10 games
    (SELECT ts."ORB%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 0) AS away_ORB_1,
    (SELECT ts."ORB%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 1) AS away_ORB_2,
    (SELECT ts."ORB%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 2) AS away_ORB_3,
    (SELECT ts."ORB%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 3) AS away_ORB_4,
    (SELECT ts."ORB%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 4) AS away_ORB_5,
    (SELECT ts."ORB%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 5) AS away_ORB_6,
    (SELECT ts."ORB%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 6) AS away_ORB_7,
    (SELECT ts."ORB%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 7) AS away_ORB_8,
    (SELECT ts."ORB%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 8) AS away_ORB_9,
    (SELECT ts."ORB%" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 9) AS away_ORB_10,

    -- Away team FT/FTr for last 10 games
    (SELECT ts."FTr" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 0) AS away_FT_FTr_1,
    (SELECT ts."FTr" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 1) AS away_FT_FTr_2,
    (SELECT ts."FTr" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 2) AS away_FT_FTr_3,
    (SELECT ts."FTr" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 3) AS away_FT_FTr_4,
    (SELECT ts."FTr" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 4) AS away_FT_FTr_5,
    (SELECT ts."FTr" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 5) AS away_FT_FTr_6,
    (SELECT ts."FTr" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 6) AS away_FT_FTr_7,
    (SELECT ts."FTr" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 7) AS away_FT_FTr_8,
    (SELECT ts."FTr" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 8) AS away_FT_FTr_9,
    (SELECT ts."FTr" FROM team_stats ts 
     JOIN game_info gi2 ON ts.game_id = gi2.game_id 
     WHERE (gi2.home_team = gi.away_team OR gi2.away_team = gi.away_team) 
       AND ts.team = gi.away_team 
       AND gi2.date < gi.date 
     ORDER BY gi2.date DESC LIMIT 1 OFFSET 9) AS away_FT_FTr_10

FROM game_info gi
ORDER BY gi.date;
                    """)

  game_info_last_20_all = cur.fetchall()

  with open(game_info_last_20_all_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(game_info_last_20_all)

game_info_last_20_all = []

with open(game_info_last_20_all_path, 'r', newline='') as file:
  reader = csv.reader(file)
  for row in reader:
    try:
      game_info_last_20_all.append([int(row[0]), row[1], row[2]] + [float(elem) for elem in row[3:]])
    except:
      pass
  #game_info_detailed_2 = [[int(row[0]), row[1], row[2]] + [float(elem) for elem in row[3:]] for row in islice(reader, 100, None)]

con.close()