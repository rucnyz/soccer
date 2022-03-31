# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 17:41
# @Author  : nieyuzhou
# @File    : extract_sql.py
# @Software: PyCharm
import os
import sqlite3
import pandas as pd

path = "./data/"
database = path + 'database.sqlite'
conn = sqlite3.connect(database)

# 获取数据
player_data = pd.read_sql("SELECT * FROM Player;", conn)
player_stats_data = pd.read_sql("SELECT * FROM Player_Attributes;", conn)
team_data = pd.read_sql("SELECT * FROM Team;", conn)
team_stats_data = pd.read_sql("SELECT * FROM Team_Attributes;", conn)
match_data = pd.read_sql("SELECT * FROM Match;", conn)
league_data = pd.read_sql("SELECT * FROM League;", conn)
country_data = pd.read_sql("SELECT * FROM Country;", conn)

player_data.to_csv(os.path.join(path, "player.csv"), index_col = 0)
player_stats_data.to_csv(os.path.join(path, "player_attr.csv"), index_col = 0)
team_data.to_csv(os.path.join(path, "team.csv"), index_col = 0)
team_stats_data.to_csv(os.path.join(path, "team_attr.csv"), index_col = 0)
match_data.to_csv(os.path.join(path, "match.csv"), index_col = 0)
league_data.to_csv(os.path.join(path, "league.csv"), index_col = 0)
country_data.to_csv(os.path.join(path, "country.csv"), index_col = 0)
