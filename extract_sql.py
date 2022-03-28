# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 17:41
# @Author  : nieyuzhou
# @File    : extract_sql.py
# @Software: PyCharm
import sqlite3
import pandas as pd

path = "./data/"  # Insert path here
database = path + 'database.sqlite'
conn = sqlite3.connect(database)

# Fetching required data tables
player_data = pd.read_sql("SELECT * FROM Player;", conn)
player_stats_data = pd.read_sql("SELECT * FROM Player_Attributes;", conn)
team_data = pd.read_sql("SELECT * FROM Team;", conn)
match_data = pd.read_sql("SELECT * FROM Match;", conn)
