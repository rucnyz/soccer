# -*- coding: utf-8 -*-
# @Time    : 2022/3/30 21:00
# @Author  : nieyuzhou
# @File    : leaderboard.py
# @Software: PyCharm
import os

import networkx as nx
import pandas as pd
import warnings

warnings.simplefilter("ignore")


def get_win_lose(x):
    if (x["home_team_goal"] - x["away_team_goal"]) > 0:
        x["outcome"] = 1
    elif (x["home_team_goal"] - x["away_team_goal"]) == 0:
        x["outcome"] = 0
    elif (x["home_team_goal"] - x["away_team_goal"]) < 0:
        x["outcome"] = -1
    return x.drop(["home_team_goal", "away_team_goal"])


def get_team_name(x):
    return [team_data[team_data["team_api_id"] == x[0]]["team_long_name"].item(), x[1]]


if __name__ == '__main__':
    data_path = "./data/"  # 数据文件夹
    match_data = pd.read_csv(os.path.join(data_path, "match.csv"), index_col = 0)
    team_data = pd.read_csv(os.path.join(data_path, "team.csv"), index_col = 0)
    # 找到所属联赛，以西甲为例
    match_data = match_data[match_data["league_id"] == 21518].loc[:,
                 ["home_team_api_id", "away_team_api_id", "home_team_goal", "away_team_goal"]]
    graph_data = match_data.apply(get_win_lose, axis = 1)
    graph_data = graph_data.groupby(["home_team_api_id", "away_team_api_id"], as_index = False).sum()
    team_list = set(graph_data["home_team_api_id"])
    team_list.update(set(graph_data["away_team_api_id"]))
    team_list = list(team_list)
    result_data = pd.DataFrame(columns = ["team_1", "team_2", "outcome"])
    for i in range(len(team_list)):
        for j in range(i + 1, len(team_list)):
            team1 = team_list[i]
            team2 = team_list[j]
            tmp = graph_data[graph_data["home_team_api_id"] == team1]
            team_12 = tmp[tmp["away_team_api_id"] == team2]
            tmp = graph_data[graph_data["home_team_api_id"] == team2]
            team_21 = tmp[tmp["away_team_api_id"] == team1]
            flag = False
            if team_12.empty and team_21.empty:
                goal = goal1 = goal2 = 0
                flag = False
            elif team_12.empty:
                goal2 = team_21.iloc[0, 2]
                goal1 = 0
            elif team_21.empty:
                goal1 = team_12.iloc[0, 2]
                goal2 = 0
            else:
                goal1 = team_12.iloc[0, 2]
                goal2 = team_21.iloc[0, 2]
            goal = goal1 - goal2
            if goal > 0:
                result_data = result_data.append(
                    {"team_1": team2, "team_2": team1, "outcome": goal},
                    ignore_index = True)
            elif goal == 0:
                if flag:
                    result_data = result_data.append(
                        {"team_1": team2, "team_2": team1, "outcome": abs(goal1)}, ignore_index = True)
                    result_data = result_data.append(
                        {"team_1": team1, "team_2": team2, "outcome": abs(goal2)}, ignore_index = True)
            else:
                result_data = result_data.append(
                    {"team_1": team1, "team_2": team2, "outcome": -goal}, ignore_index = True)
    # graph_data["outcome"] = graph_data["outcome"].where(graph_data["outcome"] >= 0, 0)
    # graph_data = graph_data.drop(graph_data[graph_data["outcome"] < 0].index)
    # 调换两列顺序
    # graph_data = graph_data[["away_team_api_id", "home_team_api_id", "outcome"]]
    G = nx.DiGraph()
    G.add_weighted_edges_from(result_data.values)
    board = nx.pagerank(G)
    board = sorted(board.items(), key = lambda s: s[1], reverse = True)
    result = list(map(get_team_name, board))
