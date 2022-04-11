# -*- coding: utf-8 -*-
# @Time    : 2022/4/10 20:39
# @Author  : nieyuzhou
# @File    : strategy.py
# @Software: PyCharm
import os

import pandas as pd

from test import test
from train import extract_feat


def get_model_odds(x):
    if x['predict'] == "Win":
        return x["B365H"]
    elif x['predict'] == "Defeat":
        return x["B365A"]
    else:
        return x["B365D"]


def display_result(investment, predict, describe):
    print(describe + ":")
    print("  成本:", end = " ")
    print(investment, "元")
    correct_perc = (predict != 0).sum() / predict.shape[0]
    print("  正确预测比例:", end = " ")
    print("{:.2f}%".format(correct_perc * 100), end = "    ")
    profit = sum(predict)
    print("  盈利:", end = " ")
    print("{:.2f} 元".format(profit - investment), end = "    ")
    profit_perc = (profit - investment) / investment
    print("  盈利比例:", end = " ")
    print("{:.2f}%".format(profit_perc * 100))
    print("---------------------------")


if __name__ == '__main__':
    data_path = "../data/"  # 数据文件夹
    country_data = pd.read_csv(os.path.join(data_path, "country.csv"), index_col = 0)
    league_data = pd.read_csv(os.path.join(data_path, "league.csv"), index_col = 0)
    match_data = pd.read_csv(os.path.join(data_path, "match.csv"), index_col = 0)
    # 选择大规模的联赛
    selected_countries = ['England', 'France', 'Germany', 'Italy', 'Spain', 'Netherlands', 'Portugal']
    countries = country_data[country_data["name"].isin(selected_countries)]
    leagues = countries.merge(league_data, on = 'id', suffixes = ('', '_y'))
    # 为了我们模型能够运行
    rows = ["country_id", "league_id", "season", "stage", "date", "match_api_id", "home_team_api_id",
            "away_team_api_id", "home_team_goal", "away_team_goal", "home_player_1", "home_player_2",
            "home_player_3", "home_player_4", "home_player_5", "home_player_6", "home_player_7",
            "home_player_8", "home_player_9", "home_player_10", "home_player_11", "away_player_1",
            "away_player_2", "away_player_3", "away_player_4", "away_player_5", "away_player_6",
            "away_player_7", "away_player_8", "away_player_9", "away_player_10", "away_player_11"]
    match_data.dropna(subset = rows, inplace = True)
    # 选出联赛
    match_data = match_data[match_data['league_id'].isin(leagues["id"])]
    matches = match_data[
        ['id', 'country_id', 'league_id', 'season', 'stage', 'date', 'match_api_id', 'home_team_api_id',
         'away_team_api_id', 'home_team_goal', 'away_team_goal', 'B365H', 'B365D', 'B365A']].copy()
    matches.dropna(inplace = True)
    # 得到比赛结果
    matches.loc[:, 'result'] = 'H'
    matches.loc[matches.home_team_goal == matches.away_team_goal, "result"] = 'D'
    matches.loc[matches.home_team_goal < matches.away_team_goal, "result"] = 'A'

    # 得到最安全和最危险的赔率
    matches.loc[:, 'safest_odds'] = matches.apply(lambda x: min(x[11], x[12], x[13]), axis = 1)
    matches.loc[:, 'longshot_odds'] = matches.apply(lambda x: max(x[11], x[12], x[13]), axis = 1)

    # 最安全和最冒险的预测
    matches.loc[:, 'safest_outcome'] = 'H'
    matches.loc[matches.B365D == matches.safest_odds, "safest_outcome"] = 'D'
    matches.loc[matches.B365A == matches.safest_odds, "safest_outcome"] = 'A'

    matches.loc[:, 'longshot_outcome'] = 'A'
    matches.loc[matches["B365D"] == matches["longshot_odds"], "longshot_outcome"] = 'D'
    matches.loc[matches["B365H"] == matches["longshot_odds"], "longshot_outcome"] = 'H'
    # 得到盈利情况，假设每次投注10元钱
    matches.loc[:, 'safest_bet_payout'] = matches["safest_odds"] * 10
    matches.loc[matches["safest_outcome"] != matches["result"], 'safest_bet_payout'] = 0

    matches.loc[:, 'longshot_bet_payout'] = matches["longshot_odds"] * 10
    matches.loc[matches["longshot_outcome"] != matches["result"], 'longshot_bet_payout'] = 0
    # 得到成本
    invest = 10 * matches.shape[0]
    # 得到最终情况
    # 用我们的模型试试
    if os.path.exists(os.path.join(data_path, "processed/bet.csv")):
        inputs = pd.read_csv(os.path.join(data_path, "processed/bet.csv"))
    else:
        player_data = pd.read_csv(os.path.join(data_path, "player.csv"), index_col = 0)
        player_stats_data = pd.read_csv(os.path.join(data_path, "player_attr.csv"), index_col = 0)
        team_data = pd.read_csv(os.path.join(data_path, "team_attr.csv"), index_col = 0)
        inputs = extract_feat(data_path, match_data, player_stats_data, team_data)
        inputs.to_csv(os.path.join(data_path, "processed/bet.csv"))
    inputs[['League_1.0', 'League_15722.0', 'League_19694.0', 'League_24558.0']] = 0
    y_pred = test(inputs.drop("match_api_id", axis = 1), data_path, 0)
    inputs["predict"] = y_pred
    rest_data = pd.merge(matches, inputs, on = 'match_api_id')
    # 得到赔率
    rest_data.loc[:, 'my_odds'] = rest_data.apply(get_model_odds, axis = 1)
    # 得到盈利情况
    rest_data.loc[:, 'my_bet_payout'] = rest_data["my_odds"] * 10
    rest_data.loc[rest_data["predict"] != rest_data["label"], 'my_bet_payout'] = 0
    display_result(10 * rest_data.shape[0], rest_data["my_bet_payout"], "我们的模型策略")
    # 安全策略预测比例甚至超过了0.5，但是仍然赔钱
    display_result(invest, matches["safest_bet_payout"], "最安全策略")
    # 最冒险的策略亏麻了
    display_result(invest, matches["longshot_bet_payout"], "最冒险策略")
