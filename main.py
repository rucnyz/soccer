# -*- coding: utf-8 -*-
# @Time    : 2022/3/20 21:56
# @Author  : nieyuzhou
# @File    : main.py
# @Software: PyCharm
import argparse
import os.path
import random
import warnings
from time import time

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from train import train

warnings.simplefilter("ignore")


def parse_my_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default = random.randint(0, 9999999))
    parser.add_argument('--visual', type = int, default = 0)
    parser.add_argument('--metric', type = str, default = "f1", choices = ["pre", "recall", "f1"])
    parser.add_argument('--average', type = str, default = "weighted", choices = ["macro", "micro", "weighted"])
    parser.add_argument('--n_jobs', type = int, default = 1)
    arg = parser.parse_args()
    if arg.metric == "pre":
        arg.metric_fn = precision_score
    elif arg.metric == "recall":
        arg.metric_fn = recall_score
    elif arg.metric == 'f1':
        arg.metric_fn = f1_score
    return arg


if __name__ == '__main__':
    start = time()
    args = parse_my_args()
    data_path = "./data/"  # 数据文件夹
    # 网格搜索
    # 得到原始数据
    player_data = pd.read_csv(os.path.join(data_path, "player.csv"), index_col = 0)
    player_stats_data = pd.read_csv(os.path.join(data_path, "player_attr.csv"), index_col = 0)
    team_data = pd.read_csv(os.path.join(data_path, "team.csv"), index_col = 0)
    match_data = pd.read_csv(os.path.join(data_path, "match.csv"), index_col = 0)

    # 去掉na项
    rows = ["country_id", "league_id", "season", "stage", "date", "match_api_id", "home_team_api_id",
            "away_team_api_id", "home_team_goal", "away_team_goal", "home_player_1", "home_player_2",
            "home_player_3", "home_player_4", "home_player_5", "home_player_6", "home_player_7",
            "home_player_8", "home_player_9", "home_player_10", "home_player_11", "away_player_1",
            "away_player_2", "away_player_3", "away_player_4", "away_player_5", "away_player_6",
            "away_player_7", "away_player_8", "away_player_9", "away_player_10", "away_player_11"]
    match_data.dropna(subset = rows, inplace = True)
    # match_data = match_data.tail(1500)
    # 开始训练
    train(data_path, match_data, player_stats_data, args)
    end = time()
    print("总计运行时间 {:.1f} 分钟".format((end - start) / 60))
