# -*- coding: utf-8 -*-
# @Time    : 2022/3/28 17:07
# @Author  : nieyuzhou
# @File    : visualization.py
# @Software: PyCharm
import os
import time

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils.get_data import get_age_for_football_players, get_pos_stats
from utils.visualize import plot_radar, plot_bar, plot_age_dependence


def plot_famous_radar(data, data_six):
    data_six = data_six.groupby(["player_api_id", "player_name"], as_index = False).mean()
    famous, average = get_famous_average(data)
    famous = pd.merge(famous, data_six, on = ['player_api_id', 'player_name'], how = 'left').groupby(
        ['player_api_id', 'player_name'], as_index = False).mean()
    average = pd.merge(average, data_six, on = ['player_api_id', 'player_name'], how = 'left').groupby(
        ['player_api_id', 'player_name'], as_index = False).mean()
    f = famous[['skill', 'movement', 'attacking', 'power', 'mentality', 'defending']].mean()
    a = average[['skill', 'movement', 'attacking', 'power', 'mentality', 'defending']].mean()
    final = pd.concat([f, a], axis = 1).T
    plot_radar(final, kinds = ['famous', 'average'], path = os.path.join(data_path, "pic/radar_famous.png"),
               title = 'famous and average player radar')


def plot_pos_radar(pos_six):
    six = pos_six.groupby(["pos"], as_index = False).mean()
    final = six.drop("pos", axis = 1)
    plot_radar(final, kinds = ["Defender", "Forward", "Goalkeeper", "Midfielder"],
               path = os.path.join(data_path, "pic/radar_pos.png"), title = "player position radar")


def get_position(data, player_stats):
    start = time.time()
    total_data = pd.DataFrame()
    home_p = "home_player_Y"
    home = "home_player_"
    away_p = "away_player_Y"
    away = "away_player_"
    for i in range(1, 12):
        data.loc[:, home + str(i)] = data.loc[:, home + str(i)].astype(int)
        data.loc[:, away + str(i)] = data.loc[:, away + str(i)].astype(int)
        home_data = data[[home + str(i), home_p + str(i), 'date']].copy()
        away_data = data[[away + str(i), away_p + str(i), 'date']].copy()
        home_data["pos"] = pd.cut(home_data[home_p + str(i)], [0, 1, 5, 9.9, 20], labels = ["gk", "def", "mid", "for"])
        away_data["pos"] = pd.cut(away_data[away_p + str(i)], [0, 1, 5, 9.9, 20], labels = ["gk", "def", "mid", "for"])
        # 重命名
        home_data = home_data.rename(columns = {home + str(i): "player_api_id"}).drop(home_p + str(i), axis = 1)
        away_data = away_data.rename(columns = {away + str(i): "player_api_id"}).drop(away_p + str(i), axis = 1)
        tmp_data = pd.concat([home_data, away_data], axis = 0).reset_index(drop = True)
        # 根据时间找到距离比赛最近的球员状态
        tmp_data = tmp_data.apply(lambda x: get_pos_stats(x, player_stats), axis = 1)
        total_data = pd.concat([total_data, tmp_data], axis = 0).reset_index(drop = True)
    print("得到球员位置信息共计用时 {:.4f} 分钟".format((time.time() - start) / 60))
    return total_data


def get_six(data, keep):
    data['skill'] = data['dribbling'] + data['curve'] + +data['free_kick_accuracy'] + data['long_passing'] + data[
        'ball_control']
    data['movement'] = data['acceleration'] + data['sprint_speed'] + data['reactions'] + data['agility'] + data[
        'balance']
    data['attacking'] = data['crossing'] + data['finishing'] + data['heading_accuracy'] + data['short_passing'] + data[
        "volleys"]
    data['power'] = data['stamina'] + data['strength'] + data['jumping'] + data['long_shots'] + data['shot_power']
    data['mentality'] = data['aggression'] + data['interceptions'] + data['positioning'] + data['vision'] + data[
        'penalties']
    data['defending'] = data['standing_tackle'] + data['sliding_tackle']

    scaler = MinMaxScaler([0, 10])
    norm_data = scaler.fit_transform(data[['skill', 'movement', 'attacking', 'power', 'mentality', 'defending']])
    norm_data = pd.DataFrame(norm_data, columns = ['skill', 'movement', 'attacking', 'power', 'mentality', 'defending'])
    norm_data[keep] = data[keep]
    return norm_data


def get_famous_average(data):
    return data[data["overall_rating"] >= 90][['player_api_id', 'player_name']], \
           data[data["overall_rating"] <= 70][['player_api_id', 'player_name']]


if __name__ == '__main__':
    data_path = "./data/"  # 数据文件夹
    match_list = ["home_player_1", "home_player_2", "home_player_3", "home_player_4", "home_player_5", "home_player_6",
                  "home_player_7", "home_player_8", "home_player_9", "home_player_10", "home_player_11",
                  "away_player_1", "away_player_2", "away_player_3", "away_player_4", "away_player_5", "away_player_6",
                  "away_player_7", "away_player_8", "away_player_9", "away_player_10", "away_player_11",
                  "home_player_Y1", "home_player_Y2", "home_player_Y3", "home_player_Y4", "home_player_Y5",
                  "home_player_Y6", "home_player_Y7", "home_player_Y8", "home_player_Y9", "home_player_Y10",
                  "home_player_Y11", "away_player_Y1", "away_player_Y2", "away_player_Y3", "away_player_Y4",
                  "away_player_Y5", "away_player_Y6", "away_player_Y7", "away_player_Y8", "away_player_Y9",
                  "away_player_Y10", "away_player_Y11", "date"]
    # 得到球员数据
    player_data = pd.read_csv(os.path.join(data_path, "player.csv"), index_col = 0).drop(
        ['player_fifa_api_id', "id"], axis = 1)
    player_stats_data = pd.read_csv(os.path.join(data_path, "player_attr.csv"), index_col = 0).drop(
        ['player_fifa_api_id', "id"], axis = 1)
    match_data = pd.read_csv(os.path.join(data_path, "match.csv"), index_col = 0)[match_list]
    player = pd.merge(player_data, player_stats_data, on = 'player_api_id', how = 'inner')
    # pos_data太多了先保存
    if os.path.exists(os.path.join(data_path, "player_pos.csv")):
        pos_data = pd.read_csv(os.path.join(data_path, "player_pos.csv"), index_col = 0)
    else:
        pos_data = get_position(match_data, player)
        pos_data["age"] = pos_data["birthday"].apply(get_age_for_football_players)
    # 按照球场位置
    pos_data.dropna(inplace = True)
    pos_six_data = get_six(pos_data, ["pos"])
    # 明星球员
    famous_six_data = get_six(player, ["player_api_id", "player_name"])
    # 画图函数，用哪个取消注释即可
    # plot_pos_radar(pos_six_data)
    # plot_famous_radar(player, famous_six_data)
    # plot_bar(["Forward", "Midfielder", "Goalkeeper", "Defender"], [431, 311, 147, 2],
    #          os.path.join(data_path, "pic/bar.png"))
    plot_age_dependence(pos_data, os.path.join(data_path, "pic/age.png"))
