# -*- coding: utf-8 -*-
# @Time    : 2022/3/28 17:07
# @Author  : nieyuzhou
# @File    : visualization.py
# @Software: PyCharm
import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils.visualize import plot_radar


def plot_famous_radar(data, data_six):
    famous, average = get_famous_average(data)
    famous = pd.merge(famous, data_six, on = ['player_api_id', 'player_name'], how = 'left').groupby(
        ['player_api_id', 'player_name'], as_index = False).mean()
    average = pd.merge(average, data_six, on = ['player_api_id', 'player_name'], how = 'left').groupby(
        ['player_api_id', 'player_name'], as_index = False).mean()
    f = famous[['skill', 'movement', 'attacking', 'power', 'mentality', 'defending']].mean()
    a = average[['skill', 'movement', 'attacking', 'power', 'mentality', 'defending']].mean()
    final = pd.concat([f, a], axis = 1).T
    plot_radar(final, kinds = ['famous', 'average'], path = os.path.join(data_path, "pic/radar_famous.png"))


def get_six(data):
    data['skill'] = data['heading_accuracy'] + data['free_kick_accuracy'] + data['short_passing'] + data['volleys'] + \
                    data['dribbling'] + data['curve'] + data['long_passing'] + data['ball_control'] + data['jumping'] \
                    + data['long_shots'] + data['positioning'] + data['vision']
    data['movement'] = data['acceleration'] + data['sprint_speed'] + data['reactions'] + data['agility'] + data[
        'balance']
    data['attacking'] = data['standing_tackle'] + data['sliding_tackle']
    data['power'] = data['stamina'] + data['strength']
    data['mentality'] = data['aggression'] + data['interceptions']
    data['defending'] = data['gk_diving'] + data['gk_handling'] + data['gk_kicking'] + data['gk_positioning'] + data[
        'gk_reflexes']

    scaler = MinMaxScaler([0, 10])
    norm_data = scaler.fit_transform(data[['skill', 'movement', 'attacking', 'power', 'mentality', 'defending']])
    norm_data = pd.DataFrame(norm_data, columns = ['skill', 'movement', 'attacking', 'power', 'mentality', 'defending'])
    norm_data[["player_api_id", "player_name"]] = data[["player_api_id", "player_name"]]
    norm_data = norm_data.groupby(['player_api_id', 'player_name'], as_index = False).mean()
    return norm_data


def get_famous_average(data):
    return data[data["overall_rating"] >= 90][['player_api_id', 'player_name']], \
           data[data["overall_rating"] <= 70][['player_api_id', 'player_name']]


if __name__ == '__main__':
    data_path = "./data/"  # 数据文件夹
    # 得到球员数据
    player_data = pd.read_csv(os.path.join(data_path, "player.csv"), index_col = 0).drop(
        ['player_fifa_api_id', "id"], axis = 1)
    player_stats_data = pd.read_csv(os.path.join(data_path, "player_attr.csv"), index_col = 0).drop(
        ['player_fifa_api_id', "id"], axis = 1).dropna()
    player = pd.merge(player_data, player_stats_data, on = 'player_api_id', how = 'inner')
    player_six = get_six(player)
    plot_famous_radar(player, player_six)
