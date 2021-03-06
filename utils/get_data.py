# -*- coding: utf-8 -*-
# @Time    : 2022/3/22 14:31
# @Author  : nieyuzhou
# @File    : get_data.py
# @Software: PyCharm
import os
from datetime import datetime
from time import time

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_overall_fifa_rankings(fifa, get_overall = False):
    """ Get overall fifa rankings from fifa data. """

    temp_data = fifa

    # Check if only overall player stats are desired
    if get_overall:

        # Get overall stats
        data = temp_data.loc[:, (fifa.columns.str.contains('overall_rating'))]
        data.loc[:, 'match_api_id'] = temp_data.loc[:, 'match_api_id']
    else:

        # Get all stats except for stat date
        cols = fifa.loc[:, (fifa.columns.str.contains('date_stat'))]
        temp_data = fifa.drop(cols.columns, axis = 1)
        data = temp_data

    # Return data
    return data


def get_last_matches(matches, date, team, x = 10):
    """ Get the last x matches of a given team. """

    # Filter team matches from matches
    team_matches = matches[(matches['home_team_api_id'] == team) | (matches['away_team_api_id'] == team)]

    # Filter x last matches from team matches
    last_matches = team_matches[team_matches.date < date].sort_values(by = 'date', ascending = False).iloc[0:x, :]

    # Return last matches
    return last_matches


def get_last_matches_against_eachother(matches, date, home_team, away_team, x = 10):
    """ 得到这两队最近x场比赛情况 """

    # 分别作为主队和客队进行寻找
    home_matches = matches[(matches['home_team_api_id'] == home_team) & (matches['away_team_api_id'] == away_team)]
    away_matches = matches[(matches['home_team_api_id'] == away_team) & (matches['away_team_api_id'] == home_team)]
    total_matches = pd.concat([home_matches, away_matches])

    # 找最近x场，如果少于x场则选取全部
    try:
        last_matches = total_matches[total_matches.date < date].sort_values(by = 'date', ascending = False).iloc[0:x, :]
    except:
        last_matches = total_matches[total_matches.date < date].sort_values(by = 'date', ascending = False).iloc[
                       0:total_matches.shape[0], :]

        # 没用了现在
        if last_matches.shape[0] > x:
            print("Error")

    return last_matches


def get_goals(matches, team):
    """ 计算该球队这些场的进球数 """

    # 得到分别作为主队和客队的进球数
    home_goals = int(matches.home_team_goal[matches.home_team_api_id == team].sum())
    away_goals = int(matches.away_team_goal[matches.away_team_api_id == team].sum())

    total_goals = home_goals + away_goals
    return total_goals


def get_goals_conceided(matches, team):
    """ 计算该球队这些场的被进球数 """

    # 得到分别作为主队和客队的进球数
    home_goals = int(matches.home_team_goal[matches.away_team_api_id == team].sum())
    away_goals = int(matches.away_team_goal[matches.home_team_api_id == team].sum())

    total_goals = home_goals + away_goals
    return total_goals


def get_wins(matches, team):
    """ 得到赢球数 """

    # 分别客队主队
    home_wins = int(matches.home_team_goal[
                        (matches.home_team_api_id == team) & (matches.home_team_goal > matches.away_team_goal)].count())
    away_wins = int(matches.away_team_goal[
                        (matches.away_team_api_id == team) & (matches.away_team_goal > matches.home_team_goal)].count())

    total_wins = home_wins + away_wins
    return total_wins


# Loading all functions
def get_match_label(match):
    """ 对一个给定比赛生成标签 """

    home_goals = match['home_team_goal']
    away_goals = match['away_team_goal']

    label = pd.DataFrame()
    label.loc[0, 'match_api_id'] = match['match_api_id']

    # 生成标签
    if home_goals > away_goals:
        label.loc[0, 'label'] = "Win"
    if home_goals == away_goals:
        label.loc[0, 'label'] = "Draw"
    if home_goals < away_goals:
        label.loc[0, 'label'] = "Defeat"
    return label.loc[0]


def get_gk(data):
    return data[['overall_rating', 'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning', 'gk_reflexes']]


def get_six(data):
    data['Pace'] = 0.45 * data['acceleration'] + 0.55 * data['sprint_speed']
    data['Shooting'] = 0.45 * data['finishing'] + 0.2 * data['long_shots'] + 0.05 * data['penalties'] + 0.2 * data[
        'shot_power'] + 0.05 * data["volleys"] + 0.05 * data['positioning']
    data['Passing'] = 0.2 * data['vision'] + 0.2 * data['crossing'] + 0.05 * data['curve'] + 0.05 * data[
        'free_kick_accuracy'] + 0.15 * data['long_passing'] + 0.35 * data['short_passing']
    data['Dribbling'] = 0.5 * data['dribbling'] + 0.3 * data['ball_control'] + 0.1 * data['agility'] + 0.05 * data[
        'balance'] + 0.05 * data['reactions']
    data['Defending'] = 0.3 * data['standing_tackle'] + 0.1 * data['sliding_tackle'] + 0.1 * data[
        'heading_accuracy'] + 0.2 * data['interceptions'] + 0.3 * data['marking']
    data['Physical'] = 0.2 * data['aggression'] + 0.25 * data['stamina'] + 0.5 * data['strength'] + 0.05 * data[
        'jumping']

    return data[['overall_rating', 'Pace', 'Shooting', 'Passing', 'Dribbling', 'Defending', 'Physical']]


def get_fifa_stats(match, player_stats):
    """ 得到一场比赛的球员数据 """

    match_id = int(match.match_api_id)
    date = match['date']
    # 考虑主客场所有球员
    players = ['home_player_1', 'home_player_2', 'home_player_3', "home_player_4", "home_player_5",
               "home_player_6", "home_player_7", "home_player_8", "home_player_9", "home_player_10",
               "home_player_11", "away_player_1", "away_player_2", "away_player_3", "away_player_4",
               "away_player_5", "away_player_6", "away_player_7", "away_player_8", "away_player_9",
               "away_player_10", "away_player_11"]
    player_stats_new = pd.DataFrame()
    names = []

    for i, player in enumerate(players):
        # 得到球员ID
        player_id = match[player]
        # 获得该球员信息
        stats = player_stats[player_stats.player_api_id == player_id]

        # 获得距离该比赛最近的该球员状态
        current_stats = stats[stats.date < date].sort_values(by = 'date', ascending = False)[:1]
        current_stats.reset_index(inplace = True, drop = True)
        # 得到六维属性
        if i == 0 or i == 11:
            gk_rating = get_gk(current_stats)
            # 重命名
            name = "{}_overall_rating".format(player)
            name1 = "{}_gk_diving".format(player)
            name2 = "{}_gk_handling".format(player)
            name3 = "{}_gk_kicking".format(player)
            name4 = "{}_gk_positioning".format(player)
            name5 = "{}_gk_reflexes".format(player)
            names.extend([name, name1, name2, name3, name4, name5])
            player_stats_new = pd.concat([player_stats_new, gk_rating], axis = 1)
        else:
            six_rating = get_six(current_stats)
            # 重命名
            name = "{}_overall_rating".format(player)
            name1 = "{}_pace".format(player)
            name2 = "{}_shooting".format(player)
            name3 = "{}_passing".format(player)
            name4 = "{}_dribbling".format(player)
            name5 = "{}_defending".format(player)
            name6 = "{}_physical".format(player)
            names.extend([name, name1, name2, name3, name4, name5, name6])
            player_stats_new = pd.concat([player_stats_new, six_rating], axis = 1)

    player_stats_new.columns = names
    player_stats_new['match_api_id'] = match_id
    player_stats_new.reset_index(inplace = True, drop = True)

    return player_stats_new.iloc[0]


def get_pos_stats(x, player_stats):
    stats = player_stats[player_stats.player_api_id == x["player_api_id"]]
    cur_stats = stats[stats.date < x["date"]].sort_values(by = 'date', ascending = False).iloc[:1, :]
    cur_stats['pos'] = x['pos']
    return cur_stats.iloc[0, :]


def get_age_for_football_players(x):
    date = x.split(" ")[0]
    today = datetime.strptime("2016-01-01", "%Y-%m-%d").date()
    born = datetime.strptime(date, "%Y-%m-%d").date()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def get_fifa_data(matches, player_stats, path = None):
    """ 得到所有比赛的FIFA中球员数据 """
    fifa_path = os.path.join(path, "middle_results/fifa_data.npy")
    # 保存数据不用每次都计算一次
    if os.path.exists(fifa_path):
        data = pd.read_pickle(fifa_path)
        print("读入每场比赛球员数据")
    else:
        start = time()
        # 去掉奇怪的比赛（没有守门员？）
        matches = matches[matches.away_player_Y1 != 3]
        matches = matches[matches.home_player_Y1 != 3]
        # 对每一场比赛得到所有球员六维属性
        data = matches.apply(lambda x: get_fifa_stats(x, player_stats), axis = 1)
        end = time()
        print("得到每场比赛球员数据所用时间 {:.1f} 分钟".format((end - start) / 60))
        data.to_pickle(fifa_path)
    return data


def get_match_features(match, matches, team):
    """ 生成比赛数据 """

    date = match.date
    home_team = match.home_team_api_id
    away_team = match.away_team_api_id

    # 得到主队和客队分别最近x场的比赛情况
    matches_home_team = get_last_matches(matches, date, home_team, x = 10)
    matches_away_team = get_last_matches(matches, date, away_team, x = 10)

    # 得到这两队最近x场的比赛情况
    last_matches_against = get_last_matches_against_eachother(matches, date, home_team, away_team, x = 3)

    # 得到主队和客队分别这些场的进球数
    home_goals = get_goals(matches_home_team, home_team)
    away_goals = get_goals(matches_away_team, away_team)
    # 得到主队和客队分别这些场的被进球数
    home_goals_conceided = get_goals_conceided(matches_home_team, home_team)
    away_goals_conceided = get_goals_conceided(matches_away_team, away_team)

    result = pd.DataFrame()
    # ID
    result.loc[0, 'match_api_id'] = match.match_api_id
    result.loc[0, 'league_id'] = match.league_id
    # 得到比赛层面的特征
    result.loc[0, 'home_team_goals_difference'] = home_goals - home_goals_conceided
    result.loc[0, 'away_team_goals_difference'] = away_goals - away_goals_conceided
    result.loc[0, 'games_won_home_team'] = get_wins(matches_home_team, home_team)
    result.loc[0, 'games_won_away_team'] = get_wins(matches_away_team, away_team)
    result.loc[0, 'games_against_won'] = get_wins(last_matches_against, home_team)
    result.loc[0, 'games_against_lost'] = get_wins(last_matches_against, away_team)
    # 得到其余比赛情况的特征
    # result.loc[0, ''] = match
    return result.loc[0]


def get_bookkeeper_data(matches, bookkeepers, horizontal = True):
    """ 生成各大赌场的赔率数据 """
    bk_data = pd.DataFrame()

    for bookkeeper in bookkeepers:

        # 找到选取的赌场
        temp_data = matches.loc[:, (matches.columns.str.contains(bookkeeper))].copy()
        temp_data.loc[:, 'bookkeeper'] = str(bookkeeper)
        temp_data.loc[:, 'match_api_id'] = matches.loc[:, 'match_api_id']

        # HDA分别代表home、draw、away。转化为赢平输
        cols = temp_data.columns.values
        cols[:3] = ['Win', 'Draw', 'Defeat']
        temp_data.columns = cols
        temp_data.loc[:, 'Win'] = pd.to_numeric(temp_data['Win'])
        temp_data.loc[:, 'Draw'] = pd.to_numeric(temp_data['Draw'])
        temp_data.loc[:, 'Defeat'] = pd.to_numeric(temp_data['Defeat'])

        if horizontal:
            # 赔率转化为概率
            temp_data = convert_odds_to_prob(temp_data)
            temp_data.drop('match_api_id', axis = 1, inplace = True)
            temp_data.drop('bookkeeper', axis = 1, inplace = True)
            # 重命名，加上赌场名字
            win_name = bookkeeper + "_" + "Win"
            draw_name = bookkeeper + "_" + "Draw"
            defeat_name = bookkeeper + "_" + "Defeat"
            temp_data.columns.values[:3] = [win_name, draw_name, defeat_name]
            # 拼接起来
            bk_data = pd.concat([bk_data, temp_data], axis = 1)
        else:
            # Aggregate vertically
            bk_data = bk_data.append(temp_data, ignore_index = True)

    # If horizontal add match api id to data
    if horizontal:
        temp_data.loc[:, 'match_api_id'] = matches.loc[:, 'match_api_id']

    return bk_data


def convert_odds_to_prob(match_odds):
    """ 赔率转为概率 """

    match_id = match_odds.loc[:, 'match_api_id']
    bookkeeper = match_odds.loc[:, 'bookkeeper']
    win_odd = match_odds.loc[:, 'Win']
    draw_odd = match_odds.loc[:, 'Draw']
    loss_odd = match_odds.loc[:, 'Defeat']

    # 归一化
    win_prob = 1 / win_odd
    draw_prob = 1 / draw_odd
    loss_prob = 1 / loss_odd

    total_prob = win_prob + draw_prob + loss_prob

    probs = pd.DataFrame()

    # Define output format and scale probs by sum over all probs
    probs.loc[:, 'match_api_id'] = match_id
    probs.loc[:, 'bookkeeper'] = bookkeeper
    probs.loc[:, 'Win'] = win_prob / total_prob
    probs.loc[:, 'Draw'] = draw_prob / total_prob
    probs.loc[:, 'Defeat'] = loss_prob / total_prob

    # Return probs and meta data
    return probs


def get_bookkeeper_probs(matches, bookkeepers):
    """ Get bookkeeper data and convert to probabilities for vertical aggregation. """

    # 赔率数据
    data = get_bookkeeper_data(matches, bookkeepers, horizontal = False)
    # 转化为概率
    probs = convert_odds_to_prob(data)
    return probs


def create_feables(matches, fifa, bookkeepers,teams, get_overall = False, horizontal = True):
    """ 创建整体的数据集，包括球员能力、球队情况、比赛数据 """

    # 球员的总体能力：overall rating
    start = time()
    fifa_stats = get_overall_fifa_rankings(fifa, get_overall)
    # 比赛数据
    match_stats = matches.apply(lambda x: get_match_features(x, matches,teams), axis = 1)
    # 对联盟数据做哑变量处理
    dummies = pd.get_dummies(match_stats['league_id']).rename(columns = lambda x: 'League_' + str(x))
    match_stats = pd.concat([match_stats, dummies], axis = 1)
    match_stats.drop(['league_id'], inplace = True, axis = 1)
    end = time()
    print("生成比赛特征所用时间 {:.1f} 分钟".format((end - start) / 60))

    # 生成标签
    start = time()
    labels = matches.apply(get_match_label, axis = 1)
    end = time()
    print("生成比赛标签所用时间 {:.1f} 分钟".format((end - start) / 60))

    # 暂时不要赌博赔率数据
    start = time()
    bk_data = get_bookkeeper_data(matches, bookkeepers, horizontal = horizontal)
    bk_data.loc[:, 'match_api_id'] = matches.loc[:, 'match_api_id']
    end = time()
    print("生成赔率数据所用时间 {:.1f} 分钟".format((end - start) / 60))

    # 结合起来
    features = pd.merge(match_stats, fifa_stats, on = 'match_api_id', how = 'left')
    # features = pd.merge(features, bk_data, on = 'match_api_id', how = 'left')
    feature_label = pd.merge(features, labels, on = 'match_api_id', how = 'left')
    # 去掉NA
    feature_label.dropna(inplace = True)
    return feature_label


def preprocess(data, norm = 1):
    label = data.loc[:, 'label'].values
    feat = data.drop('label', axis = 1).values
    if norm == 0:
        return label, feat
    else:
        scaler = MinMaxScaler([0, 1])
        norm_feat = scaler.fit_transform(feat)
        return label, norm_feat
