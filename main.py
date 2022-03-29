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

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import model_selection
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from models.ml_train import find_best_classifier, train_ensemble
from utils.get_data import get_fifa_data, create_feables
from utils.visualize import explore_data, plot_confusion_matrix, plot_training_results

warnings.simplefilter("ignore")


def parse_my_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default = random.randint(0, 9999999))
    parser.add_argument('--visual', type = int, default = 0)
    parser.add_argument('--metric', type = str, default = "f1", choices = ["pre", "recall", "f1"])
    parser.add_argument('--average', type = str, default = "weighted", choices = ["macro", "micro", "weighted"])
    arg = parser.parse_args()
    if arg.metric == "pre":
        arg.metric_fn = precision_score
    elif arg.metric == "recall":
        arg.metric_fn = recall_score
    elif arg.metric == 'f1':
        arg.metric_fn = f1_score
    return arg


def preprocess(data, norm = 1):
    label = data.loc[:, 'label'].values
    feat = data.drop('label', axis = 1).values
    if norm == 0:
        return label, feat
    else:
        scaler = MinMaxScaler([0, 1])
        norm_feat = scaler.fit_transform(feat)
        return label, norm_feat


if __name__ == '__main__':
    # TODO 测试调整参数范围有没有用 n_components
    start = time()
    args = parse_my_args()
    data_path = "./data/"  # 数据文件夹
    # 网格搜索
    n_jobs = 1
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

    # -----------------------生成特征, 准备训练-----------------------
    feature_path = os.path.join(data_path, "middle_results/final_data.npy")
    # 赌场信息，比如betway(BW)，随便选两个有名的，不然na值太多了
    bk_cols = ['B365', 'BW', 'IW', 'LB', 'PS', 'WH', 'SJ', 'VC', 'GB', 'BS']
    bk_cols_selected = ['B365', 'BW']
    # 得到每场比赛球员数据
    fifa_data = get_fifa_data(match_data, player_stats_data, path = data_path)
    # 创建整体数据集
    if os.path.exists(feature_path):
        inputs = pd.read_pickle(feature_path)
        print("读入所有数据")
    else:
        feables = create_feables(match_data, fifa_data, bk_cols_selected, get_overall = True)
        inputs = feables.drop('match_api_id', axis = 1)
        inputs.to_pickle(feature_path)
    labels, features = preprocess(inputs, norm = 1)

    # -----------------------可视化-----------------------
    if args.visual == 1:
        feature_details = explore_data(inputs, os.path.join(data_path, "pic/visual.png"))

    # -----------------------开始训练-----------------------
    X_train_calibrate, X_test, y_train_calibrate, y_test = train_test_split(features, labels, test_size = 0.2,
                                                                            random_state = args.seed, stratify = labels)
    X_train, X_calibrate, y_train, y_calibrate = train_test_split(X_train_calibrate, y_train_calibrate, test_size = 0.3,
                                                                  random_state = args.seed,
                                                                  stratify = y_train_calibrate)
    # 用训练数据做五折交叉验证
    cv_sets = model_selection.StratifiedShuffleSplit(n_splits = 5, test_size = 0.20, random_state = 5)
    cv_sets.get_n_splits(X_train, y_train)

    # 初始化所有模型
    # 使用的分类器
    # GB_clf = GradientBoostingClassifier(random_state = args.seed)
    RF_clf = RandomForestClassifier(n_estimators = 200, random_state = 2, class_weight = 'balanced')
    XT_clf = ExtraTreesClassifier(random_state = 1)
    AB_clf = AdaBoostClassifier(random_state = 3)
    GNB_clf = GaussianNB()
    KNN_clf = KNeighborsClassifier()
    LOG_clf = linear_model.LogisticRegression(multi_class = "ovr", solver = "sag", class_weight = 'balanced',
                                              random_state = args.seed)

    # clfs = [GNB_clf]
    clfs = [RF_clf, AB_clf, GNB_clf, KNN_clf, LOG_clf]

    # 使用的降维方法
    pca = PCA()
    ica = FastICA()
    # reductions = [pca]
    reductions = [pca, ica]

    # 使用的评价指标以及网格搜索的参数
    feature_len = features.shape[1]
    scorer = make_scorer(accuracy_score)
    # scorer = make_scorer(args.metric_fn, average = args.average)
    parameters_RF = {'clf__max_features': ['auto', 'log2'],
                     'reduce__n_components': np.arange(5, feature_len + 1, np.around(feature_len / 5) - 1,
                                                       dtype = int)}
    parameters_AB = {'clf__learning_rate': np.linspace(0.5, 2, 5), 'clf__n_estimators': [50, 100, 200],
                     'reduce__n_components': np.arange(5, feature_len + 1, np.around(feature_len / 5) - 1,
                                                       dtype = int)}
    parameters_GNB = {
        'reduce__n_components': np.arange(5, feature_len + 1, np.around(feature_len / 5) - 1, dtype = int)}
    parameters_KNN = {'clf__n_neighbors': [3, 5, 10],
                      'reduce__n_components': np.arange(5, feature_len + 1, np.around(feature_len / 5) - 1,
                                                        dtype = int)}
    parameters_LOG = {'clf__C': np.logspace(1, 1000, 5),
                      'reduce__n_components': np.arange(5, feature_len + 1, np.around(feature_len / 5) - 1,
                                                        dtype = int)}
    parameters = {
        clfs[0]: parameters_RF,
        clfs[1]: parameters_AB,
        clfs[2]: parameters_GNB,
        clfs[3]: parameters_KNN,
        clfs[4]: parameters_LOG,
    }

    # 简单做一个baseline
    print("----------------------------------")
    print("一个简单的模型:")
    clf = LOG_clf
    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    print("[{}] training set "
          "[balanced accuracy]: {:.4f}  "
          "[{} score]: {:.4f}".format(clf.__class__.__name__, accuracy_score(y_train, train_pred), args.metric,
                                      args.metric_fn(y_train, train_pred, average = args.average)))
    print("[{}] test set     "
          "[balanced accuracy]: {:.4f}  "
          "[{} score]: {:.4f}".format(clf.__class__.__name__, accuracy_score(y_test, test_pred), args.metric,
                                      args.metric_fn(y_test, test_pred, average = args.average)))
    # 训练所有的方法
    clfs, reductions, train_scores, test_scores = find_best_classifier(clfs, reductions, scorer, X_train, y_train,
                                                                       X_calibrate, y_calibrate, X_test, y_test,
                                                                       cv_sets, parameters, n_jobs, args)
    train_score, test_score = train_ensemble(clfs, reductions, X_train_calibrate, y_train_calibrate, X_test, y_test)
    train_scores.append(train_score)
    test_scores.append(test_score)
    # 可视化训练集和测试集结果
    plot_training_results(clfs, reductions, np.array(train_scores), np.array(test_scores),
                          path = os.path.join(data_path, "pic/train_visual.png"), metric_fn = args.metric)

    # -----------------------画混淆矩阵和赌博的预测-----------------------
    # 找到最佳的分类器和降维方法然后画混淆矩阵
    best_clf = clfs[np.argmax(test_scores)]
    best_reduce = reductions[np.argmax(test_scores)]
    print("最佳分类器为 [{}] 降维方法为 [{}].".format(best_clf.base_estimator.__class__.__name__,
                                           best_reduce.__class__.__name__))
    plot_confusion_matrix(y_test, X_test, best_clf, best_reduce, path = os.path.join(data_path, "pic/cf_visual.png"),
                          normalize = True)

    # # 画赌博预测的混淆矩阵
    # plot_bookkeeper_cf_matrix(match_data, bk_cols, os.path.join(data_path, "pic/bet_cf_visual.png"), normalize = True)
    #
    # # 用网格搜索找到最佳赌博策略
    # percentile_grid = np.linspace(0, 1, 2)
    # probability_grid = np.linspace(0, 0.5, 2)
    # best_betting = optimize_betting(best_clf, best_reduce, bk_cols_selected, bk_cols, match_data, fifa_data,
    #                                 5, 300, percentile_grid, probability_grid)
    # print("The best return of investment is: {}".format(best_betting.results))

    end = time()
    print("总计运行时间 {:.1f} 分钟".format((end - start) / 60))
