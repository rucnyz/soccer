# -*- coding: utf-8 -*-
# @Time    : 2022/4/4 11:08
# @Author  : nieyuzhou
# @File    : train.py
# @Software: PyCharm

import os

import numpy as np
import pandas as pd

from sklearn import model_selection, linear_model
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from models.ml_train import find_best_classifier, train_ensemble
from utils.get_data import get_fifa_data, create_feables, preprocess
from utils.visualize import plot_confusion_matrix, plot_training_results, explore_data


def extract_feat(data_path, match_data, player_stats_data, team_data):
    # -----------------------生成特征, 准备训练-----------------------
    feature_path = os.path.join(data_path, "middle_results/final_data.npy")
    # 赌场信息，比如betway(BW)，随便选两个有名的，不然na值太多了
    # bk_cols = ['B365', 'BW', 'IW', 'LB', 'PS', 'WH', 'SJ', 'VC', 'GB', 'BS']
    bk_cols_selected = ['B365', 'BW']
    # 得到每场比赛球员数据
    fifa_data = get_fifa_data(match_data, player_stats_data, path = data_path)
    # 创建整体数据集
    if os.path.exists(feature_path):
        inputs = pd.read_pickle(feature_path)
        print("读入所有数据")
    else:
        feables = create_feables(match_data, fifa_data, bk_cols_selected, team_data, get_overall = False)
        inputs = feables.set_index('match_api_id')
        inputs.to_pickle(feature_path)
    return inputs


def ml_training(X_train_valid, y_train_valid, X_train, X_valid, y_train, y_valid, X_test, y_test, args, data_path):
    # 用训练数据做五折交叉验证
    cv_sets = model_selection.StratifiedShuffleSplit(n_splits = 5, test_size = 0.20, random_state = 5)
    cv_sets.get_n_splits(X_train, y_train)
    # 初始化所有模型
    # 使用的分类器
    # GB_clf = GradientBoostingClassifier(random_state = args.seed)
    RF_clf = RandomForestClassifier(n_estimators = 200, random_state = 2, class_weight = 'balanced')
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
    feature_len = X_train.shape[1]
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
                                                                       X_valid, y_valid, X_test, y_test,
                                                                       cv_sets, parameters, args.n_jobs, args)
    train_score, test_score, test_pred = train_ensemble(clfs, reductions, X_train_valid, y_train_valid, X_test,
                                                        y_test)
    train_scores.append(train_score)
    test_scores.append(test_score)
    # 可视化训练集和测试集结果
    plot_training_results(clfs, reductions, np.array(train_scores), np.array(test_scores),
                          path = os.path.join(data_path, "pic/train_visual.png"), metric_fn = args.metric)
    # -----------------------画混淆矩阵-----------------------
    # 找到最佳的分类器和降维方法然后画混淆矩阵
    best_clf = clfs[np.argmax(test_scores)]
    best_reduce = reductions[np.argmax(test_scores)]
    print("最佳分类器为 [{}] 降维方法为 [{}].".format(best_clf.base_estimator.__class__.__name__,
                                           best_reduce.__class__.__name__))
    plot_confusion_matrix(y_test, X_test, best_clf, best_reduce, path = os.path.join(data_path, "pic/cf_visual.png"),
                          normalize = True)


def train(data_path, match_data, player_stats_data, team_data, args):
    inputs = extract_feat(data_path, match_data, player_stats_data, team_data)

    labels, features = preprocess(inputs, norm = 1)
    # -----------------------可视化-----------------------
    if args.visual == 1:
        explore_data(inputs, os.path.join(data_path, "pic/visual.png"))
    # -----------------------开始训练-----------------------
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(features, labels, test_size = 0.2,
                                                                    random_state = args.seed,
                                                                    stratify = labels)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size = 0.3,
                                                          random_state = args.seed,
                                                          stratify = y_train_valid)
    if args.method == "ml":
        ml_training(X_train_valid, y_train_valid, X_train, X_valid, y_train, y_valid, X_test, y_test, args, data_path)
    else:
        inputs.to_csv(os.path.join(data_path, "processed/all2.csv"))
