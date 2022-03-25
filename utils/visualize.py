# -*- coding: utf-8 -*-
# @Time    : 2022/3/25 14:33
# @Author  : nieyuzhou
# @File    : visualize.py
# @Software: PyCharm
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

from utils.get_data import get_bookkeeper_probs, get_match_label


def explore_data(inputs, path):
    """ Explore data by plotting KDE graphs. """

    # Define figure subplots
    plt.figure(figsize = (22, 22), dpi = 200)
    # fig.subplots_adjust(bottom = -1, left = 0.025, top = 2, right = 0.975)

    # Loop through features
    i = 1
    for col in inputs.columns:
        if "League" in col:
            continue
        # Set subplot and plot format
        sns.set_style("whitegrid")
        # sns.set_context("paper", font_scale = 0.5, rc = {"lines.linewidth": 1})
        plt.subplot(6, 6, 0 + i)

        # Plot KDE for all labels
        sns.kdeplot(inputs[inputs['label'] == 'Win'].loc[:, col], label = 'Win')
        sns.kdeplot(inputs[inputs['label'] == 'Draw'].loc[:, col], label = 'Draw')
        sns.kdeplot(inputs[inputs['label'] == 'Defeat'].loc[:, col], label = 'Defeat')
        plt.legend()
        i = i + 1
    plt.subplots_adjust()
    # Define plot format
    # DefaultSize = fig.get_size_inches()
    # fig.set_size_inches((DefaultSize[0] * 1.2, DefaultSize[1] * 1.2))

    plt.savefig(path)

    # Compute and print label weights
    labels = inputs.loc[:, 'label']
    class_weights = labels.value_counts() / len(labels)
    print(class_weights)

    # Store description of all features
    feature_details = inputs.describe().transpose()

    # Return feature details
    return feature_details


def plot_confusion_matrix(y_test, X_test, clf, dim_reduce, path, cmap = plt.cm.Blues, normalize = False):
    """ 画混淆矩阵 """

    # Define label names and get confusion matrix values
    labels = ["Win", "Draw", "Defeat"]
    cm = confusion_matrix(y_test, clf.predict(dim_reduce.transform(X_test)))

    if normalize:
        # Normalize
        cm = cm.astype('float') / cm.sum()

    sns.set_style("whitegrid", {"axes.grid": False})
    plt.figure(1)
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    title = "Confusion matrix of a {} with {}".format(clf.__class__.__name__,
                                                      dim_reduce.__class__.__name__)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation = 45)
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 2),
                 horizontalalignment = "center", color = "white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(path)

    # Print classification report
    y_pred = clf.predict(dim_reduce.transform(X_test))
    print(classification_report(y_test, y_pred))


def plot_training_results(clfs, dm_reductions, train_scores, test_scores, path, metric_fn):
    """ 画出训练结果图 """
    plt.figure(figsize = (8, 8), dpi = 160)
    sns.set_style("whitegrid")
    # sns.set_context("paper", font_scale = 1, rc = {"lines.linewidth": 1})
    ax = plt.subplot(111)
    w = 0.5
    x = np.arange(len(train_scores))
    ax.set_yticks(x + w)
    ax.legend((train_scores[0], test_scores[0]), ("Train Scores", "Test Scores"))
    names = []

    for i in range(0, len(clfs)):
        clf = clfs[i]
        clf_name = clf.__class__.__name__
        dm = dm_reductions[i]
        dm_name = dm.__class__.__name__

        # 储存名字
        name = "{} with {}".format(clf_name, dm_name)
        names.append(name)

    ax.set_yticklabels(names)
    plt.xlim(0.4, 0.6)
    plt.barh(x, test_scores, color = 'b', alpha = 0.6)
    title = "Test Data {} Scores".format(metric_fn)
    plt.title(title)
    plt.savefig(path)
    print(title + "绘制完成")
    print("----------------------------------")


def plot_bookkeeper_cf_matrix(matches, bookkeepers, path, normalize = True):
    """ 画赔率的混淆矩阵 """

    # 得到标签
    y_test_temp = matches.apply(get_match_label, axis = 1)

    # 得到赔率的输赢概率
    bookkeeper_probs = get_bookkeeper_probs(matches, bookkeepers)
    bookkeeper_probs.reset_index(inplace = True, drop = True)
    bookkeeper_probs.dropna(inplace = True)

    # 得到赔率代表的标签
    y_pred_temp = pd.DataFrame()
    y_pred_temp.loc[:, 'bk_label'] = bookkeeper_probs[['Win', 'Draw', 'Defeat']].idxmax(axis = 1)
    y_pred_temp.loc[:, 'match_api_id'] = bookkeeper_probs.loc[:, 'match_api_id']

    results = pd.merge(y_pred_temp, y_test_temp, on = 'match_api_id', how = 'left')
    y_test = results.loc[:, 'label']
    y_pred = results.loc[:, 'bk_label']

    # 生成混淆矩阵
    labels = ["Win", "Draw", "Defeat"]
    cm = confusion_matrix(y_test, y_pred)
    # Normalize
    if normalize:
        cm = cm.astype('float') / cm.sum()

    # 画混淆矩阵
    sns.set_style("whitegrid", {"axes.grid": False})
    plt.figure(1)
    plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)
    title = "Confusion matrix of Bookkeeper predictions!"
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation = 45)
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 2),
                 horizontalalignment = "center",
                 color = "white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(path)
    # 输出报告
    print(classification_report(y_test, y_pred))
    print("菠菜的分数 test set: {:.4f}.".format(balanced_accuracy_score(y_test, y_pred)))
