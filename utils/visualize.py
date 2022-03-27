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
    """ 画出每个特征的KDE图 """

    fig = plt.figure(figsize = (11, 11), dpi = 200)
    # fig.subplots_adjust(bottom = -1, left = 0.025, top = 2, right = 0.975)

    # 对每个特征进行循环
    i = 1
    for col in inputs.columns:
        if "League" in col or col == "label":
            continue
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale = 0.6, rc = {"lines.linewidth": 1})
        plt.subplot(6, 6, 0 + i)

        # 画KDE图
        sns.kdeplot(inputs[inputs['label'] == 'Win'].loc[:, col], label = 'Win')
        sns.kdeplot(inputs[inputs['label'] == 'Draw'].loc[:, col], label = 'Draw')
        sns.kdeplot(inputs[inputs['label'] == 'Defeat'].loc[:, col], label = 'Defeat')
        i = i + 1
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'lower right', fontsize = 14)
    plt.tight_layout()
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
    plt.figure(dpi = 180)
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    title = "Confusion matrix of a {} with {}".format(clf.base_estimator.__class__.__name__,
                                                      dim_reduce.__class__.__name__)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 2), horizontalalignment = "center",
                 color = "white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(path)

    # Print classification report
    y_pred = clf.predict(dim_reduce.transform(X_test))
    print(classification_report(y_test, y_pred))


def plot_training_results(clfs, reductions, train_scores, test_scores, path, metric_fn):
    """ 画出训练结果图 """
    plt.figure(dpi = 160)
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale = 0.8, rc = {"lines.linewidth": 1.2})
    ax = plt.subplot(111)
    w = 0
    x = np.arange(len(train_scores))
    ax.set_yticks(x + w)
    ax.legend((train_scores[0], test_scores[0]), ("Train Scores", "Test Scores"))
    names = []

    for i in range(0, len(clfs)):
        clf = clfs[i]
        clf_name = clf.base_estimator.__class__.__name__
        red = reductions[i]
        red_name = red.__class__.__name__

        # 储存名字
        name = "{} with {}".format(clf_name, red_name)
        names.append(name)

    ax.set_yticklabels(names)
    plt.xlim(min(test_scores) - 0.01, max(test_scores) + 0.01)
    plt.barh(x, test_scores, alpha = 0.6,
             color = ["grey", "gold", "darkviolet", "turquoise", "r", "g", "b", "c", "m", "y",
                      "k", "darkorange", "lightgreen", "plum", "tan",
                      "khaki", "pink", "skyblue", "lawngreen", "salmon"])
    title = "Test Data {} Scores".format(metric_fn)
    plt.title(title)
    plt.tight_layout()
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
