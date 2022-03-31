# -*- coding: utf-8 -*-
# @Time    : 2022/3/25 14:33
# @Author  : nieyuzhou
# @File    : visualize.py
# @Software: PyCharm
import itertools
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

from utils.get_data import get_bookkeeper_probs, get_match_label

colors = ["r", "g", "grey", "gold", "darkviolet", "turquoise", "b", "c", "m", "y", "k", "darkorange", "lightgreen",
          "plum", "tan", "khaki", "pink", "skyblue", "lawngreen", "salmon"]


def plot_bar(x, y, path):
    plt.figure(dpi = 140)
    plt.bar(x, y, color = colors, alpha = 0.6)
    for i, b in enumerate(y):
        plt.text(i, b + 7, b, ha = 'center', va = 'center')
    plt.title("bar")
    plt.savefig(path, transparent = True)


def plot_radar(data, kinds, path, title):
    labels = data.columns.values
    result = pd.concat([data, data[[labels[0]]]], axis = 1)
    centers = np.array(result.loc[:, :])
    n = len(labels)
    angle = np.linspace(0, 2 * np.pi, n, endpoint = False)
    angle2 = np.concatenate((angle, [angle[0]]))
    plt.figure(dpi = 160)
    for i in range(len(kinds)):
        plt.subplot(111, polar = True)
        plt.plot(angle2, centers[i], color = colors[i], linewidth = 1, label = kinds[i])
        plt.fill(angle2, centers[i], color = colors[i], alpha = 0.1)
    plt.thetagrids(angle * 180 / np.pi, labels)
    plt.legend(loc = (-0.2, 0), fontsize = 10)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, transparent = True)


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

    plt.savefig(path, transparent = True)

    labels = inputs.loc[:, 'label']
    class_weights = labels.value_counts() / len(labels)
    print(class_weights)

    feature_details = inputs.describe().transpose()

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
    title = "Confusion matrix of a stacking method"
    # .format(clf.base_estimator.__class__.__name__,dim_reduce.__class__.__name__)
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
    plt.savefig(path, transparent = True)

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
    names.append("Stacking")
    ax.set_yticklabels(names)
    plt.xlim(min(test_scores) - 0.01, max(test_scores) + 0.01)
    plt.barh(x, test_scores, alpha = 0.6,
             color = colors)
    title = "Test Data {} Scores".format(metric_fn)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, transparent = True)
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

    plt.savefig(path, transparent = True)
    # 输出报告
    print(classification_report(y_test, y_pred))
    print("菠菜的分数 test set: {:.4f}.".format(balanced_accuracy_score(y_test, y_pred)))


def plot_age_dependence(player_data, path):
    def_data = player_data[player_data["pos"] == "def"]
    forw_data = player_data[player_data["pos"] == "for"]
    gk_data = player_data[player_data["pos"] == "gk"]
    midf_data = player_data[player_data["pos"] == "mid"]

    fig = plt.figure(dpi = 140, facecolor = 'w', edgecolor = 'k')
    group = forw_data.groupby("age")["overall_rating"].mean().reset_index()
    age_forw = group["age"]
    rating_forw = group["overall_rating"]

    ff = interp1d(age_forw, rating_forw, kind = 'cubic')
    group = def_data.groupby("age")["overall_rating"].mean().reset_index()
    age_def = group["age"]
    rating_def = group["overall_rating"]
    fd = interp1d(age_def, rating_def, kind = 'cubic')

    group = gk_data.groupby("age")["overall_rating"].mean().reset_index()
    age_gk = group["age"]
    rating_gk = group["overall_rating"]
    fg = interp1d(age_gk, rating_gk, kind = 'cubic')

    group = midf_data.groupby("age")["overall_rating"].mean().reset_index()
    age_midf = group["age"]
    rating_midf = group["overall_rating"]
    fm = interp1d(age_midf, rating_midf, kind = 'cubic')

    agenew = np.linspace(17, 40, num = 100, endpoint = True)
    subplot = fig.add_subplot(111)
    subplot.tick_params(axis = 'both', which = 'major')
    plt.xlabel('Age')
    plt.ylabel('Overall Rating')
    plt.plot(agenew, ff(agenew), "-", agenew, fd(agenew), "-", agenew, fg(agenew), "-", agenew, fm(agenew), "-")
    plt.legend(['Forward', "Defender", "Goalkeeper", "Midfielder"], loc = 'best')
    plt.title("Age and Rating")
    plt.savefig(path, transparent = True)


def plot_beautiful_scatter_weight_and_height(player_data, path):
    lbs_to_kg = 0.453592
    fig = plt.figure(dpi = 160, facecolor = 'w', edgecolor = 'k')
    def_data = player_data[player_data["pos"] == "def"]
    forw_data = player_data[player_data["pos"] == "for"]
    gk_data = player_data[player_data["pos"] == "gk"]
    midf_data = player_data[player_data["pos"] == "mid"]
    def_heigh = (def_data["height"] + np.random.normal(loc = 0.0, scale = 3.0, size = len(def_data)))
    forw_heigh = forw_data["height"] + np.random.normal(loc = 0.0, scale = 3.0, size = len(forw_data))
    gk_heigh = gk_data["height"] + np.random.normal(loc = 0.0, scale = 3.0, size = len(gk_data))
    midf_heigh = midf_data["height"] + np.random.normal(loc = 0.0, scale = 3.0, size = len(midf_data))
    def_weight = (def_data["weight"] + np.random.normal(loc = 0.0, scale = 3.0, size = len(def_data))) * lbs_to_kg
    forw_weight = (forw_data["weight"] + np.random.normal(loc = 0.0, scale = 3.0, size = len(forw_data))) * lbs_to_kg
    gk_weight = (gk_data["weight"] + np.random.normal(loc = 0.0, scale = 3.0, size = len(gk_data))) * lbs_to_kg
    midf_weight = (midf_data["weight"] + np.random.normal(loc = 0.0, scale = 3.0, size = len(midf_data))) * lbs_to_kg
    subplot = fig.add_subplot(111)
    subplot.tick_params(axis = 'both', which = 'major')
    midf = subplot.scatter(midf_weight, midf_heigh, marker = 'o', color = colors[0], alpha = 0.6, s = 5)
    defend = subplot.scatter(def_weight, def_heigh, marker = 'v', color = colors[1], alpha = 0.6, s = 5)
    forw = subplot.scatter(forw_weight, forw_heigh, marker = '*', color = colors[2], alpha = 0.6, s = 5)
    gk = subplot.scatter(gk_weight, gk_heigh, marker = '^', color = colors[3], alpha = 0.6, s = 5)
    plt.xlabel('Weight (kilograms)')
    plt.ylabel('Height (centimeters)')
    plt.legend((defend, forw, gk, midf), ('Defender', 'Forward', 'Goalkeeper', 'Midfielder'), scatterpoints = 1,
               loc = 'upper left', ncol = 1, fontsize = 10)
    plt.title("Height and Weight")
    plt.tight_layout()
    plt.savefig(path, transparent = True)
