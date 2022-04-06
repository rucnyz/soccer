# -*- coding: utf-8 -*-
# @Time    : 2022/4/5 10:08
# @Author  : nieyuzhou
# @File    : gluon_train.py
# @Software: PyCharm
import os
from pprint import pprint

import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split

from utils.get_data import preprocess

if __name__ == '__main__':
    data_path = "./data/"
    label = 'label'
    feature_path = os.path.join(data_path, "all2.csv")
    inputs = pd.read_csv(feature_path, index_col = 0)
    col = inputs.columns
    labels, features = preprocess(inputs, norm = 0)
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(features, labels, test_size = 0.2,
                                                                    random_state = 5,
                                                                    stratify = labels)
    train_data = np.hstack([X_train_valid, y_train_valid.reshape(y_train_valid.shape[0], 1)])
    test_data = np.hstack([X_test, y_test.reshape(y_test.shape[0], 1)])
    train_data = pd.DataFrame(train_data, columns = col)
    test_data = pd.DataFrame(test_data, columns = col)

    y_train = train_data[label]
    y_test = test_data[label]
    train_data = train_data.drop(label, axis = 1).astype(np.float32)
    test_data = test_data.drop(label, axis = 1).astype(np.float32)
    train_data = pd.concat([train_data, y_train], axis = 1)
    test_data = pd.concat([test_data, y_test], axis = 1)
    train_data = TabularDataset(train_data)
    test_data = TabularDataset(test_data)

    predictor = TabularPredictor(label = label, path = os.path.join(data_path, "model")).fit(train_data)
    # 测试
    print("-----------------开始测试-----------------")
    test_data_nolab = test_data.drop(columns = [label])
    y_pred = predictor.predict(test_data_nolab)
    perf = predictor.evaluate_predictions(y_true = y_test, y_pred = y_pred, auxiliary_metrics = True)
    lb = predictor.leaderboard(test_data, silent = True)
    pprint(lb)
