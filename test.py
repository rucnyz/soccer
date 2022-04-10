# -*- coding: utf-8 -*-
# @Time    : 2022/4/10 10:11
# @Author  : nieyuzhou
# @File    : test.py
# @Software: PyCharm
import os
import pandas as pd

from autogluon.tabular import TabularPredictor


def test(test_data, path):
    predictor = TabularPredictor.load(os.path.join(path, "model"))
    y_pred = predictor.predict_proba(test_data)
    return y_pred


if __name__ == '__main__':
    data_path = "./data/"

    test_pos = pd.read_excel(os.path.join(data_path, "processed/pos_change.xlsx"), index_col = 0)
    test_player = pd.read_excel(os.path.join(data_path, "processed/ability_change.xlsx"), index_col = 0)
    y_pred_pos = test(test_pos.drop("label", axis = 1), data_path)
    y_pred_player = test(test_player.drop("label", axis = 1), data_path)
