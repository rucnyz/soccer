# -*- coding: utf-8 -*-
# @Time    : 2022/3/25 14:54
# @Author  : nieyuzhou
# @File    : ml_train.py
# @Software: PyCharm
from time import time

from sklearn import model_selection
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline


def predict_labels(clf, best_reduce, features, target, pred_func, average):
    """ 依据评测方法进行预测 """
    y_pred = clf.predict(best_reduce.transform(features))
    return pred_func(target, y_pred), f1_score(target, y_pred, average = average)


def train_classifier(clf, reduction, X_train, y_train, cv_sets, params, scorer, jobs):
    """ 训练 """

    start = time()
    # TODO 训练过一次后就不要用网格搜索了
    estimators = [('reduce', reduction), ('clf', clf)]
    pipeline = Pipeline(estimators)
    # 进行网格搜索
    grid_obj = model_selection.GridSearchCV(pipeline, param_grid = params, scoring = scorer, cv = cv_sets,
                                            n_jobs = jobs)
    grid_obj.fit(X_train, y_train)
    best_pipe = grid_obj.best_estimator_
    # else:
    #     # 搜索好了直接用
    #     estimators = [('reduce', reduction(n_components = best_components)), ('clf', clf(best_params))]
    #     pipeline = Pipeline(estimators)
    #     best_pipe = pipeline.fit(X_train, y_train)
    end = time()
    print("训练 {} 用时 {:.1f} 分钟".format(clf.__class__.__name__, (end - start) / 60))
    return best_pipe


def train_and_predict(clf, reduction, X_train, y_train, X_calibrate, y_calibrate, X_test, y_test, cv_sets,
                      params, scorer, jobs, args):
    """ 某个分类器的训练过程，计算指标 """
    print("----------------------------------")
    print("训练开始：分类器为 {} 降维方法为 {}".format(clf.__class__.__name__, reduction.__class__.__name__))
    # 用该分类器和降维方法训练
    best_pipe = train_classifier(clf, reduction, X_train, y_train, cv_sets, params, scorer, jobs)
    best_clf = best_pipe.named_steps['clf']
    best_reduce = best_pipe.named_steps['reduce']
    # 校准
    cal_clf = CalibratedClassifierCV(best_clf, cv = 'prefit', method = 'isotonic')
    cal_clf.fit(best_reduce.transform(X_calibrate), y_calibrate)
    # 输出结果
    acc_train, train_score = predict_labels(cal_clf, best_reduce, X_train, y_train, scorer._score_func, args.average)
    acc_test, test_score = predict_labels(cal_clf, best_reduce, X_test, y_test, scorer._score_func, args.average)
    print("[{}] training set "
          "[balanced accuracy]: {:.4f}  "
          "[{} score]: {:.4f}".format(cal_clf.base_estimator.__class__.__name__, acc_train, args.metric, train_score))
    print("[{}] test set     "
          "[balanced accuracy]: {:.4f}  "
          "[{} score]: {:.4f}".format(cal_clf.base_estimator.__class__.__name__, acc_test, args.metric, test_score))
    return cal_clf, best_reduce, acc_train, acc_test


def find_best_classifier(classifiers, reductions, scorer, X_train, y_train, X_calibrate, y_calibrate, X_test, y_test,
                         cv_sets, params, jobs,
                         args):
    """ 调参，找到最佳分类器(包括降维方法) """

    clfs_list = []
    reduce_list = []
    train_scores = []
    test_scores = []

    for red in reductions:
        for clf in classifiers:
            # 网格搜索并且测试
            best_clf, best_reduce, train_score, test_score = train_and_predict(clf = clf, reduction = red,
                                                                               X_train = X_train, y_train = y_train,
                                                                               X_calibrate = X_calibrate,
                                                                               y_calibrate = y_calibrate,
                                                                               X_test = X_test, y_test = y_test,
                                                                               cv_sets = cv_sets, params = params[clf],
                                                                               scorer = scorer, jobs = jobs,
                                                                               args = args)
            # 存储结果
            clfs_list.append(best_clf)
            reduce_list.append(best_reduce)
            train_scores.append(train_score)
            test_scores.append(test_score)
    print("----------------------------------")
    return clfs_list, reduce_list, train_scores, test_scores
