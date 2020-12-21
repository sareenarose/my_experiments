# Step 3 of PDA

#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

from PDA import *
from DTBR import *

from scipy.stats import chi2
import math


def decision_tree_bagging_regressor_optim(X, y):

    # Global Variables
    # --------------------------
    N_DATA = X.shape[0]
    N_DATA
    
    N_DICISION_TREE_LEAF_SAMPLES_ESTIMATORS = [x for x in range(5, int(N_DATA*0.1), 5)]
    
    # TreeBagger1
    # -------------------

    baggingRegressor_estimator = None
    number_of_samples_per_leaf_for_best_DT = None
    for val in N_DICISION_TREE_LEAF_SAMPLES_ESTIMATORS:
        DTregressor = DecisionTreeRegressor(min_samples_leaf=val, random_state=0)
        regr = BaggingRegressor(base_estimator=DTregressor, n_estimators=200, oob_score=True, max_samples=200, random_state=0).fit(X, y)
        if baggingRegressor_estimator == None:
            baggingRegressor_estimator = regr
            number_of_samples_per_leaf_for_best_DT = val
        else:
            new_baggingRegressor_error = 1 - regr.oob_score_
            current_baggingRegressor_error = 1 - baggingRegressor_estimator.oob_score_
            if new_baggingRegressor_error < current_baggingRegressor_error:
                baggingRegressor_estimator = regr
                number_of_samples_per_leaf_for_best_DT = val


    # TreeBagger2
    # -------------------
    N_OF_ESTIMATORS_OF_DT_FIND = [x for x in range(10, int(N_DATA*0.1), 10)]

    treeBagger = None
    number_of_estimators = None
    for val in N_OF_ESTIMATORS_OF_DT_FIND:
        DTregressor = baggingRegressor_estimator
        regr = BaggingRegressor(base_estimator=DTregressor, n_estimators=val, oob_score=True, max_samples=200, random_state=0).fit(X, y)
        if treeBagger == None:
            treeBagger = regr
            number_of_estimators = val
        else:
            current_treeBagger_error = 1 - treeBagger.oob_score_
            new_treeBagger_error = 1 - regr.oob_score_
            if new_treeBagger_error < current_treeBagger_error:
                treeBagger = regr
                number_of_estimators = val

    # TreeBagger3
    # -----------------------

    DTregressor = DecisionTreeRegressor(min_samples_leaf=number_of_samples_per_leaf_for_best_DT, random_state=0)
    regr = BaggingRegressor(base_estimator=DTregressor, n_estimators=number_of_estimators, oob_score=True, max_samples=200, random_state=0).fit(X, y)
    
    return regr


def decision_tree_bagging_regressor(X, y):

    # Global Variables
    # --------------------------
    N_DATA = X.shape[0]
    N_DATA
    
    N_DICISION_TREE_LEAF_SAMPLES_ESTIMATORS = [x for x in range(5, int(N_DATA*0.1), 5)]
    
    # TreeBagger1
    # -------------------

    baggingRegressor_estimator = None
    number_of_samples_per_leaf_for_best_DT = None
    for val in N_DICISION_TREE_LEAF_SAMPLES_ESTIMATORS:
        DTregressor = DecisionTreeRegressor(min_samples_leaf=val, random_state=0)
        regr = BaggingRegressor(base_estimator=DTregressor, n_estimators=200, oob_score=True, max_samples=200, random_state=0).fit(X, y)
        if baggingRegressor_estimator == None:
            baggingRegressor_estimator = regr
            number_of_samples_per_leaf_for_best_DT = val
        else:
            new_baggingRegressor_error = 1 - regr.oob_score_
            current_baggingRegressor_error = 1 - baggingRegressor_estimator.oob_score_
            if new_baggingRegressor_error < current_baggingRegressor_error:
                baggingRegressor_estimator = regr
                number_of_samples_per_leaf_for_best_DT = val


    # TreeBagger2
    # -------------------
    N_OF_ESTIMATORS_OF_DT_FIND = [x for x in range(10, int(N_DATA*0.1), 10)]

    treeBagger = None
    number_of_estimators = None
    for val in N_OF_ESTIMATORS_OF_DT_FIND:
        DTregressor = baggingRegressor_estimator
        regr = BaggingRegressor(base_estimator=DTregressor, n_estimators=val, oob_score=True, max_samples=200, random_state=0).fit(X, y)
        if treeBagger == None:
            treeBagger = regr
            number_of_estimators = val
        else:
            current_treeBagger_error = 1 - treeBagger.oob_score_
            new_treeBagger_error = 1 - regr.oob_score_
            if new_treeBagger_error < current_treeBagger_error:
                treeBagger = regr
                number_of_estimators = val
    
    return number_of_samples_per_leaf_for_best_DT, number_of_estimators

data = pd.read_excel('Train.xlsx')
X = data[[3, 5, 6, 7, 11]]
y = data[10]

model_k_means = decision_tree_bagging_regressor_quantile_clusters(X, y)

X = data[[3, 5, 6, 7, 11, 10]]

X['label'] = model_k_means.labels_

X[X['label'] == 0].shape

cluster_record = X[X['label'] == 0][[3, 5, 6, 7, 11, 10]]

group_of_inliers, group_of_ambigious_records, group_of_outliers = compute_PD_based_outliers(cluster_record)

group_of_inliers = pd.DataFrame(group_of_inliers)

X = cluster_record[[3, 5, 6, 7, 11]]
y = cluster_record[10]

# DT Regressor
# ------------------------------------------------------------
DT_Regressor = decision_tree_bagging_regressor_optim(X, y)

pred = DT_Regressor.predict(group_of_inliers)

# -------------------------------------------------------------

yFit = np.column_stack((y, pred))

yFit = np.column_stack((yFit, np.absolute(y - pred)))

t_cookd = np.percentile(yFit[:, 2], 0.75 * 100, axis=0)

in_ = yFit[:, 2] <= t_cookd

gr1_RO=group_of_inliers[in_]
gr1_VO=group_of_inliers[~in_]

X = gr1_RO[[3, 5, 6, 7, 11]]
y = gr1_RO[10]

leaf, estmtr = decision_tree_bagging_regressor(X, y)

group_of_outliers = np.concatenate((group_of_outliers, group_of_ambigious_records), axis=0)

# TreeBagger
# -----------------------

X = group_of_outliers[[3, 5, 6, 7, 11]]
y = group_of_outliers[10]

DTregressor = DecisionTreeRegressor(min_samples_leaf=leaf, random_state=0)
regr = BaggingRegressor(base_estimator=DTregressor, n_estimators=estmtr, oob_score=True, max_samples=200, random_state=0).fit(X, y)

pred = regr.predict(X)

# --------------------------------------

yFit = np.column_stack((y, pred))

yFit = np.column_stack((yFit, np.absolute(y - pred)))

t_cookd = np.percentile(yFit[:, 2], 0.75 * 100, axis=0)

in_ = yFit[:, 2] <= t_cookd

gr1_GLP=group_of_outliers[in_]
gr1_BLP=group_of_outliers[~in_]


