# This code optimizes the hyper parameters and train the model to perform quantile regression
#!/usr/bin/env python
# coding: utf-8

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

import numpy as np
from sklearn.cluster import KMeans

def rf_quantile(m, X, q):
    rf_preds = []
    for estimator in m.estimators_:
        rf_preds.append(estimator.predict(X))
    rf_preds = np.array(rf_preds).transpose()  # One row per record.
    return np.percentile(rf_preds, q * 100, axis=1)

def decision_tree_bagging_regressor_quantile_clusters(X, y):

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
    N_OF_ESTIMATORS_OF_DT_FIND = [x for x in range(50, int(N_DATA*0.1), 50)]

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


    quantile_ranges = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    merged_quantile = rf_quantile(regr, X, 0.1)
    for i in quantile_ranges:
        qtile = rf_quantile(regr, X, 0.9)
        merged_quantile = np.column_stack((merged_quantile, qtile))

    # Clustering
    # ----------------------
    N_CLUSTERS = int(N_DATA/200)

    model = KMeans(n_clusters=N_CLUSTERS)
    model.fit(merged_quantile)
    
    return model
