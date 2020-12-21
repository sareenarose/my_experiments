from MVOD import multivariate_outlier_detection
# MVOD.py should contain the function 'multivariate_outlier_detection' that implements
# multivariate outlier detection as in 
# https://www.github.com/AntonSemechko/Multivariate-Outliers

import pandas as pd
import numpy as np
import math

from scipy.stats.distributions import chi2


def mahalanobis(x=None, data=None, cov=None, mean=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - mean
    if not cov.any():
        #cov = np.cov(data.values.T)
        cov = np.cov(data.T)
    try:
        inv_covmat = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        arr = np.empty((len(x.index),1))
        arr[:] = np.NaN
        return arr
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()

def covarience_matrix_fn(np_array, mean):
    np_array -= mean 
    N = np_array.shape[0]
    fact = N - 1
    np_array = np.array(np_array)
    by_hand = np.dot(np_array.T, np_array.conj()) / fact
    return by_hand


def compute_PD_based_outliers(X):

    group_of_inliers = None
    group_of_outliers = None
    group_of_ambigious_records = None

    mean = X.mean(axis=0)
    cov = covarience_matrix_fn(X, mean)
    mahalanobis_distance = mahalanobis(x=X, data=X, cov=cov, mean=mean)

    p_val = 1 - chi2.cdf(mahalanobis_distance, 4)

    m, co, RD, chi_crt=multivariate_outlier_detection(X)

    c1=p_val>.05
    c2=RD<chi_crt[2]
    in_ = c1&c2
    
    in_ = pd.DataFrame(in_)
    
    group_of_inliers = X[in_]
    group_of_ambigious_records = X[c1&~c2|~c1&c2]
    group_of_outliers = X[~c1&~c2]

    m1, co1, RD, chi_crt1=multivariate_outlier_detection(group_of_inliers)

    index_RD_sorted_I1 = np.argsort(RD)

    mean = group_of_inliers.mean(axis=0)
    cov = covarience_matrix_fn(group_of_inliers, mean)
    m3 = mahalanobis(x=group_of_inliers, data=group_of_inliers, cov=cov, mean=mean)
    
    index_mahalnobis_dist_sorted_I2 = np.argsort(m3)
    
    s1=math.ceil(group_of_inliers.shape[0]*.75);
    start=group_of_inliers.shape[0]-s1;
    
    index_set = set(index_RD_sorted_I1[start:group_of_inliers.shape[0]].index).intersection(set([group_of_inliers.index[i] for i in index_mahalnobis_dist_sorted_I2][start:group_of_inliers.shape[0]]))
    
    index_DF = pd.DataFrame(index_set)

    s = np.ones((index_DF.shape[0], 1))
    s = np.concatenate((s, np.zeros((group_of_outliers.shape[0], 1))), axis=0)
    
    try:
        X = group_of_inliers.loc[index_DF[0]]
    except:
        X = group_of_inliers

    start=X.shape[0]+1
    stop=group_of_inliers.shape[0]+start-1

    X = np.concatenate((X, group_of_outliers), axis=0)

    mean = X.mean(axis=0)
    cov = covarience_matrix_fn(X, mean)
    m4 = mahalanobis(x=X, data=X, cov=cov, mean=mean)

    p_val = 1 - chi2.cdf(m4, 4)

    m5, co2, RD, chi_crt=multivariate_outlier_detection(pd.DataFrame(X), math.floor((X.shape[0]+5+1)/2), s)

    c1 = p_val[start:stop] > .05
    c2 = RD[start:stop] < chi_crt[2]
    in_ = c1&c2

    in_ = pd.DataFrame(in_)

    group_of_inliers = np.concatenate((group_of_inliers, group_of_ambigious_records[in_]), axis=0)
    group_of_ambigious_records = group_of_ambigious_records.drop(group_of_ambigious_records[~in_].index)

    return group_of_inliers, group_of_ambigious_records, group_of_outliers
