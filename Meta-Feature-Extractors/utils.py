import numpy as np
from scipy.stats import spearmanr


def distance(x, y):
    return np.linalg.norm(x - y)


def transform_to_ranks(df):
    return df.rank()


def distance_vector(X):
    dvector = []
    for i in range(X.shape[0]):
        for j in range(i + 1, X.shape[0]):
            dvector.append(distance(X[i, :], X[j, :]))
    return np.asarray(dvector)


def correlation_vector(X):
    cvector = []
    for i in range(X.shape[0]):
        for j in range(i + 1, X.shape[0]):
            cor_ = spearmanr(X[i, :], X[j, :]).correlation
            try:
                cor_ = float(cor_)
            except:
                cor_ = 0
            cvector.append(cor_)
    return np.asarray(cvector)
