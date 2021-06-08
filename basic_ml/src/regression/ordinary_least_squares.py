import numpy as np
from sklearn.linear_model import LinearRegression


def ols_sklearn(row):
    """Solve OLS using scikit-learn's LinearRegression"""
    estimator = LinearRegression()
    data = np.arange(row.shape[0]).reshape(-1, 1)  # shape (14, 1)
    # note that the intercept is built inside LinearRegression
    estimator.fit(data, row.values)
    m = estimator.coef_[0]  # note c is in estimator.intercept_
    return m


def ols_lstsq(row):
    """Solve OLS using numpy.linalg.lstsq"""
    # build X values for [0, 13]
    data = np.arange(row.shape[0])  # shape (14,)
    ones = np.ones(row.shape[0])  # constant used to build intercept
    matrix = np.vstack((data, ones)).T  # shape(14, 2)
    # lstsq returns the coefficient and intercept as the first result
    # followed by the residuals and other items
    m, _ = np.linalg.lstsq(matrix, row.values, rcond=-1)[0]
    return m


def ols_lstsq_raw(row):
    """Variant of `ols_lstsq` where row is a numpy array (not a Series)"""
    data = np.arange(row.shape[0])
    ones = np.ones(row.shape[0])
    matrix = np.vstack((data, ones)).T
    m, _ = np.linalg.lstsq(matrix, row, rcond=-1)[0]
    return m
