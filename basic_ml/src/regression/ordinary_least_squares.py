import numpy as np
from sklearn.linear_model import LinearRegression

from visualization.linear_plots import plot_line


def ols_sklearn(row):
    """
    Solve Ordinary Least Squares using scikit-learn's LinearRegression
    """
    estimator = LinearRegression()
    data = np.arange(row.shape[0]).reshape(-1, 1)
    # note that the intercept is built inside LinearRegression
    estimator.fit(data, row)
    slope = estimator.coef_[0]
    intercept = estimator.intercept_
    return slope, intercept


def ols_numpy(row):
    """
    Solve Ordinary Least Squares using numpy.linalg.lstsq

    Fits a line `y = mx + c`.
    We can rewrite the line equation as `y = Ap`,
    where A = [[x 1]] and p = [[m], [c]].

    """
    data = np.arange(row.shape[0])
    ones = np.ones(row.shape[0])
    matrix = np.vstack((data, ones)).T
    slope, intercept = np.linalg.lstsq(matrix, row, rcond=None)[0]
    return slope, intercept


if __name__ == "__main__":
    x_values = np.array([0, 1, 2, 3])
    y_values = np.array([-1, 0.2, 0.9, 2.1])
    m, c = ols_numpy(y_values)
    # m, c = ols_sklearn(y_values)
    plot_line(x_values, y_values, m, c)
    print((m, c))
