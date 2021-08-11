import numpy as np
from sklearn.linear_model import LinearRegression

from visualization.linear_plots import plot_line


def ols_sklearn(feature, y_values):
    """
    Solve Ordinary Least Squares using scikit-learn's LinearRegression
    """
    estimator = LinearRegression()
    # note that the intercept is built inside LinearRegression
    estimator.fit(feature.reshape(-1, 1), y_values)
    slope = estimator.coef_[0]
    intercept = estimator.intercept_
    return slope, intercept


def ols_numpy(feature, y_values):
    """
    Solve Ordinary Least Squares using numpy.linalg.lstsq

    Fits a line `y = mx + c`.
    We can rewrite the line equation as `y = Ap`,
    where A = [[x 1]] and p = [[m], [c]].

    """
    ones = np.ones(y_values.shape[0])
    matrix = np.vstack((feature, ones)).T
    slope, intercept = np.linalg.lstsq(matrix, y_values, rcond=None)[0]
    return slope, intercept


def ols_closed_form(feature, y_values):
    """
    Compute the closed form of linear regression

    slope = (A - B) / (C - D)
    intercept = [ sum(output) / N ] - slope * [ sum(feature) / N ]

    where,
        A = sum(feature * output)
        B = [sum(feature) * sum(output)] * 1/N
        C = sum(feature^2)
        D = sum(feature)^2 * 1/N

    Returns: the intercept and slope values as a tuple

    """
    size = feature.shape[0]
    sum_features = np.sum(feature)
    sum_squared_features = np.sum(feature * feature)
    sum_output = np.sum(y_values)

    # calculate slope
    numerator = np.sum(feature * y_values) - (sum_features * sum_output) / size
    denominator = sum_squared_features - (sum_features * sum_features) / size
    slope = numerator / denominator

    # use this computed slope to compute the intercept:
    intercept = np.mean(y_values) - slope * np.mean(feature)
    return slope, intercept


def regression_prediction(input_feature, intercept, slope):
    """
    Calculate the predicted values based on the liner regression model

    Returns:
        the estimated value
    """
    return intercept + slope * input_feature


def gds_form(feature, y_values, step_size: float = 1e-2, tolerance: float = 1e-3, max_iter: int = 1e2):
    slope, intercept = ols_closed_form(feature, y_values)
    magnitude = np.inf
    iteration = 0
    while magnitude > tolerance and iteration < max_iter:
        predictions = regression_prediction(feature, intercept, slope)

        # Compute the prediction errors (prediction - Y)
        residual_errors = predictions - y_values

        # Update the intercept
        # The derivative of the cost for the intercept
        # is the sum of the errors
        intercept -= step_size * np.sum(residual_errors)

        # Update the slope
        # The derivative of the cost for the slope
        # is the sum of the product of the errors and the input
        partial = np.sum(feature * residual_errors)
        slope -= step_size * partial

        # Compute the magnitude of the gradient
        magnitude = np.sqrt(sum(residual_errors ** 2, partial ** 2))
        iteration += 1

    return slope, intercept


if __name__ == "__main__":
    feature_data = np.array([0, 1, 2, 3])
    y_data = np.array([-1, 0.2, 0.9, 2.1])
    # feature_data = np.array([0, 1, 2, 3, 4])
    # y_data = np.array([1, 3, 7, 13, 21])
    # feature_data = np.array([0, 1, 2, 3, 4])
    # y_data = np.array([1, 2, 3, 4, 5])

    m, c = ols_numpy(feature_data, y_data)
    print(f"numpy based: ({m:.2f}, {c:.2f})")

    m, c = ols_sklearn(feature_data, y_data)
    print(f"sklearn based: ({m:.2f}, {c:.2f})")

    m, c = ols_closed_form(feature_data, y_data)
    print(f"closed form: ({m:.2f}, {c:.2f})")

    m, c = gds_form(feature_data, y_data)
    print(f"gds_form: ({m:.2f}, {c:.2f})")

    plot_line(feature_data, y_data, m, c)
