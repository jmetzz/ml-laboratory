from typing import Tuple

import numpy as np


def accuracy(true_labels, predicted_labels) -> float:
    """
    Calculate the accuracy of predictions.

    Args:
        true_labels (list or np.ndarray): The ground truth labels.
        predicted_labels (list or np.ndarray): The predicted labels by the model.

    Returns:
        float: The accuracy of the predictions, ranging from 0 to 1.
    """
    return np.sum([int(y == y_hat) for y, y_hat in zip(true_labels, predicted_labels)]) / len(true_labels)


def precision(true_labels, predicted_labels) -> float:
    """
    Calculate the precision of the predictions.

    Precision is the ratio of correctly predicted positive observations
    to the total predicted positives. It is a measure of the accuracy of
    the positive predictions.

    Args:
        true_labels (np.ndarray): The ground truth labels.
        predicted_labels (np.ndarray): The predicted labels.

    Returns:
        float: The precision of the predictions. Returns 0 if the denominator is 0.

    Raises:
        ValueError: If `true_labels` and `predicted_labels` have different lengths.
    """
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Length of true_labels and predicted_labels must be the same.")

    y_true, y_hat = np.asarray(true_labels), np.asarray(predicted_labels)

    true_positive = np.sum((y_hat == 1) & (y_true == 1))
    false_positive = np.sum((y_hat == 1) & (y_true == 0))
    denominator = true_positive + false_positive
    return true_positive / denominator if denominator > 0 else 0


def recall(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    """
    Calculate the recall of the predictions.

    Recall (or sensitivity) is the ratio of correctly predicted positive observations
    to all observations in the actual class. It measures the ability of
    the model to find all the relevant cases within a dataset.

    Args:
        true_labels (np.ndarray): The ground truth labels.
        predicted_labels (np.ndarray): The predicted labels.

    Returns:
        float: The recall of the predictions. Returns 0 if the denominator is 0.

    Raises:
        ValueError: If `true_labels` and `predicted_labels` have different lengths.
    """
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Length of true_labels and predicted_labels must be the same.")

    y_true, y_hat = np.asarray(true_labels), np.asarray(predicted_labels)

    tp = np.sum((y_hat == 1) & (y_true == 1))
    fn = np.sum((y_hat == 0) & (y_true == 1))
    denominator = tp + fn
    return tp / (tp + fn) if denominator > 0 else 0


def f1_score(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    """
    Calculate the F1 score of the predictions.

    The F1 score is the harmonic mean of precision and recall.
    It is a measure of a test's accuracy that considers both
    the precision and the recall.

    Args:
        true_labels (np.ndarray): The ground truth labels.
        predicted_labels (np.ndarray): The predicted labels.

    Returns:
        float: The F1 score of the predictions.

    Raises:
        ValueError: If `true_labels` and `predicted_labels` have different lengths.
    """
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Length of true_labels and predicted_labels must be the same.")

    pr = precision(true_labels, predicted_labels)
    re = recall(true_labels, predicted_labels)
    return 2 * (pr * re) / (pr + re) if (pr + re) != 0 else 0


def confusion_matrix(true_labels, predicted_labels) -> np.ndarray:
    """
    Calculate the confusion matrix.

    The confusion matrix is a 2x2 numpy array with the following layout:

    [
        [True Negative, False Positive],
        [False Negative, True Positive]
    ]

    Args:
        true_labels (np.ndarray): The ground truth labels.
        predicted_labels (np.ndarray): The predicted labels.

    Returns:
        np.ndarray: The confusion matrix.

    Raises:
        ValueError: If `true_labels` and `predicted_labels` have different lengths.
    """
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Length of true_labels and predicted_labels must be the same.")

    y_true, y_hat = np.asarray(true_labels), np.asarray(predicted_labels)

    # True negatives (TN): Predicted = 0, True = 0
    true_negative = np.sum((y_hat == 0) & (y_true == 0))

    # False positives (FP): Predicted = 1, True = 0
    false_positive = np.sum((y_hat == 1) & (y_true == 0))

    # False negatives (FN): Predicted = 0, True = 1
    false_negative = np.sum((y_hat == 0) & (y_true == 1))

    # True positives (TP): Predicted = 1, True = 1
    true_positive = np.sum((y_hat == 1) & (y_true == 1))

    matrix = np.array([[true_negative, false_positive], [false_negative, true_positive]])
    return matrix


def roc(true_labels: np.ndarray, prediction_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the ROC curve data points.

    The ROC curve plots the true positive rate (TPR) against the
    false positive rate (FPR) at various threshold settings.

    Args:
        true_labels (np.ndarray): The ground truth binary labels.
        prediction_scores (np.ndarray): The prediction scores for the positive class,
        not necessarily binary labels.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing arrays for
        FPR, TPR, and thresholds.

    Raises:
        ValueError: If `true_labels` and `prediction_scores` have different lengths.
    """
    if len(true_labels) != len(prediction_scores):
        raise ValueError("Length of true_labels and prediction_scores must be the same.")

    # sort prediction scores and corresponding truth values
    y_true = np.asarray(true_labels)
    y_scores = np.asarray(prediction_scores)

    desc_sorted_indices = np.argsort(y_scores)[::-1]
    y_scores = y_scores[desc_sorted_indices]
    y_true = y_true[desc_sorted_indices]

    # Get unique threshold values through differentiation
    thresholds = np.unique(y_scores)

    # Initialize lists to hold the true positive rates and false positive rates
    true_positive_rates = []
    false_positive_rates = []

    # Calculate TPR and FPR for each threshold
    for thresh in thresholds:
        # Predictions based on the threshold
        _preds = (y_scores >= thresh).astype(int)

        # True positives, false positives, true negatives, false negatives
        true_positives = np.sum((_preds == 1) & (y_true == 1))
        false_positives = np.sum((_preds == 1) & (y_true == 0))
        true_negatives = np.sum((_preds == 0) & (y_true == 0))
        false_negatives = np.sum((_preds == 0) & (y_true == 1))

        true_positive_rates.append(true_positives / (true_positives + false_negatives))
        false_positive_rates.append(false_positives / (false_positives + true_negatives))

    return np.array(false_positive_rates), np.array(true_positive_rates), thresholds


def basic_evaluation(true_labels, prediction) -> Tuple[float, float]:
    """
    Perform basic evaluation of the predictions by calculating precision and recall.

    Args:
        true_labels (np.ndarray): The ground truth labels.
        prediction (np.ndarray): The predicted labels.

    Returns:
        Tuple[float, float]: A tuple containing the precision and recall of the predictions.

    Raises:
        ValueError: If `true_labels` and `prediction` have different lengths.
    """
    if len(true_labels) != len(prediction):
        raise ValueError("Length of true_labels and prediction must be the same.")

    return precision(true_labels, prediction), recall(true_labels, prediction)


def mae(true_labels, predictions) -> float:
    """
    Calculate the Mean Absolute Error (MAE).

    MAE measures the average magnitude of the errors in a set of predictions, without considering their direction.
    It's the average over the test sample of the absolute differences between prediction and actual observation
    where all individual differences have equal weight.

    Args:
        true_labels (np.ndarray): The ground truth values.
        predictions (np.ndarray): The predicted values.

    Returns:
        float: The MAE of the predictions.

    Raises:
        ValueError: If `true_labels` and `predictions` have different lengths.
    """
    if len(true_labels) != len(predictions):
        raise ValueError("Length of true_labels and predictions must be the same.")

    y_true, y_hat = np.asarray(true_labels), np.asarray(predictions)
    return np.abs(y_true - y_hat).sum() / len(y_true)


def mse(true_labels, predictions) -> float:
    """
    Calculate the Mean Absolute Error (MAE).

    MAE measures the average magnitude of the errors in a set of predictions,
    without considering their direction.
    It's the average over the test sample of the absolute differences between
    prediction and actual observation where all individual differences have equal weight.

    Args:
        true_labels (np.ndarray): The ground truth values.
        predictions (np.ndarray): The predicted values.

    Returns:
        float: The MAE of the predictions.

    Raises:
        ValueError: If `true_labels` and `predictions` have different lengths.
    """
    if len(true_labels) != len(predictions):
        raise ValueError("Length of true_labels and predictions must be the same.")
    y_true, y_hat = np.asarray(true_labels), np.asarray(predictions)
    return np.sum((y_true - y_hat) ** 2) / len(y_true)


def rmse(true_labels, predictions) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE).

    RMSE is a measure of the differences between values predicted by
    a model and the values observed.
    It represents the square root of the second sample moment of
    the differences between predicted values and observed values
    or the quadratic mean of these differences.

    RMSE is a measure of accuracy, to compare prediction errors of
    different models or model configurations for a particular dataset and
    not between datasets, as it is scale-dependent.

    Args:
        true_labels (np.ndarray): The ground truth values, observed values.
        predictions (np.ndarray): The predicted values by the model.

    Returns:
        float: The RMSE of the predictions. It gives a measure of the
        magnitude of the error, with the same units as the data.

    Raises:
        ValueError: If `true_labels` and `predictions` have different lengths.
    """
    if len(true_labels) != len(predictions):
        raise ValueError("Length of true_labels and predictions must be the same.")

    y_true, y_hat = np.asarray(true_labels), np.asarray(predictions)
    return np.sqrt(np.sum((y_true - y_hat) ** 2) / len(y_true))


def mape(true_labels, predictions) -> float:
    """
    Calculate the Mean Absolute Percentage Error (MAPE).

    MAPE measures the average magnitude of the errors in percentage terms.
    It is calculated as the average of the absolute differences between
    the predicted values and observed values divided by the observed values,
    often expressed as a percentage.

    MAPE is a useful measure for forecasting accuracy that normalizes the
    absolute error by the observed values, making it relative and easier
    to interpret across different datasets.

    Args:
        true_labels (np.ndarray): The ground truth values, observed values.
        predictions (np.ndarray): The predicted values by the model.

    Returns:
        float: The MAPE of the predictions, expressed as a percentage.
        It represents the average absolute percent error for each
        value compared to the actual value.

    Raises:
        ValueError: If `true_labels` and `predictions` have different lengths.
        ZeroDivisionError: If any `true_labels` value is zero, which would
        result in a division by zero in the calculation.

    Note:
        MAPE can be significantly impacted by zero or near-zero values
        in the true_labels since these lead to undefined or
        very large percentage errors.
    """
    if len(true_labels) != len(predictions):
        raise ValueError("Length of true_labels and predictions must be the same.")

    y_true, y_hat = np.asarray(true_labels), np.asarray(predictions)
    if np.any(y_true == 0):
        raise ZeroDivisionError("MPE undefined for true_labels with value zero.")

    return np.mean(np.abs((y_true - y_hat) / y_true))


def mpe(true_labels, predictions) -> float:
    """
    Calculate the Mean Percentage Error (MPE).

    MPE measures the average of the percentage differences between
    the predicted values and observed values.
    Unlike MAPE, which takes the absolute value of the percentage errors,
    MPE can indicate the direction of errors (i.e., whether they are predominantly
    overestimates or underestimates).

    It is calculated as the mean of the differences between predicted
    and observed values divided by the observed values, expressed as a percentage.

    Args:
        true_labels (np.ndarray): The ground truth values, observed values.
        predictions (np.ndarray): The predicted values by the model.

    Returns:
        float: The MPE of the predictions, expressed as a percentage.
        Positive values indicate overestimation bias, while negative
        values indicate underestimation bias.

    Raises:
        ValueError: If `true_labels` and `predictions` have different lengths.
        ZeroDivisionError: If any `true_labels` value is zero,
        which would result in a division by zero in the calculation.

    Note:
        MPE is affected by zero or near-zero values in the true_labels,
        which can lead to large or undefined errors.
    """
    if len(true_labels) != len(predictions):
        raise ValueError("Length of true_labels and predictions must be the same.")

    y_true, y_hat = np.asarray(true_labels), np.asarray(predictions)
    if np.any(y_true == 0):
        raise ZeroDivisionError("MPE undefined for true_labels with value zero.")

    return np.mean((y_true - y_hat) / y_true)


if __name__ == "__main__":
    y_true = np.array([0, 1, 1, 0, 1])  # Example ground truth labels
    y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.7])  # Example prediction scores
    fpr, tpr, tr = roc(y_true, y_scores)
    print(fpr)
    print(tpr)
    print(tr)
