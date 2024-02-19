from typing import Tuple

import numpy as np

from neural_networks.base.activations import softmax


def logistic_regression(features, target, weights, iterations, learning_rate=0.01) -> np.ndarray:
    """
    Train a logistic regression model using gradient descent.

    This function updates the weights of a simple logistic regression model
    using the gradient descent algorithm. It assumes binary targets and uses a sigmoid function
    as the logistic function.

    Args:
        features (np.ndarray): The feature matrix where rows represent samples and columns represent features.
        target (np.ndarray): The target vector, where each element is the target for a sample.
        weights (np.ndarray): The initial weights vector for the logistic regression model.
        iterations (int): The number of iterations to perform gradient descent.
        learning_rate (float, optional): The learning rate for gradient descent. Defaults to 0.01.

    Returns:
        np.ndarray: The updated weights after training.

    Note:
        This implementation does not include a bias term or regularization
        and assumes binary classification.
    """
    for _ in range(iterations):
        predictions = 1 / (1 + np.exp(-np.dot(features, weights)))
        errors = target - predictions
        gradient = np.dot(features.T, errors)
        weights += learning_rate * gradient
    return weights


def logistic_regression_l2regularized(features,
                                      target,
                                      weights,
                                      bias,
                                      iterations,
                                      learning_rate=0.01,
                                      regularization_strength=0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train a logistic regression model using gradient descent with L2 regularization.

    This function updates the weights and bias of a logistic regression model using
    the gradient descent algorithm, incorporating L2 regularization to penalize large weights.

    The regularization term (regularization_strength * weights) is subtracted from the gradient
    of the loss function. This penalizes large weights, encouraging the model to find simpler,
    more generalizable patterns in the data.
    L2 regularization is applied directly to the weights but not to the bias term,
    as regularizing the bias can lead to underfitting.

    Args:
        features (np.ndarray): The feature matrix where rows represent samples and columns represent features.
        target (np.ndarray): The target vector, where each element is the target for a sample.
        weights (np.ndarray): The initial weights vector for the logistic regression model.
        bias (float): The initial bias term for the logistic regression model. The bias term allows
            the model to fit the best possible plane to the data by allowing the decision boundary
            to shift. It's updated along with the weights but is not regularized.
        iterations (int): The number of iterations to perform gradient descent.
        learning_rate (float): The learning rate for gradient descent.
            Controls the step size during gradient descent.
            A smaller learning rate requires more iterations to converge,
            while a too-large learning rate can overshoot the minimum.
        regularization_strength (float): The strength of the L2 regularization.
           Higher values mean stronger regularization.

    Returns:
        np.ndarray: The updated weights after training.
        float: The updated bias after training.

    Note:
        This implementation assumes binary classification and uses a sigmoid function as the logistic function.
    """
    _weights, _bias = np.asarray(weights), np.asarray(bias)
    for _ in range(iterations):
        # linear model
        logits = np.dot(features, _weights) + _bias
        predictions = 1 / (1 + np.exp(-logits))

        # Derivative of the loss with respect to weights
        error = target - predictions
        gradient = np.dot(features.T, error) - regularization_strength * _weights
        weights_update = learning_rate * gradient

        # Derivative of the loss with respect to bias
        bias_update = learning_rate * np.sum(error)

        # Update weights and bias
        _weights += weights_update
        _bias += bias_update

    return _weights, _bias


def multinomial_logistic_regression(features,
                                    target,
                                    weights,
                                    bias,
                                    iterations,
                                    learning_rate=0.01,
                                    regularization_strength=0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train a softmax logistic regression model using gradient descent with L2 regularization.

    This function updates the weights and bias of a softmax regression model for multi-class classification.

    Args:
        features (np.ndarray): The feature matrix (samples x features).
        target (np.ndarray): The target matrix in one-hot encoded form (samples x classes).
        weights (np.ndarray): The initial weights matrix (features x classes).
        bias (np.ndarray): The initial bias vector (1 x classes).
        iterations (int): The number of iterations for gradient descent.
        learning_rate (float): The learning rate for gradient descent.
        regularization_strength (float): The strength of L2 regularization.

    Returns:
        np.ndarray: The updated weights after training.
        np.ndarray: The updated bias after training.
    """

    _weights, _bias = np.asarray(weights), np.asarray(bias)
    for _ in range(iterations):
        # linear model
        logits = np.dot(features, _weights) + _bias

        # softmax function for predictions
        y_pred, _ = softmax(logits)

        # Gradient of the loss wrt weights
        error = y_pred - target
        gradient_w = np.dot(features.T, error) / features.shape[0] + regularization_strength * weights

        # Gradient of the loss wrt bias
        gradient_b = np.mean(error, axis=0)

        # Update weights and bias
        _weights -= learning_rate * gradient_w
        _bias -= learning_rate * gradient_b

    return _weights, _bias
