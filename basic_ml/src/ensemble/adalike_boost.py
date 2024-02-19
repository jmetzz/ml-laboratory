from typing import List

import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from ensemble.boosting import SimpleBoostClassifier


class AdaBoostLikeClassifier:
    """
    A simplified version of the adaboost algorithm.

    This algorithm combines weak learners to generate a strong learner.
    A weak learner has error probability less than 1/2, which makes it better
    than random guessing on a two-class problem, and a strong learner has
    arbitrarily small error probability.

    The rationale:
    1. Start with a Weak Learner: Assume the existence of a weak learning algorithm that
        can produce classifiers whose performance is slightly better than random guessing
        for any distribution over the training data.

    2. Iterative Process: The algorithm works by iteratively applying the
        weak learning algorithm to modified versions of the data.
        After each round, the data is modified to make the examples
        on which the current aggregate model performs poorly more important.
        The idea is to focus the weak learner on the examples that are
        hardest to classify correctly.

    3. Combining Classifiers: After several rounds of training weak learners
        on adjusted data, the final model is a combination
        (a weighted majority vote) of all the weak classifiers
        obtained in each round. The combination aims to correct the
        mistakes of the previous classifiers.

    """

    def __init__(self, learners: List):
        self.learners = learners
        self.learner_weights = []
        self.n_learners = len(learners)

    def fit(self, instances, true_labels):
        n_samples = instances.shape[0]
        weights = np.ones(n_samples) / n_samples

        for learner in self.learners:
            # Train a weak learner
            learner.fit(instances, true_labels, sample_weight=weights)
            prediction = learner.predict(instances)

            # Calculate error and learner weight, avoiding division by zero
            error = np.sum(weights * (prediction != true_labels))
            # Prevent division by zero error
            if error == 0:
                self.learner_weights.append(float("inf"))  # Assigning high weight to a perfect learner
                break
            alpha = 0.5 * np.log((1 - error) / error)

            # Update instance weights
            weights *= np.exp(-alpha * true_labels * prediction)
            weights /= np.sum(weights)

            self.learner_weights.append(alpha)

    def predict(self, instances) -> np.ndarray:
        predictions = [learner.predict(instances) for learner in self.learners]
        predictions = np.array(predictions)
        weighted_predictions = np.dot(self.learner_weights, predictions)
        return np.sign(weighted_predictions)


if __name__ == "__main__":
    n_learners = 3

    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, n_informative=2, random_state=42)
    y = np.where(y == 0, -1, 1)  # Convert labels to {-1, 1}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    learners = [DecisionTreeClassifier(max_depth=1) for _ in range(n_learners)]
    model = AdaBoostLikeClassifier(learners)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"AdaBoostLikeClassifier(random data) acc: {accuracy_score(y_test, y_pred)}")

    # Iris
    full_features_set, full_labels_set = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        full_features_set, full_labels_set, test_size=0.33, random_state=42
    )

    learners = [DecisionTreeClassifier(max_depth=1) for _ in range(n_learners)]
    model = AdaBoostLikeClassifier(learners)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"AdaBoostLikeClassifier(iris data) acc: {accuracy_score(y_test, y_pred)}")

    boosting = SimpleBoostClassifier(
        KNeighborsClassifier(n_neighbors=3),
        DecisionTreeClassifier(criterion="entropy", max_depth=2, random_state=33),
        DecisionTreeClassifier(criterion="gini", random_state=71),
    )

    boosting.fit(X_train, y_train)
    prediction = boosting.predict(X_test)
    print(f"SimpleBootClassifier(iris data) acc: {accuracy_score(y_test, y_pred)}")
