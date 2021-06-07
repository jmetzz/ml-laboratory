import random

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

import numpy as np


class BoostClassifier:
    """The original boosting algorithm by Schapire (1990).
    This algorithm combines three weak learners to generate a strong learner.
    A weak learner has error probability less than 1/2, which makes it better
    than random guessing on a two-class problem, and a strong learner has
    arbitrarily small error probability.

    This simple implementation uses 3 base-learners given as arguments to its
    constructor. One can use the same inducer algorithm with the same hyperparameters,
    or completely different inducers.

    References:
        Schapire, R. E. 1990. "The Strength of Weak Learnability." Machine Learning 5: 197–227.
    """

    def __init__(self, learner1, learner2, learner3):
        self._model = dict()
        self._classifiers_trained = 0
        self.classifiers = [learner1, learner2, learner3]
        for l in self.classifiers:
            assert l is not None

    def fit(self, instances, true_labels):
        (X1, y1), (X2, y2), (X3, y3) = self._random_split(instances, true_labels)

        # Train the 1st classifier
        self.classifiers[0].fit(X1, y1)
        self._classifiers_trained = 1

        # Train the 2nd classifier
        prediction_h1 = self.classifiers[0].predict(X1)
        prediction_h2 = self.classifiers[0].predict(X2)
        X2, y2 = self.prepare_input_h2(X1, y1, X2, y2, prediction_h1, prediction_h2)
        self.classifiers[1].fit(X2, y2)
        self._classifiers_trained = 2

        # Train the 3rd classifier
        prediction_h1 = self.classifiers[0].predict(X3)
        prediction_h2 = self.classifiers[1].predict(X3)
        X3, y3 = self.prepare_input_h3(X3, y3, prediction_h1, prediction_h2)
        if X3.size != 0:
            self.classifiers[2].fit(X3, y3)
            self._classifiers_trained = 3

    def predict(self, instances):
        prediction_h1 = self.classifiers[0].predict(instances)
        prediction_h2 = self.classifiers[1].predict(instances)
        disagree = prediction_h1 != prediction_h2

        to_classify = [(i, instances[i]) for i, _ in enumerate(disagree)]
        for idx, instance in to_classify:
            if self._classifiers_trained == 3:
                prediction_h1[idx] = self.classifiers[2].predict(instance.reshape(1, -1))
            else:
                prediction_h1[idx] = random.choice((prediction_h2[idx], prediction_h2[idx]))
        return prediction_h1

    @staticmethod
    def prepare_input_h2(X1, y1, X2, y2, prediction_h1, prediction_h2):
        wrongly_classified = prediction_h1 != y1
        correctly_classified = prediction_h2 == y2
        if np.any(wrongly_classified) and np.any(correctly_classified):
            X = np.vstack((X1[wrongly_classified], X2[correctly_classified]))
            y = np.vstack((y1[wrongly_classified].reshape(-1, 1), y2[correctly_classified].reshape(-1, 1)))
            return X, y
        elif np.any(wrongly_classified):
            return X1[wrongly_classified], y1[wrongly_classified]
        elif np.any(correctly_classified):
            return X2[correctly_classified], y2[correctly_classified]

    @staticmethod
    def prepare_input_h3(X3, y3, prediction_h1, prediction_h2):
        disagree = prediction_h1 != prediction_h2
        return X3[disagree], y3[disagree]

    @staticmethod
    def _random_split(instances, true_labels):
        X, y = shuffle(
            instances,
            true_labels,
        )
        split = len(instances) // 3
        return (X[:split], y[:split]), (X[split : split * 2], y[split : split * 2]), (X[split * 2 :], y[split * 2 :])


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    boosting = BoostClassifier(
        KNeighborsClassifier(n_neighbors=3),
        DecisionTreeClassifier(criterion="entropy", max_depth=2, random_state=33),
        DecisionTreeClassifier(criterion="gini", random_state=71),
    )

    boosting.fit(X_train, y_train)
    prediction = boosting.predict(X_test)

    acc = sum(int(x == y) for (x, y) in zip(prediction, y_test)) / len(y_test)
    print(f"Acc = {acc:.2f}")
