import random

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle


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
        for classifier in self.classifiers:
            if classifier is None:
                raise ValueError("Classifier is None")

    def fit(self, instances, true_labels):
        (features_1, labels_1), (features_2, labels_2), (features_3, labels_3) = self._random_split(
            instances, true_labels
        )

        # Train the 1st classifier
        self.classifiers[0].fit(features_1, labels_1)
        self._classifiers_trained = 1

        # Train the 2nd classifier
        prediction_h1 = self.classifiers[0].predict(features_1)
        prediction_h2 = self.classifiers[0].predict(features_2)
        features_2, labels_2 = self.prepare_input_h2(
            features_1, labels_1, features_2, labels_2, prediction_h1, prediction_h2
        )
        self.classifiers[1].fit(features_2, labels_2)
        self._classifiers_trained = 2

        # Train the 3rd classifier
        prediction_h1 = self.classifiers[0].predict(features_3)
        prediction_h2 = self.classifiers[1].predict(features_3)
        features_3, labels_3 = self.prepare_input_h3(features_3, labels_3, prediction_h1, prediction_h2)
        if features_3.size != 0:
            self.classifiers[2].fit(features_3, labels_3)
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
    def prepare_input_h2(features_1, labels_1, features_2, labels_2, prediction_h1, prediction_h2):
        wrongly_classified = prediction_h1 != labels_1
        correctly_classified = prediction_h2 == labels_2
        if np.any(wrongly_classified) and np.any(correctly_classified):
            features = np.vstack((features_1[wrongly_classified], features_2[correctly_classified]))
            labels = np.vstack(
                (
                    labels_1[wrongly_classified].reshape(-1, 1),
                    labels_2[correctly_classified].reshape(-1, 1),
                )
            )
            return features, labels
        if np.any(wrongly_classified):
            return features_1[wrongly_classified], labels_1[wrongly_classified]

        return features_2[correctly_classified], labels_2[correctly_classified]

    @staticmethod
    def prepare_input_h3(features, labels, prediction_h1, prediction_h2):
        disagree = prediction_h1 != prediction_h2
        return features[disagree], labels[disagree]

    @staticmethod
    def _random_split(instances, true_labels):
        features, labels = shuffle(
            instances,
            true_labels,
        )
        split = len(instances) // 3
        return (
            (features[:split], labels[:split]),
            (features[split : split * 2], labels[split : split * 2]),
            (features[split * 2 :], labels[split * 2 :]),
        )


if __name__ == "__main__":
    full_features_set, full_labels_set = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        full_features_set, full_labels_set, test_size=0.33, random_state=42
    )

    boosting = BoostClassifier(
        KNeighborsClassifier(n_neighbors=3),
        DecisionTreeClassifier(criterion="entropy", max_depth=2, random_state=33),
        DecisionTreeClassifier(criterion="gini", random_state=71),
    )

    boosting.fit(x_train, y_train)
    prediction = boosting.predict(x_test)

    acc = sum(int(x == y) for (x, y) in zip(prediction, y_test)) / len(y_test)
    print(f"Acc = {acc:.2f}")
