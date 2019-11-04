import random

from sklearn.tree import DecisionTreeClassifier
import numpy as np

from ensemble.ensemble_utils import Sampling


class AdaBoostM1Classifier:
    """The original AdaBoost algorithm
        This algorithm was proposed by Freund and Schapire (1996). There are many variations
        of it. However, this implementation refers to the original AdaBoot.M1 algorithm,
        which targets binary classification problems, where the class labels are either +1 or -1.

        Moreover, it uses decision trees as the weak base-learner algorithm.
        By default, decision trees with only one decision node, also known
        as Decision Stump (see https://en.wikipedia.org/wiki/Decision_stump).
    """

    def __init__(self, rounds, criterion='gini', max_depth=1, seed=31):
        self.model = dict()
        self.model['rounds'] = rounds
        self.classifiers = [DecisionTreeClassifier(criterion=criterion,
                                                   max_depth=max_depth,
                                                   random_state=seed
                                                   )
                            for _ in range(rounds)]

    def fit(self, instances, true_labels):
        """Build a binary classification ensemble model based on AdaBoost.M1 algorithm
        """
        label_values = np.unique(true_labels)
        assert (label_values.size == 2)
        self.model["label_map"] = {label_values[0]: -1, label_values[1]: 1}
        self.model["label_map_inverse"] = {-1: label_values[0], 1: label_values[1]}

        transformed_labels = [self.model["label_map"][k] for k in true_labels]

        n = len(instances)
        self._initialize_weights(n)
        # alphas_r represents the importance of the
        # classifier built at round r
        self.model["alphas"] = np.zeros((self.model['rounds'], 1))
        self.model["errors"] = np.zeros((self.model['rounds'], 1))
        for r in range(self.model['rounds']):
            while True:
                X, y = Sampling.subsample(instances, transformed_labels, self.model["weights"])
                self.classifiers[r].fit(X, y)
                predictions = self.classifiers[r].predict(instances)
                # store the current prediction in a cache to be used for weights update
                self.model["prediction_cache"] = list(zip(transformed_labels, predictions))
                error = self.calculate_error(self.model["prediction_cache"], self.model["weights"])
                # remember: it is binary classification, the weak classifier should be
                # better than random.
                if error > 0.5:
                    self._reset_weights()
                    # ignore the current classifier and try build another one
                    # using a new sample data
                    continue
                else:
                    self.model["errors"][r] = error
                    self.model["alphas"][r] = alpha = self._calculate_alpha(error)
                    self.model["weights"] = self._update_weights(alpha,
                                                                 self.model["prediction_cache"],
                                                                 self.model["weights"])
                    break

    def predict(self, instances):
        predictions = np.array([m.predict(instances) for m in self.classifiers])
        predictions = np.sign(np.sum(predictions * self.model["alphas"], axis=0))
        return [self.model["label_map_inverse"][y_hat] for y_hat in predictions ]

    def _initialize_weights(self, n):
        self.model["weights"] = np.full((n, 1), 1 / n)

    def _reset_weights(self):
        self._initialize_weights(len(self.model["weights"]))

    @staticmethod
    def _update_weights(alpha, prediction_pairs, weights):
        signed_alphas = alpha * np.array([1 if int(y_hat != y) else -1
                                          for _, (y, y_hat) in enumerate(prediction_pairs)]).reshape((-1, 1))
        new_weights = weights * np.exp(signed_alphas)
        normalization_factor = np.sum(new_weights)
        return new_weights / normalization_factor

    @staticmethod
    def calculate_error(prediction_pairs, weights):
        error = sum(weights[i] * int(y != y_hat) for i, (y, y_hat) in enumerate(prediction_pairs))
        error.reshape((1, 1))
        return error / len(weights)

    @staticmethod
    def _calculate_alpha(error):
        return 1 / 2 * np.log((1 - error) / error)


if __name__ == "__main__":
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split

    # Debug Test with toy 1-dimension dataset
    from utils import datasets

    X_train = datasets.toy_labelled_1d[:, :-1]
    y_train = datasets.toy_labelled_1d[:, -1]
    boosting = AdaBoostM1Classifier(3)
    boosting.fit(X_train, y_train)
    prediction = boosting.predict(X_train)
    acc = sum(int(x == y) for (x, y) in zip(prediction, y_train)) / len(y_train)
    print(f"Acc = {acc:.2f}")

    X, y = make_moons(n_samples=160, noise=0.3, random_state=101)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
    boosting = AdaBoostM1Classifier(5)
    boosting.fit(X_train, y_train)
    prediction = boosting.predict(X_test)
    print(prediction)
    acc = sum(int(x == y) for (x, y) in zip(prediction, y_test)) / len(y_test)
    print(f"Acc = {acc:.2f}")
