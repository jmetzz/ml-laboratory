import random

from sklearn.tree import DecisionTreeClassifier
import numpy as np

from ensemble.ensemble_utils import Sampling


class BaggingClassifier:

    def __init__(self, size):
        self.size = size
        self.models = [DecisionTreeClassifier() for _ in range(size)]

    def fit(self, instances, true_labels):
        for i in range(self.size):
            X, y = Sampling.subsample(instances, true_labels, 1.0)
            self.models[i].fit(X, y)

    def predict(self, instances):
        predictions = np.array([m.predict(instances) for m in self.models])
        n = len(instances)
        y_hat = np.zeros((n,), dtype=int)
        for i in range(n):
            labels, votes = np.unique(predictions[:, i], return_counts=True)
            y_hat[i] = labels[np.argmax(votes)]
        return y_hat


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    bagging = BaggingClassifier(5)
    bagging.fit(X_train, y_train)
    prediction = bagging.predict(X_test)

    acc = sum(int(x == y) for (x, y) in zip(prediction, y_test)) / len(y_test)
    print(f"Acc = {acc:.2f}")
