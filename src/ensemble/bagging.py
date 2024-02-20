import numpy as np
from common import sampling
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class BaggingClassifier:
    def __init__(self, size):
        self.size = size
        self.models = [DecisionTreeClassifier() for _ in range(size)]

    def fit(self, instances, true_labels):
        for idx in range(self.size):
            features, labels = sampling.subsample(instances, true_labels)
            self.models[idx].fit(features, labels)

    def predict(self, instances):
        ensemble_predictions = np.array([m.predict(instances) for m in self.models])
        num_instances = len(instances)
        predictions = np.zeros((num_instances,), dtype=int)
        for idx in range(num_instances):
            labels, votes = np.unique(ensemble_predictions[:, idx], return_counts=True)
            predictions[idx] = labels[np.argmax(votes)]
        return predictions


if __name__ == "__main__":
    full_features_set, full_labels_set = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        full_features_set, full_labels_set, test_size=0.33, random_state=42
    )
    bagging = BaggingClassifier(5)
    bagging.fit(x_train, y_train)
    prediction = bagging.predict(x_test)

    acc = sum(int(x == full_labels_set) for (x, full_labels_set) in zip(prediction, y_test)) / len(y_test)
    print(f"Acc = {acc:.2f}")
