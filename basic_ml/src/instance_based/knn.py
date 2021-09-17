import operator

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from common import distance
from common.evaluation import accuracy


class Knn:
    def __init__(self, k=3, distance_func=distance.euclidean, is_classifier: bool = True):
        self._model = dict()
        self.k = k
        self.distance = distance_func
        self._is_classifier = is_classifier

    def fit(self, instances, true_output):
        self._model["instances"] = instances
        self._model["dimension"] = instances.shape[1]
        self._model["output"] = true_output

    def predict(self, instances):
        return [self._predict(e) for e in instances]

    def _predict(self, instance):
        neighbors = self._find_neighbors(self._model["instances"], instance, self.k)
        return self._vote(neighbors, self._model["output"])

    def _find_neighbors(self, training_set, instance, k):
        distances = [(idx, self.distance(instance, e)) for idx, e in enumerate(training_set)]
        distances.sort(key=operator.itemgetter(1))
        neighbors = distances[:k] if len(distances) > k else distances
        return [idx for idx, _ in neighbors]

    def _vote(self, neighbors, true_output):
        if self._is_classifier:
            output_values, votes = np.unique(true_output[neighbors], return_counts=True)
            return output_values[np.argmax(votes)]
        else:
            return np.average(true_output[neighbors])


if __name__ == "__main__":
    features, labels = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=31)

    knn = Knn(k=3)
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_test)
    acc = accuracy(y_test, predictions)
    print(f"Accuracy is {acc:.2f}")
