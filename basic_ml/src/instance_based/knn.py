import operator

import numpy as np
from common import distance


class Knn:
    def __init__(self, k=3, distance=distance.euclidean):
        self._model = dict()
        self.k = k
        self.distance = distance

    def fit(self, instances, true_labels):
        self._model["instances"] = instances
        self._model["dimension"] = instances.shape[1]
        self._model["labels"] = true_labels

    def predict(self, instances):
        return [self._predict(e) for e in instances]

    def _predict(self, instance):
        neighbors = self._find_neighbors(self._model["instances"], instance, self.k)
        return self._vote(neighbors, self._model["labels"])

    def _find_neighbors(self, training_set, x, k):
        dim = self._model["dimension"]
        distances = [(idx, self.distance(x, e, dim)) for idx, e in enumerate(training_set)]
        distances.sort(key=operator.itemgetter(1))
        neighbors = distances[:k] if len(distances) > k else distances
        return [idx for idx, _ in neighbors]

    @staticmethod
    def _vote(neighbors, true_labels):
        labels, votes = np.unique(true_labels[neighbors], return_counts=True)
        return labels[np.argmax(votes)]


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    from common.evaluation import accuracy

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=31)

    knn = Knn(k=3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    acc = accuracy(y_test, predictions)
    print("Accuracy is {0:.2f}".format(acc))
