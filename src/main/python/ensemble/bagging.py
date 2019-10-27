"""Bagging
Bagging is a technique that repeatedly samples (with replacement) from a data set
according to a uniform distribution. Each bootstrap sample has the same size as
the original data. Some instances may appear several times in the same training set,
while others may be omitted from the training set.

After training k classifiers, a test instance is assigned to the class that receives
the highest number of votes.

The basic algorithm is:

    Let k be the number of bootstrap samples
    for i = 1 to k do
        Create a bootstrap sample D_i of size N
        Train a base classifier C_i on the bootstrap sample D_i
    end for

    C*(x) = argmax y [\sum_i delta(C_i(x) = y)]
        where delta(.) = 1 if its argument is true and 0 otherwise

Bagging improves generalization error by reducing the variance of the base classifiers.
The performance of bagging depends on the stability of the base classifier. If a base
classifier is unstable, bagging helps to reduce the errors associated with random fluctuations
in the training data. If a base classifier is stable, i.e., robust to minor perturbations in
the training set, then the error of the ensemble is primarily caused by bias in the base
classifier. In this situation, bagging may not be able to improve the performance of the
base classifiers significantly. It may even degrade the classifier's performance because
the effective size of each training set is about 37% smaller than the original data.

On the other hand, since bagging does not focus on any particular instance of the training
data, if is less susceptible to model overfitting when applied to noisy data.

In this example I use the DecisionTreeClassifier from sklearn as the base classifier.
However, the code is pretty straightforward to adapt in order to enable other classifiers
to be used as base.
"""

import random

from sklearn.tree import DecisionTreeClassifier
import numpy as np


class BaggingClassifier:

    def __init__(self, size):
        self.size = size
        self.models = dict()

    def fit(self, instances, labels):
        for i in range(self.size):
            X, y = self.subsample(instances, labels, 1.0)
            self.models[i] = DecisionTreeClassifier()
            self.models[i].fit(X, y)

    def predict(self, instances):
        predictions = np.array([m.predict(instances) for _, m in self.models.items()])
        n = len(instances)
        y_hat = np.zeros((n,), dtype=int)
        for i in range(n):
            labels, votes = np.unique(predictions[:, i], return_counts=True)
            y_hat[i] = labels[np.argmax(votes)]
        return y_hat

    @classmethod
    def subsample(cls, instances, labels, ratio=1.0):
        assert (0 < ratio <= 1.0)
        n = len(labels)
        num_examples = round(n * ratio)
        indexes = random.choices(range(n), k=num_examples)
        X = list()
        y = list()
        for i in indexes:
            X.append(instances[i])
            y.append(labels[i])
        return X, y


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
