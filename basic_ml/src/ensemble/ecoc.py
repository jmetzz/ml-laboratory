"""Error-Correcting Output Codes
TODO: describe and implement
"""
from sklearn.tree import DecisionTreeClassifier


class Ecoc:
    def __init__(self, size):
        self.size = size
        self.models = [DecisionTreeClassifier() for _ in range(size)]

    def fit(self, instances, true_labels):
        raise NotImplementedError
        # Base-learners are binary classifiers having output −1/+1
        # Thus, we need some pre-processing in each sub-sample of the
        # training data

        # K rows are the binary codes of classes in terms of the L base-learners

        # Each classifier h_i is trained to separate class i and i+1 and
        # does not use the training instances belonging to the other classes

    def predict(self, instances):
        # TODO
        raise NotImplementedError


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    bagging = Ecoc(5)
    bagging.fit(X_train, y_train)
    prediction = bagging.predict(X_test)

    acc = sum(int(x == y) for (x, y) in zip(prediction, y_test)) / len(y_test)
    print(f"Acc = {acc:.2f}")
