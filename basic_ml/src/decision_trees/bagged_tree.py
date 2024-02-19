import numpy as np
from random_tree import RandomTree
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


class BaggedTrees:
    def __init__(self, n_estimators: int = 100, leaf_size: int = 10, tree_class=RandomTree) -> None:
        self.n_trees = n_estimators
        self.leaf_size = leaf_size
        self.tree_class = tree_class
        self._trees = []

    def fit(self, instances: np.ndarray, true_labels: np.ndarray):
        for _ in range(self.n_trees):
            training_features_boot, labels_boot = resample(instances, true_labels)
            self._trees.append(self.tree_class(self.leaf_size).fit(training_features_boot, labels_boot))
        return self

    def predict(self, instances):
        return np.array([tree.predict(instances) for tree in self._trees]).mean(0)


if __name__ == "__main__":
    train_features, test_features, train_labels, test_labels = train_test_split(
        *load_breast_cancer(return_X_y=True, as_frame=False), test_size=0.33, random_state=42
    )

    model = BaggedTrees(n_estimators=10)
    model.fit(train_features, train_labels)
    y_pred = model.predict(test_features)

    print(f"Accuracy: {roc_auc_score(test_labels, y_pred)}")
