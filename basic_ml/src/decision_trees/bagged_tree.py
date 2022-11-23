import numpy as np
from random_tree import RandomTree
from sklearn.utils import resample


class BaggedTrees:
    def __init__(self, n_trees: int = 100, leaf_size: int = 10, tree_class=RandomTree) -> None:
        self.n_trees = n_trees
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
