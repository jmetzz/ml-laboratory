import random
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(init=True, frozen=True)
class Node:
    feature: int
    value: float
    left: Any = None
    right: Any = None


class RandomTree:
    def __init__(self, leaf_size: int = 10):
        self.leaf_size = leaf_size
        self._tree = None

    def fit(self, instances, true_labels):
        self._tree = self._build(instances, true_labels)
        return self

    def predict(self, instances):
        return np.array([self._predict(e, self._tree) for e in instances])

    def _build(self, instances: np.ndarray, true_labels: np.ndarray) -> Node:
        if instances.shape[0] == 0:
            return Node(None, 0.0)
        if instances.shape[0] <= self.leaf_size or len(np.unique(true_labels)) == 1:
            return Node(None, np.mean(true_labels).item())

        split_feature = random.randint(0, instances.shape[1] - 1)
        split_value = random.choice(instances[:, split_feature])

        left_index = instances[:, split_feature] <= split_value
        right_index = instances[:, split_feature] > split_value

        return Node(
            feature=split_feature,
            value=split_feature,
            left=self._build(instances[left_index], true_labels[left_index]),
            right=self._build(instances[right_index], true_labels[right_index]),
        )

    def _predict(self, instance: np, node: Node):
        if node.feature is None:
            return node.value
        if instance[node.feature] <= node.value:
            return self._predict(instance, node.left)
        return self._predict(instance, node.right)
