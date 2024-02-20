import json

import numpy as np
from utils.encoders import NumpyArrayEncoder


class CART:
    def __init__(self, max_depth, min_size):
        self.max_depth = max_depth
        self.min_size = min_size
        self._model = dict()

    def fit(self, instances, true_labels):
        self._model["dimensions"] = instances.shape[1]
        self._model["labels"] = tuple(np.unique(true_labels))
        self._model["root"] = root = self._select_split(instances, true_labels)
        self._split_node(root, 1)

    def predict(self, instances):
        return [self._predict(self._model["root"], e) for e in instances]

    @classmethod
    def _predict(cls, node, instance):
        if instance[node["index"]] < node["value"]:
            if isinstance(node["left"], dict):
                return cls._predict(node["left"], instance)
            return node["left"]

        if isinstance(node["right"], dict):
            return cls._predict(node["right"], instance)
        return node["right"]

    def _select_split(self, instances, true_labels):
        """Select the best split according to Gini Index

        Args: the dataset

        Returns: a tuple containing
            index: the index of the best attribute to split the dataset
            value: the slit value of the attribute
            groups: the instances separated into groups according to the
            split attribute and value
        """

        # define some variables to hold the best values found during computation
        best_index, best_value, best_score, best_groups = 999, 999, 999, None
        dimensions = self._model["dimensions"]

        # iterate over the attributes and instances in the group
        for feature in range(dimensions):
            for instance in instances:
                left_filter = instances[:, feature] < instance[feature]
                left_idx = np.where(left_filter)
                right_idx = np.where(np.invert(left_filter))

                left_features = instances[left_idx]
                left_labels = true_labels[left_idx].reshape((-1, 1))

                right_features = instances[right_idx]
                right_labels = true_labels[right_idx].reshape((-1, 1))

                gini = self.gini_index((left_labels, right_labels), len(left_features) + len(right_features))

                if gini < best_score:
                    best_index, best_value = feature, instance[feature]
                    best_score = gini
                    best_groups = ((left_features, left_labels), (right_features, right_labels))

        return {"index": best_index, "value": best_value, "groups": best_groups}

    def _split_node(self, node, current_depth):
        """Create child node splits for a given node

        Or makes a terminal node when it is needed.

        Args:
            node:
            current_depth:

        Returns:
        """
        (left_features, left_labels), (right_features, right_labels) = node["groups"]
        del node["groups"]

        # check for a no split
        if not left_features.any() or not right_features.any():
            node["left"] = node["right"] = self._to_terminal_node(np.vstack((left_labels, right_labels)))
            return

        # check for max depth
        if current_depth >= self.max_depth:
            node["left"], node["right"] = (
                self._to_terminal_node(left_labels),
                self._to_terminal_node(right_labels),
            )
            return

        # process left child
        if len(left_features) <= self.min_size:
            node["left"] = self._to_terminal_node(left_labels)
        else:
            # recursive step
            node["left"] = self._select_split(left_features, left_labels)
            self._split_node(node["left"], current_depth + 1)

        # process right child
        if len(right_features) <= self.min_size:
            node["right"] = self._to_terminal_node(right_labels)
        else:
            node["right"] = self._select_split(right_features, right_labels)
            self._split_node(node["right"], current_depth + 1)

    @staticmethod
    def gini_index(groups, n_instances):
        gini_value = 0.0
        for group in groups:
            size = len(group)
            if size == 0:
                continue

            # score the group based on the proportion of each class
            score = 0.0
            labels, counts = np.unique(group, return_counts=True)
            for _, count in zip(labels, counts):
                proportion = count / size
                score += proportion * proportion
            # weight the group score by its relative size
            gini_value += (1.0 - score) * (size / n_instances)

        return gini_value

    @staticmethod
    def _to_terminal_node(labels):
        """Creates a terminal node value

        The created terminal node consists of all instances present in the given group

        Returns:
            the label with the majority of elements in the group
        """
        labels, counts = np.unique(labels, return_counts=True)
        return labels[np.argmax(counts)]

    def summary(self) -> str:
        return json.dumps(self._model, cls=NumpyArrayEncoder)
