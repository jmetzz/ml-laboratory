import numpy as np


class CART:

    def __init__(self, max_depth, min_size):
        self.max_depth = max_depth
        self.min_size = min_size
        self._model = dict()

    def fit(self, instances, true_labels):
        self._model['dimensions'] = instances.shape[1]
        self._model['labels'] = np.unique(true_labels)
        self._model['root'] = root = self._select_split(instances, true_labels)
        self._split_node(root, 1)

    def predict(self, instances):
        return [self._predict(self._model['root'], e) for e in instances]

    @classmethod
    def _predict(cls, node, instance):
        if instance[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return cls._predict(node['left'], instance)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return cls._predict(node['right'], instance)
            else:
                return node['right']

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
        dimensions = self._model['dimensions']

        # iterate over the attributes and instances in the group
        for feature in range(dimensions):
            for instance in instances:
                left_filter = instances[:, feature] < instance[feature]
                left_idx = np.where(left_filter)
                right_idx = np.where(np.invert(left_filter))

                X_left = instances[left_idx]
                y_left = true_labels[left_idx].reshape((-1, 1))

                X_right = instances[right_idx]
                y_right = true_labels[right_idx].reshape((-1, 1))

                gini = self.gini_index((y_left, y_right), len(X_left) + len(X_right))

                if gini < best_score:
                    best_index, best_value = feature, instance[feature]
                    best_score = gini
                    best_groups = ((X_left, y_left), (X_right, y_right))

        return {'index': best_index, 'value': best_value, 'groups': best_groups}

    def _split_node(self, node, current_depth):
        """Create child node splits for a given node

            Or makes a terminal node when it is needed.

            Args:
                node:
                max_depth:
                min_size:
                current_depth:

            Returns:
        """
        (X_left, y_left), (X_right, y_right) = node['groups']
        del (node['groups'])

        # check for a no split
        if not X_left.any() or not X_right.any():
            node['left'] = node['right'] = self._to_terminal_node(np.vstack((y_left, y_right)))
            return

        # check for max depth
        if current_depth >= self.max_depth:
            node['left'], node['right'] = self._to_terminal_node(y_left), \
                                          self._to_terminal_node(y_right)
            return

        # process left child
        if len(X_left) <= self.min_size:
            node['left'] = self._to_terminal_node(y_left)
        else:
            # recursive step
            node['left'] = self._select_split(X_left, y_left)
            self._split_node(node['left'], current_depth + 1)

        # process right child
        if len(X_right) <= self.min_size:
            node['right'] = self._to_terminal_node(y_right)
        else:
            node['right'] = self._select_split(X_right, y_right)
            self._split_node(node['right'], current_depth + 1)

    @staticmethod
    def gini_index(groups, n_instances):
        gini_value = 0.0
        for g in groups:
            size = len(g)
            if size == 0:
                continue

            # score the group based on the proportion of each class
            score = 0.0
            labels, counts = np.unique(g, return_counts=True)
            for label, count in zip(labels, counts):
                proportion = count / size
                score += proportion * proportion
            # weight the group score by its relative size
            gini_value += (1.0 - score) * (size / n_instances)

        return gini_value

    @staticmethod
    def _to_terminal_node(labels):
        """Creates a terminal node value

            The created terminal node comprises of all instances present in the given group

            Returns:
                the label with the majority of elements in the group
        """
        labels, counts = np.unique(labels, return_counts=True)
        return labels[np.argmax(counts)]

    def summary(self):
        print(self._model)


if __name__ == '__main__':
    import pprint as pp
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from common.evaluation import accuracy

    mock_stump = {'index': 0, 'right': 1, 'value': 6.642287351, 'left': 0}
    print("Mock tree:")
    print(mock_stump)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=31)

    model = CART(max_depth=4, min_size=1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy(y_test, predictions)
    pp.pprint(model._model)
    print('Accuracy is {0:.2f}'.format(acc))

    model_stump = CART(max_depth=1, min_size=1)
    model_stump.fit(X_train, y_train)
    predictions = model_stump.predict(X_test)
    acc = accuracy(y_test, predictions)
    pp.pprint(model_stump._model)
    print('Accuracy is {0:.2f}'.format(acc))
