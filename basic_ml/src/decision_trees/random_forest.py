import numpy as np
from scipy.stats import mode
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class SimpleRandomForest:
    """
    A simple implementation of the Random Forest algorithm for classification.

    This class uses DecisionTreeClassifier from scikit-learn for simplicity.
    Each tree is trained on a bootstrap sample of the data,
    and a random subset of features is selected according to the max_features parameter.

    The fit method trains each tree on these samples and features.
    The trained trees are stored along with the indices of the features used for training.

    The predict method aggregates predictions from all trees using majority voting (mode)
    for classification.

    Parameters like max_depth, min_samples_split, and whether to bootstrap are configurable.

    Some of the many limitations are:

    This implementation lacks features like parallelization, which can significantly improve performance.
    It doesn't include advanced options for dealing with class imbalances,
    nor does it implement regression capabilities.
    It uses a fixed random state within the trees, making it deterministic
    in this simplified example. In practice, you might want to introduce more randomness.

    Methods:
        fit(X, y): Trains the Random Forest model on the input dataset.
        predict(X): Predicts class labels for the input samples.
    """

    def __init__(self, n_estimators=100, max_features="sqrt", max_depth=None, min_samples_split=2, bootstrap=True):
        """
        Initializes the SimpleRandomForest with the given parameters.

        Args:
            n_estimators (int): The number of trees in the forest.
            max_features (str): The number of features to consider when looking for the best split.
            max_depth (int, optional): The maximum depth of the trees.
            min_samples_split (int): The minimum number of samples required to split an internal node.
            bootstrap (bool): Whether bootstrap samples are used when building trees.
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self._trees = []
        self.feature_importances_ = None

    def fit(self, instances, true_labels):
        """
        Trains the Random Forest model on the input dataset.

        Args:
            instances (np.ndarray): The feature matrix where rows represent samples and columns represent features.
            true_labels (np.ndarray): The target vector, where each element is the target for a sample.
        """
        self._trees = []
        self.feature_importances_ = np.zeros(X.shape[1])
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)

            if self.bootstrap:
                indices = np.random.choice(range(len(instances)), size=len(instances))
                instance_sample, y_sample = instances[indices], true_labels[indices]
            else:
                instance_sample, y_sample = instances, true_labels

            if self.max_features == "sqrt":
                n_features = int(np.sqrt(instances.shape[1]))
            elif self.max_features == "log2":
                n_features = int(np.log2(instances.shape[1]))
            else:
                n_features = instances.shape[1]

            features_indices = np.random.choice(range(instances.shape[1]), size=n_features, replace=False)
            instance_sample = instance_sample[:, features_indices]

            tree.fit(instance_sample, y_sample)
            self._trees.append((tree, features_indices))

            # Aggregate feature importance
            for i, feature_index in enumerate(features_indices):
                self.feature_importances_[feature_index] += tree.feature_importances_[i]

        # Normalize feature importances
        self.feature_importances_ /= self.n_estimators

    def predict(self, instances) -> np.ndarray:
        """
        Predicts class labels for the input samples.

        Args:
            instances (np.ndarray): The feature matrix where rows represent samples.

        Returns:
            np.ndarray: The predicted class labels for each sample.
        """
        predictions = np.zeros((instances.shape[0], len(self._trees)))
        for i, (tree, features_indices) in enumerate(self._trees):
            predictions[:, i] = tree.predict(instances[:, features_indices])
        return mode(predictions, axis=1)[0].ravel()

    def get_feature_importances(self) -> np.ndarray:
        """
        Returns the feature importances calculated as the mean decrease in impurity.

        Returns:
            np.ndarray: Normalized feature importances.
        """
        if self.feature_importances_ is None:
            raise ValueError("Feature importances not available. Please ensure the model has been trained.")
        return self.feature_importances_


if __name__ == "__main__":
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = SimpleRandomForest(
        n_estimators=10, max_features="sqrt", max_depth=None, min_samples_split=2, bootstrap=True
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
