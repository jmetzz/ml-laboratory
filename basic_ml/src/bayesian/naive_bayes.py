from collections import Counter

import numpy as np


class NB:
    """Naive-Bayes algorithm implementation

    This algorithm follows the statistical (Bayesian) approach.
    The basic_ml assumption is that the probability of a class C is given by
    the posteriory probability P(C | x^i), where x^i refers to an entry in the test set.

    After some simplification the Bayes rule can be applied as:

    C = argmax P(C) . Product(P(x_i | C))

    where P(x_i | C)  is the conditional probability that feature i belongs to class C.

    This probability can simply be calculated by the relative values of feature i per class.

    Naive Bayes Classifier method:
        It is trained with a 2D-array X (dimensions m,n) and a 1D array Y (dimension 1,n).
        X should have one column per feature (total n) and one row per training example (total m).
        After training a hash table is filled with the class probabilities per feature.
        We start with an empty hash table nb_dict, which has the form:

        nb_dict = {
            'class1': {
                'feature1': [],
                'feature2': [],
                (...)
                'featureN': []
            }
            'class2': {
                'feature1': [],
                'feature2': [],
                (...)
                'featureN': []
            }
        }
    """

    def __init__(self):
        self.model = dict()

    def fit(self, instances, true_labels):
        # the model is implemented as a dictionary of classes and feature relative frequencies
        self.model = self._create_internal_model(instances, true_labels)
        internal_model = self.model["nb_dict"]
        labels = self.model["labels"]
        dimension = self.model["dimension"]

        # first step is to isolate all instances of the same class
        # and populate the model with all occurrences of each feature value
        # per class
        for label in labels:
            subset_l = instances[true_labels == label]
            # print(subset_l)

            # capture all occurrences of feature values in this subset
            # keep them per class label and feature
            for feature in range(dimension):
                internal_model[label][feature] = subset_l[:, feature]

        # second step is to calculate the actual probability for each feature value
        # based on the relative occurrence per class in the class
        for label in labels:
            for feature in range(dimension):
                internal_model[label][feature] = self._relative_occurrence(internal_model[label][feature])

        # Save the class probabilities
        self.model["probabilities"] = self._relative_occurrence(true_labels)

    def predict(self, instances):
        """Predict the class label for the given list of instances.

        Args:
            instances: the list of instances to classify
        Return:
            the list of predictions
        """
        return [self._predict(e) for e in instances]

    def _predict(self, instance):
        """Predict the class label for the given instance.

        Args:
            instance: the unlabelled instance to be classified
        Return:
            the class label
        """

        # First we determine the class-probability of each class,
        # and then predict the class with the highest probability
        nb_dict = self.model["nb_dict"]
        labels = self.model["labels"]
        scores = [-1] * len(labels)
        # label_map = dict()
        for index, label in enumerate(labels):
            # label_map[index] = label
            c_prob = self.model["probabilities"][label]
            for feature in range(self.model["dimension"]):
                relative_feature_values = nb_dict[label][feature]
                if instance[feature] in relative_feature_values.keys():
                    c_prob *= relative_feature_values[instance[feature]]
                else:
                    c_prob *= 0
            scores[index] = c_prob
        return labels[np.argmax(scores)]

    @staticmethod
    def _relative_occurrence(values):
        """Counts the relative occurrence of each given value in the list
        regarding the total number of elements

        Args:
            values: a list of elements

        Returns: a dictionary with the input elements as key and
            their relative frequency as values
        """
        # A Counter is a dict subclass for counting hashable objects.
        # It is an unordered collection where elements
        # are stored as dictionary keys and their counts are stored as dictionary values.
        # Counts are allowed to be any integer value including zero or negative counts.
        # The Counter class is similar to bags or multisets in other languages.
        counter_dict = dict(Counter(values))
        size = len(values)
        for key in counter_dict.keys():
            counter_dict[key] = counter_dict[key] / float(size)
        return counter_dict

    @staticmethod
    def _create_internal_model(instances, true_labels):
        model = dict()
        model["dimension"] = instances.shape[1]
        model["labels"] = np.unique(true_labels)
        model["num_labels"] = len(model["labels"])
        model["nb_dict"] = {}
        for label in model["labels"]:
            feature_map = {f: [] for f in range(model["dimension"])}
            model["nb_dict"][label] = feature_map
        model["probabilities"] = None
        return model


if __name__ == "__main__":
    from common.evaluation import accuracy
    from utils import datasets

    X_train, y_train, X_test, y_test = datasets.play_tennis()
    nb_model = NB()
    nb_model.fit(X_train, y_train)

    prediction = nb_model.predict(X_test)
    acc = accuracy(y_test, prediction)
    print(f"Accuracy is {acc:.2f}")
