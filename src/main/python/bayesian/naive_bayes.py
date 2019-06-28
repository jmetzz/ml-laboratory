from collections import Counter
import pprint as pp

"""Naive-Bayes algorithm implementation

    This algorithm follows the statistical (Bayesian) approach.
    The basic assumption is that the probability of a class C is given by
    the posteriory probability P(C | x^i), where x^i refers to an entry in the test set.

    After some simplification the Bayes rule can be applyed as:

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
                'featuren': []
            }
            'class2': {
                'feature1': [],
                'feature2': [],
                (...)
                'featuren': []
            }
        }
"""


def get_class_values(dataset):
    return list(set(row[-1] for row in dataset))


def relative_occurrence(values):
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


def train(data):
    labels = get_class_values(data)
    no_features = len(data[0]) - 1
    label_idx = no_features

    # the model is implemented as a dictionary of classes and feature relative frequencies
    nb_model = create_model(labels, no_features)

    # first step is to isolate all instances of the same class
    # and populate the model with all occurrences of each feature value
    # per class
    for l in labels:
        subset_l = [e for e in data if e[label_idx] == l]
        # print(subset_l)

        # capture all occurrences of feature values in this subset
        # keep them per class label and feature
        for feature in range(no_features):
            nb_model[l][feature] = column_values(subset_l, feature)

    # second step is to calculate the actual probability for each feature value
    # based on the relative occurrence per class in the class
    for l in labels:
        for feature in range(no_features):
            nb_model[l][feature] = relative_occurrence(nb_model[l][feature])

    classes_probabilities = relative_occurrence([row[-1] for row in data])
    return classes_probabilities, nb_model


def test(model, classes_probabilities, data):
    return [classify(model, classes_probabilities, e) for e in data]


def get_class_with_highest_probability(prediction):
    values = list(prediction.values())
    max_value_index = values.index(max(values))
    return list(prediction.keys())[max_value_index]


def classify(model, classes_probabilities, instance):
    """
    :return: C = argmax P(C) . Product(P(x_i | C))
    """
    prediction = {}
    # First we determine the class-probability of each class, and then we determine the class with the highest probability
    for l in model.keys():
        c_prob = classes_probabilities[l]
        for f in range(len(instance)):
            relative_feature_values = model[l][f]
            if instance[f] in relative_feature_values.keys():
                c_prob *= relative_feature_values[instance[f]]
            else:
                c_prob *= 0
        prediction[l] = c_prob
    return get_class_with_highest_probability(prediction)


def column_values(data, col):
    return [row[col] for row in data]


def create_model(labels, no_features):
    nb_dict = {}
    for label in labels:
        feature_map = {f: [] for f in range(no_features)}
        nb_dict[label] = feature_map
    return nb_dict


def remove_labels(input_data):
    return [d[:-1] for d in input_data]


def accuracy(test_set, predictions):
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(test_set))) * 100.0


def main():
    train_data = [['Sunny', 'Hot', 'High', 'Weak', 'No'],
                  ['Sunny', 'Hot', 'High', 'Strong', 'No'],
                  ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
                  ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
                  ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
                  ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
                  ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
                  ['Sunny', 'Mild', 'High', 'Weak', 'No'],
                  ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes']]

    test_data = [['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
                 ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
                 ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
                 ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
                 ['Rain', 'Mild', 'High', 'Strong', 'No']]

    classes_probabilities, model = train(train_data)
    # pp.pprint(model) # debug print
    # pp.pprint(classes_probabilities)

    prediction = test(model, classes_probabilities, remove_labels(test_data))
    print(" ------- Test data: -------")
    pp.pprint(test_data)
    print(" ------- Prediction result: -------")
    pp.pprint(prediction)  # debug print

    acc = accuracy(test_data, prediction)
    print('Accuracy is {0:.2f}'.format(acc))


if __name__ == '__main__':
    main()
