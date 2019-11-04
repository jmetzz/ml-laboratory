import json

import numpy as np

toy_labelled_1d = np.array([[0.1, 1],
                            [0.2, 1],
                            [0.3, 1],
                            [0.4, -1],
                            [0.5, -1],
                            [0.6, -1],
                            [0.7, -1],
                            [0.8, 1],
                            [0.9, 1],
                            [1.0, 1]])

toy_labelled_2d = np.array([[2.771244718, 1.784783929, 0],
                            [1.728571309, 1.169761413, 0],
                            [3.678319846, 2.81281357, 0],
                            [3.961043357, 2.61995032, 0],
                            [2.999208922, 2.209014212, 0],
                            [7.497545867, 3.162953546, 1],
                            [9.00220326, 3.339047188, 1],
                            [7.444542326, 0.476683375, 1],
                            [10.12493903, 3.234550982, 1],
                            [6.642287351, 3.319983761, 1]])

toy_unlabelled_2d = np.array([[2.771244718, 1.784783929],
                              [1.728571309, 1.169761413],
                              [3.678319846, 2.81281357],
                              [3.961043357, 2.61995032],
                              [2.999208922, 2.209014212],
                              [7.497545867, 3.162953546],
                              [9.00220326, 3.339047188],
                              [7.444542326, 0.476683375],
                              [10.12493903, 3.234550982],
                              [6.642287351, 3.319983761]])


def play_tennis():
    X_train = np.array([['Sunny', 'Hot', 'High', 'Weak'],
                        ['Sunny', 'Hot', 'High', 'Strong'],
                        ['Overcast', 'Hot', 'High', 'Weak'],
                        ['Rain', 'Mild', 'High', 'Weak'],
                        ['Rain', 'Cool', 'Normal', 'Weak'],
                        ['Rain', 'Cool', 'Normal', 'Strong'],
                        ['Overcast', 'Cool', 'Normal', 'Strong'],
                        ['Sunny', 'Mild', 'High', 'Weak'],
                        ['Sunny', 'Cool', 'Normal', 'Weak']])

    y_train = np.array(['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes'])

    X_test = np.array([['Rain', 'Mild', 'Normal', 'Weak'],
                       ['Sunny', 'Mild', 'Normal', 'Strong'],
                       ['Overcast', 'Mild', 'High', 'Strong'],
                       ['Overcast', 'Hot', 'Normal', 'Weak'],
                       ['Rain', 'Mild', 'High', 'Strong']])

    y_test = np.array(['Yes', 'Yes', 'Yes', 'Yes', 'No'])
    return X_train, y_train, X_test, y_test


def amazon():
    filename = 'reviews_Video_Games_5.json'
    train_summary = []
    train_review_text = []
    train_labels = []

    test_summary = []
    test_review_text = []
    test_labels = []

    with open(filename, 'r') as f:
        for (i, line) in enumerate(f):
            data = json.loads(line)

            if data['overall'] == 3:
                next
            elif data['overall'] == 4 or data['overall'] == 5:
                label = 1
            elif data['overall'] == 1 or data['overall'] == 2:
                label = 0
            else:
                raise Exception("Unexpected value " + str(data['overall']))

            summary = data['summary']
            review_text = data['reviewText']

            if i % 10 == 0:
                test_summary.append(summary)
                test_review_text.append(review_text)
                test_labels.append(label)
            else:
                train_summary.append(summary)
                train_review_text.append(review_text)
                train_labels.append(label)

    return (train_summary, train_review_text, train_labels), (test_summary, test_review_text, test_labels)


def load_amazon_smaller(size=20000):
    (train_summary, train_review_text, train_labels), (test_summary, test_review_text, test_labels) = amazon()
    return (train_summary[:size], train_review_text[:size], np.array(train_labels[:size])), (
        test_summary[:size], test_review_text[:size], np.array(test_labels[:size]))
