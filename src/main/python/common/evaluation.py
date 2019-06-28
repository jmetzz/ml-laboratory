def accuracy(test_set, predictions):
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(test_set))) * 100.0


def precision(gold_truth, predictions):
    # precision = tp / tp_fp
    raise NotImplementedError


def recall(gold_truth, predictions):
    # recall = tp / tp_fn
    raise NotImplementedError


def f1(gold_truth, predictions):
    raise NotImplementedError


def confusion_matrix(gold_truth, predictions):
    raise NotImplementedError


def basic_evaluation(gold_truth, prediction):
    return precision(gold_truth, prediction), recall(gold_truth, prediction)
