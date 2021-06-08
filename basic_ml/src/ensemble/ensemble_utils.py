import random


class Sampling:  # pylint: disable=too-few-public-methods
    @staticmethod
    def subsample(instances, labels, weights=None, ratio=1.0):
        if 1.0 < ratio <= 0.0:
            raise ValueError("Parameter ration must in bin the interval (0, 1.0]")

        num_labels = len(labels)
        num_examples = round(num_labels * ratio)
        indexes = random.choices(range(num_labels), k=num_examples, weights=weights)
        feature_list = list()
        labels_list = list()
        for idx in indexes:
            feature_list.append(instances[idx])
            labels_list.append(labels[idx])
        return feature_list, labels_list
