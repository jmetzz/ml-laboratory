import random


class Sampling:
    @staticmethod
    def subsample(instances, labels, weights=None, ratio=1.0):
        assert 0 < ratio <= 1.0
        n = len(labels)
        num_examples = round(n * ratio)
        indexes = random.choices(range(n), k=num_examples, weights=weights)
        X = list()
        y = list()
        for i in indexes:
            X.append(instances[i])
            y.append(labels[i])
        return X, y
