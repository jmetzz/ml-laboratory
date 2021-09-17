import numpy as np


class KFold:
    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        raise NotImplemented(":(")

    def get_n_splits(self, data):
        raise NotImplemented(":(")

    def split(self, data) -> tuple[np.ndarray, np.ndarray]:
        """

        Args:
            data:

        Yields:
            train : numpy.ndarray
                The training set indices for that split.
            test : numpy.ndarray
                The testing set indices for that split.

        """
        raise NotImplemented(":(")
