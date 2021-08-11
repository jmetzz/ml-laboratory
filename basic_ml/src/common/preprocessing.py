import numpy as np


class Normalizer:
    """
    Divide each feature by its 2-norm

    So that the transformed feature has norm 1.

    """

    def __init__(self):
        self._norms = None

    def fit(self, features: np.ndarray):
        self._norms = np.linalg.norm(features, axis=0)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self._norms is None:
            raise ValueError("Model not initialized. Call fit before.")
        return features / self._norms
