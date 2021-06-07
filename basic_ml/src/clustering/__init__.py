from abc import ABCMeta, abstractmethod


class BaseClustering:
    __metaclass__ = ABCMeta

    def __init__(self, data):
        self.data = data
        self.size = len(data)
        self.num_features = len(self.data[0])
        self.dimension = len(data[0])
        self.clusters = {}

    @abstractmethod
    def model(self):
        raise NotImplementedError(
            "BaseClustering does not implement ``build`` method." "A specialization class should implement it."
        )

    @abstractmethod
    def build(self):
        raise NotImplementedError(
            "BaseClustering does not implement ``build`` method." "A specialization class should implement it."
        )

    @classmethod
    def calculate_centroids(cls, data, k, clusters):
        centroids = [0.0] * k
        for c in range(k):
            points = [data[j] for j in range(len(data)) if clusters[j] == c]
            centroids[c] = BaseClustering.mean(points)
        return centroids

    @classmethod
    def mean(cls, points):
        # TODO - Improvements on readability and performance:
        # by using vectorization and matrix multiplication formula, we have:
        # [\mathbf{1}^T\mathbf{M}]_j= \sum_i \mathbf{1}_i \mathbf{M}_{i,j} =\sum_i \mathbf{M}_{i,j}.
        # Hence, the column-wise sum of \mathbf{M} is \mathbf{1}^T\mathbf{M}.
        ncols = len(points[0])
        m = [0] * ncols
        for col in range(ncols):
            for p in points:
                m[col] += p[col]
            m[col] = m[col] / len(points)
        return m
