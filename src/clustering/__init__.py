from abc import ABCMeta, abstractmethod


class BaseClustering(metaclass=ABCMeta):
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
    def calculate_centroids(cls, data, num_centroids, clusters):
        centroids = [0.0] * num_centroids
        for centroid in range(num_centroids):
            points = [data[j] for j in range(len(data)) if clusters[j] == centroid]
            centroids[centroid] = BaseClustering.mean(points)
        return centroids

    @classmethod
    def mean(cls, points):
        # TODO - Improvements on readability and performance:
        # by using vectorization and matrix multiplication formula, we have:
        # [\mathbf{1}^T\mathbf{M}]_j= \sum_i \mathbf{1}_i \mathbf{M}_{i,j} =\sum_i \mathbf{M}_{i,j}.
        # Hence, the column-wise sum of \mathbf{M} is \mathbf{1}^T\mathbf{M}.
        ncols = len(points[0])
        mean_value = [0] * ncols
        for col in range(ncols):
            for point in points:
                mean_value[col] += point[col]
            mean_value[col] = mean_value[col] / len(points)
        return mean_value
