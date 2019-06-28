import random

from clustering import BaseClustering
from common import distance
from utils import data_helper


class KMeans(BaseClustering):
    """K-means clustering algorithm implementation

    K-means is often referred to as Lloyd’s algorithm.
    In basic terms, the algorithm has three steps.
    The first step chooses the initial centroids, with
    the most basic method being to choose k samples from
    the dataset X. After initialization, K-means consists of
    looping between the two other steps. The first step
    assigns each sample to its nearest centroid. The second step
    creates new centroids by taking the mean value of all of
    the samples assigned to each previous centroid.
    The difference between the old and the new centroids are computed
    and the algorithm repeats these last two steps until this value is
    less than a threshold. In other words, it repeats until
    the centroids do not move significantly.

    Given enough time, K-means will always converge, however this may be
    to a local minimum. This is highly dependent on the initialization of
    the centroids. As a result, the computation is often done several times,
    with different initializations of the centroids.

        Given the inputs x1,x2,x3,…,xn and value of K

        Step 1 - Pick K random points as cluster centers called centroids.
        Step 2 - Assign each xi_i to nearest cluster by calculating its distance to each centroid.
        Step 3 - Find new cluster center by taking the average of the assigned points.
        Step 4 - Repeat Step 2 and 3 until none of the cluster assignments change.
    """

    def __init__(self, data, function_dist=distance.euclidean):
        super().__init__(data)
        self.function_dist = function_dist
        self.centroids = []

    def build(self, k=3):
        clusters = [0] * self.num_features
        prev_centroids = [[-1] * self.num_features] * k
        new_centroids = self._initial_centroids(k, self.data)

        while self._centroids_changed(prev_centroids, new_centroids):
            clusters = self._find_clusters(self.data, new_centroids)
            prev_centroids = new_centroids
            new_centroids = BaseClustering.calculate_centroids(self.data, k, clusters)
        self._update_model(new_centroids, clusters)

    def model(self):
        return self.centroids, self.clusters

    def _find_clusters(self, data, centroids, function_dist=distance.euclidean):
        """Distribute the instances in the clusters represented by the centroids

        :param data: the dataset of instances
        :param centroids: the centroid vectors with the same structure as the instances in the dataset
        :param function_dist: the function used to calculate the distance between two instances
        :return: a list representing the cluster assignment for each instance in the dataset
        """
        clusters = [0] * self.size
        for idx in range(self.size):
            distances = self._dist(data[idx], centroids)
            cluster = distances.index(min(distances))
            clusters[idx] = cluster
        return clusters

    def _dist(self, instance, centroids):
        ncol = len(instance)
        return [self.function_dist(instance, c, ncol) for c in centroids]

    @staticmethod
    def _centroids_changed(prev_centroids, new_centroids):
        for i in range(len(prev_centroids)):
            for z1, z2 in zip(prev_centroids[i], new_centroids[i]):
                if z1 != z2:
                    return True
        return False

    @staticmethod
    def _initial_centroids(k, data):
        indexes = set(random.sample(range(0, len(data)), k))
        return [data[i] for i in indexes]

    def _update_model(self, new_centroinds, clusters):
        self.centroids = new_centroinds
        self.clusters = {c: [] for c in set(clusters)}
        for e in range(len(self.data)):
            self.clusters.get(clusters[e]).append(e)


def report(centroids, model):
    indexed_centroids = zip(range(len(centroids)), centroids)
    for i, c in indexed_centroids:
        print(f"C-{i}: {c}")

    print("\nClusters: [ cluster-id | instances]")
    for c, e in model.items():
        print(f"C-{c}: {e}")


if __name__ == '__main__':
    data = data_helper.toy_unlabeled_dataset

    kmeans = KMeans(data)
    kmeans.build()
    centroids, model = kmeans.model()
    report(centroids, model)
