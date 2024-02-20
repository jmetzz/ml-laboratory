import random

from clustering import BaseClustering
from common import distance
from utils import datasets


class KMeans(BaseClustering):
    """K-means clustering algorithm implementation

    K-means is often referred to as Lloyd’s algorithm.
    In basic_ml terms, the algorithm has three steps.
    The first step chooses the initial centroids, with
    the most basic_ml method being to choose k samples from
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

    def _find_clusters(self, data, centroids):
        """Distribute the instances in the clusters represented by the centroids

        :param data: the dataset of instances
        :param centroids: the centroid vectors with the same structure as the instances in the dataset
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
        for idx, _ in enumerate(prev_centroids):
            for centroid_1, centroid_2 in zip(prev_centroids[idx], new_centroids[idx]):
                if centroid_1 != centroid_2:
                    return True
        return False

    @staticmethod
    def _initial_centroids(num_clusters, data):
        indexes = set(random.sample(range(0, len(data)), num_clusters))
        return [data[i] for i in indexes]

    def _update_model(self, new_centroids, clusters):
        self.centroids = new_centroids
        self.clusters = {c: [] for c in set(clusters)}
        for element in range(len(self.data)):
            self.clusters.get(clusters[element]).append(element)


def report(centroids, model):
    indexed_centroids = zip(range(len(centroids)), centroids)
    for idx, centroid in indexed_centroids:
        print(f"C-{idx}: {centroid}")

    print("\nClusters: [ cluster-id | instances]")
    for centroid, element in model.items():
        print(f"C-{centroid}: {element}")


if __name__ == "__main__":
    dataset = datasets.TOY_2D_UNLABELLED

    kmeans = KMeans(dataset)
    kmeans.build()
    clustering_centroids, model = kmeans.model()
    report(clustering_centroids, model)
