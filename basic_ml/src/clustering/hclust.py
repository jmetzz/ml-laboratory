import heapq
import pprint as pp

from scipy.spatial import distance

from utils import datasets


class HClustering:
    def __init__(self, data, k):
        self.k = k
        self.data = data
        self.size = len(data)
        self.dimension = len(data[0])
        self.clusters = {}

    def clusterize(self):
        current_clusters = self._initialize_clusters()
        heap = self._build_priority_queue()  # pairwise distance based in the form ([dist, [[i], [j]])
        old_clusters = []
        while len(current_clusters) > self.k:
            candidate_cluster = heapq.heappop(heap)
            if not self._valid_heap_node(candidate_cluster, old_clusters):
                continue
            elements = candidate_cluster[1]
            new_cluster = self._create_cluster(elements)
            new_cluster_elements = new_cluster["elements"]

            for e in elements:
                old_clusters.append(e)
                del current_clusters[str(e)]
            self._add_heap_entry(heap, new_cluster, current_clusters)
            current_clusters[str(new_cluster_elements)] = new_cluster
        self.clusters = current_clusters
        return self.clusters

    def _create_cluster(self, instances):
        new_cluster = {}
        # flatten the list of instances
        element_indexes = sum(instances, [])
        element_indexes.sort()
        centroid = self._compute_centroid(self.data, element_indexes)
        new_cluster.setdefault("centroid", centroid)
        new_cluster.setdefault("elements", element_indexes)
        return new_cluster

    def _initialize_clusters(self):
        # every cluster is represented as '[idx] <,[idx]>' and this value is
        # used as the key on the clusters dictionary
        for idx in range(self.size):
            key = str([idx])
            self.clusters.setdefault(key, {})
            self.clusters[key].setdefault("centroid", self.data[idx])
            self.clusters[key].setdefault("elements", [idx])
        return self.clusters

    def _build_priority_queue(self):
        """Transform list x into a heap, in-place, in linear time using
        heapq from standard library
        :return: the generated heap
        """
        distance_list = self._compute_pairwise_distance()
        heapq.heapify(distance_list)
        return distance_list

    def _compute_pairwise_distance(self):
        result = []
        # calculate the pairwise distance between points
        # only for the upper triangular matrix
        for i in range(self.size - 1):
            for j in range(i + 1, self.size):
                dist = distance.euclidean(self.data[i], self.data[j], self.dimension)
                ## duplicate dist, need to be remove, and there is no difference to use tuple only
                ## leave second dist here is to take up a position for tie selection
                ## result.append((dist, [dist, [[i], [j]]]))

                # saves the distance and the indexes of the instances
                result.append(([dist, [[i], [j]]]))
        return result

    def _compute_centroid_two_clusters(self, current_clusters, data_points_index):
        size = len(data_points_index)
        dim = self.dimension
        centroid = [0.0] * dim
        for index in data_points_index:
            dim_data = current_clusters[str(index)]["centroid"]
            for i in range(dim):
                centroid[i] += float(dim_data[i])
        for i in range(dim):
            centroid[i] /= size
        return centroid

    def _compute_centroid(self, data, instances_indexes):
        points = [data[j] for j in instances_indexes]
        return self._mean(points)

    def _mean(self, instances):
        num_cols = self.dimension
        num_instances = len(instances)
        m = [0.0] * num_cols
        for col in range(num_cols):
            for point in instances:
                m[col] += float(point[col])
            m[col] = m[col] / num_instances
        return m

    def _valid_heap_node(self, heap_node, old_clusters):
        pair_data = heap_node[1]
        for cluster in old_clusters:
            if cluster in pair_data:
                return False
        return True

    def _add_heap_entry(self, heap, new_cluster, current_clusters):
        for existing_cluster in current_clusters.values():
            dist = distance.euclidean(existing_cluster["centroid"], new_cluster["centroid"], self.dimension)
            new_entry = self._create_heap_entry(existing_cluster, new_cluster, dist)
            heapq.heappush(heap, new_entry)
            # heapq.heappush(heap, (dist, new_entry))

    def _create_heap_entry(self, existing_cluster, new_cluster, dist):
        heap_entry = []
        heap_entry.append(dist)
        heap_entry.append([new_cluster["elements"], existing_cluster["elements"]])
        return heap_entry


def main(data, k):
    hc = HClustering(data, k)
    clusters = hc.clusterize()
    # cluster_labels = clusters.values()
    for c in clusters.values():
        pp.pprint(c)

    # gold_standard = {}  # initialize gold standard based on the class labels
    # precision, recall = eval.evaluate(clusters, gold_standard)


if __name__ == "__main__":
    """
    Arguments:
        filename: a text file name for the input data
        k: a value k for the desired number of clusters.

    Returns:
        clusters: output k clusters, with each cluster contains a set of data points (index for input data)
        precision
        recall
    """
    main(datasets.toy_labelled_2d, 2)
