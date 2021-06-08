import math


def euclidean(instance1, instance2, dimension):
    distance = 0
    for dim in range(dimension):
        distance += pow((instance1[dim] - instance2[dim]), 2)
    return math.sqrt(distance)
