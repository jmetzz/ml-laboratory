import math


def euclidean(instance1, instance2, dimension):
    distance = 0
    for x in range(dimension):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)
