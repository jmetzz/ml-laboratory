from utils.datasets import linear_forward_test_case

from neural_networks.base.networks import DidacticCourseraNetwork

if __name__ == "__main__":
    A, W, b = linear_forward_test_case()

    print("test case created")
    net = DidacticCourseraNetwork()
    Z, linear_cache = net.linear_forward(A, W, b)
    print("Z = " + str(Z))
