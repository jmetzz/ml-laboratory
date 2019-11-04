import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from scipy import ndimage
from utils import data_helper

if __name__ == "__main__":

    dataset_path = r'../data/raw/cat'
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = data_helper.load_from_h5(dataset_path, 'catvnoncat')

    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]

    # Reshape the training and test examples
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    # standardize our dataset.
    # One common preprocessing step in machine learning is to center and standardize your dataset,
    # meaning that you substract the mean of the whole numpy array from each example, and then divide
    # each example by the standard deviation of the whole numpy array. But for picture datasets,
    # it is simpler and more convenient and works almost as well to just divide every row of the dataset
    # by 255 (the maximum value of a pixel channel).
    train_set_x = train_set_x_flatten/255.
    test_set_x = test_set_x_flatten/255.

    # To design a simple algorithm to distinguish cat images from non-cat images, base on Logistic Regression,
    # perform the following steps:
    # - Initialize the parameters of the model
    # - Learn the parameters for the model by minimizing the cost
    # - Use the learned parameters to make predictions (on the test set)
    # - Analyse the results and conclude


