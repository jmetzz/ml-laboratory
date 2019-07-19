import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from utils import data_helper

dataset_path = r'../data/raw/cat'
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = data_helper.load_h5_dataset(dataset_path,
                                                                                                  'catvnoncat')

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
