from keras.datasets import imdb
import numpy as np

def load_imdb_pickle(vocabulary_size=1000):
    """Loads the keras imdb dataset

    Since numpy version 1.16.4 sets allow_pickle to False by default,
    we need to overwrite this parameter to be able to load the dataset into memory.

    We should, obviously, reset the default parameters later.
    """

    # save np.load
    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    # call load_data with allow_pickle implicitly set to true
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocabulary_size)

    # restore np.load for future normal usage
    np.load = np_load_old
    return (X_train, y_train), (X_test, y_test)
