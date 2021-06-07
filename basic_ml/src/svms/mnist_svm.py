"""A classifier program for recognizing handwritten digits from the MNIST data set

    This example uses an SVM classifier from skearn python library,
    which provides a simple Python interface to a fast C-based library
     for SVMs known as LIBSVM (https://www.csie.ntu.edu.tw/~cjlin/libsvm/).
"""

from sklearn import svm

from utils.dataset_loader import MNISTLoader

if __name__ == "__main__":
    training_data, validation_data, test_data = MNISTLoader.load_data("../../../../data/processed/MNIST/mnist.pkl.gz")

    clf = svm.SVC(gamma="auto")
    clf.fit(training_data[0], training_data[1])

    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))

    print("Baseline classifier using an SVM.")
    print(f"{num_correct} of {len(test_data[1])} values correct.")
