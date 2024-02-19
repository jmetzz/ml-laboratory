import matplotlib.pyplot as plt
import numpy as np
from common import evaluation


def plot_roc_curve(fpr, tpr, label=None):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], "k--")  # dashed diagonal
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()


# Example usage

if __name__ == "__main__":
    y_true = np.array([0, 1, 1, 0, 1])  # Example ground truth labels
    y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.7])  # Example prediction scores

    fpr, tpr, thresholds = evaluation.roc(y_true, y_scores)
    plot_roc_curve(fpr, tpr, label="Example Model")
