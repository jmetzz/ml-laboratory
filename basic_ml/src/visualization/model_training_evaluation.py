import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame


def get_training_metrics(history):
    # This is needed depending on if you used the pretrained model or you trained it yourself
    if not isinstance(history, DataFrame):
        history = history.history_data

    acc = history["sparse_categorical_accuracy"]
    val_acc = history["val_sparse_categorical_accuracy"]

    loss = history["loss"]
    val_loss = history["val_loss"]

    return acc, val_acc, loss, val_loss


def plot_train_eval(history: pd.DataFrame) -> None:
    acc, val_acc, loss, val_loss = get_training_metrics(history)

    acc_plot = pd.DataFrame({"training accuracy": acc, "evaluation accuracy": val_acc})
    acc_plot = sns.lineplot(data=acc_plot)
    acc_plot.set_title("training vs evaluation accuracy")
    acc_plot.set_xlabel("epoch")
    acc_plot.set_ylabel("sparse_categorical_accuracy")
    plt.show()

    loss_plot = pd.DataFrame({"training loss": loss, "evaluation loss": val_loss})
    loss_plot = sns.lineplot(data=loss_plot)
    loss_plot.set_title("training vs evaluation loss")
    loss_plot.set_xlabel("epoch")
    loss_plot.set_ylabel("loss")
    plt.show()


if __name__ == "__main__":
    rng = np.random.default_rng()

    history_data = pd.DataFrame(
        data=rng.random((100, 4)),
        columns=["loss", "sparse_categorical_accuracy", "val_loss", "val_sparse_categorical_accuracy"],
    )
    plot_train_eval(history_data)
