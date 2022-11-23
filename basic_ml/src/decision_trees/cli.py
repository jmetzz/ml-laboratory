import numpy as np
import pandas as pd
import typer
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from common.evaluation import accuracy
from decision_trees.bagged_tree import BaggedTrees
from decision_trees.cart import CART

app = typer.Typer()


@app.command()
def run_cart(max_depth: int = 4, min_size: int = 1, avg_strategy: str = "ovr", verbose: bool = False):
    train_features, test_features, train_labels, test_labels = train_test_split(
        *load_iris(return_X_y=True), test_size=0.33, random_state=31
    )

    model = CART(max_depth, min_size)
    model.fit(train_features, train_labels)
    predictions = model.predict(test_features)

    acc = accuracy(test_labels, predictions)
    # prediction probability in the shape of (n_samples, n_classes)
    # is expected to calculate the auc.
    # Thus, let's simulate this with probability of 1 for the
    # predicted class and 0 for the others
    probas = np.zeros((len(test_labels), 3))
    for idx, predicted_label_idx in enumerate(predictions):
        probas[idx, predicted_label_idx] = 1.0

    auc = roc_auc_score(y_true=test_labels, y_score=probas, multi_class=avg_strategy)

    print(f"Accuracy is {acc:.2f}")
    print(f"AUC is {auc:.2f}")

    if verbose:
        print(model.summary())


@app.command()
def random_vs_bagged(test_size: float = 0.33, max_num_trees: int = 100, num_trees_increment: int = 5):
    train_features, test_features, train_labels, test_labels = train_test_split(
        *load_breast_cancer(return_X_y=True, as_frame=False), test_size=test_size, random_state=42
    )
    auc_random_forest = []
    auc_bagged_trees = []
    num_trees_map = []

    for num_trees in range(1, max_num_trees, num_trees_increment):
        rf_clf = RandomForestClassifier(n_estimators=num_trees)
        bt_clf = BaggedTrees(n_trees=num_trees)

        rf_clf.fit(train_features, train_labels)
        bt_clf.fit(train_features, train_labels)

        # the probability of the class with the “greater label” should be provided.
        # and thus, [:, 1] is used.
        prediction_rf = rf_clf.predict_proba(test_features)[:, 1]
        prediction_bagged = bt_clf.predict(test_features)

        auc_random_forest.append(roc_auc_score(test_labels, prediction_rf))
        auc_bagged_trees.append(roc_auc_score(test_labels, prediction_bagged))
        num_trees_map.append(num_trees)

    df = pd.DataFrame({"num_trees": num_trees_map, "AUC RF": auc_random_forest, "AUC BT": auc_bagged_trees})
    print(df.tail())


def plot_auc():
    y = np.array([1, 1, 2, 2])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr, thresholds = roc_curve(y, scores, pos_label=2)


if __name__ == "__main__":
    app()
