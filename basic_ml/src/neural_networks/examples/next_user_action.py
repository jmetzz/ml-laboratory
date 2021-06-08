import time

import numpy as np
import pandas as pd
from sklearn import preprocessing

from neural_networks.base import networks
from neural_networks.base.costs import CrossEntropyCost
from neural_networks.base.initializers import sqrt_connections_ratio
from utils.data_helper import as_vector
from utils.model_helper import save_pickle_model


def prepare_training_data(df, num_features, class_feature, num_classes):
    y = df[class_feature].to_numpy()

    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(df.drop([class_feature], axis=1))

    inputs = [np.reshape(x, (num_features, 1)) for x in X]
    labels = [as_vector(label - 1, num_classes) for label in y]
    return zip(inputs, labels)


def prepare_test_data(df, num_features, class_feature):
    y = df[class_feature].to_numpy()

    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(df.drop([class_feature], axis=1))

    inputs = [np.reshape(x, (num_features, 1)) for x in X]
    labels = [label - 1 for label in y]
    return zip(inputs, labels)


if __name__ == "__main__":
    path_to_data = "../data/raw/user_actions"
    # Training data:
    df_train = pd.read_csv(f"{path_to_data}/train_processed_2.csv", sep="\t")
    df_train = df_train.drop(["user_id", "timestamp"], axis=1)
    training_set = prepare_training_data(df_train, 14, "action", 5)

    # Test data
    # Normalize test examples
    df_test = pd.read_csv(f"{path_to_data}/test_processed_2.csv", sep="\t")
    df_test = df_test.drop(["user_id", "timestamp"], axis=1)
    test_set = prepare_test_data(df_test, 14, "action")

    net = networks.ImprovedNetwork(
        [14, 30, 5],
        cost_function=CrossEntropyCost,
        weight_initializer=sqrt_connections_ratio,
    )

    evaluation = net.sdg(
        training_set,
        epochs=10,
        eta=0.75,
        batch_size=100,
        lmbda=0.5,
        monitor=["train_acc"],
        debug=True,
    )

    acc, cost = net.evaluate(test_set, ["acc", "cost"])

    print("Test performance:")
    print(f"\tAcc: {acc}")
    print(f"\tCost: {cost}")
    ts = time.time()
    net.export(f"../models/model_30e_10_0.75_100_0.5-{ts}.json")
    save_pickle_model(f"../models/model_30e_10_0.75_100_0.5-{ts}.pickle", net)
