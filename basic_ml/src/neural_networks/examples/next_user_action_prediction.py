import numpy as np
import pandas as pd
from sklearn import preprocessing
from utils.model_helper import load_pickle_model


def prepare_test_data(df, num_features, class_feature):
    y = df[class_feature].to_numpy()

    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(df.drop([class_feature], axis=1))

    inputs = [np.reshape(x, (num_features, 1)) for x in X]
    labels = [label - 1 for label in y]
    return zip(inputs, labels)


if __name__ == "__main__":
    path_to_data = "../data/raw/user_actions"
    df_test = pd.read_csv(f"{path_to_data}/test_processed_2.csv", sep="\t")
    df_test = df_test.drop(["user_id", "timestamp"], axis=1)
    test_set = prepare_test_data(df_test, 14, "action")

    net = load_pickle_model("../models/model.pickle")
    acc, cost = net.evaluate(test_set, ["acc", "cost"])

    print("Test performance:")
    print(f"\tAcc: {acc}")
    print(f"\tCost: {cost}")
