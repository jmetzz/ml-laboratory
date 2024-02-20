import os

import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    cols_names = [
        "Class",
        "age",
        "menopause",
        "tumor-size",
        "inv-nodes",
        "node-caps",
        "deg-malig",
        "breast",
        "breast-quad",
        "irradiat",
    ]
    # read the data
    DATASET_PATH = r"../../resources/data/external/breast_cancer/breast-cancer.data"

    print(os.getcwd())
    df = pd.read_csv(DATASET_PATH, header=None, names=cols_names).replace(
        {"?": "unknown"}
    )  # NaN are represented by '?'

    data = df.drop(columns="Class")
    labels = df["Class"].copy()
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.5, random_state=42)
    print(data_train.shape)
