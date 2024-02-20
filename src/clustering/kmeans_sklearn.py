import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder


def kmeans_clustering(data: pd.DataFrame, n_clusters: int = 3, verbose: bool = False):
    X = data.dropna()
    # converting the non-numeric to numeric values
    encoders = {var: LabelEncoder() for var in data.select_dtypes(exclude=["number"]).columns}
    for var, encoder in encoders.items():
        X[var] = encoder.fit_transform(X[var])

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X)
    inertia = kmeans.predict(X)
    if verbose:
        print(kmeans.inertia_)

    for var, encoder in encoders.items():
        X[var] = encoder.inverse_transform(X[var])

    X["Clusters"] = inertia
    return X


def calculate_elbow(data: np.ndarray) -> list:
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans.fit(data)
        # inertia method returns wcss for that model
        wcss.append(kmeans.inertia_)
    return wcss


def plot_elbow(wcss):
    plt.figure(figsize=(10, 5))
    sns.lineplot(range(1, 11), wcss, marker="o", color="green")
    plt.title("The Elbow Method")
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.show()
    plt.close()


if __name__ == "__main__":
    data = None  # collect some data ...

    # find best number of clusters
    plot_elbow(calculate_elbow(data))
    # use the plot to choose the value of k,
    # then change the parameter below:
    clustering = kmeans_clustering(data, n_clusters=3)
