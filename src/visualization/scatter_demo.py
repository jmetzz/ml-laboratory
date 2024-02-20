import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs

sns.set()  # for plot styling

features, labels = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

plt.scatter(features[:, 0], features[:, 1], s=50)
plt.show()
