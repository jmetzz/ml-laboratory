import numpy as np
import pandas as pd
import scipy as sc
from pandas import Series
from scipy import stats


def ent(data: Series) -> float:
    p_data = data.value_counts() / len(data)  # calculates the probabilities
    entropy = sc.stats.entropy(p_data, base=2)  # input probabilities to get the entropy
    # entropy = sc.stats.entropy(p_data)  # input probabilities to get the entropy
    return entropy


if __name__ == '__main__':
    data = pd.Series(np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]))
    print(ent(data))

    data = pd.Series(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    print(ent(data))
