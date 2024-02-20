import numpy as np
import pandas as pd
import scipy
from pandas import Series


def ent(data: Series) -> float:
    p_data = data.value_counts() / len(data)  # calculates the probabilities
    entropy = scipy.stats.entropy(p_data, base=2)  # input probabilities to get the entropy
    # entropy = sc.stats.entropy(p_data)  # input probabilities to get the entropy
    return entropy


if __name__ == "__main__":
    input_data = pd.Series(np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]))
    print(ent(input_data))

    input_data = pd.Series(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    print(ent(input_data))
