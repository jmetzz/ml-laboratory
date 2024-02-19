from pathlib import Path

import numpy as np
import pandas as pd
from utils.data_helper import normalize

from regression.base import predict_output


def coordinate_descent_step(feature_matrix, feature_idx, weights, output, l1_penalty):
    """
    Implementation of cyclical coordinate descent with normalized features

    Cycle through coordinates 0 to (d-1) in order,
    and assume the features were normalized.

    The formula for optimizing each coordinate is as follows:

               ┌ (ro[i] + lambda/2)     if ro[i] < -lambda/2
        w[i] = ├ 0                      if -lambda/2 <= ro[i] <= lambda/2
               └ (ro[i] - lambda/2)     if ro[i] > lambda/2
        where

        ro[i] = SUM[ [feature_i]*(output - prediction + w[i]*[feature_i]) ].

    Note that we do not regularize the weight of the constant feature (intercept) w[0],
    so, for this weight, the update is simply:

        w[0] = ro[i]
    """
    prediction = predict_output(feature_matrix, weights)

    ro_i = np.sum(
        feature_matrix[:, feature_idx] * (output - prediction + weights[feature_idx] * feature_matrix[:, feature_idx])
    )

    if feature_idx == 0:  # intercept -- do not regularize
        return ro_i

    if ro_i < -l1_penalty / 2.0:
        return ro_i + l1_penalty / 2.0

    if ro_i > l1_penalty / 2.0:
        return ro_i - l1_penalty / 2.0

    return 0.0


def lasso_cyclical_coordinate_descent(
    feature_matrix, initial_weights, output, l1_penalty=1e-2, tolerance=1e-2, max_iterations=1e3
):
    new_weights = initial_weights.copy()
    deltas = np.zeros(initial_weights.shape)
    converged = False
    it = 0
    while not converged:
        for idx, current_weight in enumerate(new_weights):
            new_weights[idx] = coordinate_descent_step(feature_matrix, idx, new_weights, output, l1_penalty)
            deltas[idx] = abs(new_weights[idx] - current_weight)
        converged = max(deltas) < tolerance or it >= max_iterations
        it += 1
    return new_weights


def get_numpy_data(data_df, features, output):
    data_df["constant"] = 1
    features = ["constant"] + features
    features_sframe = data_df[features]
    output_sarray = data_df[output]
    return features_sframe.to_numpy(), output_sarray.to_numpy()


if __name__ == "__main__":
    USER_HOME = Path("/Users/jean.metz")
    PARENT_HOME = Path(USER_HOME, "workspace", "jmetzz", "ml-laboratory")
    PROJECT_HOME = Path(PARENT_HOME, "basic_ml")
    DATA_HOME = Path(PARENT_HOME, "data", "processed", "king_county_house_sales")

    dtype_dict = {
        "bathrooms": float,
        "waterfront": int,
        "sqft_above": int,
        "sqft_living15": float,
        "grade": int,
        "yr_renovated": int,
        "price": float,
        "bedrooms": float,
        "zipcode": str,
        "long": float,
        "sqft_lot15": float,
        "sqft_living": float,
        "floors": float,
        "condition": int,
        "lat": float,
        "date": str,
        "sqft_basement": int,
        "yr_built": int,
        "id": str,
        "sqft_lot": int,
        "view": int,
    }

    train_data = pd.read_csv(Path(DATA_HOME, "wk3_kc_house_train_data.csv"), dtype=dtype_dict)
    test_data = pd.read_csv(Path(DATA_HOME, "wk3_kc_house_test_data.csv"), dtype=dtype_dict)
    train_data["constant"] = 0.0
    test_data["constant"] = 0.0

    features_names = [
        "constant",  # the intercept
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "view",
        "condition",
        "grade",
        "sqft_above",
        "sqft_basement",
        "yr_built",
        "yr_renovated",
    ]
    target_variable = "price"

    # train_feature_matrix, train_target_values = get_numpy_data(train_data, features_names, target_variable)
    train_feature_matrix, train_target_values = train_data[features_names], train_data[target_variable]

    # test_feature_matrix, test_target_values = get_numpy_data(test_data, features_names, target_variable)
    test_feature_matrix, test_target_values = test_data[features_names], test_data[target_variable]

    normalized_train_features, feature_norms = normalize(train_feature_matrix)

    new_weights = lasso_cyclical_coordinate_descent(
        feature_matrix=normalized_train_features.to_numpy(),
        initial_weights=np.zeros(len(features_names)),
        output=train_target_values.to_numpy(),
        l1_penalty=1e7,
        tolerance=1.0,
    )

    print(list(new_weights))
