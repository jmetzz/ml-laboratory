import pandas as pd


def polynomial_dataframe(feature: pd.Series, degree: int) -> pd.DataFrame:
    """

    Creates a data frame  with increasing integer powers up to ‘degree’ from the given features

    The first column of the new data frame is equal to ‘feature’
    and the remaining columns equal to ‘feature’ to increasing integer powers up to ‘degree’.

    Args:
        feature: a pandas.Series to use as base
        degree: the max desired polynomial degree

    Returns:
        a pandas.DataFrame
    """

    if degree < 1:
        raise ValueError("Degree must be greater of zero.")

    poly_dataframe = pandas.DataFrame(data=feature, columns=['power_1'])

    if degree == 1:
        return poly_dataframe

    for power in range(2, degree + 1):
        # first we'll give the column a name:
        col_name = f"power_{power}"
        # assign poly_dataframe[name] to be feature^power; use apply(*)
        poly_dataframe[col_name] = feature.apply(lambda x: x ** power)

    return poly_dataframe


if __name__ == '__main__':
    s = pd.Series([1, 2, 3, 4])
    print(polynomial_dataframe(s, 3))

