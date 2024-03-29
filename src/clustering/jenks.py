""" The Jenks optimization method
(https://en.wikipedia.org/wiki/Jenks_natural_breaks_optimization),
also called the Jenks natural breaks classification method,
is a data clustering method designed to determine the best
arrangement of values into different classes.
This is done by seeking to minimize each class's average deviation
from the class mean, while maximizing each class's
deviation from the means of the other groups. In other words,
the method seeks to reduce the variance within classes and maximize
the variance between classes.
"""
import pandas as pd

# pylint: disable=C0103,R0914,R0912


def get_jenks_breaks(data_list, number_class):
    """source: http://danieljlewis.org/files/2010/06/Jenks.pdf"""
    data_list.sort()
    mat1 = []
    for line in range(len(data_list) + 1):  # noqa: B007
        temp = []
        for col in range(number_class + 1):  # noqa: B007
            temp.append(0)
        mat1.append(temp)
    mat2 = []
    for line in range(len(data_list) + 1):  # noqa: B007
        temp = []
        for col in range(number_class + 1):  # noqa: B007
            temp.append(0)
        mat2.append(temp)
    for line in range(1, number_class + 1):
        mat1[1][line] = 1
        mat2[1][line] = 0
        for col in range(2, len(data_list) + 1):
            mat2[col][line] = float("inf")
    v = 0.0
    for line in range(2, len(data_list) + 1):
        s1 = 0.0
        s2 = 0.0
        w = 0.0
        for m in range(1, line + 1):
            i3 = line - m + 1
            val = float(data_list[i3 - 1])
            s2 += val * val
            s1 += val
            w += 1
            v = s2 - (s1 * s1) / w
            i4 = i3 - 1
            if i4 != 0:
                for col in range(2, number_class + 1):
                    if mat2[line][col] >= (v + mat2[i4][col - 1]):
                        mat1[line][col] = i3
                        mat2[line][col] = v + mat2[i4][col - 1]
        mat1[line][1] = 1
        mat2[line][1] = v
    k = len(data_list)
    kclass = []
    for line in range(number_class + 1):  # noqa: B007
        kclass.append(min(data_list))
    kclass[number_class] = float(data_list[len(data_list) - 1])
    count_num = number_class
    while count_num >= 2:  # print "rank = " + str(mat1[k][count_num])
        idx = int((mat1[k][count_num]) - 2)
        # print ("val = " + str(data_list[idx]))
        kclass[count_num - 1] = data_list[idx]
        k = int(mat1[k][count_num] - 1)
        count_num -= 1
    return kclass


# Use Jenks breaks to find the lower and upper price cuts
data_df = pd.DataFrame()
jbreaks = get_jenks_breaks(data_list=data_df["average_price"].values.tolist(), number_class=3)
