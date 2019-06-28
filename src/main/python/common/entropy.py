#!/usr/bin/env python3
"""Entropy is a measure of disorder. In other words,
Entropy is an indicator of how messy your data is.

Let us imagine we have a set of N items.
These items fall into two categories, n have Label 1
and m=N-n have Label 2. As we have seen, to get
our data a bit more ordered, we want to group them by labels.
We introduce the ratio p=n/N and q=m/N=1-p

The entropy of out set is given by the following equation:

 E = -p lob2(p) - q log2(q)

The entropy is an absolute measure which provides a number between 0 and 1,
independently of the size of the set. At least if we consider the
binary scenario and log with base 2.
"""

import fileinput
import string
from math import log


def _identity(value): return value


def range_bytes(): return range(256)


def range_binary(): return range(2)


def range_printable(): return (ord(c) for c in string.printable)


def H(data, iterator=range_bytes, convert=_identity, base=2):
    if not data:
        return 0
    entropy = 0
    for x in iterator():
        p_x = float(data.count(convert(x))) / len(data)
        if p_x > 0:
            entropy += - p_x * log(p_x, base)
    return entropy


def shannon_entropy(data):
    return H(data, range_binary)


def _resolve_base(data):
    base = len(set(data))
    if base < 2:
        base = 2
    return base


def column(matrix, i):
    return [row[i] for row in matrix]


def main():
    for row in fileinput.input():
        string_input = row.rstrip('\n')
        print("%s: %f" % (string_input, H(string_input, range_printable)))


if __name__ == '__main__':
    for str in ['gargleblaster', 'tripleee', 'magnus', 'lkjasdlk',
                'aaaaaaaa', 'sadfasdfasdf', '7&wS/p(', 'aabb']:
        print("%s: %f" % (str, H(str, range_printable, chr)))

    # str_data = ''.join(str(e) for e in column(dh.toy_labeled_dataset, 2))
    # print("%s: %f" %("dataset", H(column(dh.toy_labeled_dataset, 2), range_binary)))

    # dataset1 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    # dataset2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    # print("%s: %f" % ("dataset 1", H(dataset1, range_binary)))
    # print("%s: %f" % ("dataset 2", H(dataset2, range_binary)))

    # print("%s: %f" % ("dataset 1", shannon_entropy(dataset1)))
    # print("%s: %f" % ("dataset 2", shannon_entropy(dataset2)))
