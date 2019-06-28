import pprint as pp

from utils import data_helper


def gini_index(groups, classes):
    """Calculate the Gini index for a split dataset.
    A perfect separation results in a Gini score of 0,
    whereas the worst case split that results in
    50/50 classes in each group result in a Gini
    score of 0.5 (for a 2 class problem)

    Args:
        groups: the list of groups representing the data split.
        Each group is comprised of data instances, where the last
        attribute represents the class/label

        classes: the possible class values (labels) included in the domain

    Returns: a float point number representing the Gini index for the given split
    """
    n_instances = sum([len(group) for group in groups])
    gini = 0.0
    for g in groups:
        size = len(g)
        if size == 0:
            continue
        score = 0.0
        # score the group based on the proportion of each class
        for class_val in classes:
            proportion = [row[-1] for row in g].count(class_val) / size
            score += proportion * proportion
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)

    return gini


def split_on_attribute(index, value, dataset):
    """Separate a given dataset into two lists of rows.

    Args:
        index: the attribute index
        value: the split point of this attribute
        dataset: the list of instances the represent the dataset

    Returns: A tuple with the lists of instances according to the split,
        where left list has instances for which the value of the given attribute
        is less than split value, and the right list contains the other instances
    """
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)

    return left, right


def get_class_values(dataset):
    return list(set([row[-1] for row in dataset]))


def select_split(dataset):
    """Select the best split according to Gini Index

    Args: the dataset

    Returns: a tuple containing
        index: the index of the best attribute to split the dataset
        value: the slit value of the attribute
        groups: the instances separated into groups according to the
        split attribute and value
    """
    class_values = get_class_values(dataset)

    # define some variables to hold the best values found during computation
    b_index, b_value, b_score, b_groups = 999, 999, 999, None

    # iterate over the attributes
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = split_on_attribute(index, row[index], dataset)
            gini = gini_index(groups, class_values)

            # debug info:
            #     print('X%d < %.3f Gini=%.3f' % ((index+1), row[index], gini))
            # ----

            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups

    return {'index': b_index, 'value': b_value, 'groups': b_groups}


def to_terminal_node(group):
    """Creates a terminal node value comprising of all instances present in the given group

    Returns:
        the majority label, i.e., the label used to classify the biggest amount of instances in the group
    """
    # get the label of each instance in the group
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def split_node(node, max_depth, min_size, depth):
    """Create child node splits for a given node, or makes a terminal node
        when it is needed.

    Args:
        node:
        max_depth:
        min_size:
        depth:

    Returns:
    """
    left, right = node['groups']
    del (node['groups'])

    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal_node(left + right)
        return

    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal_node(left), to_terminal_node(right)
        return

    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal_node(left)
    else:
        # recursive step
        node['left'] = select_split(left)
        split_node(node['left'], max_depth, min_size, depth + 1)

    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal_node(right)
    else:
        node['right'] = select_split(right)
        split_node(node['right'], max_depth, min_size, depth + 1)


def build_tree(train, max_depth, min_size):
    root = select_split(train)
    split_node(root, max_depth, min_size, 1)
    return root


def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


if __name__ == '__main__':
    # test Gini values
    #   two groups of data with 2 rows in each group
    #   Each group contains instances for which the last attribute represents the class
    # print(gini_index([[[1, 0], [1, 0]], [[1, 1], [1, 1]]], [0, 1]))
    # print(gini_index([[[1, 1], [1, 0]], [[1, 1], [1, 0]]], [0, 1]))
    # print(gini_index([dataset[:7], dataset[7:]], [0, 1]))

    # test split dataset
    # l, r = split(0, 4, dataset)
    # print("Left:")
    # pp.pprint(l)

    # print("\nRight:")
    # pp.pprint(r)

    # test select split
    # best_split = select_split(dataset)
    # print('Split: [X%d < %.3f]' % ((best_split['index']+1), best_split['value']))

    print("-" * len("Testing"))
    print("Testing")
    print("-" * len("Testing"))

    stump = {'index': 0, 'right': 1, 'value': 6.642287351, 'left': 0}
    print("Mock tree:")
    pp.pprint(stump)
    # print("\n\tprediction: {}".format(predict(stump, [2.771244718, 1.784783929, 0])))

    # print("\nTrained tree:")
    # tree = build_tree(data_helper.toy_labeled_dataset, 4, 1)
    # print("\n\tprediction: {}".format(predict(tree, [2.771244718, 1.784783929, 0])))
    dataset_path = r'../data/raw/iris.data'
    tree = build_tree(data_helper.load_dataset(dataset_path), 4, 1)
    pp.pprint(tree)

    print("\n\tprediction: {}".format(predict(tree, [6.2, 3.4, 5.4, 2.3, 0])))
