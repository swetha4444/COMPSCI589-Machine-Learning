import numpy as np
import pandas as pd
from tree_helper import calcInfoGain, split_dataset, calcAvgGiniCoeff
import matplotlib as plt
import math
from data_preprocessing import PreProcesser

class Node:
    def __init__(self, branches = None, feature = None, depth = 0, label = None):
        self.branches = branches if branches != None else {} #branches are marked as dictonary
        self.feature = feature
        self.label = label
        self.depth = depth

'''
    metric: id3 (information gain) cart (gini)
    stopping_criteria: 85 
'''

def fit(X, y, depth=0, metric = "id3", stopping_criteria=None):
    # Base Case - Heuristics: If current data has 85% of the same class, don't split further
    if stopping_criteria is not None:
        values, counts = np.unique(y, return_counts=True)
        if max(counts) / len(y) >= stopping_criteria / 100:  # Convert percentage to fraction
            return Node(label=values[np.argmax(counts)], branches=None)

    # Base Case 1: No more features to split - take majority value of split to be leaf
    if X.shape[1] == 0:  # No more features left to split on
        return Node(label=np.bincount(y).argmax(), branches=None)  # Majority class leaf node

    # Base Case 2: All values y of split are same, no need to split further
    if len(np.unique(y)) == 1:
        return Node(label=y[0], branches=None)

    # Getting best split by comparing each split criteria
    best_feature_index, best_criteria_value, best_subsets = None, -math.inf, None
    for i in range(X.shape[1]):  # For each feature
        subsets = split_dataset(X, y, i)  # Get subsets for each feature
        y_subsets = [subset[1] for subset in subsets]
        # criteria_value = calcInfoGain(y, y_subsets)
        criteria_value = calcInfoGain(y,y_subsets) if(metric == "id3") else -1 * calcAvgGiniCoeff(y,y_subsets)

        if criteria_value > best_criteria_value:
            best_criteria_value = criteria_value
            best_subsets = subsets
            best_feature_index = i

    # Base Case 3: Zero value of Info Gain, no need to split further as data is same after split
    if best_criteria_value == 0:
        return Node(label=np.bincount(y).argmax(), branches=None)  # Majority class leaf node

    # Create the decision node and recursively build the tree
    branches = {}
    for subset in best_subsets:
        X_subset, y_subset = subset
        branch = fit(X_subset, y_subset, depth=depth + 1, stopping_criteria=stopping_criteria)
        branches[np.unique(X_subset[:, best_feature_index])[0]] = branch

    return Node(feature=best_feature_index, branches=branches)


def _predictX_i(node: Node, X_i):
    # If leaf node
    if node.label is not None:
        return node.label
    # If node is a decision node
    value = X_i[node.feature]
    if value in node.branches:
        return _predictX_i(node.branches[value], X_i)  # Recursive call to next branch
    else:
        # Get all labels in the current node's branch (subtree) for the probable label
        all_labels = []
        for branch in node.branches.values():
            all_labels.extend(get_all_labels(branch))
        # Return the majority class from the current node's subtree
        probable_label = np.bincount(all_labels).argmax()
        return probable_label
    
def get_all_labels(node: Node):
    if node.label is not None:
        return [node.label]
    labels = []
    for branch in node.branches.values():
        labels.extend(get_all_labels(branch))  # Recurse through the branches
    return labels

def predict(root, X):
    return [_predictX_i(root,x) for x in X]

def parsetree(node: Node, encoder_dict):
    tabs = "\t" * node.depth  # Create the tab indentation based on the node depth
    # If it's a leaf node, print the decoded label
    if node.label is not None:
        label = [key for key, value in encoder_dict[list(encoder_dict)[-1]].items() if value == node.label][0]
        print(f"{tabs}leaf: {label}")
    else:
        # If it's not a leaf node, print the feature name using its index
        feature_index = node.feature
        # Get the feature name by finding the column in the encoder_dict
        feature_name = list(encoder_dict)[feature_index]
        print(f"{tabs}Feature: {feature_name}")
        
        # For each branch of the current node, recursively print the tree
        for b_value, branch in node.branches.items():
            # Decode the branch value back to the original value (category) of the feature
            decoded_value = [key for key, value in encoder_dict[feature_name].items() if value == b_value][0]
            print(f"{tabs}|   Value {decoded_value}:")
            parsetree(branch, encoder_dict)


