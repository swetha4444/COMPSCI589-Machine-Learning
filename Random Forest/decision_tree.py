import numpy as np
import pandas as pd
import math
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, branches=None, label=None, is_categorical=None, depth=0):
        self.feature = feature
        self.threshold = threshold  # Only used for numerical splits
        self.branches = branches if branches is not None else {}  # Dictionary for children
        self.label = label  # Label if leaf node (generic type)
        self.is_categorical = is_categorical  # Type of feature split
        self.depth = depth  # Store depth mainly for potential pruning or visualization

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=10, min_info_gain=1e-7, numerical_split_method='average'):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_info_gain = min_info_gain
        self.numerical_split_method = numerical_split_method

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        if X.size == 0 or y.size ==0:
          raise ValueError("X and Y cannot be empty")
        self.root = self._grow_tree(X, y, depth=0)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _is_feature_categorical(self, X_column):
        if pd.api.types.is_object_dtype(X_column) or pd.api.types.is_string_dtype(X_column):
            try:
                X_column.astype(float)
                return False
            except ValueError:
                return True
        if pd.api.types.is_categorical_dtype(X_column):
            return True
        return not pd.api.types.is_numeric_dtype(X_column)

    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or
                n_labels == 1 or
                n_samples < self.min_samples_split or n_features == 0):
            most_common_label = self._most_common_label(y)
            return Node(label=most_common_label, depth=depth)

        best_gain = -1.0
        best_criteria = None
        best_feature_idx = None
        best_is_categorical = None

        for feature_idx in range(n_features):
            X_column = X[:, feature_idx]
            is_categorical = self._is_feature_categorical(X_column)

            if is_categorical:
                unique_values = np.unique(X_column)
                if len(unique_values) <= 1:
                    continue
                gain = self._information_gain(y, X_column, unique_values, is_categorical=True)
                criteria = unique_values
            else:
                if self.numerical_split_method == 'average':
                    X_column_numeric = X_column.astype(float)
                    if len(np.unique(X_column_numeric)) <= 1:
                        continue
                    threshold = np.mean(X_column_numeric)
                    gain = self._information_gain(y, X_column_numeric, threshold, is_categorical=False)
                    criteria = threshold
                else:
                    raise ValueError(f"Invalid numerical_split_method: {self.numerical_split_method}")

            if gain > best_gain:
                best_gain = gain
                best_criteria = criteria
                best_feature_idx = feature_idx
                best_is_categorical = is_categorical

        if best_gain < self.min_info_gain:
            most_common_label = self._most_common_label(y)
            return Node(label=most_common_label, depth=depth)

        branches = {}
        node_threshold = None

        if best_is_categorical:
            unique_values = best_criteria
            for value in unique_values:
                idxs = np.where(X[:, best_feature_idx] == value)[0]
                if len(idxs) == 0:
                    branches[value] = Node(label=self._most_common_label(y), depth=depth + 1)
                else:
                    branches[value] = self._grow_tree(X[idxs, :], y[idxs], depth + 1)
        else:
            threshold = best_criteria
            node_threshold = threshold
            X_column_numeric = X[:, best_feature_idx].astype(float)

            left_idxs = np.where(X_column_numeric <= threshold)[0]
            if len(left_idxs) == 0:
                branches[0] = Node(label=self._most_common_label(y), depth=depth + 1)
            else:
                branches[0] = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)

            right_idxs = np.where(X_column_numeric > threshold)[0]
            if len(right_idxs) == 0:
                branches[1] = Node(label=self._most_common_label(y), depth=depth + 1)
            else:
                branches[1] = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(feature=best_feature_idx, threshold=node_threshold, branches=branches, is_categorical=best_is_categorical, depth=depth)

    def _information_gain(self, y, X_column, split_criteria, is_categorical):
        parent_entropy = self._entropy(y)
        n = len(y)
        weighted_child_entropy = 0.0

        if is_categorical:
            unique_values = split_criteria
            for value in unique_values:
                idxs = np.where(X_column == value)[0]
                if len(idxs) > 0:
                    weight = len(idxs) / n
                    weighted_child_entropy += weight * self._entropy(y[idxs])
        else:
            threshold = split_criteria
            left_idxs = np.where(X_column <= threshold)[0]
            right_idxs = np.where(X_column > threshold)[0]

            if len(left_idxs) > 0:
                weight_left = len(left_idxs) / n
                weighted_child_entropy += weight_left * self._entropy(y[left_idxs])

            if len(right_idxs) > 0:
                weight_right = len(right_idxs) / n
                weighted_child_entropy += weight_right * self._entropy(y[right_idxs])

        information_gain = parent_entropy - weighted_child_entropy
        return information_gain

    def _entropy(self, y):
        if len(y) == 0:
            return 0
        hist = np.bincount(y) #generic type
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        if len(y) == 0:
            return None
        counter = Counter(y)
        if not counter:
            return None
        return counter.most_common(1)[0][0]

    def _traverse_tree(self, x, node):
        if node.label is not None:
            return node.label

        feature_value = x[node.feature]

        if node.is_categorical:
            if feature_value in node.branches:
                return self._traverse_tree(x, node.branches[feature_value])
            else:
                return self._get_subtree_majority_label(node)
        else:
            try:
                val_numeric = float(feature_value)
            except ValueError:
                return self._get_subtree_majority_label(node)

            if val_numeric <= node.threshold:
                branch_key = 0
            else:
                branch_key = 1

            if branch_key in node.branches:
                return self._traverse_tree(x, node.branches[branch_key])
            else:
                return self._get_subtree_majority_label(node)

    def _get_subtree_majority_label(self, node):
        labels = []
        queue = [node]

        while queue:
            current_node = queue.pop(0)
            if current_node.label is not None:
                labels.append(current_node.label)
            else:
                for child_node in current_node.branches.values():
                    queue.append(child_node)

        return self._most_common_label(np.array(labels)) if labels else None