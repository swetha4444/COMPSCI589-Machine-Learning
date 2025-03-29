import numpy as np
import pandas as pd

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None, is_categorical=False):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label
        self.is_categorical = is_categorical


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=10, min_info_gain=1e-5):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.current_depth = 0
        self.min_info_gain = min_info_gain

    def fit(self, X, y):
        self.root = self._grow_tree(X, y, self.current_depth)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria
        if (n_labels == 1 or n_samples < self.min_samples_split or depth >= self.max_depth):
            most_common_label = np.argmax(np.bincount(y.astype(int)))
            return Node(label=most_common_label)

        best_gain = -1
        best_feature = None
        best_threshold = None
        best_left_idx = None
        best_right_idx = None
        best_is_categorical = False

        for feature_idx in range(n_features):
            X_column = X[:, feature_idx]
            is_categorical = pd.api.types.is_object_dtype(X_column)
            if is_categorical:
                thresholds = np.unique(X_column)
            else:
                thresholds = np.mean(X_column)  # Using average for numerical threshold

            gain = self._information_gain(y, X_column, thresholds, is_categorical)

            if gain > best_gain and gain >= self.min_info_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = thresholds
                best_is_categorical = is_categorical

        if best_gain == -1:  # No suitable split found
            most_common_label = np.argmax(np.bincount(y.astype(int)))
            return Node(label=most_common_label)

        left_idx, right_idx = self._split(X[:, best_feature], best_threshold, best_is_categorical)
        left = self._grow_tree(X[left_idx, :], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feature, best_threshold, left, right, is_categorical=best_is_categorical)

    def _information_gain(self, y, X_column, threshold, is_categorical):
        parent_entropy = self._entropy(y)
        left_idx, right_idx = self._split(X_column, threshold, is_categorical)
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0
        n = len(y)
        n_l, n_r = len(left_idx), len(right_idx)
        e_l, e_r = self._entropy(y[left_idx]), self._entropy(y[right_idx])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        return parent_entropy - child_entropy

    def _split(self, X_column, threshold, is_categorical):
        if is_categorical:
            left_idx = np.where(X_column == threshold)[0]
            right_idx = np.where(X_column != threshold)[0]
        else:
            left_idx = np.where(X_column <= threshold)[0]
            right_idx = np.where(X_column > threshold)[0]
        return left_idx, right_idx

    def _entropy(self, y):
        hist = np.bincount(y.astype(int))
        ps = hist / np.sum(hist)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _traverse_tree(self, x, node):
        if node.label is not None:
            return node.label
        if node.is_categorical:
            if x[node.feature] == node.threshold:
                return self._traverse_tree(x, node.left)
            return self._traverse_tree(x, node.right)
        else:
            if x[node.feature] <= node.threshold:
                return self._traverse_tree(x, node.left)
            return self._traverse_tree(x, node.right)