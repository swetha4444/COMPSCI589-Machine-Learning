import numpy as np
import math
from decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None, min_info_gain=1e-5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
        self.min_info_gain = min_info_gain

    def fit(self, X, y):
        n_samples, n_total_features = X.shape
        if not self.n_features or self.n_features > n_total_features:
            self.n_features = int(math.sqrt(n_total_features))

        self.trees = []
        for _ in range(self.n_trees):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X[indices], y[indices]

            # Feature subsampling
            feat_indices = np.random.choice(n_total_features, self.n_features, replace=False)
            X_boot_subsampled = X_boot[:, feat_indices]

            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                min_info_gain=self.min_info_gain)
            tree.fit(X_boot_subsampled, y_boot)
            self.trees.append((tree, feat_indices))

    def predict(self, X):
        tree_preds = []
        for tree, feat_indices in self.trees:
            X_subsampled = X[:, feat_indices]
            tree_preds.append(tree.predict(X_subsampled))
        tree_preds = np.swapaxes(np.array(tree_preds), 0, 1) # (N_samples, N_trees)

        # Majority vote
        predictions = [np.argmax(np.bincount(tree_preds_sample.astype(int))) for tree_preds_sample in tree_preds]
        return np.array(predictions)