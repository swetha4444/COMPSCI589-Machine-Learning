import numpy as np
from collections import defaultdict
from calculate_accuracy import CalculateAccuracy
import random

class CrossValidator:
    def __init__(self, model, n_splits=5, beta = 1):
        self.model = model
        self.n_splits = n_splits
        self.results = defaultdict(list)
        self.beta = beta

    # def stratified_k_fold(self, X, y):
    #     y = np.array(y)
    #     classes = np.unique(y)
    #     indices = np.arange(len(y))
    #     folds = [[[], []] for _ in range(self.n_splits)]

    #     for c in classes:
    #         class_indices = indices[y == c]
    #         n_class_samples = len(class_indices)
    #         fold_sizes = np.full(self.n_splits, n_class_samples // self.n_splits, dtype=int)
    #         fold_sizes[:n_class_samples % self.n_splits] += 1
    #         class_indices_splits = np.split(np.random.permutation(class_indices), np.cumsum(fold_sizes)[:-1])

    #         for i in range(self.n_splits):
    #             folds[i][0].extend(class_indices_splits[i])
    #             folds[i][1].extend(class_indices_splits[i])

    #     for i in range(self.n_splits):
    #         folds[i][0] = np.array(folds[i][0], dtype=int)
    #         folds[i][1] = np.array(folds[i][1], dtype=int)
    #     return folds

    # def stratified_k_fold(self, X, y):
    #     y = np.array(y)
    #     X = np.array(X)
        
    #     # Get unique classes and initialize fold structure
    #     classes = np.unique(y)
    #     folds = [[] for _ in range(self.n_splits)]  # Initialize folds as lists, not tuples
    #     class_indices = {c: np.where(y == c)[0] for c in classes}

    #     # Create fold counts dictionary
    #     fold_counts = {c: np.zeros(self.n_splits, dtype=int) for c in classes}
    #     total_samples = len(y)
    #     samples_per_fold = total_samples // self.n_splits
    #     remainder = total_samples % self.n_splits

    #     # Distribute samples proportionally across folds
    #     for i in range(self.n_splits):
    #         fold_size = samples_per_fold + (1 if i < remainder else 0)
    #         for c in classes:
    #             class_proportion = len(class_indices[c]) / total_samples
    #             n_samples_for_class = int(round(fold_size * class_proportion))
    #             fold_counts[c][i] = n_samples_for_class

    #     # Assign samples to folds while ensuring disjoint sets
    #     assigned_indices = {c: [] for c in classes}  # Keep track of assigned indices per class
    #     for i in range(self.n_splits):
    #         train_indices = []
    #         val_indices = []

    #         for c in classes:
    #             np.random.shuffle(class_indices[c])
    #             n_val = fold_counts[c][i]
    #             val_indices.extend(class_indices[c][:n_val])

    #             # Ensure training indices are disjoint (remove validation indices from class pool)
    #             remaining_class_indices = class_indices[c][n_val:]

    #             # Check if we have already assigned this index to a previous fold, if so skip
    #             remaining_class_indices = [idx for idx in remaining_class_indices if idx not in assigned_indices[c]]

    #             # Ensure train indices are disjoint by adding only non-repeated indices
    #             train_indices.extend(remaining_class_indices)

    #             # Mark these indices as assigned for this fold
    #             assigned_indices[c].extend(class_indices[c][:n_val])

    #         # Convert indices to integers explicitly to ensure they're valid
    #         folds[i] = [list(map(int, set(train_indices))), list(map(int, set(val_indices)))]  # Update folds correctly

    #     return folds

    def stratified_k_fold(self, X, y, random_state=42):
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)
            
        # Convert y to numpy array for convenience
        y = np.array(y)
        
        # Get unique classes and initialize fold structure
        classes = np.unique(y)
        folds = [[] for _ in range(self.n_splits)]  # Initialize folds as lists, not tuples
        class_indices = {c: np.where(y == c)[0] for c in classes}
        
        # Create fold counts dictionary
        fold_counts = {c: np.zeros(self.n_splits, dtype=int) for c in classes}
        total_samples = len(y)
        samples_per_fold = total_samples // self.n_splits
        remainder = total_samples % self.n_splits
        
        # Distribute samples proportionally across folds
        for i in range(self.n_splits):
            fold_size = samples_per_fold + (1 if i < remainder else 0)
            for c in classes:
                class_proportion = len(class_indices[c]) / total_samples
                n_samples_for_class = int(round(fold_size * class_proportion))
                fold_counts[c][i] = n_samples_for_class
        
        # Assign samples to folds while ensuring disjoint sets
        assigned_indices = {c: [] for c in classes}  # Keep track of assigned indices per class
        for i in range(self.n_splits):
            train_indices = []
            val_indices = []

            for c in classes:
                np.random.shuffle(class_indices[c])  # Shuffle the class indices
                n_val = fold_counts[c][i]
                val_indices.extend(class_indices[c][:n_val])
                remaining_class_indices = class_indices[c][n_val:]
                remaining_class_indices = [idx for idx in remaining_class_indices if idx not in assigned_indices[c]]
                train_indices.extend(remaining_class_indices)
                assigned_indices[c].extend(class_indices[c][:n_val])
            folds[i] = [list(map(int, set(train_indices))), list(map(int, set(val_indices)))]  # Update folds correctly

        return folds

    def run(self, X, y):
        folds = self.stratified_k_fold(X, y)

        for i in range(self.n_splits):
            train_idx = np.concatenate([folds[j][0] for j in range(self.n_splits) if j != i])
            test_idx = folds[i][1]

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            calc_acc = CalculateAccuracy(y_test, y_pred)
            self.results['accuracy'].append(calc_acc.accuracy_percentage())
            self.results['precision'].append(calc_acc.precision() * 100)
            self.results['recall'].append(calc_acc.recall() * 100)
            self.results['f1'].append(calc_acc.f1_score(beta=self.beta) * 100)

        return self.results

    def get_mean_metrics(self):
        mean_metrics = {}
        for metric, values in self.results.items():
            mean_metrics[metric] = np.mean(values)
        return mean_metrics

    def get_std_metrics(self):
        std_metrics = {}
        for metric, values in self.results.items():
            std_metrics[metric] = np.std(values)
        return std_metrics