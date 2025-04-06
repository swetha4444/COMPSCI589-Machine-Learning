import numpy as np
from collections import defaultdict

class CrossValidator:
    def __init__(self, model, n_splits=5):
        self.model = model
        self.n_splits = n_splits
        self.results = defaultdict(list)

    def stratified_k_fold(self, X, y):
        y = np.array(y)
        classes = np.unique(y)
        indices = np.arange(len(y))
        folds = [[[], []] for _ in range(self.n_splits)]

        for c in classes:
            class_indices = indices[y == c]
            n_class_samples = len(class_indices)
            fold_sizes = np.full(self.n_splits, n_class_samples // self.n_splits, dtype=int)
            fold_sizes[:n_class_samples % self.n_splits] += 1
            class_indices_splits = np.split(np.random.permutation(class_indices), np.cumsum(fold_sizes)[:-1])

            for i in range(self.n_splits):
                folds[i][0].extend(class_indices_splits[i])
                folds[i][1].extend(class_indices_splits[i])

        for i in range(self.n_splits):
            folds[i][0] = np.array(folds[i][0], dtype=int)
            folds[i][1] = np.array(folds[i][1], dtype=int)
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

            from calculate_accuracy import CalculateAccuracy
            accuracy_calculator = CalculateAccuracy(y_test, y_pred)
            self.results['accuracy'].append(accuracy_calculator.accuracy_percentage())
            self.results['precision'].append(accuracy_calculator.precision() * 100) # Store as percentage
            self.results['recall'].append(accuracy_calculator.recall() * 100)       # Store as percentage
            self.results['f1'].append(accuracy_calculator.f1_score() * 100)         # Store as percentage
        return self.results

    def print_results(self):
        print("\n--- Cross-Validation Results ---")
        for metric, values in self.results.items():
            print(f"Mean {metric.capitalize()}: {np.mean(values):.2f}%")
            print(f"Std {metric.capitalize()}: {np.std(values):.2f}%")
        print("-----------------------------\n")

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