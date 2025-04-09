import numpy as np
from collections import defaultdict
from calculate_accuracy import CalculateAccuracy

class CrossValidator:
    def __init__(self, model, n_splits=5, beta = 1):
        self.model = model
        self.n_splits = n_splits
        self.results = defaultdict(list)
        self.beta = beta

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