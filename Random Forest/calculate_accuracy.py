import numpy as np

class CalculateAccuracy:
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.classes = np.unique(self.y_true)
        self._calculate_confusion_matrix()

    def _calculate_confusion_matrix(self):
        # Assuming binary classification with classes 0 (negative), 1 (positive)
        self.TP = np.sum((self.y_true == 1) & (self.y_pred == 1))
        self.TN = np.sum((self.y_true == 0) & (self.y_pred == 0))
        self.FP = np.sum((self.y_true == 0) & (self.y_pred == 1))
        self.FN = np.sum((self.y_true == 1) & (self.y_pred == 0))
        # print(f"TP: {self.TP}, FP: {self.FP}, TN: {self.TN}, FN: {self.FN}")

    def print_confusion_values(self):
        print(f"TP: {self.TP}, FP: {self.FP}, TN: {self.TN}, FN: {self.FN}")

    def accuracy_percentage(self):
        correct = np.sum(self.y_true == self.y_pred)
        return round(correct / len(self.y_true) * 100, 2)

    def precision(self, average='micro'):
        if average == 'macro':
            precisions = []
            for c in self.classes:
                tp = np.sum((self.y_true == c) & (self.y_pred == c))
                fp = np.sum((self.y_true != c) & (self.y_pred == c))
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                precisions.append(precision)
            return np.mean(precisions)

        elif average == 'micro':
            tp = self.TP
            fp = self.FP
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0

        elif average == 'weighted':
            precisions = []
            weights = []
            for c in self.classes:
                tp = np.sum((self.y_true == c) & (self.y_pred == c))
                fp = np.sum((self.y_true != c) & (self.y_pred == c))
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                precisions.append(precision)
                weights.append(np.sum(self.y_true == c))
            return np.average(precisions, weights=weights)

        else:
            raise ValueError("Invalid average parameter")

    def recall(self, average='micro'):
        if average == 'macro':
            recalls = []
            for c in self.classes:
                tp = np.sum((self.y_true == c) & (self.y_pred == c))
                fn = np.sum((self.y_true == c) & (self.y_pred != c))
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                recalls.append(recall)
            return np.mean(recalls)

        elif average == 'micro':
            tp = self.TP
            fn = self.FN
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0

        elif average == 'weighted':
            recalls = []
            weights = []
            for c in self.classes:
                tp = np.sum((self.y_true == c) & (self.y_pred == c))
                fn = np.sum((self.y_true == c) & (self.y_pred != c))
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                recalls.append(recall)
                weights.append(np.sum(self.y_true == c))
            return np.average(recalls, weights=weights)

        else:
            raise ValueError("Invalid average parameter")

    def f1_score(self, average='micro', beta=1):
        precision = self.precision(average=average)
        recall = self.recall(average=average)
        if (precision + recall) == 0:
            return 0.0
        beta_squared = beta ** 2
        return ((1 + beta_squared) * precision * recall) / (beta_squared * precision + recall)
