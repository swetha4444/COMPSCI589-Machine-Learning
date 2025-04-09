import numpy as np

class CalculateAccuracy:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def accuracy_percentage(self):
        correct = np.sum(self.y_true == self.y_pred)
        return round(correct / len(self.y_true) * 100, 2)

    def precision(self, average='micro'):
        classes = np.unique(self.y_true)
        if average == 'macro':
            precisions = []
            for c in classes:
                tp = np.sum((self.y_true == c) & (self.y_pred == c))
                fp = np.sum((self.y_true != c) & (self.y_pred == c))
                if (tp + fp) == 0:
                    precision = 0.0
                else:
                    precision = tp / (tp + fp)
                precisions.append(precision)
            return np.mean(precisions)
        elif average == 'micro':
            tp = np.sum(self.y_true == self.y_pred)
            return tp / len(self.y_true)
        elif average == 'weighted':
            precisions = []
            weights = []
            for c in classes:
                tp = np.sum((self.y_true == c) & (self.y_pred == c))
                fp = np.sum((self.y_true != c) & (self.y_pred == c))
                if (tp + fp) == 0:
                    precision = 0.0
                else:
                    precision = tp / (tp + fp)
                precisions.append(precision)
                weights.append(np.sum(self.y_true == c))
            return np.average(precisions, weights=weights)
        else:
            raise ValueError("Invalid average parameter")

    def recall(self, average='micro'):
        classes = np.unique(self.y_true)
        if average == 'macro':
            recalls = []
            for c in classes:
                tp = np.sum((self.y_true == c) & (self.y_pred == c))
                fn = np.sum((self.y_true == c) & (self.y_pred != c))
                if (tp + fn) == 0:
                    recall = 0.0
                else:
                    recall = tp / (tp + fn)
                recalls.append(recall)
            return np.mean(recalls)
        elif average == 'micro':
            tp = np.sum(self.y_true == self.y_pred)
            return tp / len(self.y_true)
        elif average == 'weighted':
            recalls = []
            weights = []
            for c in classes:
                tp = np.sum((self.y_true == c) & (self.y_pred == c))
                fn = np.sum((self.y_true == c) & (self.y_pred != c))
                if (tp + fn) == 0:
                    recall = 0.0
                else:
                    recall = tp / (tp + fn)
                recalls.append(recall)
                weights.append(np.sum(self.y_true == c))
            return np.average(recalls, weights=weights)
        else:
            raise ValueError("Invalid average parameter")

    def f1_score(self, average='micro',beta = 1):
        precision = self.precision(average=average)
        recall = self.recall(average=average)
        
        if (precision + recall) == 0:
            return 0.0
            
        # Calculate F-beta score
        beta_squared = beta ** 2
        return ((1 + beta_squared) * precision * recall) / (beta_squared * precision + recall)