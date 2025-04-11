import pandas as pd
import numpy as np
import math
from statistics import multimode
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('Random Forest/titanic.csv')
features = list(data.columns[:-1])
target = data.columns[-1]

def bootstrap_sample(data):
    return data.sample(n=len(data), replace=True)

def entropy(labels):
    total = len(labels)
    counts = {}
    entropy_val = 0.0
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    for count in counts.values():
        probability = count / total
        entropy_val -= probability * math.log2(probability)
    return entropy_val

def info_gain(data, feat):
    y = data.columns[-1]
    total_entropy = entropy(data[y])
    if np.issubdtype(data[feat].dtype, np.number):
        sorted_values = sorted(data[feat].unique())
        if len(sorted_values) < 2:
            return 0, None
        split_points = [(sorted_values[i] + sorted_values[i+1])/2 for i in range(len(sorted_values)-1)]
        best_gain = -float('inf')
        best_thresh = None
        for threshold in split_points:
            left = data[data[feat] <= threshold]
            right = data[data[feat] > threshold]
            if len(left) == 0 or len(right) == 0:
                continue
            gain = total_entropy - (
                len(left)/len(data) * entropy(left[y]) +
                len(right)/len(data) * entropy(right[y])
            )
            if gain > best_gain:
                best_gain = gain
                best_thresh = threshold
        return best_gain, best_thresh
    else:
        gain = total_entropy
        values = data[feat].unique()
        for v in values:
            subset = data[data[feat] == v]
            gain -= len(subset)/len(data) * entropy(subset[y])
        return gain, None

def build_tree_random(data, features, m_features):
    target = data.columns[-1]
    if len(set(data[target])) == 1:
        return data[target].iloc[0]
    if len(features) == 0:
        return multimode(data[target])[0]

    best, best_gain, best_thresh = None, -float('inf'), None
    m = max(1, int(len(features)**0.5))
    m_features = min(m, len(features))  # Ensure m_features does not exceed total features
    chosen_feats = np.random.choice(features, m_features, replace=False)

    for feat in chosen_feats:
        gain, threshold = info_gain(data, feat)
        if gain > best_gain:
            best, best_gain, best_thresh = feat, gain, threshold

    if best is None:
        return multimode(data[target])[0]

    branches = {}
    new_features = [f for f in features if f != best]

    if best_thresh is not None:
        left = data[data[best] <= best_thresh]
        right = data[data[best] > best_thresh]
        branches[("<=", best_thresh)] = build_tree_random(left, new_features, m_features)
        branches[(">", best_thresh)] = build_tree_random(right, new_features, m_features)
    else:
        for val in data[best].unique():
            subset = data[data[best] == val]
            branches[val] = build_tree_random(subset, new_features, m_features)

    return (best, branches, multimode(data[target])[0], best_thresh)

def predict(tree, instance):
    if not isinstance(tree, tuple):
        return tree
    feature, branches, majority, threshold = tree
    val = instance[feature]
    if threshold is not None:
        branch_key = ("<=", threshold) if val <= threshold else (">", threshold)
    else:
        branch_key = val
    return predict(branches.get(branch_key, majority), instance)

def predict_forest(forest, instance):
    return multimode([predict(tree, instance) for tree in forest])[0]

def train_random_forest(data, features, ntree):
    m = max(1, int(len(features)**0.5))
    return [build_tree_random(bootstrap_sample(data), features, m) for _ in range(ntree)]

def stratified_k_fold_split(data, target_col, k=5, seed=42):
    np.random.seed(seed)
    folds = [[] for _ in range(k)]
    class_indices = {label: [] for label in data[target_col].unique()}
    for idx, label in enumerate(data[target_col]):
        class_indices[label].append(idx)
    for indices in class_indices.values():
        np.random.shuffle(indices)
        for i, idx in enumerate(indices):
            folds[i % k].append(idx)
    stratified_folds = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = [idx for j in range(k) if j != i for idx in folds[j]]
        stratified_folds.append((
            data.iloc[train_idx].reset_index(drop=True),
            data.iloc[test_idx].reset_index(drop=True)
        ))
    return stratified_folds

def evaluate_random_forest_kfold(data, features, target, ntree, k=5):
    folds = stratified_k_fold_split(data, target, k)
    a_l, p_l, r_l, f1_l = [], [], [], []

    for train, test in folds:
        forest = train_random_forest(train, features, ntree)
        y_true = test[target].tolist()
        y_pred = [predict_forest(forest, row.drop(target).to_dict()) for _, row in test.iterrows()]

        labels = list(set(y_true + y_pred))
        labels.sort()
        positive_label = labels[1] if len(labels) == 2 else labels[0]

        TP = sum((yt == yp == positive_label) for yt, yp in zip(y_true, y_pred))
        TN = sum((yt == yp != positive_label) for yt, yp in zip(y_true, y_pred))
        FP = sum((yt != positive_label and yp == positive_label) for yt, yp in zip(y_true, y_pred))
        FN = sum((yt == positive_label and yp != positive_label) for yt, yp in zip(y_true, y_pred))

        acc = (TP + TN) / len(y_true) if len(y_true) else 0
        prec = TP / (TP + FP) if TP + FP else 0
        rec = TP / (TP + FN) if TP + FN else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0

        a_l.append(acc)
        p_l.append(prec)
        r_l.append(rec)
        f1_l.append(f1)

    return {
        "accuracy": (np.mean(a_l), np.std(a_l)),
        "precision": (np.mean(p_l), np.std(p_l)),
        "recall": (np.mean(r_l), np.std(r_l)),
        "f1": (np.mean(f1_l), np.std(f1_l))
    }

# 🧪 Run evaluation and plot
ntree_values = [1, 5, 10, 20, 30, 40, 50]
results = {metric: [] for metric in ["accuracy", "precision", "recall", "f1"]}

print("{:<8} {:<10} {:<10} {:<10} {:<10}".format("ntree", "accuracy", "precision", "recall", "f1"))
for ntree in ntree_values:
    scores = evaluate_random_forest_kfold(data, features, target, ntree, k=5)
    print("{:<8} {:.4f}     {:.4f}     {:.4f}     {:.4f}".format(
        ntree, scores["accuracy"][0], scores["precision"][0], scores["recall"][0], scores["f1"][0]
    ))
    for metric in results:
        results[metric].append(scores[metric][0])

# 📈 Plotting
plt.figure(figsize=(12, 6))
for metric in results:
    plt.plot(ntree_values, results[metric], marker='o', label=metric.capitalize())
plt.title("Random Forest Evaluation Metrics (WDBC Dataset)")
plt.xlabel("Number of Trees")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
