import multiprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_bootstrap_sample(data):
    return data.sample(n=len(data), replace=True)

def _get_entropy(y):
    d = len(y)
    if d == 0:
        return 0
    _, y_mapped = np.unique(y, return_inverse=True)
    counts = np.bincount(y_mapped)
    probs = counts / d
    return -np.sum(probs * np.log2(probs))

def _get_info_gain(data, split_attribute):
    label_col = data.columns[-1]  # <-- Fixed to use last column as label
    dataset_entropy = _get_entropy(data[label_col])
    if pd.api.types.is_numeric_dtype(data[split_attribute]):
        return _get_numeric_info_gain(data, split_attribute, label_col, dataset_entropy)
    else:
        info_gain = dataset_entropy
        splits = data[split_attribute].unique()
        for split in splits:
            split_subset = data[data[split_attribute] == split]
            info_gain -= len(split_subset) / len(data) * _get_entropy(split_subset[label_col])
        return info_gain, None

def _get_numeric_info_gain(data, split_attribute, label_col, dataset_entropy):
    sorted_thresh = np.sort(data[split_attribute].unique())
    if len(sorted_thresh) < 2:
        return 0, None
    splits = (sorted_thresh[:-1] + sorted_thresh[1:]) / 2
    best_gain, threshold = -float('inf'), None
    for split in splits:
        left = data[data[split_attribute] <= split]
        right = data[data[split_attribute] > split]
        if left.empty or right.empty:
            continue
        gain = dataset_entropy - (
            (len(left) / len(data)) * _get_entropy(left[label_col]) +
            (len(right) / len(data)) * _get_entropy(right[label_col])
        )
        if gain > best_gain:
            best_gain = gain
            threshold = split
    return best_gain, threshold

def create_decision_tree(data, attributes, M):
    label_col = data.columns[-1]
    if data[label_col].nunique() == 1:
        return data[label_col].iloc[0]
    if not attributes:
        return data[label_col].mode()[0]
    m = min(M, len(attributes))
    attributes = np.random.choice(attributes, m, replace=False)
    best_gain, best_split, best_threshold = -float('inf'), None, None
    for attr in attributes:
        gain, threshold = _get_info_gain(data, attr)
        if gain > best_gain:
            best_gain = gain
            best_split = attr
            best_threshold = threshold
    if best_gain == 0 or best_split is None:
        return data[label_col].mode()[0]
    branches = {}
    new_attributes = [attr for attr in attributes if attr != best_split]
    if best_threshold is not None:
        left = data[data[best_split] <= best_threshold]
        right = data[data[best_split] > best_threshold]
        if left.empty or right.empty:
            return data[label_col].mode()[0]
        branches[("le", best_threshold)] = create_decision_tree(left, new_attributes, M)
        branches[("gt", best_threshold)] = create_decision_tree(right, new_attributes, M)
    else:
        for val in data[best_split].unique():
            subset = data[data[best_split] == val]
            branches[val] = create_decision_tree(subset, new_attributes, M)
    return (best_split, branches, data[label_col].mode()[0], best_threshold)

def cross_validation_stratified(data, k=5, seed=9):
    np.random.seed(seed)
    data = data.sample(frac=1).reset_index(drop=True)
    label_col = data.columns[-1]
    target = data[label_col]
    folds = [[] for _ in range(k)]
    for label in target.unique():
        idx = list(data[target == label].index)
        np.random.shuffle(idx)
        for i, index in enumerate(idx):
            folds[i % k].append(index)
    return [(data.iloc[sum(folds[:i] + folds[i+1:], [])].reset_index(drop=True),
             data.iloc[folds[i]].reset_index(drop=True)) for i in range(k)]

def tree_prediction(tree, row):
    while isinstance(tree, tuple):
        attr, branches, default, threshold = tree
        val = row[attr]
        if threshold is not None:
            tree = branches.get(("le", threshold) if val <= threshold else ("gt", threshold), default)
        else:
            tree = branches.get(val, default)
    return tree

def fit_random_forest(data, attributes, ntree):
    M = max(1, int(np.sqrt(len(attributes))))
    return [create_decision_tree(get_bootstrap_sample(data), attributes, M) for _ in range(ntree)]

def predict_random_forest(forest, row):
    votes = [tree_prediction(tree, row) for tree in forest]
    return pd.Series(votes).mode()[0]

def calculate_metrics(y_true, y_pred):
    tp, fp, fn, tn = 0, 0, 0, 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            if true == 1:
                tp += 1
            else:
                tn += 1
        else:
            if true == 1:
                fn += 1
            else:
                fp += 1
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return accuracy, precision, recall, f1

def plot_metrics(results, dataset_name):
    metrics = ["accuracy", "precision", "recall", "f1"]
    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        plt.errorbar(results["nTree"], results[metric], yerr=np.std(results[metric]), fmt='o')

        plt.plot(results["nTree"], results[metric], marker='o')
        plt.title(f"{metric.capitalize()} vs nTree")
        plt.xlabel("Number of Trees")
        plt.ylabel(metric.capitalize())
        plt.ylim(0, 1)
        plt.legend([metric])
        plt.grid(True)
    plt.tight_layout()
    plt.suptitle(f"Random Forest Performance on {dataset_name}", fontsize=16, y=1.02)
    plt.savefig(f"{dataset_name}_random_forest_metrics.png")
    plt.show()

def run_random_forest(filename, nTrees=[1, 5, 10, 20, 30, 40, 50], k=5):
    print(f"Running Random Forest on {filename}")
    data = pd.read_csv(filename)
    label_col = data.columns[-1] 
    results = {'nTree': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    for ntree in nTrees:
        accs, precs, recs, f1s = [], [], [], []
        for train_df, test_df in cross_validation_stratified(data, k):
            features = list(train_df.columns[:-1])  # everything except the last column
            forest = fit_random_forest(train_df, attributes=features, ntree=ntree)
            y_true = test_df[label_col].tolist()
            y_pred = [predict_random_forest(forest, row) for _, row in test_df.iterrows()]
            acc, prec, rec, f1 = calculate_metrics(y_true, y_pred)
            accs.append(acc)
            precs.append(prec)
            recs.append(rec)
            f1s.append(f1)
        results['nTree'].append(ntree)
        results['accuracy'].append(np.mean(accs))
        results['precision'].append(np.mean(precs))
        results['recall'].append(np.mean(recs))
        results['f1'].append(np.mean(f1s))
        print(f"Trees: {ntree}, Accuracy: {np.mean(accs):.4f}, Precision: {np.mean(precs):.4f}, Recall: {np.mean(recs):.4f}, F1: {np.mean(f1s):.4f}")
    plot_metrics(results, dataset_name=filename.split("/")[-1].replace(".csv", ""))

# Example run
if __name__ == "__main__":
    # Use multi procesisg library to run datatsets in paralel
    with multiprocessing.Pool(processes=2) as pool:
        pool.apply_async(run_random_forest, ("Random Forest/wdbc.csv",))
        pool.apply_async(run_random_forest, ("Random Forest/raisin.csv",))
        pool.apply_async(run_random_forest, ("Random Forest/titanic.csv",))
        pool.apply_async(run_random_forest, ("Random Forest/loan.csv",))  # Make sure this CSV file has label as the LAST column
        pool.close()
        pool.join()
    # run_random_forest("Random Forest/raisin.csv")  # Make sure this CSV file has label as the LAST column