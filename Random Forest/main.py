import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from random_forest import RandomForest
from cross_validator import CrossValidator
from data_preprocessing import Preprocessor
import numpy as np

def run_experiment(df, ntree_values, filename):
    # Initialize preprocessor with the dataframe
    processor = Preprocessor(df)
    # Get preprocessed data with one-hot encoding
    X, y = processor.preprocess()
    
    print_data_info(df, filename)
    print("\nOne-hot encoded features:", processor.encoded_columns)
    print("Final feature shape:", X.shape)

    all_mean_metrics = defaultdict(list)
    all_std_metrics = defaultdict(list)

    for ntree in ntree_values:
        # Use the same parameters as before
        model = RandomForest(n_trees=ntree, max_depth=10, min_samples_split=5, min_info_gain=0.01)
        cv = CrossValidator(model, n_splits=5)
        results = cv.run(X, y)
        cv.print_results()

        mean_metrics = cv.get_mean_metrics()
        std_metrics = cv.get_std_metrics()

        for metric, value in mean_metrics.items():
            all_mean_metrics[metric].append(value)
        for metric, value in std_metrics.items():
            all_std_metrics[metric].append(value)

    plot_metrics_vs_ntree(all_mean_metrics, all_std_metrics, ntree_values, filename)


def print_data_info(df, filename):
    print(f"\n--- Data Info for {filename} ---")
    print(df.head())
    print("\nData Shape:", df.shape)
    print("\nColumn Info:")
    df.info(verbose=False)
    print("\nLabel Value Counts:")
    print(df[df.columns[-1]].value_counts())
    print("\n---------------------------\n")


def plot_metrics_vs_ntree(all_mean_metrics, all_std_metrics, ntree_values, filename):
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Metrics vs. Number of Trees for {filename}', fontsize=16)

    for i, metric in enumerate(metrics):
        row = i // 2
        col = i % 2
        axes[row, col].plot(ntree_values, all_mean_metrics[metric], marker='o', label='Mean')
        axes[row, col].fill_between(ntree_values, np.array(all_mean_metrics[metric]) - np.array(all_std_metrics[metric]), np.array(all_mean_metrics[metric]) + np.array(all_std_metrics[metric]), alpha=0.3, label='Std Dev')
        axes[row, col].set_xlabel('Number of Trees (ntree)')
        axes[row, col].set_ylabel(metric)
        axes[row, col].set_xticks(ntree_values)
        axes[row, col].set_ylim(0,100)
        axes[row, col].grid(True)
        axes[row, col].set_title(metric)
        axes[row, col].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main():
    df = pd.read_csv('Random Forest/titanic.csv') #make sure that this is the same dataset as in the second code.
    ntree_values = [1, 5, 10, 20, 30, 40, 50]
    run_experiment(df, ntree_values, 'loan') #make sure that the filename is correct.
    #tune_and_plot_ntree_impact(titanic_df, 'titanic') #remove or comment this line.
if __name__ == "__main__":
    main()