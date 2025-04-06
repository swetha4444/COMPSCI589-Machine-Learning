import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from random_forest import RandomForest
from cross_validator import CrossValidator
from data_preprocessing import Preprocessor
import numpy as np
import itertools

def run_experiment(df, ntree_values, filename):
    processor = Preprocessor(df)
    processor.preprocess()
    X, y = processor.X, processor.y

    print_data_info(df, filename)

    all_mean_metrics = defaultdict(list)
    all_std_metrics = defaultdict(list) # Add this line

    for ntree in ntree_values:
        model = RandomForest(n_trees=ntree, max_depth=10, min_samples_split=2, min_info_gain=0.001)
        cv = CrossValidator(model, n_splits=5)
        results = cv.run(X, y)
        cv.print_results()

        mean_metrics = cv.get_mean_metrics()
        std_metrics = cv.get_std_metrics() # Add this line

        for metric, value in mean_metrics.items():
            all_mean_metrics[metric].append(value)
        for metric, value in std_metrics.items(): # Add this loop
            all_std_metrics[metric].append(value)

    plot_metrics_vs_ntree(all_mean_metrics, all_std_metrics, ntree_values, filename) #Modify this line


def print_data_info(df, filename):
    print(f"\n--- Data Info for {filename} ---")
    print(df.head())
    print("\nData Shape:", df.shape)
    print("\nColumn Info:")
    df.info(verbose=False)
    print("\nLabel Value Counts:")
    print(df[df.columns[-1]].value_counts())
    print("\n---------------------------\n")


def plot_metrics_vs_ntree(all_mean_metrics, all_std_metrics, ntree_values, filename): #Modify this line
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Metrics vs. Number of Trees for {filename}', fontsize=16)

    for i, metric in enumerate(metrics):
        row = i // 2
        col = i % 2
        axes[row, col].plot(ntree_values, all_mean_metrics[metric], marker='o', label='Mean') # modify this line
        axes[row, col].fill_between(ntree_values, np.array(all_mean_metrics[metric]) - np.array(all_std_metrics[metric]), np.array(all_mean_metrics[metric]) + np.array(all_std_metrics[metric]), alpha=0.3, label='Std Dev') #Add this line.
        axes[row, col].set_xlabel('Number of Trees (ntree)')
        axes[row, col].set_ylabel(metric)
        axes[row, col].set_xticks(ntree_values)
        axes[row, col].set_ylim(0,100)
        axes[row, col].grid(True)
        axes[row, col].set_title(metric)
        axes[row, col].legend() #add this line

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def tune_and_plot_ntree_impact(df, filename):
    """
    Tunes max_depth and min_samples_split, and plots metrics vs. n_trees for each combination.

    Args:
        df (pd.DataFrame): The input DataFrame.
        filename (str): Name of the dataset.
    """

    processor = Preprocessor(df)
    processor.preprocess()
    X, y = processor.X, processor.y

    ntree_values = [10, 20, 30, 40, 50]  # Vary n_trees
    max_depth_values = [5, 10]
    min_samples_split_values = [2, 5]

    param_combinations = list(itertools.product(max_depth_values, min_samples_split_values))
    num_combinations = len(param_combinations)

    fig, axes = plt.subplots(num_combinations, 4, figsize=(16, 4 * num_combinations), sharex='col')
    fig.suptitle(f'n_trees Tuning for {filename}', fontsize=16)

    metrics = ['accuracy', 'precision', 'recall', 'f1']

    for i, (max_depth, min_samples_split) in enumerate(param_combinations):
        for j, metric in enumerate(metrics):
            metric_values = []
            std_values = [] #add this line.
            for n_trees in ntree_values:
                model = RandomForest(n_trees=n_trees, max_depth=max_depth, min_samples_split=min_samples_split)
                cv = CrossValidator(model, n_splits=3)  # Reduced folds for speed
                results = cv.run(X, y)
                mean_metrics = cv.get_mean_metrics()
                std_metrics = cv.get_std_metrics() #add this line.
                metric_values.append(mean_metrics[metric])
                std_values.append(std_metrics[metric]) #add this line.

            axes[i, j].plot(ntree_values, metric_values, marker='o', label='Mean') #modify this line.
            axes[i, j].fill_between(ntree_values, np.array(metric_values) - np.array(std_values), np.array(metric_values) + np.array(std_values), alpha=0.3, label='Std Dev') #add this line.
            axes[i, j].set_xlabel('n_trees')
            axes[i, j].set_ylabel(metric)
            axes[i, j].set_xticks(ntree_values)
            axes[i, j].set_ylim(0,100)
            axes[i, j].grid(True)
            if i == 0:
                axes[i, j].set_title(metric)
            if j == 0:
                axes[i, j].set_ylabel(f'depth={max_depth}\nsplit={min_samples_split}')
            axes[i,j].legend() #add this line.

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main():
    # Load Datasets
    # wdbc_df = pd.read_csv('Random Forest/wdbc.csv')
    loan_df = pd.read_csv('Random Forest/loan.csv')
    # titanic_df = pd.read_csv('Random Forest/titanic.csv')
    # raisin_df = pd.read_csv('Random Forest/raisin.csv')

    # Define ntree values to test
    ntree_values = [1, 5, 10, 20, 30, 40, 50]

    # Run experiments
    # run_experiment(wdbc_df, ntree_values, 'WDBC')
    run_experiment(loan_df, ntree_values, 'Loan')
    # run_experiment(titanic_df, ntree_values, 'Titanic')
    # run_experiment(raisin_df, ntree_values, 'Raisin')

    # Tune n_trees and plot
    # tune_and_plot_ntree_impact(wdbc_df, 'WDBC')
    # tune_and_plot_ntree_impact(loan_df, 'Loan')
    # tune_and_plot_ntree_impact(titanic_df, 'Titanic')
    # tune_and_plot_ntree_impact(raisin_df, 'Raisin')


if __name__ == "__main__":
    main()