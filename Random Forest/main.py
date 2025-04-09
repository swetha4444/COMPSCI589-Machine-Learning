import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from itertools import product
from random_forest import RandomForest
from cross_validator import CrossValidator
from data_preprocessing import Preprocessor

def tune_parameters(X, y, param_grid, cv_splits=5):
    """
    Tune Random Forest parameters using grid search with cross-validation.
    Returns list of results for each parameter combination.
    """
    results = []
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                         for v in product(*param_grid.values())]
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    for params in param_combinations:
        model = RandomForest(**params)
        cv = CrossValidator(model, n_splits=cv_splits)
        cv_results = cv.run(X, y)
        
        results.append({
            'params': params,
            'mean_metrics': cv.get_mean_metrics(),
            'std_metrics': cv.get_std_metrics()
        })
    
    return results

def plot_parameter_tuning(results, param_name, dataset_name):
    """Plot parameter tuning results with enhanced visualization."""
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    param_values = sorted(list(set(r['params'][param_name] for r in results)))
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Show fixed parameters in title
    fixed_params = {k: v for k, v in results[0]['params'].items() if k != param_name}
    param_info = ', '.join([f"{k}={v}" for k, v in fixed_params.items()])
    
    fig.suptitle(f'Parameter Tuning: Impact of {param_name}\n'
                 f'Dataset: {dataset_name}\nFixed Parameters: {param_info}',
                 fontsize=8, y=0.95)
    
    colors =['#fedd51', '#8ab25e', '#ef78a8', '#f3a6a6', '#bb7641']
    for i, metric in enumerate(metrics):
        row, col = i // 2, i % 2
        means = []
        stds = []
        
        for value in param_values:
            relevant_results = [r for r in results if r['params'][param_name] == value]
            means.append(np.mean([r['mean_metrics'][metric] for r in relevant_results]))
            stds.append(np.mean([r['std_metrics'][metric] for r in relevant_results]))
        
        axes[row, col].errorbar(param_values, means, yerr=stds,
                              marker='o', markersize=8,
                              color=colors[i], linewidth=2,
                              capsize=5, capthick=2,
                              label=f'{metric.capitalize()} Score')
        
        # Add value labels
        for x, y, std in zip(param_values, means, stds):
            axes[row, col].annotate(f'{y:.2f}±{std:.2f}',
                                  (x, y),
                                  textcoords="offset points",
                                  xytext=(0,10),
                                  ha='center',
                                  fontsize=8)
        
        axes[row, col].set_xlabel(param_name, fontsize=6)
        axes[row, col].set_ylabel(f'{metric.capitalize()} Score', fontsize=6)
        axes[row, col].set_title(f'{metric.capitalize()}', fontsize=8, pad=20)
        axes[row, col].grid(True, linestyle='--', alpha=0.7)
        axes[row, col].legend(loc='lower right')
        axes[row, col].set_ylim(0, min(100, max(means) + max(stds) + 10))
    
    plt.tight_layout()
    plt.show()

def plot_final_results(metrics_data, dataset_name, params):
    """Plot final results for best parameters."""
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    param_info = (f"Best Parameters: max_depth={params['max_depth']}, "
                 f"min_samples_split={params['min_samples_split']}, "
                 f"min_info_gain={params['min_info_gain']}")
    
    fig.suptitle(f'Random Forest Performance with Best Parameters\n'
                 f'Dataset: {dataset_name}\n{param_info}',
                 fontsize=10, y=0.95)
    
    colors =['#fedd51', '#8ab25e', '#ef78a8', '#f3a6a6', '#bb7641']
    for i, metric in enumerate(metrics):
        row, col = i // 2, i % 2
        means = metrics_data[f'{metric}_mean']
        stds = metrics_data[f'{metric}_std']
        
        axes[row, col].errorbar(metrics_data['ntrees'], means, yerr=stds,
                              marker='o', markersize=8,
                              color=colors[i], linewidth=2,
                              capsize=5, capthick=2,
                              label=f'{metric.capitalize()} Score')
        
        for x, y, std in zip(metrics_data['ntrees'], means, stds):
            axes[row, col].annotate(f'{y:.2f}±{std:.2f}',
                                  (x, y),
                                  textcoords="offset points",
                                  xytext=(0,10),
                                  ha='center',
                                  fontsize=8)
        
        axes[row, col].set_xlabel('Number of Trees', fontsize=8)
        axes[row, col].set_ylabel(f'{metric.capitalize()} Score', fontsize=8)
        axes[row, col].set_title(f'{metric.capitalize()}', fontsize=9, pad=20)
        axes[row, col].grid(True, linestyle='--', alpha=0.7)
        axes[row, col].legend(loc='lower right')
        axes[row, col].set_xticks(metrics_data['ntrees'])
        axes[row, col].set_ylim(0, min(100, max(means) + max(stds) + 10)) 
    
    plt.tight_layout()
    plt.show()

def run_experiment(dataset_name,beta=1, accW=50, f1W=50):
    """Run complete experiment including parameter tuning and final evaluation."""
    # Load and preprocess data
    print("\n" + "="*80)
    print(f"RANDOM FOREST EXPERIMENT: {dataset_name.upper()}")
    print("="*80)
    
    df = pd.read_csv(f'Random Forest/{dataset_name}.csv')
    processor = Preprocessor(df)
    X, y = processor.preprocess()
    
    # Parameter tuning phase
    print("\nPhase 1: Parameter Tuning")
    print("="*80)
    
    param_grid = {
        'n_trees': [5],  # Fixed number of trees for tuning as we are plotting against trees at the end
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_info_gain': [0.001, 0.01, 0.1]
    }
    
    results = tune_parameters(X, y, param_grid)

    
    scored_configs = []
    for result in results:
        # Calculate metrics
        mean_metrics = result['mean_metrics']
        std_metrics = result['std_metrics']
        
        # Calculate stability (inverse of average std)
        stability = 1 / (1 + np.mean([
            std_metrics['accuracy'],
            std_metrics['precision'],
            std_metrics['recall'],
            std_metrics['f1']
        ]))
        
        # Calculate overall score (weighted average)
        score = (accW * mean_metrics['accuracy'] +
                f1W * mean_metrics['f1'])
        
        # Adjust score by stability
        final_score = 0.7 * score + 0.3 * stability
        
        scored_configs.append((final_score, result))
        
        # Print configuration details
        params_str = ", ".join(f"{k}={v}" for k, v in result['params'].items())
        print(f"{params_str:<50} "
              f"{mean_metrics['accuracy']:8.2f} "
              f"{mean_metrics['precision']:8.2f} "
              f"{mean_metrics['recall']:8.2f} "
              f"{mean_metrics['f1']:8.2f} "
              f"{stability:8.2f}")
    
    # Select best configuration
    best_result = max(scored_configs, key=lambda x: x[0])[1]
    best_params = best_result['params']
    
    print("\nSelected Best Parameters:")
    print("-"*40)
    for param, value in best_params.items():
        print(f"{param:20}: {value}")
    
    # Plot parameter tuning results
    for param in ['max_depth', 'min_samples_split', 'min_info_gain']:
        plot_parameter_tuning(results, param, dataset_name)
    
    # Final evaluation phase
    print("\nPhase 2: Final Evaluation with Best Parameters")
    print("="*80)
    
    ntree_values = [1, 5, 10, 20, 30, 40, 50]
    metrics_data = {
        'ntrees': ntree_values,
        'accuracy_mean': [], 'accuracy_std': [],
        'precision_mean': [], 'precision_std': [],
        'recall_mean': [], 'recall_std': [],
        'f1_mean': [], 'f1_std': []
    }
    
    print(f"\n{'Trees\t'} {'Accuracy\t'} {'Precision\t'} {'Recall\t\t'} {'F1\t'}")
    print("-"*80)
    
    for ntree in ntree_values:
        params = best_params.copy()
        params['n_trees'] = ntree
        model = RandomForest(**params)
        cv = CrossValidator(model, n_splits=5, beta=beta)
        cv_results = cv.run(X, y)
        
        mean_metrics = cv.get_mean_metrics()
        std_metrics = cv.get_std_metrics()
        
        # Store results
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            metrics_data[f'{metric}_mean'].append(mean_metrics[metric])
            metrics_data[f'{metric}_std'].append(std_metrics[metric])
        
        print(f"{ntree:6d} "
              f"{mean_metrics['accuracy']:6.2f}±{std_metrics['accuracy']:4.2f}    "
              f"{mean_metrics['precision']:6.2f}±{std_metrics['precision']:4.2f}    "
              f"{mean_metrics['recall']:6.2f}±{std_metrics['recall']:4.2f}    "
              f"{mean_metrics['f1']:6.2f}±{std_metrics['f1']:4.2f}")
    
    print("-"*80)
    plot_final_results(metrics_data, dataset_name, best_params)

def checkClassImbalance(dataset):
    df = pd.read_csv(f'Random Forest/{dataset}.csv')
    print(f"Class distribution in {dataset}:")
    class_distribution = df['label'].value_counts(normalize=True).reset_index()
    class_distribution.columns = ['Class', 'Proportion']
    print(class_distribution)

def main():
    # Loan Data set
    checkClassImbalance('loan')
    run_experiment('loan', beta=0.5, accW=30, f1W=70)

    # Raisin Data set
    checkClassImbalance('raisin')
    run_experiment('raisin', beta=1)

    # WDBC Data set
    checkClassImbalance('wdbc')
    run_experiment('wdbc', beta=2,  accW=20, f1W=80)

    # Titanic Data set
    checkClassImbalance('titanic')
    run_experiment('titanic', beta=1,  accW=40, f1W=60)

if __name__ == "__main__":
    main()