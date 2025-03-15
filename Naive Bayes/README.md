# Sentiment Analysis using Multinomial Naive Bayes

## Project Structure
```
Naive Bayes/
├── naive_bayes.py         # Core Naive Bayes implementation
├── train_class.py         # Training class object implementation
├── model_sampler.py       # Model sampling and evaluation utilities
├── computation.py         # Helper functions for computations
├── calculate_accuracy.py  # Accuracy and evaluation metrics calculation utilities
├── main.py                # Main experiment runner
└── starter_code/
    ├── utils.py           # Data loading and preprocessing utilities
    ├── train-positive.csv # Positive review data
    └── train-negative.csv # Negative review data
```

## Usage

Run experiments using the command flags in main.py:
```python
runCommand = [True,False,False,False,False]  # Enable/disable experiments
plot_saturation_graph = False  # Enable learning curve plot
```
Each value in the runCommand list is to run the code for the respective question (Q1,Q2,Q3,Q4,Q6). The plotSaturationGraph is the variable set to see how the accuracy saturates after a certain percentage of training data.

1. **Baseline Model (Q1)**
   - 20% training data, 20% test data
   - No Laplace smoothing
   - Basic Multinomial Naive Bayes

2. **Laplace Smoothing Analysis (Q2)**
   - 20% training data, 20% test data
   - Laplace factors: [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
   - Log probability implementation

3. **Full Dataset Analysis (Q3)**
   - 100% training data, 100% test data
   - Optimal Laplace factor (α=1)
   - Log probability implementation

4. **Reduced Training Set (Q4)**
   - 30% training data, 100% test data
   - Optimal Laplace factor
   - Log probability implementation

5. **Imbalanced Training Set (Q6)**
   - 10% positive, 50% negative training data
   - 100% test data
   - Optimal Laplace factor

## Using the NaiveBayesSampler

The `NaiveBayesSampler` class handles model training, evaluation, and visualization.
### Parameters

- `labels`: List of class labels (e.g., ["positive", "negative"])
- `title`: Title for plots and output
- `laplaceRange`: List of Laplace smoothing factors to test
- `logProb`: Whether to use log probabilities (recommended for numerical stability)
- `plotCM`: Whether to plot confusion matrices

```python
# Create a sampler instance
sampler = NaiveBayesSampler(
    labels=["positive", "negative"],
    title="Standard MNB with 20% train and 20% test",
    laplaceRange=[0.0001, 0.001, 0.01, 0.1, 1, 10],
    logProb=True,
    plotCM=True
)

# Train and evaluate the model
sampler.sampler(
    trainData=trainData,  # Dictionary mapping class labels to training documents
    X_test=test_docs,    # List of test documents
    y_test=test_labels,  # List of true labels
    bow=vocabulary       # Set of all unique words
)
```

```python
# Print individual metrics
print("Accuracy: ",sampler.accuracies)
print("Precision: ",sampler.precision)
print("Recall: ",sampler.recall)

# Plot individual metrics
sampler.plotAccuracy()     # Accuracy vs Laplace factor
sampler.plotPrecision()    # Precision vs Laplace factor
sampler.plotRecall()       # Recall vs Laplace factor

# Plot all metrics together
sampler.superimposePrint() # Combined plot of accuracy, precision, recall
```
