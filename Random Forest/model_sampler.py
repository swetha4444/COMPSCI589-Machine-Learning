import pandas as pd
from data_preprocessing import PreProcesser
from calculate_accuracy import CalculateAccuracy, calculate_precision, calculate_recall, calculate_f1_score # Added other metrics
import numpy as np
import matplotlib.pyplot as plt
from decision_tree import fit, Node, predict, parsetree
import progressbar
import time
from sklearn.model_selection import StratifiedKFold # Changed the split
from collections import defaultdict #For majority voting

class DecisionTreeSampler: # Old class is removed and functions are moved to Random Forest and CrossValidator
    def __init__(self,df, test_data=True, sampling_runs: int=100, metric = "id3", stopping_criteria = None):
        self.df = df
        self.processorObj = PreProcesser(df)
        self.processorObj.preprocess()
        self.sampling_runs = sampling_runs
        self.test_data = test_data
        self.metric = metric
        self.stopping_criteria = stopping_criteria

def run(self):
    widgets = ['Loading: ', progressbar.AnimatedMarker()]
    bar = progressbar.ProgressBar(widgets=widgets).start()
    # We want to shuffle and split the data for each run (combination of k and sampling)
    self.accuracies = []
    for i in range(self.sampling_runs):
        bar.update(i)
        X_train, X_test, y_train, y_test = self.processorObj.split()
        # self.processorObj.X,self.processorObj.X,self.processorObj.y,self.processorObj.y
        
        # Now we want to train the model and get accuracy
        root = fit(X_train, y_train, metric=self.metric, stopping_criteria= self.stopping_criteria)
        if self.test_data:
            predictions = predict(root, X_test)
            calcModel = CalculateAccuracy(y_test, predictions)
        else:
            predictions = predict(root, X_train)
            calcModel = CalculateAccuracy(y_train, predictions)
            # We use the Calculate Accuracy class to calculate the accuracy
        self.accuracies.append(calcModel.accuracy_percentage())
        self.mean, self.std, self.var = np.round(np.mean(self.accuracies),2), np.round(np.std(self.accuracies),2), np.round(np.var(self.accuracies),2)
    return self.mean, self.std

def plot(self):
    plt.figure(figsize=(12,6))
    plt.plot(range(self.sampling_runs), self.accuracies, linestyle='-', marker='o')
    plt.xlabel('Run Number')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for different samples of data - ' + ('Test Data' if self.test_data else 'Training Data'))
    plt.show()

def plotHistogram(self):
    plt.figure(figsize=(6,6))
    self.accuracies = np.round(self.accuracies, 4)
    # Bin settings to ensure proper grouping
    width = 0.3 if self.test_data == True else 10  # Adjust to control closeness
    bins = np.arange(min(self.accuracies), max(self.accuracies) + width, width) if self.test_data else 10
    plt.hist(self.accuracies,bins=bins,color='skyblue',alpha=0.7,)
    plt.axvline(self.mean,linestyle='dotted',color='blue',label=f'Mean: {self.mean}')
    # plt.axvline(self.mean-