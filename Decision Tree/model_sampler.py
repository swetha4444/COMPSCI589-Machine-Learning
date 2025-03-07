import pandas as pd
from data_preprocessing import PreProcesser
from calculate_accuracy import CalculateAccuracy
import numpy as np
import matplotlib.pyplot as plt
from decision_tree import fit, Node, predict, parsetree
import progressbar
import time

class DecisionTreeSampler:
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
        plt.axvline(self.mean-self.std,linestyle='dotted',color='red',label=f'Mean-Std: {self.mean-self.std}')
        plt.axvline(self.mean+self.std,linestyle='dotted',color='orange',label=f'Mean+Std: {self.mean+self.std}')
        plt.xlabel('Accuracy %')
        plt.ylabel('Count')
        plt.legend()
        plt.title('Accuracy of model over 100 runs - ' + ('Test Data' if self.test_data else 'Training Data'))
        plt.show()

def compareHistogram(accuracies1,accuracies2,label1,label2,test_data=True):
        plt.figure(figsize=(6,6))
        width = 0.3
        concatAcc = np.concatenate((accuracies1,accuracies2))
        bins = np.arange(min(concatAcc), max(concatAcc) + width, width)  if test_data else 10
        plt.hist(accuracies1,bins=bins,color='skyblue',alpha=0.3,label=f'{label1} with accuracy {np.mean(accuracies1)}')
        plt.hist(accuracies2,bins=bins,color='red',alpha=0.5,label=f'{label2} with accuracy {np.mean(accuracies2)}')
        plt.xlabel('Accuracy %')
        plt.ylabel('Count')
        plt.legend()
        plt.title('Accuracy Comparision of both models over 100 runs - ' + ('Test Data' if test_data else 'Training Data'))
        plt.show()

def variance_plot(df):
    test_sizes = [10,20,30,40]
    runs = 100
    results = {}
    processorObj = PreProcesser(df)
    processorObj.preprocess()
    plt.figure(figsize=(12, 5))

    for i,s in enumerate(test_sizes):
        accuracies = []
        for _ in range(runs):
            X_train, X_test, y_train, y_test = processorObj.split()
            root = fit(X_train, y_train, metric='id3', stopping_criteria=None)
            predictions = predict(root, X_test)
            calcModel = CalculateAccuracy(y_test, predictions)
            accuracies.append(calcModel.accuracy_percentage())
        results[s] = {
            'mean': np.mean(accuracies),
            'variance': np.var(accuracies),
            'accuracies': accuracies
        }
        # Compute statistics
        mean = np.mean(accuracies)
        variance = np.var(accuracies)

        # Histogram of accuracies
        plt.hist(accuracies, bins=15, alpha=0.6, label=f'Test Size {s}% (Var: {variance:.4f})')
        plt.axvline(mean, color=f'C{i}', linestyle='dashed', linewidth=2)

    plt.legend()
    plt.xlabel('Frequency')
    plt.ylabel('Accuracy')
    plt.title('Variance in accuracy for different Train-Test Split on Accuracy Variance')
    plt.show()


