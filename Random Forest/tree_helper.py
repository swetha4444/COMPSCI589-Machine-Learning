import numpy as np

def calcEntropy(data):
    classes, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    return -np.sum(probabilities * np.log2(probabilities))

def calcInfoGain(y, y_subsets):
    total_entropy = calcEntropy(y)
    weighted_entropy = 0
    for subset in y_subsets:
        weighted_entropy += (len(subset) / len(y)) * calcEntropy(subset)
    return total_entropy - weighted_entropy

def calcGiniIndex(subset):
    classes,counts = np.unique(subset, return_counts = True)
    return 1 - np.sum(np.square(counts/len(subset)))

def calcAvgGiniCoeff(y,y_subsets):
    average_gini = 0
    #Get average Gini coeff
    for subset in y_subsets:
        average_gini += (len(subset) / len(y)) * calcGiniIndex(subset)
    return average_gini

def split_dataset(X, y, feature_index):
    unique_values = np.unique(X[:, feature_index]) #get unique values from feature column
    subsets = []
    for value in unique_values:
        mask = X[:, feature_index] == value #Trying to choose rows belonging to that value
        subsets.append((X[mask], y[mask]))#getting values of X and y for which feature = values
    return subsets