import pandas as pd
from data_preprocessing import PreProcesser
from data_analysing import Analyser
import numpy as np
import matplotlib.pyplot as plt
from decision_tree import fit, Node, predict, parsetree
from model_sampler import DecisionTreeSampler, compareHistogram, variance_plot
import sys


#Create Processor Object
file_path = 'Decision Tree/car.csv'
df = pd.read_csv(file_path)
carAnalyser = Analyser(df)
#Basic Histogram
decisionTreeSamplerTestData = DecisionTreeSampler(df,metric="id3")
decisionTreeSamplerTrainData = DecisionTreeSampler(df,test_data=False,metric="id3")

#Gini Histogram
decisionTreeSamplerTestDataGini = DecisionTreeSampler(df,metric="cart")
decisionTreeSamplerTrainDataGini = DecisionTreeSampler(df,test_data=False,metric="cart")

#Stoppinf Criteria Histogram
sc=85
decisionTreeSamplerTestDataSC = DecisionTreeSampler(df,metric="id3",stopping_criteria=sc)
decisionTreeSamplerTrainDataSC = DecisionTreeSampler(df,test_data=False,metric="id3",stopping_criteria=sc)


'''
    Data Analysing
'''
# Create Analyser Object
# carAnalyser.info()
# carAnalyser.show_unique_categories()
# carAnalyser.corr_label_data()


'''
    Sample Prediction
'''
# carProcessor = PreProcesser(df)
# carProcessor.preprocess()
# X_train, X_test, y_train, y_test = carProcessor.split()      
# root: Node = fit(carProcessor.X,carProcessor.y, metric="id3")
# print(carProcessor.encoder_dict)
# parsetree(root,carProcessor.encoder_dict)
# y_pred = predict(root, X_test)
# accuracy = np.mean(y_pred == y_test)
# accuracyPercentage = round(accuracy, 2)*100

# rootSC: Node = fit(X_train, y_train, metric="id3",stopping_crietria=85)
# # parsetree(rootSC,carProcessor.encoder_dict)
# y_pred = predict(rootSC, X_test)
# accuracySC = np.mean(y_pred == y_test)
# accuracyPercentageSC = round(accuracySC, 2)*100


'''
Test Gini
'''
test_mean1,test_std1 = decisionTreeSamplerTestDataGini.run()
train_mean1,train_std1 = decisionTreeSamplerTrainDataGini.run()

'''
Test Information Gain
'''
train_mean, train_std = decisionTreeSamplerTrainData.run()
test_mean, test_std = decisionTreeSamplerTestData.run()

'''
Test Stopping Criteria
'''
train_mean2 , train_std2 = decisionTreeSamplerTrainDataSC.run()
test_mean2 , test_std2 = decisionTreeSamplerTestDataSC.run()

'''
Output Graphs
'''
#Basic IG graphs
decisionTreeSamplerTrainData.plotHistogram()
print(f"Train Mean: {train_mean}\nTrain Standard Deviation: {train_std}")
decisionTreeSamplerTestData.plotHistogram()
print(f"Train Mean: {test_mean}\nTrain Standard Deviation: {test_std}")

decisionTreeSamplerTrainDataGini.plotHistogram()
print(f"Train Mean: {train_mean}\nTrain Standard Deviation: {train_std}")
decisionTreeSamplerTestData.plotHistogram()
print(f"Train Mean: {test_mean}\nTrain Standard Deviation: {test_std}")
decisionTreeSamplerTestDataGini.plotHistogram()
print(f"Train Mean: {test_mean1}\nTrain Standard Deviation: {test_std1}")

#Compare Gini and IG on Test Data
compareHistogram(decisionTreeSamplerTrainData.accuracies,decisionTreeSamplerTrainDataGini.accuracies,"Information Gain", "Gini Coefficient",False)
compareHistogram(decisionTreeSamplerTestData.accuracies,decisionTreeSamplerTestDataGini.accuracies,"Information Gain", "Gini Coefficient")


#Compare IG and Stopping Criteria
decisionTreeSamplerTrainDataSC.plotHistogram()
compareHistogram(decisionTreeSamplerTrainData.accuracies,decisionTreeSamplerTrainDataSC.accuracies,"Without Stopping Criteria", f"Stopping Criteria={sc}",False)
decisionTreeSamplerTestDataSC.plotHistogram()
compareHistogram(decisionTreeSamplerTestData.accuracies,decisionTreeSamplerTestDataSC.accuracies,"Without Stopping Criteria", f"Stopping Criteria={sc}")
print(f"Train Mean: {test_mean2}\nTrain Standard Deviation: {test_std2}")
print(f"Test Mean: {test_mean2}\nTest Standard Deviation: {test_std2}")

#Testing robustness of decision tree
variance_plot(df)