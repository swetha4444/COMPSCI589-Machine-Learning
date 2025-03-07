import pandas as pd
from model_sampler import KKNSampler, plotComparision
from data_preprocessing import PreProcesser
from data_analysing import Analyser
from KNN_Instance import KNNModel
from calculate_accuracy import CalculateAccuracy
import numpy as np
import matplotlib.pyplot as plt

#Create Processor Object
file_path = 'KNN/wdbc.csv'
df = pd.read_csv(file_path,header=None)
wbdcProcessor = PreProcesser(df)
wbdcAnalyser = Analyser(df)
knnSamplerTrainN = KKNSampler(df,k_range=range(1, 52, 2), test_data=False, sampling_runs=20)
knnSamplerTestN = KKNSampler(df,k_range=range(1, 52, 2), test_data=True, sampling_runs=20)
knnSamplerTrain = KKNSampler(df,k_range=range(1, 52, 2), test_data=False, sampling_runs=20,normalized=False)
knnSamplerTest = KKNSampler(df,k_range=range(1, 52, 2), test_data=True, sampling_runs=20,normalized=False)

'''
    Data Analysing
'''
#Create Analyser Object
# wbdcAnalyser.info()
# wbdcAnalyser.calculate_A_Priori()
# wbdcAnalyser.plot()
# wbdcAnalyser.plot_pie()

'''
    KNN Classifier
'''
# Run sampler for Normalized data
knnSamplerTrainN.run()
knnSamplerTrainN.plot()
knnSamplerTestN.run()
knnSamplerTestN.plot()

# Run sampler for Non-Normalized data
knnSamplerTrain.run()
knnSamplerTrain.plot(False)
knnSamplerTest.run()
knnSamplerTest.plot(False)


plotComparision(knnSamplerTestN,knnSamplerTest,
                'With Normalization', 'Without Normalization',
                'Comparing with and without normalizing test data')
