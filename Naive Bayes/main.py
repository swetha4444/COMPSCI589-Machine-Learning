from starter_code.utils import load_training_set, load_test_set
from collections import Counter
from naive_bayes import NaiveBayes
import itertools
from calculate_accuracy import CalculateAccuracy
from model_sampler import NaiveBayesSampler
from computation import trainDataFormatter, extendList
import numpy as np

'''
Q1: 20% test; 20% train and no laplace smoothening
'''
# percentage_positive_instances_train = 0.2
# percentage_negative_instances_train = 0.2
# percentage_positive_instances_test = 0.2
# percentage_negative_instances_test = 0.2
# (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
# (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)
# sampler = NaiveBayesSampler(labels=["positive","negative"],title="Standard MNB with 20% train and 20% test",plotCM=True)
# trainData = trainDataFormatter(labels=["positive","negative"],trainData=[pos_train,neg_train])
# y = extendList(pos_test,neg_test)
# actual = extendList(list(itertools.repeat("positive", len(pos_test))),list(itertools.repeat("negative", len(neg_test))))
# sampler.sampler(trainData=trainData,X_test=y,y_test=actual,bow=vocab)
# print("Accuracy: ",sampler.accuracies)
# print("Precision: ",sampler.precisions)
# print("Recall: ",sampler.recalls)

'''
Q2: 20% test; 20% train and with laplace smoothening factor ranges with log
'''
percentage_positive_instances_train = 0.2
percentage_negative_instances_train = 0.2
percentage_positive_instances_test = 0.2
percentage_negative_instances_test = 0.2
(pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
(pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)
laplaceRange = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
sampler = NaiveBayesSampler(labels=["positive","negative"],
                            title="Standard MNB with 20% train and 20% test",
                            plotCM=False,laplaceRange=laplaceRange,logProb=True)
trainData = trainDataFormatter(labels=["positive","negative"],trainData=[pos_train,neg_train])
y = extendList(pos_test,neg_test)
actual = extendList(list(itertools.repeat("positive", len(pos_test))),list(itertools.repeat("negative", len(neg_test))))
sampler.sampler(trainData=trainData,X_test=y,y_test=actual,bow=vocab)
sampler.plotAccuracy()
sampler.plotPrecision()
sampler.plotRecall()

'''
Q3; 100% test and 100% train with the best laplace smoothening factor and log prob
'''
max_alpha = laplaceRange[np.argmax(sampler.accuracies)]
print("Max alpha: ",max_alpha)
percentage_positive_instances_train = 1
percentage_negative_instances_train = 1
percentage_positive_instances_test = 1
percentage_negative_instances_test = 1
(pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
(pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)
sampler = NaiveBayesSampler(labels=["positive","negative"],
                            title="Standard MNB with 100% train and 100% test",
                            plotCM=True,laplaceRange=[max_alpha],logProb=True)

