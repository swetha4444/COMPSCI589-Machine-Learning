from starter_code.utils import load_training_set, load_test_set
from collections import Counter
from naive_bayes import NaiveBayes
import itertools
from calculate_accuracy import CalculateAccuracy
from model_sampler import NaiveBayesSampler
from computation import trainDataFormatter, extendList
import numpy as np
import matplotlib.pyplot as plt

runCommand = [True,False,False,False,False]
plotSaturationGraph = False

'''
Q1: 20% test; 20% train and no laplace smoothening
'''
if runCommand[0]:
    percentage_positive_instances_train = 0.2
    percentage_negative_instances_train = 0.2
    percentage_positive_instances_test = 0.2
    percentage_negative_instances_test = 0.2
    (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)
    sampler = NaiveBayesSampler(labels=["positive","negative"],title="Standard MNB with 20% train and 20% test",plotCM=True)
    trainData = trainDataFormatter(labels=["positive","negative"],trainData=[pos_train,neg_train])
    y = extendList(pos_test,neg_test)
    actual = extendList(list(itertools.repeat("positive", len(pos_test))),list(itertools.repeat("negative", len(neg_test))))
    sampler.sampler(trainData=trainData,X_test=y,y_test=actual,bow=vocab)
    print("Accuracy: ",sampler.accuracies[0])
    print("Precision: ",sampler.precision[0])
    print("Recall: ",sampler.recall[0])

'''
Q2: 20% test; 20% train and with laplace smoothening factor ranges with log
'''
if runCommand[1]:
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
    sampler.superimposePrint()

    # For alpha=1
    sampler = NaiveBayesSampler(labels=["positive","negative"],
                                title="Standard MNB with 20% train and 20% test",
                                plotCM=True,laplaceRange=[1],logProb=True)
    trainData = trainDataFormatter(labels=["positive","negative"],trainData=[pos_train,neg_train])
    y = extendList(pos_test,neg_test)
    actual = extendList(list(itertools.repeat("positive", len(pos_test))),list(itertools.repeat("negative", len(neg_test))))
    sampler.sampler(trainData=trainData,X_test=y,y_test=actual,bow=vocab)
    print("Accuracy: ",sampler.accuracies[0])
    print("Precision: ",sampler.precision[0])
    print("Recall: ",sampler.recall[0])

    

'''
Q3: 100% test and 100% train with the best laplace smoothening factor and log prob
'''
if runCommand[2]:
    max_alpha = 1
    print("Max alpha: ",max_alpha)
    percentage_positive_instances_train = 1
    percentage_negative_instances_train = 1
    percentage_positive_instances_test = 1
    percentage_negative_instances_test = 1
    (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)
    sampler = NaiveBayesSampler(labels=["positive","negative"],
                                title="Standard MNB with 100% train and 100% test",
                                plotCM=False,laplaceRange=[max_alpha],logProb=True)
    trainData = trainDataFormatter(labels=["positive","negative"],trainData=[pos_train,neg_train])
    y = extendList(pos_test,neg_test)
    actual = extendList(list(itertools.repeat("positive", len(pos_test))),list(itertools.repeat("negative", len(neg_test))))
    sampler.sampler(trainData=trainData,X_test=y,y_test=actual,bow=vocab)
    print("Accuracy: ",sampler.accuracies[0])
    print("Precision: ",sampler.precision[0])
    print("Recall: ",sampler.recall[0])

'''
Q4: 100% test and 30% train with the best laplace smoothening factor and log prob
'''
if runCommand[3]:
    percentage_positive_instances_train = 0.3
    percentage_negative_instances_train = 0.3
    percentage_positive_instances_test = 1
    percentage_negative_instances_test = 1
    (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)
    sampler = NaiveBayesSampler(labels=["positive","negative"],
                                title="Standard MNB with 100% train and 100% test",
                                plotCM=True,laplaceRange=[1],logProb=True)
    trainData = trainDataFormatter(labels=["positive","negative"],trainData=[pos_train,neg_train])
    y = extendList(pos_test,neg_test)
    actual = extendList(list(itertools.repeat("positive", len(pos_test))),list(itertools.repeat("negative", len(neg_test))))
    sampler.sampler(trainData=trainData,X_test=y,y_test=actual,bow=vocab)
    print("Accuracy: ",sampler.accuracies[0])
    print("Precision: ",sampler.precision[0])
    print("Recall: ",sampler.recall[0])


    

'''
Q6: 100% test and 10:50 train with the best laplace smoothening factor and log prob
'''
if runCommand[4]:
    percentage_positive_instances_train = 0.1
    percentage_negative_instances_train = 0.5
    percentage_positive_instances_test = 1
    percentage_negative_instances_test = 1
    (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)
    sampler = NaiveBayesSampler(labels=["positive","negative"],
                                title="Standard MNB with 100% train and 100% test",
                                plotCM=True,laplaceRange=[1],logProb=True)
    trainData = trainDataFormatter(labels=["positive","negative"],trainData=[pos_train,neg_train])
    y = extendList(pos_test,neg_test)
    actual = extendList(list(itertools.repeat("positive", len(pos_test))),list(itertools.repeat("negative", len(neg_test))))
    sampler.sampler(trainData=trainData,X_test=y,y_test=actual,bow=vocab)
    print("Accuracy: ",sampler.accuracies[0])
    print("Precision: ",sampler.precision[0])
    print("Recall: ",sampler.recall[0])


'''
ANNEX: Plot to find the saturation threshold for the MNB model:
'''
if plotSaturationGraph:
    percentage_train = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    accuracies, precisions, recalls = [],[],[]
    percentage_positive_instances_test = 1
    percentage_negative_instances_test = 1
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

    for percentage in percentage_train:
        percentage_positive_instances_train = percentage
        percentage_negative_instances_train = percentage
        (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
        sampler = NaiveBayesSampler(labels=["positive","negative"],
                                    title="",
                                    plotCM=False,laplaceRange=[1],logProb=True)
        trainData = trainDataFormatter(labels=["positive","negative"],trainData=[pos_train,neg_train])
        y = extendList(pos_test,neg_test)
        actual = extendList(list(itertools.repeat("positive", len(pos_test))),list(itertools.repeat("negative", len(neg_test))))
        sampler.sampler(trainData=trainData,X_test=y,y_test=actual,bow=vocab)
        accuracies.append(sampler.accuracies[0])
        precisions.append(sampler.precision[0])
        recalls.append(sampler.recall[0])

    plt.figure(figsize=(10, 6))
    plt.plot(percentage_train, accuracies, marker='x', label='Accuracy', 
         linewidth=2, markersize=8, color='blue')
    plt.ylim(0,100)
    # plt.annotate()
    plt.xlabel('Training Percentage')
    plt.ylabel('Performance')
    plt.title('Performance vs. Training Percentage (Multinomial Naïve Bayes)')
    plt.xticks(percentage_train, [f'{int(x*100)}%' for x in percentage_train], 
            fontsize=10)
    plt.yticks(fontsize=10)

    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.show()