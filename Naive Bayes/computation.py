from collections import Counter
import random
import matplotlib as plt
import numpy as np
import matplotlib.pyplot as pltp

def generateWordFreq(doc_vector):
    return Counter(doc_vector).items()

def trainDataFormatter(labels,trainData):
    trainDataMap = {}
    for i in range(len(labels)):
        trainDataMap[labels[i]] = trainData[i]
    return trainDataMap

def extendList(*lists):
    finalList = []
    for list in lists:
        finalList.extend(list)
    return finalList

def randomColorGenerator():
    return plt.colors.to_hex((random.random(),random.random(),random.random()))
