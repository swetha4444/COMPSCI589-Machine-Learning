from collections import Counter

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