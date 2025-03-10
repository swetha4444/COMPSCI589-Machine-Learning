from train_class import TrainClassObject
import numpy as np
import math
import random
import progressbar
import time
from decimal import *

class NaiveBayes:
    def __init__(self,laplaceFactor=0,logProb=False):
        '''
            laplaceFactor: factor to implement Laplace Smoothening
            logProb: apply log to probabilities
        '''
        self.laplaceFactor = laplaceFactor
        self.logProb = logProb
    
    def fit(self,trainData,bow):
        '''
            trainData: map of className:classData
            trainObjectList: internal list of train objects for each class
        '''
        total_classes = len(trainData)
        widgets = ['Training Model: ', progressbar.Percentage(), ' ',
                progressbar.Bar(), ' ', progressbar.ETA()]
        bar = progressbar.ProgressBar(maxval=total_classes, widgets=widgets).start()
        
        self.trainObjectList: list[TrainClassObject] = []
        totalDataCount = sum(len(data) for data in trainData.values())
        
        for i, (className,data) in enumerate(trainData.items()):
            trainObject = TrainClassObject(className,bow,Decimal(len(data)/totalDataCount),
                                        self.laplaceFactor,self.logProb)
            trainObject.createFrequencyMatrix(data)
            self.trainObjectList.append(trainObject)
            bar.update(i)
        bar.finish()

    def predict_X(self,testDoc):
        predClass = None
        currP = 0 if not self.logProb else -math.inf
        labels = []
        for trainObject in self.trainObjectList:
            labels.append(trainObject.className)
            tempP = trainObject.calculatePosteriorProbality(testDoc)
            if tempP > currP:
                currP = tempP
                predClass = trainObject.className
        if predClass is None:
            return random.choice(labels)
        return predClass
    
    def predict(self,y):
        pred = []
        # widgets = ['Testing Model: ', progressbar.AnimatedMarker()]
        # bar = progressbar.ProgressBar(widgets=widgets).start()
        i = 0
        for doc in y:
            # bar.update(i)
            i+=1
            pred.append(self.predict_X(doc))
        # bar.finish()
        return pred
    
    def accuracy(self,pred,test):
        return np.mean(np.array(pred)==np.array(test))
    
    
    



