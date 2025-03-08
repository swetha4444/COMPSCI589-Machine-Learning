from train_class import TrainClassObject
import numpy as np

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
        self.trainObjectList: list[TrainClassObject] = []
        totalDataCount = sum(len(data) for data in trainData.values())
        for className,data in trainData:
            trainObject = TrainClassObject(className,bow,float(len(data)/totalDataCount),self.laplaceFactor,self.logProb)
            trainObject.createFrequencyMatrix()
            self.trainObjectList.append(trainObject)

    def predict_X(self,testDoc):
        predClass = None
        currP = 0
        for trainObject in self.trainObjectList:
            tempP = trainObject.calculatePosteriorProbality(testDoc)
            if tempP > currP:
                currP = tempP
                predClass = trainObject.className
        return predClass
    
    def predict(self,y):
        pred = []
        for doc in y:
            pred.append(self.predict_X(doc))
        return pred
    
    def accuracy(self,pred,test):
        return np.mean(np.array(pred)==np.array(test))
    
    


