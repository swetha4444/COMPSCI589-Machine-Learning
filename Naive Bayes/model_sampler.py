from naive_bayes import NaiveBayes
import numpy as np
from calculate_accuracy import CalculateAccuracy
import matplotlib.pyplot as plt
from computation import randomColorGenerator

class NaiveBayesSampler:
    def __init__(self, labels, title, laplaceRange = [0], logProb = False, plotCM = False):
        self.laplaceRange = laplaceRange
        self.logProb = logProb
        self.labels = labels
        self.title = title
        self.accuracies = []
        self.precision = []
        self.recall = []
        self.plotCM = plotCM
    
    def createModel(self, trainData, bow, laplaceFactor):
        model = NaiveBayes(laplaceFactor=laplaceFactor, logProb=self.logProb)
        model.fit(trainData=trainData, bow=bow)
        return model

    def run(self,model,X_test):
        return model.predict(X_test)
    
    def sampler(self, trainData, X_test, y_test, bow):
        '''
        Used to run the model with different laplace factors
        '''
        for laplaceFactor in self.laplaceRange:
            model = NaiveBayes(laplaceFactor=laplaceFactor,logProb=True)
            model.fit(trainData=trainData,bow=bow)
            pred = model.predict(X_test)
            accObj = CalculateAccuracy(test=y_test,pred=pred,labels=self.labels)

            print("lf:",laplaceFactor, " Accuracy: ",accObj.accuracy())
            accObj.confusion_matrix()
            if self.plotCM:
                accObj.plotConfusionMatrix(title="Confusion Matrix for Laplace Factor: "+str(laplaceFactor))
            self.accuracies.append(accObj.accuracy())
            self.precision.append(accObj.precision())
            self.recall.append(accObj.recall())

    def superimposePrint(self):
        self.plotMetricVsLaplaceFactor(metricValues=[self.accuracies,self.precision,self.recall],
                                        metricNames=["Accuracy","Precision","Recall"],
                                        plotTile="All metric vs Laplace Factor")

    def plotAccuracy(self):
        plt.figure(figsize=(10, 7))
        x_positions = range(len(self.laplaceRange))
        plt.plot(x_positions, self.accuracies, marker='o')
        plt.xticks(x_positions, labels=[str(val) for val in self.laplaceRange])
        # plt.plot(self.laplaceRange, self.accuracies, marker='o')
        plt.grid()
        plt.xlabel('Laplace Factor')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Laplace Factor: '+self.title)
        # plt.legend()
        plt.show()

    def plotPrecision(self):
        plt.figure(figsize=(10, 7))
        x_positions = range(len(self.laplaceRange))
        plt.plot(x_positions, self.precision, marker='o')
        plt.xticks(x_positions, labels=[str(val) for val in self.laplaceRange])
        plt.grid()
        plt.xlabel('Laplace Factor')
        plt.ylabel('Precision')
        plt.title('Precision vs Laplace Factor: '+self.title)
        # plt.legend()
        plt.show()
    
    def plotRecall(self):
        plt.figure(figsize=(10, 7))
        x_positions = range(len(self.laplaceRange))
        plt.plot(x_positions, self.recall, marker='o')
        plt.xticks(x_positions, labels=[str(val) for val in self.laplaceRange])
        plt.grid()
        plt.xlabel('Laplace Factor')
        plt.ylabel('Recall')
        plt.title('Recall vs Laplace Factor: '+self.title)
        # plt.legend()
        plt.show()

    def plotMetricVsLaplaceFactor(self,metricValues=[],metricNames=[],yLabel="",plotTile=""):
        plt.figure(figsize=(10, 7))
        # plot all values in one graph
        for i in range(len(metricValues)):
            x_positions = range(len(self.laplaceRange))
            plt.plot(x_positions, metricValues[i], marker='o',color=randomColorGenerator(),label=metricNames[i])
            plt.xticks(x_positions, labels=[str(val) for val in self.laplaceRange])
            plt.grid()
        plt.xlabel('Laplace Factor')
        plt.ylabel('Metric '+yLabel)
        plt.title(plotTile+': '+self.title)
        plt.legend()
        plt.show()

        