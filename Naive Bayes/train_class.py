from computation import generateWordFreq

from collections import Counter
import math
class TrainClassObject:
    def __init__(self,className,bow,prioriProb,laplaceFactor=0,logProb=False):
        self.className = className
        self.bow = bow
        self.freqMap = dict.fromkeys(bow,0)
        self.prioriProb = prioriProb
        self.laplaceFactor = laplaceFactor
        self.logProb = logProb
    
    def generateMultinomialFrequency(self,classData):
        for doc in classData:
            for word,count in generateWordFreq(doc):
                # if doc in self.freqMap.keys():
                # print(word,count,self.freqMap[word],"\n")
                self.freqMap[word] += count

    def computeTotalWords(self):
        self.totalCount = sum(self.freqMap.values())
    
    def calculatePosteriorProbality(self,testDoc):
        prob = 1 if not self.logProb else 0
        # print(self.freqMap)
        # print(self.logProb,prob)
        for word in testDoc:
            n_wi = (self.freqMap[word] + self.laplaceFactor) if word in self.freqMap.keys() else (0 + self.laplaceFactor)
            n_y = self.totalCount + (self.laplaceFactor * len(self.bow))
            if(self.logProb):
                prob += math.log10(n_wi/n_y)
            else:
                prob *= n_wi/n_y
        return self.prioriProb * prob if not self.logProb else math.log10(self.prioriProb) + prob
    
    def createFrequencyMatrix(self,classData):
        self.generateMultinomialFrequency(classData)
        self.computeTotalWords()

    



    
