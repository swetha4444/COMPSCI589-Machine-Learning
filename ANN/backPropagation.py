import numpy as np
from layer import Layer

class BackPropagation:
    def __init__(self, layers, y, stepSize=0.01, regularization=0):
        self.stepSize = stepSize
        self.layers = layers
        self.y = y
        self.regularization = regularization

    def calculateBlame(self):
        self.calculateOutputError()
        for i in range(len(self.layers)-2, -1, -1):
            activationTerm = self.layers[i].a * (1 - self.layers[i].a)
            self.layers[i].blame = np.dot(self.layers[i].weight.T, self.layers[i+1].blame) * activationTerm
            self.layers[i].blame = self.layers[i].blame[1:] 

    def calculateOutputError(self):
        self.layers[-1].blame = self.layers[-1].a - self.y

    def calculateGradient(self):
        for i in range(len(self.layers)-2, -1, -1):
            self.layers[i].gradient = np.dot(self.layers[i+1].blame, self.layers[i].a.T)
            
    def calculateAvgGradient(self, m):
        for i in range(len(self.layers) - 2, -1, -1):
            self.layers[i].gradient /= m
            self.layers[i].gradient[:, 1:] += (self.regularization / m) * self.layers[i].weight[:, 1:]  # Exclude bias

    def updateWeights(self):
        for i in range(len(self.layers)-2, -1, -1):
            self.layers[i].weight -= self.stepSize * self.layers[i].gradient
            self.layers[i].gradient = np.zeros(self.layers[i].gradient.shape)
    
    def printLayers(self):
        for layer in self.layers:
            layer.printBlame()
            layer.printWeight()
            layer.printGradient()
            print("--------------------------------------------------")

