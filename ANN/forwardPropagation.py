from layer import Layer
import numpy as np

class ForwardPropagation:
    def __init__(self, layers,batchSize=1):
        self.batchSize = batchSize
        self.layers = layers
        self.J = 0

    def g(self, z):
        return 1 / (1 + np.exp(-z))
    
    def forward(self, x):
        self.layers[0].a = np.concatenate((np.array([[1]]), x), axis=0)
        for i in range(1, len(self.layers)):
            z = np.dot(self.layers[i - 1].weight, self.layers[i - 1].a)
            if i == len(self.layers) - 1:
                self.layers[i].a = self.g(z)
            else:
                self.layers[i].a = np.concatenate((np.array([[1]]), self.g(z)), axis=0)

    def printLayers(self):
        for layer in self.layers:
            layer.printA()
            layer.printWeight()

    def calculateAvgError(self, y):
        y = np.array(y)
        # Compute the cost function
        predictions = self.layers[-1].a  # Output layer activations
        self.J += -np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions)) / len(y)
        return self.J


    def calculateAvgGradient(self):
        self.J /= self.batchSize