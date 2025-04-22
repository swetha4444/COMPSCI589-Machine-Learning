from layer import Layer
import numpy as np

class ForwardPropagation:
    def __init(self, layers):
        self.layers = layers

    def g(self, z):
        return 1 / (1 + np.exp(-z))
    
    def forward(self, x):
        self.layers[0].a = np.concatenate((np.array([[1]]), x), axis=0)
        for i in range(1, len(self.layers)):
            self.layers[i].l = i
            if i == len(self.layers) - 1: # Last layer
                self.layers[i].a = self.g(np.dot(self.layers[i-1].weight, self.layers[i-1].a)) 
            else:
                self.layers[i].a = np.concatenate((np.array([[1]]), self.g(np.dot(self.layers[i-1].weight, self.layers[i-1].a))), axis=0)

    def printLayers(self):
        for layer in self.layers:
            layer.printA()
            layer.printWeight()
            print("--------------------------------------------------")

