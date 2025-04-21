import numpy as np

class Layer:
    def __init__(self, inputSize, outputSize, l):
        self.a = np.zeros((inputSize,1))
        self.blame = np.zeros((inputSize,1))
        self.weight = np.zeros((outputSize, inputSize))
        self.gradient = np.zeros((outputSize, inputSize))
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.l = l

    def printA(self):
        print("a"+self.l+": ")
        self.matrixPrint(self.a)
    
    def printBlame(self):
        print("delta"+self.l+": ")
        self.matrixPrint(self.blame)
    
    def printWeight(self):
        print("Theta"+self.l+": ")
        self.matrixPrint(self.weight)

    def printGradient(self):
        print("Gradients of Theta"+self.l+": ")
        self.matrixPrint(self.gradient)

    def matrixPrint(self, matrix):
        for row in matrix:
            for col in row:
               print(end="\t")
               print(col,end ="  ")
            print("")
        print()
    
    def getTranspose(self, matrix):
        return np.transpose(matrix)
