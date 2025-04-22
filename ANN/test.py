'''
This is a test script for the ANN class.
Initializing the network with the following structure (number of neurons per layer excluding bias): [1 2 1]

Initial Theta1 (the weights of each neuron, including the bias weight, are stored in the rows):
	0.40000  0.10000  
	0.30000  0.20000  

Initial Theta2 (the weights of each neuron, including the bias weight, are stored in the rows):
	0.70000  0.50000  0.60000  


Training set
	Training instance 1
		x: [0.13000]
		y: [0.90000]
'''

import numpy as np
from layer import Layer
from forwardPropagation import ForwardPropagation

neuronsPerLayer = [1, 2, 1]
layers = []

# Initializing the layers
inputLayer = Layer(neuronsPerLayer[0], neuronsPerLayer[1], 1)
inputLayer.weight = np.array([[0.4, 0.1], [0.3, 0.2]])
layers.append(inputLayer)

hiddenLayer = Layer(neuronsPerLayer[1], neuronsPerLayer[2], 2)
hiddenLayer.weight = np.array([[0.7, 0.5, 0.6]])
layers.append(hiddenLayer)

outputLayer = Layer(neuronsPerLayer[2], 0, 3)
layers.append(outputLayer)

# Initializing the forward propagation
forwardPropagation = ForwardPropagation()
forwardPropagation.layers = layers

# Training set
x = np.array([[0.13]])
y = np.array([[0.9]])
forwardPropagation.forward(x)
forwardPropagation.printLayers()
print("--------------------------------------------------")
