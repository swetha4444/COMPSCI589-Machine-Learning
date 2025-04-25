'''
Regularization parameter lambda=0.250

Initializing the network with the following structure (number of neurons per layer): [2 4 3 2]

Initial Theta1 (the weights of each neuron, including the bias weight, are stored in the rows):
	0.42000  0.15000  0.40000  
	0.72000  0.10000  0.54000  
	0.01000  0.19000  0.42000  
	0.30000  0.35000  0.68000  

Initial Theta2 (the weights of each neuron, including the bias weight, are stored in the rows):
	0.21000  0.67000  0.14000  0.96000  0.87000  
	0.87000  0.42000  0.20000  0.32000  0.89000  
	0.03000  0.56000  0.80000  0.69000  0.09000  

Initial Theta3 (the weights of each neuron, including the bias weight, are stored in the rows):
	0.04000  0.87000  0.42000  0.53000  
	0.17000  0.10000  0.95000  0.69000  


Training set
	Training instance 1
		x: [0.32000   0.68000]
		y: [0.75000   0.98000]
	Training instance 2
		x: [0.83000   0.02000]
		y: [0.75000   0.28000]
'''

import numpy as np
from backPropagation import BackPropagation
from layer import Layer
from forwardPropagation import ForwardPropagation

neuronsPerLayer = [2, 4, 3, 2]
layers = []
# Initializing the layers
inputLayer = Layer(neuronsPerLayer[0], neuronsPerLayer[1], l=1)
inputLayer.weight = np.array([[0.42, 0.15, 0.4], [0.72, 0.1, 0.54], [0.01, 0.19, 0.42], [0.3, 0.35, 0.68]])
layers.append(inputLayer)

hiddenLayer1 = Layer(neuronsPerLayer[1], neuronsPerLayer[2], l=2)
hiddenLayer1.weight = np.array([[0.21, 0.67, 0.14, 0.96, 0.87], [0.87, 0.42, 0.2, 0.32, 0.89], [0.03, 0.56, 0.8, 0.69, 0.09]])
layers.append(hiddenLayer1)

hiddenLayer2 = Layer(neuronsPerLayer[2], neuronsPerLayer[3], l=3)
hiddenLayer2.weight = np.array([[0.04, 0.87, 0.42, 0.53], [0.17, 0.1, 0.95, 0.69]])
layers.append(hiddenLayer2)

outputLayer = Layer(neuronsPerLayer[3], 0, l=4)
layers.append(outputLayer)

print("-------------------------------------")
print("Testing Forward Propagation")
print("-------------------------------------")
forwardPropagation = ForwardPropagation(layers=layers,regularization=0.25, batchSize=2)
x_train = [np.array([[0.32], [0.68]]), np.array([[0.83], [0.02]])]
y_train = [np.array([[0.75], [0.98]]), np.array([[0.75], [0.28]])]
for i in range(len(x_train)):
    print(f"Training instance {i+1}")
    x = x_train[i]
    y = y_train[i]
    forwardPropagation.forward(x)
    print(f"x: {x.T}")
    print(f"y: {y.T}")
    print()
    for layer in layers:
        print(f"Layer {layer.l}") 
        layer.printA()
        layer.printWeight()
        forwardPropagation.y = y
    print("Error: ", forwardPropagation.calculateError(i))
    print("--------------------------------------------------")
forwardPropagation.calculateAvgError()
print("Overall Error: ", forwardPropagation.J)


print("--------------------------------------------------")
print("Testing Back Propagation")
print("-----------------------------------------------")
forwardPropagation = ForwardPropagation(layers=layers, batchSize=2, regularization=0.25)
backPropagation = BackPropagation(layers, batchSize=2, regularization=0.25)

for i in range(len(x_train)):
    print(f"Training instance {i+1}")
    instanceGradientTracker = {}
    x = x_train[i]
    y = y_train[i]
    forwardPropagation.forward(x)
    backPropagation.y = y
    backPropagation.calculateBlame()
    instanceGradientTracker = backPropagation.calculateGradient()

    for layer in layers:
        print(f"Layer {layer.l}") 
        layer.printBlame()
        layer.printGradient(instanceGradientTracker)
    print("--------------------------------------------------")

backPropagation.calculateAvgGradient()
for layer in layers:
    print(f"Average Gradient of Layer {layer.l}")
    layer.matrixPrint(layer.gradient)
    # print("--------------------------------------------------")

