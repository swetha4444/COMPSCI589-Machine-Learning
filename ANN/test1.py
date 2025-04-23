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
Training instance 2
		x: [0.42000]
		y: [0.23000]
'''

import numpy as np
from backPropagation import BackPropagation
from layer import Layer
from forwardPropagation import ForwardPropagation

neuronsPerLayer = [1, 2, 1]
layers = []

# Initializing the layers
inputLayer = Layer(neuronsPerLayer[0], neuronsPerLayer[1], l=1)
inputLayer.weight = np.array([[0.4, 0.1], [0.3, 0.2]])
layers.append(inputLayer)

hiddenLayer = Layer(neuronsPerLayer[1], neuronsPerLayer[2], l=2)
hiddenLayer.weight = np.array([[0.7, 0.5, 0.6]])
layers.append(hiddenLayer)

outputLayer = Layer(neuronsPerLayer[2], 0, l=3)
layers.append(outputLayer)




# print("-------------------------------------")
# print("Testing Forward Propagation")
# print("-------------------------------------")

forwardPropagation = ForwardPropagation(layers=layers)


x_train = [np.array([[0.13]]), np.array([[0.42]])]
y_train = [np.array([[0.9]]), np.array([[0.23]])]


# for i in range(len(x_train)):
# 	print(f"Training instance {i+1}")
# 	x = x_train[i]
# 	y = y_train[i]
# 	forwardPropagation.forward(x)
# 	print(f"x: {x.T}")
# 	print(f"y: {y.T}")
# 	print()
# 	for layer in layers:
# 		print(f"Layer {layer.l}") 
# 		layer.printA()
# 		# layer.printBlame()
# 		layer.printWeight()
# 		# layer.printGradient()
# 	print("Error: ", forwardPropagation.calculateError(y))
# 	print("--------------------------------------------------")
# forwardPropagation.calculateAvgGradient(m=len(x_train))
# print("Overall Error: ", forwardPropagation.J)

print()
print("-------------------------------------")
print("Testing Backward Propagation")
print("-------------------------------------")
for i in range(len(x_train)):
	print(f"Training instance {i+1}")
	x = x_train[i]
	y = y_train[i]
	forwardPropagation.forward(x)
	backPropagation = BackPropagation(layers, y)
	backPropagation.calculateBlame()
	backPropagation.calculateGradient()
	for layer in layers:
		print(f"Layer {layer.l}") 
		# layer.printA()
		layer.printBlame()
		# layer.printWeight()
		layer.printGradient()



backPropagation.calculateAvgGradient(2)
for layer in layers:
	print(f"Layer {layer.l}")
	layer.printGradient()
	print("--------------------------------------------------")



