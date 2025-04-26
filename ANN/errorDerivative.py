'''
mplement the method (discussed in class) that allows you to numerically
check the gradients computed by backpropagation. This will allow you to further ensure that all gradients
computed by your implementation of backpropagation are correct; that way, you will be confident that you
trust your implementation of the neural network training procedure. You should present in your report
the estimated gradients for the two neural networks described in the provided benchmark files. First,
estimate the gradients using ε = 0.1, and then ε = 0.000001. If you choose to work on this extra-credit
task, please include the corresponding source code along with your Gradescope submission.

'''

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

def errorDerivative(layers, x, y, weightIndex=(1,1,2), epsilon=0.000001):
    layers[weightIndex[0]-1].weight[weightIndex[1]][weightIndex[2]] += epsilon
    forwardPropagation1 = ForwardPropagation(layers=layers, batchSize=2)
    for i in range(len(x)):
        x_instance = x[i]
        y_instance = y[i]
        forwardPropagation1.forward(x_instance)
        forwardPropagation1.y = y_instance
        forwardPropagation1.calculateError(i)
    forwardPropagation1.calculateAvgError()
    err1 = forwardPropagation1.J
    print(f"Error with theta + epsilon: {err1}")

    layers[weightIndex[0]-1].weight[weightIndex[1]][weightIndex[2]] -=2* epsilon
    forwardPropagation2 = ForwardPropagation(layers=layers, batchSize=2)
    for i in range(len(x)):
        x_instance = x[i]
        y_instance = y[i]
        forwardPropagation2.forward(x_instance)
        forwardPropagation2.y = y_instance
        forwardPropagation2.calculateError(i)
    forwardPropagation2.calculateAvgError()
    err2 = forwardPropagation2.J
    print(f"Error with theta - epsilon: {err2}")
    print("Gradient error derivative: ", (err1 - err2) / (2 * epsilon))
    print()
    return (err1 - err2) / (2 * epsilon)

def actualErrorDerivative(layers, x, y, weightIndex=(1,1,2)):
    forwardPropagation = ForwardPropagation(layers=layers, batchSize=2)
    backPropagation = BackPropagation(layers, batchSize=2)
    for i in range(len(x)):
        x_instance = x[i]
        y_instance = y[i]
        forwardPropagation.forward(x_instance)
        backPropagation.y = y_instance
        backPropagation.calculateBlame()
        backPropagation.calculateGradient()
    backPropagation.calculateAvgGradient()
    print("Value of Gradient from backpropagation: ", layers[weightIndex[0]-1].gradient[weightIndex[1]][weightIndex[2]])
    return layers[weightIndex[0]-1].gradient[weightIndex[1]][weightIndex[2]]


if __name__ == "__main__":
    test1 = True
    test2 = False
    if test1:
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
        x_train = [np.array([[0.13]]), np.array([[0.42]])]
        y_train = [np.array([[0.9]]), np.array([[0.23]])]
        errorDerivative(layers, x_train, y_train, weightIndex=(1,1,1), epsilon=0.000001)
        actualErrorDerivative(layers, x_train, y_train, weightIndex=(1,1,1))
    else:
        pass