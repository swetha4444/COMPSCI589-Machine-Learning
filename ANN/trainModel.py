import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from layer import Layer
from dataProcess import DataPreprocessor
from forwardPropagation import ForwardPropagation
from backPropagation import BackPropagation
import progressbar

class TrainModel:
    def __init__(self, preprocessor: DataPreprocessor, layers: list, epsilon, batchSize=10, regularization=0, stepSize=0.01):
        self.stepSize = stepSize
        self.preprocessor = preprocessor
        self.layersSkeleton = layers  # Number of neurons per layer excluding bias
        self.epsilon = epsilon
        self.batchSize = batchSize  # Initialize batchSize before calling getEpoch
        self.regularization = regularization
        self.epoch = self.getEpoch()  # Now call getEpoch after batchSize is initialized

    def getEpoch(self):
        # Calculate the total length of the first k-1 folds
        lenTrainFolds = sum(len(trainFold) for trainFold in self.preprocessor.foldedDataIndex[:-1])
        self.epoch = lenTrainFolds // self.batchSize
        return self.epoch
    
    def initialiseGaussianWeights(self):
        for layer in self.layers:
            layer.weight = np.random.normal(0, 1, layer.weight.shape)
            layer.printWeight()
        print("Weights initialized successfully")
        
    def buildModel(self):
        self.layers = []
        inputLayer = Layer(self.layersSkeleton[0], self.layersSkeleton[1], l=1)
        self.layers.append(inputLayer)
        for i in range(1, len(self.layersSkeleton) - 1):
            hiddenLayer = Layer(self.layersSkeleton[i], self.layersSkeleton[i + 1], l=i + 1)
            self.layers.append(hiddenLayer)
        outputLayer = Layer(self.layersSkeleton[-1], 0, l=len(self.layersSkeleton))
        self.layers.append(outputLayer)
        self.initialiseGaussianWeights()
        print("Model built successfully")
        self.forwardPropagation = ForwardPropagation(self.layers, batchSize=self.batchSize, regularization=self.regularization)
        self.backPropagation = BackPropagation(self.layers, self.forwardPropagation, batchSize=self.batchSize, regularization=self.regularization, stepSize=self.stepSize)
    
    def train(self, stoppingCriterion='epochs'):
        self.buildModel()
        for k in range(self.preprocessor.kFold):
            print(f"Training on fold {k + 1}")
            X_train, y_train, X_test, y_test = self.preprocessor.getTrainTestSplit(k)
            if stoppingCriterion == 'epochs':
                # Print progress bar for epochs
                bar = progressbar.ProgressBar(maxval=self.epoch, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
                bar.start()
                for epoch in range(self.epoch):
                    bar.update(epoch + 1)
                    # Process data in batches
                    for i in range(0, len(X_train), self.batchSize):
                        X_train_batch = X_train[i:i + self.batchSize]
                        y_train_batch = y_train[i:i + self.batchSize]
                        self.trainEpoch(X_train_batch, y_train_batch)
                bar.finish()

    def trainEpoch(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        for i in range(len(X_train)):
            x = X_train[i].reshape(-1, 1)  # Reshape input to (n_features, 1)
            y = y_train[i]  # Scalar label
            self.backPropagation.y = y
            self.forwardPropagation.y = y
            self.forwardPropagation.forward(x)  # Pass reshaped input
            self.forwardPropagation.calculateError(i)
            self.backPropagation.calculateBlame()
            self.backPropagation.calculateGradient()
        self.forwardPropagation.calculateAvgError()
        self.backPropagation.calculateAvgGradient()
        self.backPropagation.updateWeights()

if __name__ == "__main__":
    preprocessor = DataPreprocessor(filePath='ANN/datasets/raisin.csv')
    preprocessor.load_data()
    preprocessor.encodeCategorical()
    preprocessor.normalizeData()
    preprocessor.stratifiedKFold()
    preprocessor.printDataDetails()

    # first layer nerusopm = number of attr in dataset
    layers = [preprocessor.data.shape[1] - 1, 5, 1]  
    epsilon = 0.01
    batchSize = 10
    regularization = 0.01

    model = TrainModel(preprocessor, layers, epsilon, batchSize, regularization)
    model.train()
    print("Training completed.")
    print("Model weights after training:")
    for layer in model.layers:
        print(f"Layer {layer.l} weights:\n{layer.weight}")




