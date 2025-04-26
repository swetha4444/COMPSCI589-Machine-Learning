import numpy as np
from calculateAcuracy import calculateAccuracy, calculatePrecision, calculateRecall, calculateF1Score
from layer import Layer
from dataProcess import DataPreprocessor
from forwardPropagation import ForwardPropagation
from backPropagation import BackPropagation
import progressbar

class TrainModel:
    def __init__(self, preprocessor: DataPreprocessor, layers: list, epsilon=0.01, batchSize=10, regularization=0.25, stepSize=0.1, threshold=0.5):
        self.threshold = threshold
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
    
    def kFoldTrainTest(self, stoppingCriterion='epochs'):
        kAccuracy = []
        kPrecision = []
        kRecall = []
        kF1Score = []
        self.buildModel()
        for k in range(self.preprocessor.kFold):
            print(f"\nTraining on fold {k + 1}")
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
            else:
                # stop if change in error avg is less than epsilon
                currErr = 0
                prevErr = 0
                while True:
                    for i in range(0, len(X_train), self.batchSize):
                        X_train_batch = X_train[i:i + self.batchSize]
                        y_train_batch = y_train[i:i + self.batchSize]
                        self.trainEpoch(X_train_batch, y_train_batch)
                    currErr = self.forwardPropagation.J
                    print(f"\t\tΔError: {abs(currErr - prevErr)}")
                    if abs(currErr - prevErr) < self.epsilon:
                        break
                    prevErr = currErr
                   
            acc,pre,rec,f1 = self.testModel(X_test, y_test)
            kAccuracy.append(acc)
            kPrecision.append(pre)
            kRecall.append(rec)
            kF1Score.append(f1)
            print(f"Testing on fold {k + 1} completed")
            print("--------------------------------------------------------")
        return np.mean(kAccuracy), np.mean(kPrecision), np.mean(kRecall), np.mean(kF1Score)

    def trainEpoch(self, X_train, y_train):
        X_train = np.array(X_train, dtype=np.float64)  # Ensure X_train is a NumPy array with float64 dtype
        y_train = np.array(y_train, dtype=np.float64)  # Ensure y_train is a NumPy array with float64 dtype

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

    def testModel(self, X_test, y_test):
        # return avg metric values of all k folds
        X_test = np.array(X_test, dtype=np.float64)  # Ensure X_test is a NumPy array with float64 dtype
        y_test = np.array(y_test, dtype=np.float64)  # Ensure y_test is a NumPy array with float64 dtype

        y_pred = []
        for i in range(len(X_test)):
            x = X_test[i].reshape(-1, 1)  # Reshape input to (n_features, 1)
            self.forwardPropagation.forward(x)
            y_pred.append(self.forwardPropagation.layers[-1].a[0, 0])  # Get the probability of class 1
        y_pred = np.array(y_pred)
        y_pred_binary = (y_pred >= self.threshold).astype(int)  # Lower threshold to 0.5
        accuracy = calculateAccuracy(y_test, y_pred_binary)
        precision = calculatePrecision(y_test, y_pred_binary,labels=[0, 1])
        recall = calculateRecall(y_test, y_pred_binary,labels=[0, 1])
        f1 = calculateF1Score(y_test, y_pred_binary,labels=[0, 1])

        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")
        print("Model testing completed successfully")
        return accuracy, precision, recall, f1

if __name__ == "__main__":
    preprocessor = DataPreprocessor(filePath='ANN/datasets/loan.csv')
    preprocessor.load_data()
    preprocessor.encodeCategorical()
    preprocessor.normalizeData()
    preprocessor.stratifiedKFold()
    preprocessor.printDataDetails()

    layers = [preprocessor.data.shape[1] - 1, 10,12, 1]  
    epsilon = 0.01
    batchSize = 10
    regularization = 0.01
    stepSize = 0.01

    model = TrainModel(preprocessor, layers, epsilon, batchSize, regularization, stepSize=stepSize)
    model.kFoldTrainTest(stoppingCriterion='epochs')
    print("Training completed.")
