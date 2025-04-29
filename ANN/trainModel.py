import numpy as np
from calculateAcuracy import calculateAccuracy, calculatePrecision, calculateRecall, calculateF1Score
from layer import Layer
from dataProcess import DataPreprocessor
from forwardPropagation import ForwardPropagation
from backPropagation import BackPropagation
import progressbar
import alive_progress
from alive_progress import alive_bar

class TrainModel:
    def __init__(self, preprocessor: DataPreprocessor, layers: list, epsilon=0.01,
                  batchSize=10, regularization=0.25, stepSize=0.1, threshold=0.5,
                  epoch=100, patience=10):
        self.threshold = threshold
        self.stepSize = stepSize
        self.preprocessor = preprocessor
        self.layersSkeleton = layers  # Number of neurons per layer excluding bias
        self.epsilon = epsilon
        self.batchSize = batchSize  # Initialize batchSize before calling getEpoch
        self.regularization = regularization
        self.epoch = epoch
        # self.getEpoch()  # Now call getEpoch after batchSize is initialized
        self.patience = patience

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
        kLoss = []
        self.buildModel()

        # Initialize lists to store metrics for each epoch across all folds
        epochAccuracies = np.zeros(self.epoch, dtype=np.float64)  # Ensure float64 type
        epochPrecisions = np.zeros(self.epoch, dtype=np.float64)  # Ensure float64 type
        epochRecalls = np.zeros(self.epoch, dtype=np.float64)
        epochF1Scores = np.zeros(self.epoch, dtype=np.float64)
        epochLosses = np.zeros(self.epoch, dtype=np.float64)

        for k in range(self.preprocessor.kFold):
            print(f"\nTraining on fold {k + 1}")
            X_train, y_train, X_test, y_test = self.preprocessor.getTrainTestSplit(k)

            # Initialize metrics for this fold
            foldEpochAccuracy = []
            foldEpochPrecision = []
            foldEpochRecall = []
            foldEpochF1Score = []
            foldEpochLoss = []

            if stoppingCriterion == 'epochs':
                # Print progress bar for epochs, not percentage but epoch number on the bar
                # use alive bar pip install alive-progress
                with alive_bar(self.epoch, title="Training Progress", bar='classic2', spinner='waves',) as bar:
                    for epoch in range(self.epoch):
                        # bar.update(epoch + 1)
                        # Train on batches
                        for i in range(0, len(X_train), self.batchSize):
                            X_train_batch = X_train[i:i + self.batchSize]
                            y_train_batch = y_train[i:i + self.batchSize]
                            self.trainEpoch(X_train_batch, y_train_batch)

                        # Evaluate on the test set for this epoch
                        acc, pre, rec, f1, loss = self.testModel(X_test, y_test)
                        foldEpochAccuracy.append(acc)
                        foldEpochPrecision.append(pre)
                        foldEpochRecall.append(rec)
                        foldEpochF1Score.append(f1)
                        foldEpochLoss.append(loss)
                        bar()
                    # bar.finish()

            else:
                # Stop if change in error avg is less than epsilon
                for epoch in range(self.epoch):
                    currErr = 0
                    prevErr = 0
                    while True:
                        for i in range(0, len(X_train), self.batchSize):
                            X_train_batch = X_train[i:i + self.batchSize]
                            y_train_batch = y_train[i:i + self.batchSize]
                            self.trainEpoch(X_train_batch, y_train_batch)
                        currErr = self.forwardPropagation.J
                        # print(f"\t\tΔError: {abs(currErr - prevErr)}")
                        if abs(currErr - prevErr) < self.epsilon:
                            break
                        prevErr = currErr
                    acc, pre, rec, f1, loss = self.testModel(X_test, y_test)
                    foldEpochAccuracy.append(acc)
                    foldEpochPrecision.append(pre)
                    foldEpochRecall.append(rec)
                    foldEpochF1Score.append(f1)
                    foldEpochLoss.append(currErr)
                    # bar()
                    
            # Append fold metrics to the epoch-level lists
            if len(epochAccuracies) == 0:
                epochAccuracies = np.array(foldEpochAccuracy)
                epochPrecisions = np.array(foldEpochPrecision)
                epochRecalls = np.array(foldEpochRecall)
                epochF1Scores = np.array(foldEpochF1Score)
                epochLosses = np.array(foldEpochLoss)
            else:
                epochAccuracies += np.array(foldEpochAccuracy)
                epochPrecisions += np.array(foldEpochPrecision)
                epochRecalls += np.array(foldEpochRecall)
                epochF1Scores += np.array(foldEpochF1Score)
                epochLosses += np.array(foldEpochLoss)

            print(f"Testing on fold {k + 1} completed")
            print("--------------------------------------------------------")

        # Take the average of metrics across all folds for each epoch
        epochAccuracies /= self.preprocessor.kFold
        epochPrecisions /= self.preprocessor.kFold
        epochRecalls /= self.preprocessor.kFold
        epochF1Scores /= self.preprocessor.kFold
        epochLosses /= self.preprocessor.kFold

        # Append the averaged metrics to kAccuracy, kPrecision, etc.
        kAccuracy.extend(epochAccuracies)
        kPrecision.extend(epochPrecisions)
        kRecall.extend(epochRecalls)
        kF1Score.extend(epochF1Scores)
        kLoss.extend(epochLosses)

        # Print final metrics
        print("\nFinal Metrics Across All Epochs:")
        for epoch, (acc, pre, rec, f1, loss) in enumerate(zip(epochAccuracies, epochPrecisions, epochRecalls, epochF1Scores, epochLosses), start=1):
            print(f"Epoch {epoch}: Accuracy: {acc:.2f}%, Precision: {pre:.2f}%, Recall: {rec:.2f}%, F1 Score: {f1:.2f}%, Loss: {loss:.2f}%")

        return kAccuracy, kPrecision, kRecall, kF1Score, kLoss

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
        self.forwardPropagation.calculateAvgError(len(y_train))
        self.backPropagation.calculateAvgGradient()
        self.backPropagation.updateWeights()

    def testModel(self, X_test, y_test):
        # Ensure X_test and y_test are NumPy arrays
        X_test = np.array(X_test, dtype=np.float64)
        y_test = np.array(y_test, dtype=np.float64)

        y_pred = []
        #self.forwardPropagation.J = 0  # Reset J

        for i in range(len(X_test)):
            x = X_test[i].reshape(-1, 1)  # Reshape input to (n_features, 1)
            self.forwardPropagation.forward(x)  # Perform forward propagation
            y_pred.append(self.forwardPropagation.layers[-1].a[0, 0])  # Get the probability of class 1
            self.forwardPropagation.calculateError(i)  # Calculate error for this sample

        self.forwardPropagation.calculateAvgError(len(y_test))  # Calculate average error (J)
        y_pred = np.array(y_pred)
        y_pred_binary = (y_pred >= self.threshold).astype(int)  # Convert probabilities to binary predictions

        # Calculate other metrics
        accuracy = calculateAccuracy(y_test, y_pred_binary)
        precision = calculatePrecision(y_test, y_pred_binary, labels=[0, 1])
        recall = calculateRecall(y_test, y_pred_binary, labels=[0, 1])
        f1 = calculateF1Score(y_test, y_pred_binary, labels=[0, 1])

        return accuracy, precision, recall, f1, self.forwardPropagation.J


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
