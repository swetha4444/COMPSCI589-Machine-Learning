'''
Train a neural network and evaluate it using the stratified cross-validation technique discussed in
class. You should train neural networks using different values for the regularization parameter, λ,
and using different architectures. You can decide which architectures you will evaluate. As an
example, consider testing networks with 1, 2, 3, and 4 hidden layers, and with various numbers of
neurons per layer: e.g., 2, 4, 8, 16.
2. For each trained neural network (i.e., for each combination of architecture and regularization that
you tested), you should measure the resulting model’s accuracy and F1 score.
5
3. Based on these experiments, you should create, for each dataset and for each of the metrics described
above, a table summarizing the corresponding results (i.e., you should show the value of each
performance metric for each type of neural network that you tested, on both datasets). You should
test at least 6 neural network architectures on each dataset
'''
import numpy as np
from layer import Layer
from forwardPropagation import ForwardPropagation
from backPropagation import BackPropagation
from dataProcess import DataPreprocessor
from trainModel import TrainModel
import matplotlib.pyplot as plt
import seaborn as sns


WDBC_LAYERS_SKELETON = [[1], [20,1], [18,1],[15, 22, 1], [18,20,18,1]] 
LOAN_LAYERS_SKELETON = [[5,1], [12,1],[5, 10, 1], [10,5,8,1]]

TITANIC_LAYERS_SKELETON = [[20,1], [8,1],[16,4,8,1],[8,4,2,1]]
RAISIN_LAYERS_SKELETON =  [[1], [5,1],[10, 8, 1],[10, 16, 8, 4, 1],[10, 8, 6, 8, 1]]
class ModelSampler:
    EPSILON = 0.01
    REGULARIZATION_VALUES = [0.01,0.025]
    STEP_SIZE_VALUES = [0.01, 0.05]
    BATCH_SIZE_VALUES = [5, 10]
    K_FOLD = 5
    EPOCHS = 100
    
    def __init__(self, filePath, splice = None):
        self.filePath = filePath
        self.preprocessor = DataPreprocessor(filePath=filePath, kFold=self.K_FOLD, splice = None, randomSeed=9)
        self.preprocessor.load_data()
        self.preprocessor.encodeCategorical()
        self.preprocessor.normalizeData()
        self.preprocessor.stratifiedKFold()
        self.preprocessor.printDataDetails()
        self.trainModels = []
        self.accuracy = []
        self.f1Score = []
        self.confusionMatrix = []
        self.precision = []
        self.recall = []

    def sampleModels(self, layerSkeleton, regularization=0.01, stepSize=0.01, batchSize=10, thresholdValue=0.5, stoppingCriterionCategory='epochs'):
        # Store all the models and their metrics to plot
        accuracy = []
        f1Score = []
        precision = []
        recall = []
        loss = []
        modelAccuracy = []
        modelF1Score = []
        models = []

        for layers in layerSkeleton:
            # Add input layer size to the beginning of the architecture
            l = layers.copy()
            l.insert(0, self.preprocessor.data.shape[1] - 1)

            # Create and train the model
            model = TrainModel(self.preprocessor, l, self.EPSILON, batchSize, regularization=regularization, stepSize=stepSize, threshold=thresholdValue, epoch=self.EPOCHS)
            print("\n\n")
            print(f"Model with layers {l}, regularization {regularization}, batch size {batchSize}, step size {stepSize} created successfully")

            # Train the model using k-fold cross-validation and get Learbing curve (mtric vs epoch list)
            accLC, preLC, recLC, f1LC, lossLC = model.kFoldTrainTest(stoppingCriterion=stoppingCriterionCategory)

            # Append metrics and model to their respective lists
            accuracy.append(accLC)
            f1Score.append(f1LC)
            precision.append(preLC)
            recall.append(recLC)
            loss = np.squeeze(lossLC)
            models.append(model)
            modelAccuracy.append(model.finalModalAccuracy)
            modelF1Score.append(model.finalModalF1Score)

            # plotLearningCurve(accLC, f1LC, preLC, recLC, title="Model Performance")
            # plotLearningCurveLoss(loss, title="Model Learning Curve of {} with architecture {} regularization={}, stepSize={}, batchSize={}".format(self.filePath.split('/')[2],l,regularization, stepSize, batchSize))
            

        # Plot the metrics
        plotMetrics(modelAccuracy, modelF1Score,  models, title="Model Performance of {} with regularization={}, stepSize={}, batchSize={}".format(self.filePath.split('/')[2],regularization, stepSize, batchSize))
        print("Model sampling completed successfully")

def plotLearningCurve(accuracy, f1Score, precision, recall, title="Model Performance"):
    # Set Seaborn style and color palette
    sns.set(style="whitegrid")
    palette = sns.color_palette("Set2")

    # Create the plot
    plt.figure(figsize=(14, 8))

    # Generate epoch numbers for the x-axis
    epochs = list(range(1, len(accuracy) + 1))

    metric_names = ["Accuracy", "F1 Score", "Recall", "Precision"]
    metric_values = [accuracy, f1Score, recall, precision]

    # Plot each metric
    for i, (metric_name, values) in enumerate(zip(metric_names, metric_values)):
        plt.plot(
            epochs,
            values,
            # marker='o',
            label=metric_name,
            color=palette[i],
            linestyle='-'
        )

    # Customize the plot
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10, loc='lower right')
    plt.tight_layout()
    plt.show()

def plotLearningCurveLoss(loss, title="Model Learning Curve"):
    # Set Seaborn style and color palette
    sns.set(style="whitegrid")
    palette = sns.color_palette("Set2")

    # Create the plot
    plt.figure(figsize=(14, 8))

    # Generate epoch numbers for the x-axis
    epochs = list(range(1, len(loss) + 1))

    metric_names = ["Loss"]
    metric_values = [loss]

    # Plot each metric
    for i, (metric_name, values) in enumerate(zip(metric_names, metric_values)):
        plt.plot(
            epochs,
            values,
            # marker='o',
            label=metric_name,
            color=palette[i],
            linestyle='-'
        )

    # Customize the plot
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10, loc='lower right')
    plt.tight_layout()
    plt.show()


def plotMetrics(accuracy, f1Score, models, title="Model Performance"):
    # Convert models to string representations of their architectures
    model_architectures = [f"Layers: {model.layersSkeleton}" for model in models]

    # Calculate mean and standard deviation for each metric
    metrics = {
        "Accuracy": accuracy,
        "F1 Score": f1Score,
    }
    metric_means = {key: np.mean(values) for key, values in metrics.items()}
    metric_stds = {key: np.std(values) for key, values in metrics.items()}

    # Set Seaborn style and color palette
    sns.set(style="whitegrid")
    palette = sns.color_palette("Set2")

    # Create the plot
    plt.figure(figsize=(14, 8))

    # Plot each metric with error bars (mean ± std deviation)
    for i, (metric_name, values) in enumerate(metrics.items()):
        plt.errorbar(
            model_architectures,
            values,
            yerr=metric_stds[metric_name],
            fmt='o',
            label=f"{metric_name} (Mean: {metric_means[metric_name]:.2f})",
            color=palette[i],
            capsize=5,
            capthick=2,
            markersize=8,
            linestyle='--'
        )

    # Annotate metric values on every point
    for i, (acc, f1) in enumerate(zip(accuracy, f1Score)):
        plt.annotate(f"{acc:.2f}", (i, accuracy[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
        plt.annotate(f"{f1:.2f}", (i, f1Score[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

    plt.xlabel('Model Architecture', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(ticks=range(len(model_architectures)), labels=model_architectures, rotation=45, ha='right', fontsize=10)
    plt.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.show()
def moving_average(data, window_size=5):
    """
    Compute the moving average of a list of values.
    :param data: List of values (e.g., loss values).
    :param window_size: Size of the moving average window.
    :return: Smoothed list of values.
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')



if __name__ == "__main__":
    # layer_skeletons = [
    # [2, 1], [4, 1], [8, 1], [16, 1],  # 1 Hidden Layer
    # [2, 4, 1], [4, 8, 1], [8, 16, 1], [16, 8, 1],  # 2 Hidden Layers
    # [2, 4, 8 , 1], [4, 8, 16, 1], [8, 16, 8, 1], [16, 8, 4, 1],  # 3 Hidden Layers
    # [2, 4, 8, 16, 1], [4, 8, 16, 8, 1], [8, 16, 8, 4, 1], [16, 8, 4, 2, 1]  # 4 Hidden Layers
    # ]
    modelSampler = ModelSampler(filePath='ANN/datasets/raisin.csv')
    for reg in modelSampler.REGULARIZATION_VALUES:
        for step in modelSampler.STEP_SIZE_VALUES:
            for batch in modelSampler.BATCH_SIZE_VALUES:
                print(f"Sampling models with regularization={reg}, stepSize={step}, batchSize={batch}")
                modelSampler.sampleModels(layerSkeleton = RAISIN_LAYERS_SKELETON ,regularization=reg, stepSize=step, batchSize=batch,stoppingCriterionCategory='error')
                print("Model sampling completed successfully")
    # modelSampler.sampleModels(layerSkeleton = LOAN_LAYERS_SKELETON ,regularization=0.01, stepSize=0.1, batchSize=32,stoppingCriterionCategory='error')
    print("Model sampling completed successfully")

    # Raisin = [7,4,8,16,1]
