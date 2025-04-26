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

class ModelSampler:
    EPSILON = 0.01
    REGULARIZATION_VALUES = [0.01,0.025]
    STEP_SIZE_VALUES = [0.01, 0.05]
    BATCH_SIZE_VALUES = [5, 10]
    LAYERS_SKELETON = [[1], [5,1],[3, 1],[3, 2, 1]]
    # WDBC: [[1], [20,1], [18,1],[15, 22, 1], [18,20,18,1]] 
    # LOAN: [[1], [5,1], [12,1],[5, 10, 1], [14,5,8,1]]
    # TITANIC: [[1], [5,1],[3, 1],[3, 2, 1]]
    # RAISIN: [[1], [5,1],[6,5, 1],[5, 6, 5, 1]]
    K_FOLD = 5
    
    def __init__(self, filePath):
        self.filePath = filePath
        self.preprocessor = DataPreprocessor(filePath=filePath, kFold=self.K_FOLD)
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

    def sampleModels(self, regularization=0.01, stepSize=0.01, batchSize=10):
        # Store all the models and their metrics to plot
        accuracy = []
        f1Score = []
        precision = []
        recall = []
        models = []

        for layers in self.LAYERS_SKELETON:
            # Add input layer size to the beginning of the architecture
            l = layers.copy()
            l.insert(0, self.preprocessor.data.shape[1] - 1)

            # Create and train the model
            model = TrainModel(self.preprocessor, l, self.EPSILON, batchSize, regularization=regularization, stepSize=stepSize, threshold=0.5)
            print("\n\n")
            print(f"Model with layers {l}, regularization {regularization}, batch size {batchSize}, step size {stepSize} created successfully")

            # Train the model using k-fold cross-validation and get averaged metrics
            acc, pre, rec, f1 = model.kFoldTrainTest()

            # Append metrics and model to their respective lists
            accuracy.append(acc)
            precision.append(pre)
            recall.append(rec)
            f1Score.append(f1)
            models.append(model)

        # Plot the metrics
        plotMetrics(accuracy, f1Score, precision, recall, models, title="Model Performance of {} with regularization={}, stepSize={}, batchSize={}".format(self.filePath.split('/')[2],regularization, stepSize, batchSize))
        print("Model sampling completed successfully")


def plotMetrics(accuracy, f1Score, precision, recall, models, title="Model Performance"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Convert models to string representations of their architectures
    model_architectures = [f"Layers: {model.layersSkeleton}" for model in models]

    # Calculate mean and standard deviation for each metric
    metrics = {
        "Accuracy": accuracy,
        "F1 Score": f1Score,
        "Precision": precision,
        "Recall": recall
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
    for i, (acc, f1, pre, rec) in enumerate(zip(accuracy, f1Score, precision, recall)):
        plt.annotate(f"{acc:.2f}", (i, accuracy[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
        plt.annotate(f"{f1:.2f}", (i, f1Score[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
        plt.annotate(f"{pre:.2f}", (i, precision[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
        plt.annotate(f"{rec:.2f}", (i, recall[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

    # Customize the plot
    plt.xlabel('Model Architecture', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.title(title, fontsize=14)
    # plt.ylim(0,100)
    plt.xticks(ticks=range(len(model_architectures)), labels=model_architectures, rotation=45, ha='right', fontsize=10)
    plt.legend(fontsize=10, loc='upper left')
    plt.tight_layout()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    modelSampler = ModelSampler(filePath='ANN/datasets/titanic.csv')
    modelSampler.sampleModels(regularization=0.01, stepSize=0.01, batchSize=10)
    print("Model sampling completed successfully")


