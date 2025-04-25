import numpy as np
import pandas as pd

class DataPreprocessor:
    def __init__(self, filePath, labelColumn = 'label', kFold = 5, randomSeed = 42,):
        self.kFold = kFold
        self.randomSeed = randomSeed
        self.filePath = filePath
        self.data = None
        self.labelColumn = labelColumn

    def load_data(self):
        self.data = pd.read_csv(self.filePath)
        print(f"Data loaded successfully from {self.filePath}")
    
    def encodeCategorical(self):
        self.data = pd.get_dummies(self.data, drop_first=True)
        print("Categorical variables encoded successfully")
    
    def normalizeData(self):
        numeric_columns = self.data.select_dtypes(include=[np.number])
        self.data[numeric_columns.columns] = (numeric_columns - numeric_columns.min()) / (numeric_columns.max() - numeric_columns.min())
        print("Data normalized successfully")

    def stratifiedKFold(self): 
        np.random.seed(self.randomSeed)
        self.data = self.data.sample(frac=1, random_state=self.randomSeed).reset_index(drop=True)
        labels = self.data[self.labelColumn].values

        uniqueLabels = np.unique(labels)
        self.foldedDataIndex = [[] for _ in range(self.kFold)]
        for label in uniqueLabels:
            labelData = self.data[self.data[self.labelColumn] == label]
            labelDataIndices = labelData.index.tolist()
            foldSize = len(labelData) // self.kFold
            for i in range(self.kFold):
                start = i * foldSize
                end = (i + 1) * foldSize if i != self.kFold - 1 else len(labelData)  # last fold gets remainder
                self.foldedDataIndex[i].extend(labelDataIndices[start:end])
        return self.foldedDataIndex
    
    def printDataDetails(self):
        print(f"Data shape: {self.data.shape}")
        print(f"Label distribution:\n{self.data[self.labelColumn].value_counts()}")
        print(f"Data has been split into {self.kFold} folds")
        for i, fold in enumerate(self.foldedDataIndex):
            print(f"\tFold {i+1}: {len(fold)} samples")

if __name__ == "__main__":
    preprocessor = DataPreprocessor(filePath='ANN/datasets/raisin.csv')
    preprocessor.load_data()
    preprocessor.encodeCategorical()
    preprocessor.normalizeData()
    folds = preprocessor.stratifiedKFold()
    preprocessor.printDataDetails()



