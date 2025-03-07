# Data Pre processing module
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class PreProcesser:
    def __init__(self, data=None, normalized=True):
        self.data = data
        self.normalized = normalized

    def setData(self):
        # self.data = sklearn_shuffle(self.data);
        self.X = self.data.iloc[:, :-1] #selecting all columns except the last one
        self.y = self.data.iloc[:, -1] #selecting the last column
        
    def split(self,testPercent=20, random_state=None):
        return train_test_split(self.X,self.y, test_size=testPercent/100, random_state=random_state,shuffle=True)

    def normalize(self):
        if self.normalized:  self.X = ((self.X-self.X.min()) / (self.X.max() - self.X.min()))
        self.X = self.X.to_numpy()

    def preprocess(self):
        self.setData();
        self.normalize()

    def save(self, file_name):
        # Combining X and y into a single DataFrame without header
        self.data = pd.concat([pd.DataFrame(self.X), pd.DataFrame(self.y)], axis=1)
        self.data.to_csv(file_name, index=False, header=False)
        print("Data Saved\n")
    
    def load(self,file_name):
        self.data = pd.read_csv(file_name, header=None)
        print("Data Loaded\n")
