import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

class PreProcesser:
    def __init__(self, data):
        self.data = data
    
    def encodeData(self):
        # Encoding categorical data
        self.encoder_dict = {}  
        for col in self.data.columns:
            unique_values = self.data[col].unique() 
            self.encoder_dict[col] = {value: idx for idx, value in enumerate(unique_values)}  
            self.data[col] = self.data[col].map(self.encoder_dict[col]) 

    def setData(self):
        # Assuming the last column is the label (class)
        self.X = self.data.iloc[:, :-1].values  # Select all columns except the last one
        self.y = self.data.iloc[:, -1].values   # Select the last column as the label
    
    def preprocess(self):
        self.encodeData()
        self.setData()
    
    def split(self,testPercent=20, random_state=None):
        return train_test_split(self.X,self.y, test_size=testPercent/100,shuffle=True, random_state=None)