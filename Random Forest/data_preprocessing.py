import pandas as pd
import numpy as np

class Preprocessor:
    def __init__(self, df):
        self.df = df.copy()  # Work on a copy!
        self.X = None
        self.y = None
        self.encoder_dict = {}
        self.numerical_cols = []
        self.categorical_cols = []
        self.label_col = None  # To store the label column name

    def preprocess(self):
        self._identify_cols()
        self._encode_categorical()
        self._handle_missing_values()
        self.X = self.df.drop(columns=[self.label_col]).values
        self.y = self.df[self.label_col].values
        self.y = self.y.astype(int)  # Ensure labels are integers

    def _identify_cols(self):
        # Find the label column (assumes it's the last column)
        self.label_col = self.df.columns[-1]

        for column in self.df.columns:
            if column != self.label_col:  # Don't include the label column here
                if pd.api.types.is_object_dtype(self.df[column]):
                    self.categorical_cols.append(column)
                else:
                    self.numerical_cols.append(column)

    def _encode_categorical(self):
        for column in self.categorical_cols:
            unique_values = self.df[column].unique()
            self.encoder_dict[column] = {val: idx for idx, val in enumerate(unique_values)}
            self.df[column] = self.df[column].map(self.encoder_dict[column])

    def _handle_missing_values(self):
        for column in self.df.columns:
            if column != self.label_col and self.df[column].isnull().any(): # Skip label column
                if column in self.numerical_cols:
                    self.df[column] = self.df[column].fillna(self.df[column].mean())
                else:
                    self.df[column] = self.df[column].fillna(self.df[column].mode()[0])