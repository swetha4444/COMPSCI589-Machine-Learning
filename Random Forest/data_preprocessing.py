import pandas as pd
import numpy as np

class Preprocessor:
    def __init__(self, df, label_col=None):
        self.df = df.copy()
        self.X = None
        self.y = None
        self.numerical_cols = []
        self.categorical_cols = []
        self.label_col = label_col
        self.encoded_columns = []

    def preprocess(self):
        self._identify_label()
        self._identify_cols()
        self._handle_missing_values()
        self._one_hot_encode()
        
        self.X = self.df.drop(columns=[self.label_col]).values
        self.y = self.df[self.label_col].values
        self.y = self.y.astype(int)
        return self.X, self.y

    def _identify_label(self):
        if self.label_col is None:
            if 'label' in self.df.columns:
                self.label_col = 'label'
            elif 'target' in self.df.columns:
                self.label_col = 'target'
                self.df.rename(columns={'target': 'label'}, inplace=True)
            else:
                self.label_col = self.df.columns[-1]
                self.df.rename(columns={self.label_col: 'label'}, inplace=True)
                self.label_col = 'label'

    def _identify_cols(self):
        for column in self.df.columns:
            if column != self.label_col:
                if pd.api.types.is_object_dtype(self.df[column]) or pd.api.types.is_categorical_dtype(self.df[column]):
                    self.categorical_cols.append(column)
                else:
                    self.numerical_cols.append(column)

    def _handle_missing_values(self):
        for column in self.df.columns:
            if column != self.label_col and self.df[column].isnull().any():
                if column in self.numerical_cols:
                    self.df[column] = self.df[column].fillna(self.df[column].mean())
                else:
                    self.df[column] = self.df[column].fillna(self.df[column].mode()[0])

    def _one_hot_encode(self):
        if self.categorical_cols:
            # Perform one-hot encoding
            encoded_df = pd.get_dummies(self.df[self.categorical_cols], drop_first=True)
            self.encoded_columns = encoded_df.columns.tolist()
            
            # Drop original categorical columns and add encoded ones
            self.df = pd.concat([
                self.df.drop(columns=self.categorical_cols),
                encoded_df
            ], axis=1)
            
            # Update numerical columns list to include encoded columns
            self.numerical_cols.extend(self.encoded_columns)