import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


class Analyser:
    def __init__(self,data=None):
        self.data = data;

    def show_unique_categories(self):
        for col in self.data.columns:
            unique_values = self.data[col].unique()
            print(unique_values)

    def info(self):
        print(self.data.info())
        print(self.data.describe())

    def target_distribution(self):
        plt.figure(figsize=(8, 5))
        sns.countplot(x=self.data['class'], palette='Set2')
        plt.title("Target Data Distribution")
        plt.xlabel("Target Category")
        plt.ylabel("Count")
        plt.show()

    
    def corr_label_data(self):
        # Loop through all the columns (excluding target 'class')
        for i, category in enumerate(self.data.columns[:-1]):  # Assuming the target is the last column
            sns.countplot(x=self.data[category], hue=self.data['class'])  # Swarm plot for categorical target
            plt.title(f"Distribution of Class by {category}")
            plt.show()



