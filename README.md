# COMPSCI-589-Machine-Learning


## Decision Tree
The code includes:
- **Data Preprocessing**: Encoding data, splitting the data
- **Decision Tree Training**: **Information Gain (ID3)** or **Gini Index (CART)** as the splitting criterion.
- **Stopping Criterion**: A simple heuristic to stopping tree growth when a node is already 85% pure.
- **Model Evaluation**: Calculating accuracy and visualizing results using histograms and variance plots.

### Usage (main.py)
- **Creating sampler module with parameter**:
    ```python
    class DecisionTreeSampler(
        df: Any, #Pass dataset as dataframe
        test_data: bool = True, #To evaluate on test data pass True, for train data pass False
        sampling_runs: int = 100,
        metric: str = "id3", #'id3' for Information Gain; "cart" for Gini
        stopping_criteria: Any | None = None #Value eg. 85 for enabling pruning
    )
    #Example object creation
    decisionTreeSampler = DecisionTreeSampler(df,test_data=False,metric="id3",stopping_criteria=85)
    ```
- **Run the model**:
The run function return standard deviation and mean.
    ```python
    train_mean , train_std = decisionTreeSampler.run()
    ```
- **Visualize the results**:
The plotHistogram function plots the accuracy distribution with mean and standard deviation
    ```python
    decisionTreeSampler.plotHistogram()
    ```
- **Compare the results**:
The compareHistogram superimposes any 2 list of accuracies passed to it to compare both the graphs.
    ```python
    (function) def compareHistogram(
        accuracies1: Any, 
        accuracies2: Any,
        label1: Any,
        label2: Any,
        test_data: bool = True #to print if its for test or train data
    ) -> None
    #Example
    decisionTreeSampler.plotHistogram()
    ```
- **Test for robustness of the model**:
variance_plot runs the model 100 times using id3 criteria for different test and train splits
    ```python
    (function) def variance_plot(df: Any) -> None
    #Example
    variance_plot(df)
    ```
- **(Optional) Basic data analytics performed on the data**:
The analyser object performs some basic analysis on the data and gives some visualizations
    ```python
    analyserObj = Analyser(df) #pass the dataset as dataframe
    analyserObj.info()
    analyserObj.show_unique_categories()
    analyserObj.corr_label_data()
    ```

## K Nearest Neighbors
This code includes:
- **Data Preprocessing**: Normalizes the data using Min Max, splitting data into training and test sets.
- **KNN Training and Evaluation**: Building and evaluating model for different values of K.
- **Visualization**: Generating plots to compare the performance 

### Usage (main.py)
- **Creating sampler module with parameter**:
    ```python
    class KKNSampler(
        df: Any, #pass the dataset as dataframe
        k_range: tuple, #pass the range of k to be tested
        test_data: bool = True, #to test on train data pass False else pass True
        sampling_runs: int = 20, #number of runs for each K
        normalized: bool = True #to normalize the data pass True else False to test on raw data
    )
    #Example object creation
    knnSampler = KKNSampler(df,k_range=range(1, 52, 2), test_data=False, sampling_runs=20)
    ```
- **Run and plot accuracies the model**:
The run function runs the model for said times and saves the mean, standard deviation for each values of K. The plot function plots a line graph with standard deviation.
    ```python
    knnSamplerTrainN.run()
    knnSamplerTrainN.plot() #Pass false to display title for 'Not Normalized data'
    ```
- **Compare the results**:
The plotComparision superimposes any 2 sampler objects to it to compare both the models.
    ```python
    (function) def plotComparision(
        knn1: KKNSampler, #sampler object 1
        knn2: KKNSampler, #sampler object 2
        label1: Any, #label for model 1 to display in graph
        label2: Any, #label for model 2 to display in graph
        title: Any #overall title to display
    ) -> None
    #Example
    plotComparision(knnSamplerTestN, knnSamplerTest, 'With Normalization', 'Without Normalization', 'Comparing with and without normalizing test data')
    ```
- **(Optional) Basic data analytics performed on the data**:
The analyser object performs some basic analysis on the data and gives some visualizations
    ```python
    analyserObj = Analyser(df) #pass the dataset as dataframe
    analyserObj.info()
    analyserObj.calculate_A_Priori()
    analyserObj.plot()
    analyserObj.plot_pie()
    ```
