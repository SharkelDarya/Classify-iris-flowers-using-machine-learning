# Classify-iris-flowers-using-machine-learning
Classifying iris flowers using various machine learning algorithms.

<p align="center">
      <img src="https://i.ibb.co/NSS0bc2/iris-2.webp" alt="Project Logo" width="746">
</p>

<p align="center">
   <img src="https://img.shields.io/badge/IDE-PyCharm%2023-B7F352" alt="IDE">
</p>

## About
The project is a machine learning-based classification using the publicly available Iris dataset of 150 records. 
The idea is to classify irises based on their features, such as plot length, plot width, petal length and petal width. </br>
Machine learning models used:
- **-** **`K-Nearest Neighbors (KNN)`**
- **-** **`Support Vector Machine (SVM)`**
- **-** **`Random Forest`**
## Documentation

### Libraries
**-** **`NumPy`**, **`pandas`**, **`matplotlib`**, **`seaborn`**, **`scikit-learn`**

### Data preparation
- The data is loaded from the file "iris.csv".
- A feature matrix X and a label vector y are generated.
### Data visualization
- Graphs are created to visually analyze the relationship between features and class distributions.
### K-Nearest Neighbours (KNN):
- The KNN model is initialized with the number of neighbors (k=2).
- The model is trained on training data.
- Predictions are made on test data.
- Results include accuracy, classification report and confusion matrix.
### Maszyna wektorów nośnych (SVM):
- SVM model with linear kernel and parameter C=2 is initialized and trained.
- Predictions are made on test data.
- Results include accuracy, classification report and confusion matrix.
### Random Forest:
- Random Forest model with 100 trees is initialized and trained.
- Predictions are made on test data.
- The validity of the features of the validity chart is calculated.
- Results include accuracy, classification report and confusion matrix.
### Attribute importance analysis:
- The validity of each feature is displayed and a validity graph is plotted.
### Cross-validation and grid search:
- KFold is used to perform cross-validation.
- A grid search is performed for the Random Forest model to find the optimal parameters.
- Cross-validation results and optimal model parameters are derived.
### Przewidywanie na nowych danych:
- Prognozy są wykonywane dla trzech nowych zestawów danych.
## Developers

- Darya Sharkel (https://github.com/SharkelDarya)

