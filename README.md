# Auto Insurance Claim Prediction with Machine Learning

This repository contains an analysis and implementation of various machine learning models to predict auto insurance 
claims using Porto Seguro's "Safe Driver Prediction" dataset. The goal is to develop accurate models to predict the 
likelihood that a driver will file an insurance claim in the next year, thereby helping to set more accurate 
premiums. The start of this project was part of the Machine Learning course at Erasmus University, completed with Stein Dijkstra and Liza 
Mints.

## Project Overview
Predicting insurance claims is a crucial task for setting proper premiums. Porto Seguro, one of Brazil’s largest insurance companies, provided a dataset containing customer, vehicle, and driving-related features. The dataset is used to build machine learning models that can predict whether a driver will file a claim in the upcoming year.

## 1. Dataset
The dataset used in this project is the Porto Seguro Safe Driver Prediction dataset, which includes:
- **Target Variable**: Whether a driver will file an insurance claim (binary).
- **Features**: 91 features that include demographic and vehicle-related information.

Data files:
- `train.csv`: Training data with target labels.
- `test.csv`: Test data without target labels, used for model validation.

## 2. Methodology

### Data Preprocessing
1. **Train-Test Split**: The dataset is split into 80% training and 20% testing.
2. **Feature Scaling**: Various scaling techniques like MinMaxScaler and StandardScaler are applied to preprocess features.

### Model Selection
The following machine learning models are applied and tuned:
- **Neural Network (NN)**: A feedforward neural network optimized using a grid search.
- **Random Forest Classifier (RF)**: A random forest model optimized for parameters like depth, number of estimators, and more.
- **AdaBoost with Decision Tree (AB Tree)**: A boosting model with a decision tree as the base estimator.

#### Hyperparameter Tuning
Grid search is used for hyperparameter tuning:
- **Neural Network**: Optimized by adjusting batch size, neuron count, activation functions, initialization methods, and optimizers.
- **Random Forest**: Optimized by tuning the number of trees, depth, and feature selection.
- **AdaBoost**: Both the boosting parameters and base decision tree parameters are tuned.

#### Model Evaluation
Each model is evaluated using **classification accuracy**, and the **McNemar test** is used to compare the performance of the neural network to a random baseline.

## 3. Results

The best-performing model was a **Feedforward Neural Network (NN)** with an accuracy of **58.74%** on the test set. Below are the results from the different models:

| Model         | Accuracy | Hyperparameters                                                                                 |
|---------------|----------|-------------------------------------------------------------------------------------------------|
| **Neural Network** | 58.74%   | Layers: 16, 8, 8; Activation: ReLU; Epochs: 300; Dropout: 0.5; Optimizer: RMSprop; Batch size: 256 |
| **Random Forest**  | 57.87%   | Estimators: 200; Criterion: gini; Max depth: 10; Max features: log2; Min sample split: 40    |
| **AdaBoost Tree**  | 57.23%   | Estimators: 50; Learning rate: 0.5; Criterion: entropy; Max depth: 2; Max features: auto; Min sample split: 40 |

The neural network performed better than both Random Forest and AdaBoost, and no combination of models (via averaging) exceeded its accuracy.

## 4. Model Details

### Neural Network (NN)
- **Structure**: A three-layer neural network with 16, 8, and 8 neurons, respectively.
- **Activation Function**: ReLU to avoid vanishing gradients.
- **Optimizer**: RMSprop was used as it is efficient for deep learning tasks.
- **Tuning**: The model was tuned using grid search, testing various batch sizes, neuron configurations, and activation functions.

### Random Forest (RF)
- **Best Configuration**: The optimal random forest had 200 estimators, a maximum depth of 10, and used the "gini" criterion for impurity calculation.
- **Feature Selection**: The "log2" max features worked best to prevent overfitting while ensuring model efficiency.

### AdaBoost with Decision Tree (AB Tree)
- **Boosting**: AdaBoost was applied with a decision tree as the base estimator. While it performed slightly worse than the NN and RF, it remained a competitive option.

### Model Combination
An attempt was made to combine the models using a linear averaging approach. However, this did not outperform the neural network.

## 5. Files in the Repository

- **main.py**: The main Python script containing all functions for data preprocessing, model training, hyperparameter tuning, and evaluation.
- **Report.pdf**: Full project report with methodology, findings, and conclusions.
- **train.csv**: Training dataset.
- **test.csv**: Test dataset used for model validation.


## 6. Installation and Requirements

The project requires the following libraries:
```plaintext
numpy
pandas
seaborn
matplotlib
scipy
scikit-learn
tensorflow
keras
