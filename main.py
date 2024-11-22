# Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping
import scipy.stats

# Open a file to save results
results_file = open("results/results.txt", "w")


# Load Data
def load_data():
    train_path = 'data/train_original.csv'
    test_path = 'data/test_original.csv'
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


train, test = load_data()

X_train_complete = train.drop('target', axis=1)
y_train_complete = train['target']
X_test_complete = test

# Split Data
X = train.drop('target', axis=1)
y = train['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)


# Baseline Evaluation
def evaluate_baseline(y_test, file):
    predictions_coin = np.random.randint(0, 2, y_test.shape)
    file.write("Baseline Random Guess Evaluation:\n")
    file.write(classification_report(y_test, predictions_coin))
    file.write("\n")
    file.write(str(confusion_matrix(y_test, predictions_coin)))
    file.write("\n\n")


evaluate_baseline(y_test, results_file)


# Random Forest Model
def random_forest_model(X_train, y_train, file):
    param_grid = {
        'n_estimators': [100, 200, 400, 800],
        'max_features': ['auto', 'sqrt', 'log2'],
        'min_samples_split': [5, 20, 40],
        'max_depth': [2, 5, 10, 50],
        'criterion': ['gini', 'entropy']
    }
    rfc = RandomForestClassifier(random_state=42)
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=3, verbose=2)
    CV_rfc.fit(X_train, y_train)
    file.write("Random Forest Best Params:\n")
    file.write(str(CV_rfc.best_params_))
    file.write("\nBest Score: ")
    file.write(str(CV_rfc.best_score_))
    file.write("\n\n")
    return CV_rfc.best_estimator_


rfc_best = random_forest_model(X_train, y_train, results_file)


# AdaBoost Model
def adaboost_model(X_train, y_train, file):
    param_grid = {
        'n_estimators': [10, 50, 100],
        'learning_rate': [1, 0.5],
        'estimator__max_features': ['auto', 'sqrt', 'log2'],
        'estimator__min_samples_split': [10, 40],
        'estimator__max_depth': [2, 5, 10, None],
        'estimator__criterion': ['gini', 'entropy']
    }
    DTC = DecisionTreeClassifier(random_state=42)
    ABC = AdaBoostClassifier(estimator=DTC)
    CV_ABC = GridSearchCV(ABC, param_grid=param_grid, scoring='accuracy', cv=3, verbose=2)
    CV_ABC.fit(X_train, y_train)
    file.write("AdaBoost Best Params:\n")
    file.write(str(CV_ABC.best_params_))
    file.write("\nBest Score: ")
    file.write(str(CV_ABC.best_score_))
    file.write("\n\n")
    return CV_ABC.best_estimator_


ABC_best = adaboost_model(X_train, y_train, results_file)


# Neural Network Model
def neural_network_model(X_train, y_train):
    layers = [16, 8, 8]
    initializers = 'glorot_uniform'
    activations = 'relu'
    optimizers = 'RMSprop'
    dropouts = 0.5
    batch_size = 256
    epochs = 300

    stopper = EarlyStopping(monitor='accuracy', patience=10, verbose=2)
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i == 0:
            model.add(Dense(nodes, kernel_initializer=initializers, input_dim=X_train.shape[1]))
        else:
            model.add(Dense(nodes))
        model.add(Activation(activations))
        model.add(Dropout(dropouts))
    model.add(Dense(units=1, kernel_initializer=initializers, activation='sigmoid'))
    model.compile(optimizer=optimizers, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[stopper])
    return model


bestNN = neural_network_model(X_train, y_train)


# Model Evaluation
def evaluate_model(model, X_test, y_test, model_name, file):
    predictions = model.predict(X_test)
    if isinstance(predictions[0], np.ndarray):  # if NN probabilities
        predictions = (predictions > 0.5).astype(int).flatten()
    file.write(f"{model_name} Evaluation:\n")
    file.write(classification_report(y_test, predictions))
    file.write("\n")
    file.write(str(confusion_matrix(y_test, predictions)))
    file.write("\n\n")


evaluate_model(rfc_best, X_test, y_test, "Random Forest", results_file)
evaluate_model(ABC_best, X_test, y_test, "AdaBoost", results_file)
evaluate_model(bestNN, X_test, y_test, "Neural Network", results_file)


# McNemar Test
def mcnemar_test(predictions1, predictions2, y_test, file):
    results = np.zeros((2, 2))
    for i in range(len(y_test)):
        if predictions1[i] == y_test[i] and predictions2[i] == y_test[i]:
            results[0, 0] += 1
        elif predictions1[i] != y_test[i] and predictions2[i] == y_test[i]:
            results[1, 0] += 1
        elif predictions1[i] == y_test[i] and predictions2[i] != y_test[i]:
            results[0, 1] += 1
        elif predictions1[i] != y_test[i] and predictions2[i] != y_test[i]:
            results[1, 1] += 1
    file.write("McNemar Test Results:\n")
    file.write(str(results))
    file.write("\n\n")


# Random baseline vs Neural Network
predictions_coin = np.random.randint(0, 2, y_test.shape)
predictions_nn = (bestNN.predict(X_test) > 0.5).astype(int).flatten()
mcnemar_test(predictions_coin, predictions_nn, y_test, results_file)

# Close the results file
results_file.close()
