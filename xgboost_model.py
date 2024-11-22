import os
import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold


# Gini Coefficient Calculation
def gini_coefficient(y_true, y_pred):
    arr = np.array([y_true, y_pred]).transpose()
    arr = arr[np.lexsort((arr[:, 0], -arr[:, 1]))]
    total_losses = arr[:, 0].sum()
    gini_sum = arr[:, 0].cumsum().sum() / total_losses
    gini_sum -= (len(y_true) + 1) / 2
    return gini_sum / len(y_true)


def normalized_gini(y_true, y_pred):
    return gini_coefficient(y_true, y_pred) / gini_coefficient(y_true, y_true)


# Load pre-split data
def load_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    X_train = train.drop('target', axis=1)
    y_train = train['target']

    X_test = test.drop('target', axis=1)
    y_test = test['target']

    return X_train, X_test, y_train, y_test


# Bayesian Optimization with Stratified K-Fold Cross-Validation
def bayesian_optimization(X, y, init_points=10, n_iter=20, n_splits=5):
    def xgb_cv(colsample_bytree, learning_rate, max_depth, min_child_weight, reg_alpha, reg_lambda, subsample):
        max_depth = int(max_depth)
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        gini_scores = []

        for train_idx, val_idx in kf.split(X, y):
            X_train_k, X_val_k = X.iloc[train_idx], X.iloc[val_idx]
            y_train_k, y_val_k = y.iloc[train_idx], y.iloc[val_idx]

            model = XGBClassifier(
                colsample_bytree=colsample_bytree,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                subsample=subsample,
                n_estimators=1000,
                eval_metric='logloss',
                random_state=42
            )

            model.fit(X_train_k, y_train_k)
            y_pred_val = model.predict_proba(X_val_k)[:, 1]
            gini_val = normalized_gini(y_val_k, y_pred_val)
            gini_scores.append(gini_val)

        return np.mean(gini_scores)

    param_bounds = {
        'colsample_bytree': (0.5, 0.9),
        'learning_rate': (0.01, 0.2),
        'max_depth': (3, 10),
        'min_child_weight': (1, 10),
        'reg_alpha': (0, 1),
        'reg_lambda': (0, 1),
        'subsample': (0.5, 0.9)
    }

    optimizer = BayesianOptimization(
        f=xgb_cv,
        pbounds=param_bounds,
        random_state=42,
        verbose=2
    )

    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    best_params = optimizer.max['params']
    best_params['max_depth'] = int(best_params['max_depth'])
    best_score = optimizer.max['target']

    return best_params, best_score


# Retrain with Best Parameters and Save Model based on Test Gini Score
def train_and_save_best_model(X_train, y_train, X_test, y_test, best_params, model_name="xgboost"):
    model_folder = "models"
    os.makedirs(model_folder, exist_ok=True)

    model = XGBClassifier(
        colsample_bytree=best_params['colsample_bytree'],
        learning_rate=best_params['learning_rate'],
        max_depth=int(best_params['max_depth']),
        min_child_weight=best_params['min_child_weight'],
        reg_alpha=best_params['reg_alpha'],
        reg_lambda=best_params['reg_lambda'],
        subsample=best_params['subsample'],
        n_estimators=1000,
        eval_metric='logloss',
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred_test = model.predict_proba(X_test)[:, 1]
    gini_test = normalized_gini(y_test, y_pred_test)
    model_filename = f"{model_name}_test_gini{gini_test:.4f}.pkl"
    model_path = os.path.join(model_folder, model_filename)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path} with Test Set Normalized Gini Score: {gini_test:.4f}")

    return model, gini_test


# Main Execution
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    # Perform Bayesian optimization with Stratified K-Fold Cross-Validation
    best_params, best_score = bayesian_optimization(X_train, y_train)

    print(f"\nBest Validation Gini Score: {best_score:.4f}")
    print(f"Best Hyperparameters: {best_params}")

    # Retrain the model on full training set
    model, gini_test = train_and_save_best_model(X_train, y_train, X_test, y_test, best_params)
