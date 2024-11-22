import os
import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
from bayes_opt import BayesianOptimization
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


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


gini_scorer = make_scorer(normalized_gini, needs_proba=True)


# Load data
def load_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    X_train = train.drop('target', axis=1)
    y_train = train['target']

    X_test = test.drop('target', axis=1)
    y_test = test['target']

    return X_train, X_test, y_train, y_test


# Train Denoising Autoencoder
def train_dae(X_train):
    input_dim = X_train.shape[1]
    hidden_dim = 64

    # Define Denoising Autoencoder architecture
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(hidden_dim, activation='relu')(input_layer)
    encoded = Dense(hidden_dim // 2, activation='relu')(encoded)
    decoded = Dense(hidden_dim, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    # Compile and train the DAE
    dae = Model(inputs=input_layer, outputs=decoded)
    dae.compile(optimizer='adam', loss='mse')

    # Add noise to training data
    X_train_noisy = X_train + 0.1 * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
    dae.fit(X_train_noisy, X_train, epochs=10, batch_size=64, validation_split=0.1, verbose=1)

    # Create encoder model for feature extraction
    encoder = Model(inputs=dae.input, outputs=encoded)
    return encoder


# Get denoised features
def get_denoised_features(encoder, X):
    return encoder.predict(X)


# Bayesian Optimization for Base Models
def tune_model(X, y, model_type):
    def model_cv(**params):
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        gini_scores = []

        for train_idx, val_idx in kf.split(X, y):
            X_train_k, X_val_k = X[train_idx], X[val_idx]
            y_train_k, y_val_k = y[train_idx], y[val_idx]

            if model_type == 'xgb':
                model = XGBClassifier(
                    max_depth=int(params['max_depth']),
                    learning_rate=params['learning_rate'],
                    colsample_bytree=params['colsample_bytree'],
                    subsample=params['subsample'],
                    n_estimators=100,
                    random_state=42
                )
            elif model_type == 'lgb':
                model = LGBMClassifier(
                    max_depth=int(params['max_depth']),
                    learning_rate=params['learning_rate'],
                    colsample_bytree=params['colsample_bytree'],
                    subsample=params['subsample'],
                    n_estimators=100,
                    random_state=42
                )
            elif model_type == 'cat':
                model = CatBoostClassifier(
                    depth=int(params['depth']),
                    learning_rate=params['learning_rate'],
                    subsample=params['subsample'],
                    n_estimators=100,
                    random_state=42,
                    verbose=0
                )

            model.fit(X_train_k, y_train_k)
            y_pred = model.predict_proba(X_val_k)[:, 1]
            gini_scores.append(normalized_gini(y_val_k, y_pred))

        return np.mean(gini_scores)

    # Parameter bounds for tuning
    if model_type == 'xgb' or model_type == 'lgb':
        param_bounds = {
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.2),
            'colsample_bytree': (0.5, 0.9),
            'subsample': (0.5, 0.9)
        }
    elif model_type == 'cat':
        param_bounds = {
            'depth': (3, 10),
            'learning_rate': (0.01, 0.2),
            'subsample': (0.5, 0.9)
        }

    optimizer = BayesianOptimization(f=model_cv, pbounds=param_bounds, random_state=42)
    optimizer.maximize(init_points=10, n_iter=20)
    best_params = optimizer.max['params']
    best_params = {k: int(v) if 'depth' in k or 'max_depth' in k else v for k, v in best_params.items()}
    return best_params


# Train Base Models with Denoised Features
def train_base_models(X_train_combined, y_train, X_test_combined):
    # Train and tune models
    xgb_params = tune_model(X_train_combined, y_train, model_type='xgb')
    xgb_model = XGBClassifier(**xgb_params, n_estimators=100, random_state=42)
    xgb_model.fit(X_train_combined, y_train)
    xgb_train_pred = xgb_model.predict_proba(X_train_combined)[:, 1]
    xgb_test_pred = xgb_model.predict_proba(X_test_combined)[:, 1]

    lgb_params = tune_model(X_train_combined, y_train, model_type='lgb')
    lgb_model = LGBMClassifier(**lgb_params, n_estimators=100, random_state=42)
    lgb_model.fit(X_train_combined, y_train)
    lgb_train_pred = lgb_model.predict_proba(X_train_combined)[:, 1]
    lgb_test_pred = lgb_model.predict_proba(X_test_combined)[:, 1]

    cat_params = tune_model(X_train_combined, y_train, model_type='cat')
    cat_model = CatBoostClassifier(**cat_params, n_estimators=100, random_state=42, verbose=0)
    cat_model.fit(X_train_combined, y_train)
    cat_train_pred = cat_model.predict_proba(X_train_combined)[:, 1]
    cat_test_pred = cat_model.predict_proba(X_test_combined)[:, 1]

    # Stack base model predictions as input to the meta-model
    meta_X_train = np.column_stack((xgb_train_pred, lgb_train_pred, cat_train_pred))
    meta_X_test = np.column_stack((xgb_test_pred, lgb_test_pred, cat_test_pred))

    return meta_X_train, meta_X_test, y_train


# Train and save meta-model
def train_meta_model(meta_X_train, y_train, meta_X_test, y_test):
    meta_model = LogisticRegression()
    meta_model.fit(meta_X_train, y_train)
    meta_y_pred = meta_model.predict_proba(meta_X_test)[:, 1]
    meta_gini_score = normalized_gini(y_test, meta_y_pred)

    # Save the meta-model
    model_folder = "models"
    os.makedirs(model_folder, exist_ok=True)
    model_filename = f"metamodel_gini{meta_gini_score:.4f}.pkl"
    model_path = os.path.join(model_folder, model_filename)
    with open(model_path, 'wb') as f:
        pickle.dump(meta_model, f)
    print(f"Meta-Model saved to {model_path} with Test Gini Score: {meta_gini_score:.4f}")

    return meta_model, meta_gini_score


# Main Execution
if __name__ == "__main__":
    # Load the data
    X_train, X_test, y_train, y_test = load_data()

    # Train DAE and get denoised features
    encoder = train_dae(X_train)  # Train the DAE and get the encoder part
    X_train_denoised = get_denoised_features(encoder, X_train)
    X_test_denoised = get_denoised_features(encoder, X_test)

    # Combine original features with denoised features
    X_train_combined = np.concatenate([X_train, X_train_denoised], axis=1)
    X_test_combined = np.concatenate([X_test, X_test_denoised], axis=1)

    # Train the base models (XGBoost, LightGBM, and CatBoost) on combined features
    meta_X_train, meta_X_test, y_train = train_base_models(X_train_combined, y_train, X_test_combined)

    # Train and save the meta-model
    train_meta_model(meta_X_train, y_train, meta_X_test, y_test)
