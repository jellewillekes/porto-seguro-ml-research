import os
import pandas as pd
from sklearn.model_selection import train_test_split


# Function to load the original data and split it into train, validation, test, and new data
def load_and_split_data(train_path='train_original.csv', test_size=0.2, validation_size=0.25, new_data_size=5,
                        random_state=42):
    """
    Load original data, split into train, validation, test, and new data for individual predictions.

    Parameters:
    - train_path: Path to the original training dataset (CSV file).
    - test_size: Percentage of data to be used as test set.
    - validation_size: Percentage of training data to be used as validation set.
    - new_data_size: Number of samples to reserve as new individual data for predictions.
    - random_state: Seed for reproducibility.

    Saves:
    - train.csv, validation.csv, test.csv, new_data.csv in the current 'data' folder.
    """
    # Load original data
    data = pd.read_csv(train_path)
    X = data.drop('target', axis=1)
    y = data['target']

    # Step 1: Split data into train+validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Step 2: Split train+validation into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=validation_size, stratify=y_train_val, random_state=random_state
    )

    # Step 3: Save the train, validation, and test sets in the 'data' folder
    pd.concat([X_train, y_train], axis=1).to_csv('train.csv', index=False)
    pd.concat([X_val, y_val], axis=1).to_csv('validation.csv', index=False)
    pd.concat([X_test, y_test], axis=1).to_csv('test.csv', index=False)
    print("Train, validation, and test sets saved in the 'data' folder.")

    # Step 4: Save a small sample from the test set as new data for individual predictions
    new_data = pd.concat([X_test, y_test], axis=1).sample(n=new_data_size, random_state=random_state)
    new_data.to_csv('new_data.csv', index=False)
    print("New data sample for predictions saved as 'new_data.csv'.")


if __name__ == "__main__":
    load_and_split_data()
