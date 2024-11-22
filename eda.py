# eda.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load Data
def load_data(file_path='test_original.csv'):
    """
    Load the test dataset from a local CSV file.

    Parameters:
    - file_path: Path to the test CSV file.

    Returns:
    - test: The test DataFrame.
    """
    test = pd.read_csv(file_path)
    return test


# Show basic info about the data
def basic_info(data):
    """
    Display basic information about the dataset.

    Parameters:
    - data: DataFrame to describe.
    """
    print("Data Shape:", data.shape)
    print("\nData Info:\n")
    print(data.info())
    print("\nMissing Values:\n", data.isnull().sum())


# Check for class imbalance
def class_distribution(data, target_col='target'):
    """
    Plot the distribution of the target variable to check for class imbalance.

    Parameters:
    - data: DataFrame containing the target column.
    - target_col: Name of the target column.
    """
    if target_col in data.columns:
        sns.countplot(x=target_col, data=data)
        plt.title("Class Distribution")
        plt.xlabel("Target")
        plt.ylabel("Count")
        plt.show()
    else:
        print(f"Column '{target_col}' not found in dataset.")


# Feature analysis
def feature_analysis(data):
    """
    Display basic statistics and plot distributions for numeric features.

    Parameters:
    - data: DataFrame to analyze.
    """
    print("\nBasic Statistics for Numeric Features:\n")
    print(data.describe())

    # Plot distribution of each feature
    numeric_features = data.select_dtypes(include=[np.number]).columns
    num_features = len(numeric_features)
    cols = 3  # Set the number of columns for subplots
    rows = (num_features // cols) + (num_features % cols > 0)  # Calculate rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))  # Adjust figure size for clarity
    fig.suptitle("Feature Distributions", fontsize=16)
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    # Plot each feature
    for i, feature in enumerate(numeric_features):
        sns.histplot(data[feature].dropna(), kde=True, ax=axes[i])
        axes[i].set_title(f"Distribution of {feature}")

    # Turn off any unused axes
    for i in range(num_features, len(axes)):
        axes[i].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
    plt.show()


# Correlation analysis
def correlation_heatmap(data):
    """
    Plot a heatmap of correlations between numeric features.

    Parameters:
    - data: DataFrame to analyze.
    """
    plt.figure(figsize=(12, 8))
    correlation = data.corr()
    sns.heatmap(correlation, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.show()


# Main execution
if __name__ == "__main__":
    # Load test data
    test_data = load_data()

    # Basic information and class distribution
    basic_info(test_data)

    # Class distribution check (assuming target column if available in test data)
    # Replace 'target' with a different column if test data lacks a target
    class_distribution(test_data)

    # Feature analysis for potential model improvement insights
    feature_analysis(test_data)

    # Correlation heatmap to explore feature relationships
    correlation_heatmap(test_data)

    print("EDA Complete.")
