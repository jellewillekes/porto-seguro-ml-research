import os
import pickle
import pandas as pd
import re

# Define the path to the models folder
MODELS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


# Function to find the model with the highest Gini score
def load_best_model(models_folder=MODELS_FOLDER):
    # Get a list of all files in the models folder with 'gini' in the filename
    model_files = [f for f in os.listdir(models_folder) if 'gini' in f]

    if not model_files:
        raise FileNotFoundError("No model files found in the 'models' folder.")

    # Extract the Gini score from each model filename
    gini_scores = {}
    for model_file in model_files:
        match = re.search(r"gini([0-9]+\.[0-9]+)", model_file)
        if match:
            gini_score = float(match.group(1))
            gini_scores[model_file] = gini_score

    # Find the model with the highest Gini score
    best_model_file = max(gini_scores, key=gini_scores.get)
    print(f"Loading model: {best_model_file} with Gini score: {gini_scores[best_model_file]}")

    # Load the best model
    best_model_path = os.path.join(models_folder, best_model_file)
    with open(best_model_path, 'rb') as f:
        best_model = pickle.load(f)

    return best_model


# Load a specified observation and its target value from new_data.csv
def load_single_observation(index, data_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "new_data.csv")):
    # Load the data and select the specified row by index
    new_data = pd.read_csv(data_path)

    if index < 1 or index > len(new_data):
        raise ValueError(f"Index out of range. Please provide an index between 1 and {len(new_data)}.")

    observation = new_data.drop('target', axis=1).iloc[index - 1].values.reshape(1, -1)  # Drop 'target' and reshape
    actual_value = new_data['target'].iloc[index - 1]  # Get the actual target value for the specified row
    return observation, actual_value


# Main prediction function
def predict_single_observation(index=1):
    # Load the best model
    model = load_best_model()

    # Load the specified observation and the actual value
    observation, actual_value = load_single_observation(index)

    # Make a prediction
    prediction_proba = model.predict_proba(observation)[0, 1]  # Probability of the positive class
    prediction_class = model.predict(observation)[0]  # Predicted class label

    # Print the prediction and actual value
    print(f"Observation {index}:")
    print(f"Predicted Probability of Claim: {prediction_proba:.4f}")
    print(f"Predicted Class: {prediction_class}")
    print(f"Actual Value (Claim): {actual_value}")


if __name__ == "__main__":
    # Specify case (person) for whom to predict claim
    observation_index = 5  # Set integer between 1 and 5
    predict_single_observation(observation_index)
