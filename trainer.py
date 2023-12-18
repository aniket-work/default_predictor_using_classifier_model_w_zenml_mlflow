from typing import Any, Dict, List, Tuple, Annotated
from sklearn.ensemble import RandomForestClassifier
from zenml import step
from config import ModelConfig
import mlflow
import numpy as np

@step(experiment_tracker="mlflow_tracker")
def train_model(
    X_train: List[Dict[str, float]],
    y_train: List[float],
    config: ModelConfig
) -> Annotated[RandomForestClassifier, "model"]: 
    """
    # Training Random Forest for Defaulter Detection
    
    Trains a RandomForestClassifier model using input training data (X_train) 
    and corresponding target labels (y_train) for the purpose of detecting defaulters.

    Parameters:
    - X_train (List[Dict[str, float]]): Training dataset containing credit-related features.
    - y_train (List[float]): Target labels indicating default or non-default for each training instance.
    - config (ModelConfig): Configuration object specifying model hyperparameters.

    Returns:
    - model (RandomForestClassifier): Trained RandomForestClassifier model.
    """
    params = config.model_params  # Obtain model parameters from ModelConfig

    # Convert dictionaries to arrays
    X_train_array = np.array([list(d.values()) for d in X_train])
    y_train_array = np.array(y_train)

    # Initialize and train RandomForestClassifier with specified parameters
    model = RandomForestClassifier(**config.model_params)
    model.fit(X_train_array, y_train_array)  # Fit the model using training data

    # Log trained model and parameters to MLflow
    mlflow.sklearn.log_model(model, config.model_name)  # Log trained model
    for param in params.keys():
        mlflow.log_param(f'{param}', params[param])  # Log each parameter used for training

    return model  # Return the trained model
