from typing import Any, Dict, List, Tuple, Annotated
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from zenml import step
import mlflow
import numpy as np

@step(experiment_tracker="mlflow_tracker")
def evaluate_model(
    model: Any,
    X_test: List[Dict[str, float]],
    y_test: List[float]
) -> None:
    """
    # Evaluation of Model for Defaulter Detection
    
    Evaluates the performance of a machine learning model for defaulter detection 
    using a test dataset (X_test) and corresponding true labels (y_test).

    Parameters:
    - model (Any): Trained machine learning model for defaulter detection.
    - X_test (List[Dict[str, float]]): Test dataset containing credit-related features.
    - y_test (List[float]): True labels indicating default or non-default for each test instance.

    Evaluation Metrics:
    - Calculates evaluation metrics including Accuracy, Precision, Recall, and F1-score
      to assess the model's performance in defaulter detection.

    MLflow Logging:
    - Logs evaluation metrics to MLflow for experiment tracking and reproducibility.

    Example Usage:
    - evaluate_model(trained_model, X_test_dict, y_test_list)
    - After evaluation, metrics will be logged to MLflow for analysis and comparison.
    """
    # Convert dictionaries to arrays
    X_test_array = np.array([list(d.values()) for d in X_test])
    y_test_array = np.array(y_test)

    # Calculate metrics using the loaded model
    metrics = {
        "accuracy": accuracy_score(y_test_array, model.predict(X_test_array)),
        "precision": precision_score(y_test_array, model.predict(X_test_array), average='macro'),
        "recall": recall_score(y_test_array, model.predict(X_test_array), average='macro'),
        "f1": f1_score(y_test_array, model.predict(X_test_array), average='macro')
    }

    # Log metrics in MLflow
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)

    pass
