import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import ClassifierMixin

from zenml.steps import step, Output
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow

@enable_mlflow
@step(enable_cache=False)
def evaluate_model(model: ClassifierMixin, X_test: np.ndarray, y_test_label: np.ndarray) -> Output():
    
    y_model_predictions = model.predict(X_test)

    # Calculate metrics
    metrics = {
        
        '''
          Accuracy represents the overall correctness of predictions made by the model. However, in fraud detection scenarios, 
          accuracy alone might be misleading because fraud instances are usually rare compared to non-fraudulent ones. 
          If the dataset is imbalanced (which is often the case in fraud detection where genuine transactions vastly outnumber fraudulent ones), 
          a high accuracy might be achieved by simply labeling everything as non-fraudulent, ignoring the actual fraud cases. 
          Therefore, while accuracy provides an overall picture, it might not be the most reliable metric for fraud detection models.  
        '''
        "accuracy"  : accuracy_score(y_test_label, y_model_predictions),

        '''
        Precision measures the ratio of correctly predicted fraudulent cases to all cases predicted as fraudulent. 
        In fraud detection, precision is vital as it tells us the proportion of flagged cases that are actually fraudulent. 
        A high precision score indicates that when the model flags a transaction as fraudulent, it's highly likely to be correct. 
        It helps in minimizing false positives, reducing the number of legitimate transactions incorrectly flagged as fraudulent, 
        thus saving resources and maintaining trust with customers.
        '''
        "precision" : precision_score(y_test_label, y_model_predictions, average='macro'),
        
        '''
        Recall, also known as sensitivity or true positive rate, measures the ratio of correctly predicted fraudulent cases 
        to the actual total fraudulent cases. In fraud detection, high recall is crucial as it ensures that the model captures as many 
        fraudulent cases as possible. Missing fraudulent transactions (false negatives) can be extremely costly and damaging. 
        A high recall means that the model effectively identifies most of the fraudulent transactions, minimizing the number of undetected frauds.
        '''
        "recall"    : recall_score(y_test_label, y_model_predictions, average='macro'),

        '''
        F1-score is the harmonic mean of precision and recall. It provides a balance between precision and recall. 
        In fraud detection, achieving a balance between precision and recall is important. 
        A high F1-score indicates that the model has both high precision and high recall. 
        It's a useful metric when there's an uneven class distribution (like in fraud detection), offering a single value that 
        represents a balance between correctly identifying fraudulent transactions and minimizing false positives.
        '''
        "f1"        : f1_score(y_test_label, y_model_predictions, average='macro')
    }

    # Log metrics in MLflow
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)

    pass
