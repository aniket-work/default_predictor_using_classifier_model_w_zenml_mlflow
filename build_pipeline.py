from zenml import pipeline, step

from data_preprocessing import engineer_features, split_train_test, feature_scaling
from trainer import train_model
from test_model import evaluate_model
from data_preprocessing import load_data

@pipeline
def execute_pipeline():
    
    """   
    This pipeline orchestrates the steps involved in training and deploying
    a machine learning model for default detection.
    
    Steps in the pipeline:
    
    1. `get_data`: Loads the raw data required for training the model.
    
    2. `feature_engineering`: Processes and engineers features in the raw data.
       This step enhances the dataset by creating new relevant features or transforming existing ones.
    
    3. `split_train_test`: Splits the processed data into training and testing sets.
       It prepares the data for training the model by separating features and target variables.
    
    4. `scale_data`: Scales the features in the training and testing sets to a similar range.
       This step is often crucial for certain models to perform optimally.
    
    5. `train_model`: Trains a machine learning model using the preprocessed training data.
       It builds the model based on the training set to make predictions.
    
    6. `evaluate_model`: Evaluates the trained model's performance using the test set.
       Metrics like recall might be calculated to assess how well the model identifies default cases.
    
    7. `deployment_trigger`: Decides whether the model should be deployed based on evaluation metrics.
       For instance, it may trigger deployment if the recall metric meets a predefined threshold.
       [Note: 'deployment_trigger' appears to be a new step added to handle deployment decisions]
    
    8. `model_deployer`: Deploys the trained model if the deployment decision is affirmative.
       This step could involve integrating the model into a production environment for real-time use.
    
    The pipeline defines a sequence of actions, starting from data preparation, model training,
    evaluation, and potential deployment based on specific criteria, providing an end-to-end process.
    """
   
  
    data = load_data()

    data = engineer_features(data = data)

    X_train, X_test, y_train, y_test = split_train_test(data = data)

    X_train, X_test = feature_scaling(X_train = X_train, X_test = X_test)

    model = train_model(X_train = X_train, y_train = y_train)

    evaluate_model(model, X_test = X_test, y_test = y_test) 
   
   
    
if __name__ == '__main__':
   pipeline_to_execute = execute_pipeline()
   print("0")
    
   print("1")
   #pipeline_to_execute.execute()
   # Assuming you have MLflow imported and running
   import mlflow
   print("2")
   tracking_uri = mlflow.get_tracking_uri()
   print("3")
   print(f"MLflow Tracking URI: {tracking_uri}")


