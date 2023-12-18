
from zenml.pipelines import pipeline
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from build_pipeline import execute_pipeline
from data_preprocessing import engineer_features, split_train_test, feature_scaling
from trainer import train_model
from test_model import evaluate_model

def main():
    train = execute_pipeline(
        engineer_features=engineer_features(),
        split_train_test=split_train_test(),
        feature_scaling=feature_scaling(),
        train_model=train_model(),
        evaluate_model=evaluate_model()
    )
    train.run()

if __name__ == '__main' :
    main()