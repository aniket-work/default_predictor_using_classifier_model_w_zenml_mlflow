from typing import Any, Dict,  Tuple, Annotated, List
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from zenml import pipeline, step
from sklearn.model_selection import train_test_split
import numpy as np


@step(experiment_tracker="mlflow_tracker")
def load_data() -> Annotated[List[dict], "data"]:
    """
    Method to load the dataset for default detection.
    Reads a dataset from a CSV file containing financial information.
    Prepares the data for default detection analysis by converting it into a list of dictionaries.
    """
    data = pd.read_csv('customer_data/Default_Fin.csv')
    return data.to_dict('records')



@step(experiment_tracker="mlflow_tracker")
def engineer_features(data: List[dict]) -> Annotated[List[dict], "dataframe"]:
    """
    Method to engineer features for default detection.
    
    This method creates additional features for default detection analysis.
    It combines 'Bank Balance' and 'Annual Salary' columns into a new
    feature 'Savings'. Also, it converts specific columns to categorical
    types, preparing the data for machine learning models.
    """

    dataframe = pd.DataFrame(data).copy()
    dataframe['Savings'] = dataframe['Bank Balance'] + dataframe['Annual Salary']
    dataframe['Employed'] = dataframe['Employed'].astype('category')
    dataframe['Defaulted'] = dataframe['Defaulted'].astype('category')
    return dataframe.to_dict('records')


@step(experiment_tracker="mlflow_tracker")
def split_train_test(data: List[dict]) -> Tuple[
    Annotated[List[dict], "X_train"],
    Annotated[List[dict], "X_test"],
    Annotated[List[int], "y_train"],
    Annotated[List[int], "y_test"]
]:

    """
    # Method to split the data into training and testing sets for model evaluation
    
    This method divides the dataset into training and testing sets. 
    It separates features and the target variable, 'Defaulted', to 
    prepare the data for training and evaluating machine learning models. 
    The split is performed with 70% of the data for training and 30% for testing.
    """

    # Convert data into a DataFrame
    df = pd.DataFrame(data)

    X = df.drop('Defaulted', axis=1)
    y = df['Defaulted']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Convert DataFrames to lists of dictionaries and lists
    X_train_dict = X_train.to_dict(orient='records')
    X_test_dict = X_test.to_dict(orient='records')
    y_train_list = y_train.tolist()
    y_test_list = y_test.tolist()

    return X_train_dict, X_test_dict, y_train_list, y_test_list


@step(experiment_tracker="mlflow_tracker")
def feature_scaling(
    X_train: List[Dict[str, float]],
    X_test: List[Dict[str, float]]
) -> Tuple[
    Annotated[List[Dict[str, float]], "X_train_scaled"],
    Annotated[List[Dict[str, float]], "X_test_scaled"]
]:
    """
    # Feature Scaling for Defaulter Detection
    
    This step is responsible for scaling the credit-related features within the dataset
    to ensure uniformity and equality in their impact on the defaulter detection model.
    
    Parameters:
    - X_train (List[Dict[str, float]]): Training set containing credit-related features.
    - X_test (List[Dict[str, float]]): Test set containing credit-related features.
    
    Returns:
    - X_train_scaled (List[Dict[str, float]]): Scaled training set features.
    - X_test_scaled (List[Dict[str, float]]): Scaled test set features.
    
    Scaling Method:
    - MinMaxScaler from sklearn.preprocessing is used for scaling.
    - MinMaxScaler scales features to a range between 0 and 1 by default.
    
    Scaling Process:
    - A MinMaxScaler object is instantiated to scale the features.
    - X_train_scaled is obtained by fitting and transforming the training set with the scaler.
    - X_test_scaled is obtained by transforming the test set using the same fitted scaler
      to ensure consistency in scaling.
    
    Purpose and Importance:
    - Scaling is crucial in defaulter detection to prevent bias towards features
      with larger magnitudes, ensuring equal importance to all features.
    - It maintains the relationships between features while standardizing their scales,
      aiding in accurate model training and prediction.
    
    Example Usage:
    - X_train_scaled, X_test_scaled = feature_scaling(X_train, X_test)
    - Once scaled, these features can be used for training and evaluating
      a machine learning model for defaulter detection.
    """
    scaler = MinMaxScaler()  # Initialize MinMaxScaler
    
    # Convert dictionaries to arrays
    X_train_array = np.array([list(d.values()) for d in X_train])
    X_test_array = np.array([list(d.values()) for d in X_test])
    
    # Fit and transform the training set
    X_train_scaled = scaler.fit_transform(X_train_array)
    
    # Transform the test set using the same fitted scaler
    X_test_scaled = scaler.transform(X_test_array)
    
    # Convert scaled arrays back to dictionaries
    X_train_scaled_dicts = [dict(zip(X_train[0].keys(), row)) for row in X_train_scaled]
    X_test_scaled_dicts = [dict(zip(X_test[0].keys(), row)) for row in X_test_scaled]
    
    return X_train_scaled_dicts, X_test_scaled_dicts  # Return the scaled features as dictionaries


