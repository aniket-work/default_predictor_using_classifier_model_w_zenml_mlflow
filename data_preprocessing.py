from typing import Any, Dict
import pandas as pd
from zenml.steps import step, Output
from sklearn.model_selection import train_test_split
import numpy as np

@step
def load_data() -> Output(data=pd.DataFrame): 

    """
    # Method to load the dataset for fraud detection
    
    This method reads a dataset from a CSV file containing financial 
    information. It prepares the data for fraud detection analysis by 
    converting it into a pandas DataFrame.
    """

    data = pd.read_csv('../customer_data/Default_Fin.csv')
    return data

@step
def engineer_features(data: pd.DataFrame) -> Output(dataframe=pd.DataFrame):

    """
    # Method to engineer features for fraud detection
    
    This method creates additional features for fraud detection analysis. 
    It combines 'Bank Balance' and 'Annual Salary' columns into a new 
    feature 'Savings'. Also, it converts specific columns to categorical 
    types, preparing the data for machine learning models.
    """

    dataframe = data.copy()
    dataframe['Savings'] = dataframe['Bank Balance'] + dataframe['Annual Salary']
    dataframe['Employed'] = dataframe['Employed'].astype('category')
    dataframe['Defaulted?'] = dataframe['Defaulted?'].astype('category')
    return dataframe

@step
def split_train_test(data: pd.DataFrame) -> Output(
    X_train=np.ndarray, X_test=np.ndarray, y_train=np.ndarray, y_test=np.ndarray):
    
    """
    # Method to split the data into training and testing sets for model evaluation
    
    This method divides the dataset into training and testing sets. 
    It separates features and the target variable, 'Defaulted', to 
    prepare the data for training and evaluating machine learning models. 
    The split is performed with 70% of the data for training and 30% for testing.
    """

    X = data.drop('Defaulted', axis=1).values
    y = data['Defaulted'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test
