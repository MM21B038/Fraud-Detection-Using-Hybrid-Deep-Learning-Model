# FraudDetectionHybrid/data_processing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Loads data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data, target_column):
    """Preprocesses the data by scaling features and separating target variable."""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Splits the data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
