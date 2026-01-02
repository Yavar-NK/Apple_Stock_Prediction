import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import kagglehub
import os
import joblib

def get_data():
    # Downloading the latest version of the dataset
    path = kagglehub.dataset_download("khoongweihao/aaplcsv")
    csv_file_path = os.path.join(path, 'AAPL.csv')
    return pd.read_csv(csv_file_path)

def process_data(df):
    # Selecting the 'Open' price column
    apple_processed = df.iloc[:, 1:2].values
    
    # Normalizing the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    apple_scaled = scaler.fit_transform(apple_processed)
    
    # Creating models directory if it doesn't exist
    if not os.path.exists('models'): 
        os.makedirs('models')
    
    # Saving the scaler for future use in prediction
    joblib.dump(scaler, 'models/scaler.pkl')
    return apple_scaled, scaler

def create_sequences(data, window=60):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)