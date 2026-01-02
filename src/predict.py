import numpy as np
import joblib
import os
from keras.models import load_model
from src.preprocess import get_data

def run_prediction():
    # Define paths
    model_path = 'models/apple_stock_model.h5'
    scaler_path = 'models/scaler.pkl'
    
    # Check if model files exist
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Error: Model files not found. Please run src/train.py first.")
        return

    # Load model and scaler
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # Fetch data
    df = get_data()
    
    # Prepare last 60 days for prediction
    last_60_days = df['Open'].values[-60:].reshape(-1, 1)
    last_60_days_scaled = scaler.transform(last_60_days)
    
    # Reshape for LSTM input
    X_input = np.array([last_60_days_scaled])
    X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))
    
    # Perform prediction
    prediction_scaled = model.predict(X_input)
    prediction_final = scaler.inverse_transform(prediction_scaled)
    
    print("-" * 30)
    print(f"Predicted Price for Tomorrow: ${prediction_final[0][0]:.2f}")
    print("-" * 30)

if __name__ == "__main__":
    run_prediction()