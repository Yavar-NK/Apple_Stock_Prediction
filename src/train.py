import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from src.preprocess import get_data, process_data, create_sequences
import numpy as np

def train():
    print("Loading data and starting preprocessing...")
    df = get_data()
    scaled_data, _ = process_data(df)
    
    X, y = create_sequences(scaled_data)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    print("Building LSTM model architecture...")
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("Starting model training (this may take a few minutes)...")
    model.fit(X, y, epochs=50, batch_size=32)
    
    model.save('models/apple_stock_model.h5')
    print("Training completed successfully! Model saved in 'models/' directory.")

if __name__ == "__main__":
    train()