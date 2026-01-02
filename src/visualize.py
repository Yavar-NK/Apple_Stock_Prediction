import matplotlib.pyplot as plt
import joblib
from keras.models import load_model
from preprocess import get_data

def show_results():
    
    model = load_model('models/apple_stock_model.h5')
    scaler = joblib.load('models/scaler.pkl')
    df = get_data()
    
    
    apple_total = df['Open']
    
    plt.figure(figsize=(12, 6))
    
    plt.show()

if __name__ == "__main__":
    show_results()