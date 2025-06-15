import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

@st.cache_data
def load_sensex_data():
    df = yf.download('^BSESN', start='2008-01-01', end=None)
    df = df[['Close']].dropna()
    return df

def create_sequences(data, window_size):
    X = []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
    return np.array(X)

def plot_predictions(actual, predicted):
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label='Actual Price')
    plt.plot(predicted, label='Predicted Price')
    plt.title("Sensex Price: Actual vs Predicted")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt)

st.set_page_config(page_title="📈 Sensex LSTM Forecast", layout="centered")
st.title("📊 Stock Price Prediction ")
st.caption("Using Yahoo Finance data + Keras LSTM model")

df = load_sensex_data()
st.subheader("📜 Historical Sensex Chart")
st.line_chart(df['Close'])

st.subheader("⚙️ Model Settings")
window_size = st.slider("Select number of previous days:", 30, 120, 60, step=10)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Close']])

if len(scaled_data) <= window_size:
    st.warning("Not enough data for the selected window size.")
    st.stop()

X = create_sequences(scaled_data, window_size)
X = X.reshape((X.shape[0], X.shape[1], 1))
actual = df['Close'].values[window_size:]

model = load_model('C:/Users/lokesh/Downloads/lstm_model.h5')

predicted_scaled = model.predict(X)
predicted = scaler.inverse_transform(predicted_scaled)

st.subheader("📉 Prediction Results")
plot_predictions(actual, predicted.flatten())

last_window = scaled_data[-window_size:]
next_input = last_window.reshape((1, window_size, 1))
next_pred_scaled = model.predict(next_input)
next_price = scaler.inverse_transform(next_pred_scaled)[0][0]

st.success(f" Predicted Next Day Closing Price: ₹{next_price:,.2f}")
