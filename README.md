
# 📈 Sensex Price Prediction using LSTM (Deep Learning)

Welcome to the **BSE Sensex Forecasting Web App** — a deep learning-powered tool built with **Streamlit** and **LSTM Neural Networks** to predict the **next day's closing price** of India's stock market index: **Sensex**.

---

## 🚀 Overview

Stock price prediction is a challenging problem due to the highly non-linear and dynamic nature of the financial markets. This project applies advanced **sequence modeling** with **LSTM (Long Short-Term Memory)** to capture the temporal dependencies in historical BSE Sensex data.

The app allows users to:
- Fetch or upload Sensex data
- Preprocess and scale time series
- Train an LSTM-based model
- Visualize performance
- Predict the **next trading day’s closing price**

---

## 🧠 Models Implemented

### 1. 🔹 Basic LSTM
- A simple one-layer LSTM with a Dense output layer.
- Suitable for capturing short-term patterns.

### 2. 🔸 Stacked LSTM with Dropout (Current Model)
- Two LSTM layers: (100 units → 50 units) with Dropout.
- Helps in reducing overfitting and learning deeper sequence patterns.
- Architecture:
  ```python
  model = Sequential()
  model.add(LSTM(128, return_sequences=True))
  model.add(Dropout(0.3))
  model.add(LSTM(64))
  model.add(Dropout(0.2))
  model.add(Dense(1))
  ```

### 3. 🔁 Future Additions (Coming Soon)
- **GRU-based models**: for faster training
- **CNN-LSTM**: for capturing local and long-term dependencies
- **Attention-based LSTM**: for interpretability
- **Prophet** and **ARIMA** for hybrid benchmarking

---

## 🛠 Technologies Used

| Component      | Tool/Library        |
|----------------|---------------------|
| Data Source     | `yfinance`, CSV Upload |
| Frontend        | `Streamlit`         |
| Modeling        | `TensorFlow`, `Keras` |
| Processing      | `NumPy`, `Pandas`   |
| Evaluation      | `scikit-learn`      |
| Visualization   | `Matplotlib`, `Streamlit` charts |

---

## 📦 Setup Instructions

### 🐍 Install Dependencies

```bash
pip install streamlit tensorflow pandas numpy scikit-learn yfinance matplotlib
```

### ▶️ Run the App

```bash
streamlit run frontend.py
```

---

## 📊 How It Works

1. **Data Preparation**
   - Downloaded using Yahoo Finance or uploaded manually
   - Normalized with `MinMaxScaler`
   - Transformed into supervised learning format using a sliding window

2. **Model Training**
   - LSTM network trained using a 60-day (Variable) lookback window
   - Dropout layers help reduce overfitting

3. **Prediction & Inverse Scaling**
   - Next day’s price is predicted and inverse-transformed for readability
   - RMSE and MAE are calculated to assess accuracy

---

## 📈 Sample Output

| Metric | Value |
|--------|-------|
| RMSE   | ~948 |
| MAE    | ~731  |
| Sensex Close | ~80,000 |

**Relative Error < 1.5%**, which is very strong for financial time series.

---

## 📌 Features To Add

- 📁 CSV Upload and custom ticker inputs
- ⏩ Multi-day forecasting (e.g., 7/30 days)
- 🧮 Technical indicators (RSI, MACD, Bollinger Bands)
- 📤 Export predictions
- 💾 Save/load trained models
- 📅 Display actual upcoming Sensex value (via live API)

---

## 🧪 Evaluation Metrics

- **RMSE (Root Mean Squared Error)** – Penalizes large errors
- **MAE (Mean Absolute Error)** – Measures average prediction accuracy
- **R² Score** (future addition)

---

## 🙌 Credits

- Yahoo Finance Data API via `yfinance`
- Deep learning powered by `TensorFlow` & `Keras`
- Streamlit for rapid UI deployment

---

## 📬 Connect

Have suggestions or want to contribute?

- 📧 Email: lgarg092@gmail.com
- 🌐 LinkedIn: https://www.linkedin.com/in/lokesh-garg-58b11528b/
- 💻 GitHub: https://github.com/Lokeshg012

---

> *"Predicting the future isn't magic, it's machine learning."*

