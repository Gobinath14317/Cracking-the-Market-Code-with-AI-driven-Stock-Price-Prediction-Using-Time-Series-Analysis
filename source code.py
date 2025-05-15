import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np

st.title("📈 Stock Price Prediction using Time Series")

# Input stock symbol
stock = st.text_input("Enter Stock Symbol (Eg: AAPL, GOOGL, MSFT):", "AAPL")
data = yf.download(stock, start="2015-01-01", end="2024-12-31")

st.subheader("Raw Data")
st.write(data.tail())

# Visualize
st.subheader("Stock Price Chart")
fig, ax = plt.subplots()
ax.plot(data['Close'], label='Close Price')
ax.set_title(f'{stock} Closing Price')
st.pyplot(fig)

# Predict using Linear Regression
st.subheader("Linear Regression Prediction")
data['Date'] = data.index
data['Date'] = pd.to_datetime(data['Date'])
data['Date_ordinal'] = data['Date'].map(pd.Timestamp.toordinal)

X = data[['Date_ordinal']]
y = data[['Close']]

model = LinearRegression()
model.fit(X, y)
future = pd.date_range(start='2025-01-01', end='2025-12-31')
future_ordinal = pd.DataFrame(future.map(pd.Timestamp.toordinal), columns=['Date_ordinal'])

prediction = model.predict(future_ordinal)
st.line_chart(prediction, height=300)

# Show prediction
st.subheader("Sample Prediction Values")
future_df = pd.DataFrame({
    "Date": future,
    "Predicted Close": prediction.flatten()
})
st.write(future_df.head())

