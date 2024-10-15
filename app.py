import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

# Load sample data
data = pd.date_range(start='1/1/2020', periods=100, freq='D')
series = pd.Series(np.random.randn(100).cumsum(), index=data)

st.title('Time Series Forecasting App')
st.write("""
## Introduction
This app explains various concepts in time series forecasting including:
- Stationarity
- Holt's Method
- Holt-Winters Method
- Train-Test Split using Forward Chaining
- ARIMA and SARIMA
- Ljung-Box Test
- Jacques-Bera Normality Test
- Prophet
- DeepAR
""")

st.write("### Sample Time Series Data")
st.write("""
To demonstrate the various time series forecasting methods, we start with a sample time series data.
This data is generated to simulate a typical time series that might be seen in real-world scenarios.
""")
st.line_chart(series)

# Helper function to calculate MAE
def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

# Stationarity
st.write("## Stationarity")
st.write("""
A time series is said to be stationary if its statistical properties such as mean and variance remain constant over time.
Testing for stationarity helps in determining if the data requires transformation before modeling.
We use the Augmented Dickey-Fuller (ADF) test to check for stationarity. If the p-value is less than 0.05, we reject the null hypothesis,
which means the series is stationary.
""")
result = adfuller(series)
st.write(f'ADF Statistic: {result[0]}')
st.write(f'p-value: {result[1]}')
for key, value in result[4].items():
    st.write('Critical Value (%s): %.3f' % (key, value))
if result[1] <= 0.05:
    st.write("The series is stationary.")
else:
    st.write("The series is not stationary.")

# Train-Test Split using Forward Chaining
st.write("## Train-Test Split using Forward Chaining")
st.write("""
In time series forecasting, it's crucial to maintain the temporal order when splitting data into training and test sets.
Forward chaining involves using earlier time points for training and later time points for testing,
ensuring that we are not leaking future information into the model.
""")
train_size = int(len(series) * 0.8)
train, test = series[0:train_size], series[train_size:len(series)]
st.line_chart({'Train': train, 'Test': test})

# Model Forecasting and Comparison
models = {
    "Holt's Linear Trend": None,
    "Holt-Winters Seasonal": None,
    "ARIMA": None,
    "SARIMA": None,
    "Prophet": None
}
forecasts = {}

# Holt's Linear Trend Method
st.write("## Holt's Linear Trend Method")
st.write("""
Holt’s Linear Trend method is an extension of simple exponential smoothing to capture linear trends in the data.
It involves smoothing equations for the level and the trend.
We apply Holt’s method to forecast the next 10 data points.
""")
model_holt = ExponentialSmoothing(train, trend='add').fit()
holt_forecast = model_holt.forecast(steps=len(test))
models["Holt's Linear Trend"] = model_holt
forecasts["Holt's Linear Trend"] = holt_forecast
st.line_chart(holt_forecast)

# Holt-Winters Seasonal Method
st.write("## Holt-Winters Seasonal Method")
st.write("""
The Holt-Winters Seasonal method extends Holt’s method by adding seasonal components to capture seasonality in the data.
There are two types of seasonality: additive and multiplicative. We use the additive method here.
The model is used to forecast the next 10 data points considering seasonality.
""")
model_hw = ExponentialSmoothing(train, seasonal='add', seasonal_periods=12).fit()
hw_forecast = model_hw.forecast(steps=len(test))
models["Holt-Winters Seasonal"] = model_hw
forecasts["Holt-Winters Seasonal"] = hw_forecast
st.line_chart(hw_forecast)

# ARIMA
st.write("## ARIMA")
st.write("""
ARIMA (AutoRegressive Integrated Moving Average) is a popular model for time series forecasting.
It combines autoregression, differencing (to achieve stationarity), and moving average components.
We fit an ARIMA model to the training data and forecast the next data points.
""")
model_arima = ARIMA(train, order=(5,1,0)).fit()
arima_forecast = model_arima.forecast(steps=len(test))
models["ARIMA"] = model_arima
forecasts["ARIMA"] = arima_forecast
st.line_chart(arima_forecast)

# SARIMA
st.write("## SARIMA")
st.write("""
SARIMA (Seasonal ARIMA) extends ARIMA to handle seasonality in the data.
It includes seasonal components in the autoregression, differencing, and moving average processes.
We fit a SARIMA model to the training data and forecast the next data points.
""")
model_sarima = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12)).fit()
sarima_forecast = model_sarima.fore
