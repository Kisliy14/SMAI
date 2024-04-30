import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import requests
from datetime import datetime
from pandas_datareader import data as pdr

st.title("Stock Price Visualization")

days = st.sidebar.slider("Select number of days", min_value=1, max_value=60, value=60)

st.sidebar.subheader("Select Stock(s)")
selected_stocks = st.sidebar.text_input("Enter stock symbols separated by commas (e.g., AAPL,MSFT)", value="AAPL")
stocks = [s.strip().upper() for s in selected_stocks.split(',')]
value_variable = selected_stocks
stocks = value_variable


if st.sidebar.button("Построить графики цен на акции"):
    def plot_stock_price(value_variable, days):
        yf.pdr_override()
        data = pdr.get_data_yahoo(value_variable, start='2012-01-01', end=datetime.now())
        plt.figure(figsize=(10, 6))
        plt.plot(data['Adj Close'])
        plt.title(f"{value_variable} Stock Price")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        st.pyplot(plt)

yf.pdr_override()
data = pdr.get_data_yahoo(value_variable, start='2012-01-01', end=datetime.now())
plt.figure(figsize=(10, 6))
plt.plot(data['Adj Close'])
plt.title(f"{value_variable} Stock Price")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
st.pyplot(plt)

url = 'http://127.0.0.1:8000/predict/' 
params = {'days': days,'stocks': stocks,'value_variable': value_variable}
response = requests.post(url, params=params)

df = pd.DataFrame(response.json()['dataframe'], columns=['Close'])
final_df = pd.DataFrame(response.json()['finaldf'], columns=['Close'])
future_df = pd.DataFrame(response.json()['futureclose'], columns=['Future Close'])

st.title("Predictions Price Visualization")
plt.figure(figsize=(16, 6))
plt.title('Future Close Price Prediction')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price in USD ($)', fontsize=18)
plt.plot(final_df['Close'], label='Actual Data')
plt.axvline(x=df.index[-1], color='r', linestyle='--', label='End of Actual Data')
plt.legend()
st.pyplot()

st.title("Predictions Price's")
st.write(future_df)

