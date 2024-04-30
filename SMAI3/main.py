
import pandas as pd
import numpy as np
import pickle
import yfinance as yf
from fastapi import FastAPI,Form
from fastapi.responses import JSONResponse
from sklearn.preprocessing import MinMaxScaler
from typing import Optional


app = FastAPI()

@app.post("/predict/") 
async def predict(days: int ,stocks: str, value_variable: str, start_date: Optional[str] = '2012-01-01'):
    from pandas_datareader import data as pdr
    from datetime import datetime
    yf.pdr_override()
    tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
    tech_list.extend(stocks)
    end = datetime.now()
    start = datetime(end.year - 1, end.month, end.day)
    for stock in tech_list:
        globals()[stock] = yf.download(stock, start, end)        
    df = pdr.get_data_yahoo(value_variable, start='2012-01-01', end=datetime.now())
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = int(np.ceil( len(dataset) * .95 ))

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    train_data = scaled_data[0:int(training_data_len), :]

    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    test_data = scaled_data[training_data_len - 60: , :]
    future_days = 60
    x_future = []
    x_test = []

    for i in range(future_days, len(test_data)):
        x_future.append(test_data[i-future_days:i, 0])
    x_future = np.array(x_future)
    x_future = np.reshape(x_future, (x_future.shape[0], x_future.shape[1], 1))
    
    from keras.models import Sequential
    from keras.layers import Dense, LSTM

    model = pickle.load(open('........../models/model.pkl','rb'))

    test_data = scaled_data[training_data_len - 60: , :]
    future_days = 60
    x_future = []
    last_60_days = scaled_data[-60:]
    x_test = []
    y_test = dataset[training_data_len:, :]


    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
        x_future.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    train = data[:training_data_len].copy()
    valid = data[training_data_len:].copy()
    valid.loc[:, 'Прогнозы'] = predictions
    x_future = np.array(x_future)
    x_future = np.reshape(x_future, (x_future.shape[0], x_future.shape[1], 1))
    future_predictions = model.predict(x_future)
    future_predictions = scaler.inverse_transform(future_predictions)
    future_dates = pd.date_range(df.index[-1], periods=future_days+1, freq='B')[1:]
    future_df = pd.DataFrame(index=future_dates, columns=['Future Close'])
    future_df.index.name = 'Date'
    future_df['Future Close'] = future_predictions[:60].flatten()
    selected_days = days
    future_df = pd.DataFrame(index=future_dates[:selected_days], columns=['Future Close'])
    future_df['Future Close'] = future_predictions[:selected_days].flatten()
    final_df = pd.concat([df['Close'], future_df['Future Close']])

    return JSONResponse({
    "finaldf": final_df.tolist(), 
    "futureclose": future_df['Future Close'].tolist(), 
    "dataframe": df['Close'].tolist()
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)

#ticker