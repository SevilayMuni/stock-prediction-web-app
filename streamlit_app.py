import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from datetime import date
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from keras.models import load_model
from keras.layers import LSTM

class CustomLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)  # Remove unrecognized argument
        super().__init__(*args, **kwargs)

model = load_model('./app_model.h5', custom_objects={"LSTM": CustomLSTM}, compile=False)
google_model = load_model('./2nd-Google-LSTM-Model.h5', custom_objects={"LSTM": CustomLSTM}, compile=False)

# Page Layout
st.set_page_config(layout="wide")
tab1, tab2, tab3 = st.tabs(["APPLE Stock", "GOOGLE Stock", "Dashboard"])

with tab1: 
    st.header('🔮 StockSense AI Web Application')
	st.info('StockSense AI uses real-time stock values via Yahoo Finance.')
    st.write(' ')

# Define function to get raw data
def raw_data():
    # Determine end and start dates for dataset download
    end = datetime.now()
    start = datetime(end.year, end.month - 2, end.day)

    # Download Apple's dataset between start and end dates
    apple_df = yf.download('AAPL', start=start, end=end)

    # Rename columns of the Apple DataFrame
    column_dict = {'Open': 'open', 'High': 'high', 'Low': 'low',
                   'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'}
    apple_df = apple_df.rename(columns=column_dict)
    apple_df.index.names = ['date']
    return apple_df
raw_apple_df = raw_data()

# Define function to calculate 'On Balance Volume (OBV)'
def On_Balance_Volume(Close, Volume):
    change = Close.diff()
    OBV = np.cumsum(np.where(change > 0, Volume, np.where(change < 0, -Volume, 0)))
    return OBV

scaler = MinMaxScaler(feature_range = (0, 1))
def apple_process():
    # Determine end and start dates for dataset download
    end = datetime.now()
    start = datetime(end.year, end.month - 2, end.day)

    # Download Apple's dataset between start and end dates
    apple_df = yf.download('AAPL', start=start, end=end)

    # Rename columns of the Apple DataFrame
    column_dict = {'Open': 'open', 'High': 'high', 'Low': 'low',
                   'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'}
    apple_df = apple_df.rename(columns=column_dict)
    apple_df.index.names = ['date']

    # Add additional calculated features
    apple_df['garman_klass_volatility'] = ((np.log(apple_df['high']) - np.log(apple_df['low'])) ** 2) / 2 - \
                                          (2 * np.log(2) - 1) * ((np.log(apple_df['adj_close']) - np.log(apple_df['open'])) ** 2)
    apple_df['dollar_volume'] = (apple_df['adj_close'] * apple_df['volume']) / 1e6
    apple_df['obv'] = On_Balance_Volume(apple_df['close'], apple_df['volume'])
    apple_df['ma_3_days'] = apple_df['adj_close'].rolling(3).mean()

    # Filter and preprocess the dataset
    apple_dset = apple_df[['adj_close', 'garman_klass_volatility', 'dollar_volume', 'obv', 'ma_3_days']]
    apple_dset.dropna(axis=0, inplace=True)
    apple_test_scaled = scaler.fit_transform(apple_dset)
    return apple_test_scaled

apple_dataset = apple_process()

def feed_model(dataset, n_past, model, scaler):
    # Create X from the dataset
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
        dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
        dataY.append(dataset[i,0])
    testX = np.array(dataX)
    
    # Make predictions using the model
    pred_initial = model.predict(testX)
    
    # Repeat predictions and reshape to original scale
    pred_array = np.repeat(pred_initial, 5, axis = -1)
    preds = scaler.inverse_transform(np.reshape(pred_array, (len(pred_initial), 5)))[:5, 0]
    return preds

prediction = feed_model(apple_dataset, 21, model, scaler).tolist()
# create a dataframe
pred_df = pd.DataFrame({'Predicted Day': ['Tomorrow', '2nd Day', '3rd Day', '4th Day', '5th Day'],
                        'Adj. Closing Price($)': [ '%.2f' % elem for elem in prediction]})

# set the index to the 'name' column
pred_df.set_index('Predicted Day', inplace=True)

# Display result
title = """<div style="font-family: Arial, sans-serif; font-size: 18px; line-height: 1.6;"><strong>Apple Share For Next 5 Days</strong></div>"""

tab1.col1, tab1.col2 = tab1.columns(2)
with tab1.col1:
    st.markdown(title, unsafe_allow_html=True)
    st.dataframe(pred_df)

actual_values  = raw_apple_df['adj_close'].values.tolist()

# Calculate the comparison between predicted next price and last actual price
if actual_values and prediction:
    last_actual_price = actual_values[-1][0]
    next_predicted_price = prediction[0]

    percent_change = (next_predicted_price - last_actual_price) / last_actual_price * 100

    insight = f"""
    <div style="font-family: Arial, sans-serif; font-size: 16px; line-height: 1.6;">
        <strong>The next predicted stock price is:</strong> <span style="color: #4CAF50;">${next_predicted_price:.2f}</span><br>
        <strong>Last actual price:</strong> <span style="color: #FF5722;">${last_actual_price:.2f}</span><br>
        <strong>Change:</strong> <span style="color: {'#4CAF50' if percent_change >= 0 else '#FF5722'};">{percent_change:+.2f}%</span>
    </div>
    """
else:
    insight = "<div style='font-family: Arial, sans-serif;'>Not enough data to generate insights.</div>"

# Display the insight using Markdown with HTML formatting
with tab1.col2:
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.markdown(insight, unsafe_allow_html=True)

multi = '''This project's predictive AI is multivariate LSTM neural networks.  
Long Short-Term Memory (LSTM) networks, a variant of recurrent neural networks (RNNs), have proven effective for time series forecasting, particularly when dealing with sequential data like stock prices.    
Stock price movement is influenced by a variety of factors; thus, multivariate time series forecasting is used. The deep learning model captures the underlying patterns and relationships in the data due to domain-based feature engineering.'''

tab1.col1, tab1.col2, tab1.col3 = tab1.columns(3)
with tab1.col1:
    with st.popover("AI Model Infographics"):
        st.markdown(multi)
        st.link_button("Predictive AI Code by SMG", "https://github.com/SevilayMuni/Multivariate-TimeSeries-Forecast-LSTM-Apple-Google-Stocks/tree/main/Apple-Stock-LSTM-Model")

with tab1.col2:
    with st.popover("Variables Used by AI"):
        st.image("./images/variable-table.png")

with tab1.col3:
    with st.popover("Model Evaluation"):
        st.image("./images/variable-table.png")


dedication = """<div style="font-family: Arial, sans-serif; font-size: 16px; line-height: 1.6;"><i>The StockSense AI is dedicated to my dearest, Ceyhun Utku Girgin.</i>"""
with tab2.container(border = True):
    st.markdown(dedication, unsafe_allow_html=True)
    st.markdown(''':rainbow[End-to-end project is done by] :blue-background[Sevilay Munire Girgin]''')

tab1.warning('This work is not investment advice! It is merely a data science research.', icon="❗")

#-----------------------TAB2---------------
tab2.header('🔮 StockSense AI Web Application')
tab2.info('StockSense AI uses real-time stock values via Yahoo Finance.')
tab2.write(' ')

# Define function to get raw data
def raw_google_data():
    # Determine end and start dates for dataset download
    end = datetime.now()
    start = datetime(end.year, end.month - 2, end.day)

    # Download Apple's dataset between start and end dates
    google_df = yf.download('GOOGL', start=start, end=end)
    
    column_dict = {'Open': 'open', 'High': 'high', 'Low': 'low',
                   'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'}
    google_df = google_df.rename(columns=column_dict)
    google_df.index.names = ['date']
    return google_df
raw_google_df = raw_google_data()

def google_process(df):
    # Add additional calculated features
    df['dollar_volume'] = (df['adj_close'] * df['volume']) / 1e6
    df['obv'] = On_Balance_Volume(df['close'], df['volume'])
    df['ma_3_days'] = df['adj_close'].rolling(3).mean()
    df['macd'] = df['close'].ewm(span = 12, adjust = False).mean() - df['close'].ewm(span = 26, adjust = False).mean()
    # Filter and preprocess the dataset
    google_dset = df[['adj_close', 'volume', 'dollar_volume', 'obv', 'ma_3_days', 'macd']]
    google_dset.dropna(axis=0, inplace=True)
    google_test_scaled = scaler.fit_transform(google_dset)
    return google_test_scaled

google_dataset = google_process(raw_google_df)

def feed_google_model(dataset, n_past, modelname, scaler):
    # Create X from the dataset
    GdataX = []
    GdataY = []
    for i in range(n_past, len(dataset)):
        GdataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
        GdataY.append(dataset[i,0])
    GtestX = np.array(GdataX)
    
    # Make predictions using the model
    pred_google = modelname.predict(GtestX)
    
    # Repeat predictions and reshape to original scale
    pred_google_array = np.repeat(pred_google, 6, axis = -1)
    preds_google = scaler.inverse_transform(np.reshape(pred_google_array, (len(pred_google), 6)))[:5, 0]
    return preds_google

google_prediction = feed_google_model(google_dataset, 21, google_model, scaler).tolist()
# create a dataframe
google_pred_df = pd.DataFrame({'Predicted Day': ['Tomorrow', '2nd Day', '3rd Day', '4th Day', '5th Day'], 'Adj. Closing Price($)': [ '%.2f' % elem for elem in google_prediction]})

# set the index to the 'name' column
google_pred_df.set_index('Predicted Day', inplace=True)

# Display result
title2 = """<div style="font-family: Arial, sans-serif; font-size: 18px; line-height: 1.6;"><strong>Google Share For Next 5 Days</strong></div>"""

tab2.col1, tab2.col2 = tab2.columns(2)
with tab2.col1:
    st.markdown(title2, unsafe_allow_html=True)
    st.dataframe(google_pred_df)

actual_google_values  = raw_google_df['adj_close'].values.tolist()

# Calculate the comparison between predicted next price and last actual price
if actual_google_values and google_prediction:
    last_actual_price = actual_google_values[-1][0]
    next_predicted_price = google_prediction[0]

    percent_change = (next_predicted_price - last_actual_price) / last_actual_price * 100

    insight2 = f"""
    <div style="font-family: Arial, sans-serif; font-size: 16px; line-height: 1.6;">
        <strong>The next predicted Google price is:</strong> <span style="color: #4CAF50;">${next_predicted_price:.2f}</span><br>
        <strong>Last actual Google price:</strong> <span style="color: #FF5722;">${last_actual_price:.2f}</span><br>
        <strong>Change:</strong> <span style="color: {'#4CAF50' if percent_change >= 0 else '#FF5722'};">{percent_change:+.2f}%</span>
    </div>
    """
else:
    insight = "<div style='font-family: Arial, sans-serif;'>Not enough data to generate insights.</div>"

# Display the insight using Markdown with HTML formatting
with tab2.col2:
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.markdown(insight2, unsafe_allow_html=True)

tab2.col1, tab2.col2, tab2.col3 = tab2.columns(3)
with tab2.col1:
    with st.popover("AI Model Infographics"):
        st.markdown(multi)
        st.link_button("Predictive AI Code by SMG", "https://github.com/SevilayMuni/Multivariate-TimeSeries-Forecast-LSTM-Apple-Google-Stocks/tree/main/Apple-Stock-LSTM-Model")

with tab2.col2:
    with st.popover("Variables Used by AI"):
        st.image("./images/variable-table.png")

with tab2.col3:
    with st.popover("Model Performance"):
        st.image("./images/variable-table.png")

with tab2.container(border = True):
    st.markdown(dedication, unsafe_allow_html=True)
    st.markdown(''':rainbow[End-to-end project is done by] :blue-background[Sevilay Munire Girgin]''')

tab2.warning('This work is not investment advice! It is merely a data science research.', icon="❗")

#--------TAB3----------
import plotly.graph_objects as go
from datetime import date

# Tab 3 Content: Stock Dashboard
tab3.subheader("StockSense AI: Interactive Stock Dashboard")
tab3.markdown(''':blue-background[Analyze real-time stock data]''')


import streamlit as st
import yfinance as yf
from datetime import date
import pandas as pd
import plotly.graph_objects as go

# Fetch and process data
def load_data(ticker, start_date):
    stock_data = yf.download(ticker, start=start_date)
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)
    stock_data.reset_index(inplace=True)
    return stock_data

# Technical Indicators
def calculate_indicators(data):
    # On-Balance Volume (OBV)
    data['OBV'] = (data['Volume'] * ((data['Close'] > data['Close'].shift(1)) * 2 - 1)).cumsum()

    # Moving Averages
    data['SMA_50'] = data['Close'].rolling(window=50).mean()  # 50-day Simple Moving Average
    data['SMA_200'] = data['Close'].rolling(window=200).mean()  # 200-day SMA
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()  # 50-day Exponential Moving Average
    data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()  # 200-day EMA

    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    data['BB_Mid'] = data['Close'].rolling(window=20).mean()
    data['BB_Upper'] = data['BB_Mid'] + (data['Close'].rolling(window=20).std() * 2)
    data['BB_Lower'] = data['BB_Mid'] - (data['Close'].rolling(window=20).std())

    return data

# Plot line chart
def plot_line_chart(data, x_col, y_cols, title):
    fig = go.Figure()
    for col in y_cols:
        fig.add_trace(go.Scatter(x=data[x_col], y=data[col], mode='lines', name=col))
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title="Value",
        template="plotly_white"
    )
    return fig

# Streamlit App
st.title("StockSense AI: Enhanced Stock Dashboard")

# Inputs
START_DATE = "2015-01-01"
ticker_list = ['AAPL', 'AMZN', 'AMD', 'GOOGL', 'INTC', 'META', 'MSFT', 'NVDA', 'TSLA']
selected_stock = st.selectbox('Select stock:', ticker_list)

technical_indicator = st.selectbox(
    'Select Technical Indicator:',
    [
        'Open-Close', 
        'High-Low', 
        'Stock Volume', 
        'OBV (On-Balance Volume)', 
        'SMA/EMA', 
        'RSI (Relative Strength Index)', 
        'Bollinger Bands'
    ])

# Fetch data
data = load_data(selected_stock, START_DATE)
data = calculate_indicators(data)

st.write("Raw Data:")
st.write(data.tail())

# Display selected chart
if technical_indicator == 'Open-Close':
    fig = plot_line_chart(data, 'Date', ['Open', 'Close'], f"Open-Close for {selected_stock}")
elif technical_indicator == 'High-Low':
    fig = plot_line_chart(data, 'Date', ['High', 'Low'], f"High-Low for {selected_stock}")
elif technical_indicator == 'Stock Volume':
    fig = plot_line_chart(data, 'Date', ['Volume'], f"Stock Volume for {selected_stock}")
elif technical_indicator == 'OBV (On-Balance Volume)':
    fig = plot_line_chart(data, 'Date', ['OBV'], f"OBV for {selected_stock}")
elif technical_indicator == 'SMA/EMA':
    fig = plot_line_chart(data, 'Date', ['Close', 'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200'], f"SMA/EMA for {selected_stock}")
elif technical_indicator == 'RSI (Relative Strength Index)':
    fig = plot_line_chart(data, 'Date', ['RSI'], f"RSI for {selected_stock}")
elif technical_indicator == 'Bollinger Bands':
    fig = plot_line_chart(data, 'Date', ['Close', 'BB_Mid', 'BB_Upper', 'BB_Lower'], f"Bollinger Bands for {selected_stock}")

st.plotly_chart(fig)
