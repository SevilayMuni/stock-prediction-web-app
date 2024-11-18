import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
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

tab1.header('🔮 StockSense AI Web Application')
#info = """<div style="font-family: Arial, sans-serif; font-size: 18px; line-height: 1.6;"><strong><i>The AI uses real-time stock values via Yahoo Finance</i></strong></div>"""
tab1.info('StockSense AI uses real-time stock values via Yahoo Finance.')
tab1.write(' ')

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

tab1.markdown("---")
tab1.markdown(''':rainbow[End-to-end project is done by] :blue-background[Sevilay Munire Girgin]''')
tab1.warning('This work is not investment advice! It is merely a data science research.', icon="❗")

#-----------------------
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

tab2.markdown("---")
tab2.markdown(''':rainbow[End-to-end project is done by] :blue-background[Sevilay Munire Girgin]''')
tab2.warning('This work is not investment advice! It is merely a data science research.', icon="❗")

#--------TAB3----------
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import pytz
import ta


# Tab 3 Content: Stock Dashboard
tab3.markdown("## Interactive Stock Dashboard")
tab3.markdown("Select a stock ticker, time period, chart type, and technical indicators to explore real-time stock data.")

# Stock Options for Selectbox
stocks = {
    "Apple": "AAPL",
    "Google": "GOOGL",
    "Nvidia": "NVDA",
    "Tesla": "TSLA",
    "Microsoft": "MSFT",
    "Gamestop": "GME"
}

# Input Fields for User Interaction
ticker_name = tab3.selectbox("Choose Stock", options=list(stocks.keys()), index=0, help="Select a stock to view its data.")
ticker = stocks[ticker_name]
time_period = tab3.selectbox("Time Period", ["1d", "1wk", "1mo", "6mo", "1y", "max"], help="Choose the time period for the stock data.")
chart_type = tab3.selectbox("Chart Type", ["Candlestick", "Line"], help="Choose how the stock price data will be visualized.")
indicators = tab3.multiselect("Technical Indicators", ["SMA 20", "EMA 20"], help="Select technical indicators to overlay on the chart.")

# Fetch and Display Stock Data
if tab3.button("Generate Dashboard"):
    with st.spinner("Fetching stock data..."):
        # Map time periods to intervals
        interval_mapping = {
            '1d': '1m',
            '1wk': '30m',
            '1mo': '1d',
            '6mo': '1d',
            '1y': '1wk',
            'max': '1wk'
        }

        # Fetch data from Yahoo Finance
        end_date = datetime.now()
        try:
            if time_period == '1wk':
                start_date = end_date - timedelta(days=7)
                data = yf.download(ticker, start=start_date, end=end_date, interval=interval_mapping[time_period])
            else:
                data = yf.download(ticker, period=time_period, interval=interval_mapping[time_period])

            # Check if data is empty
            if data.empty:
                tab3.error(f"No data found for the ticker '{ticker}' and selected time period.")
            else:
                # Preprocess data
                data.reset_index(inplace=True)

                # Calculate Technical Indicators
                data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
                data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)

                # Flatten Indicators to Ensure 1D Arrays
                data['SMA_20'] = data['SMA_20'].to_numpy().flatten()
                data['EMA_20'] = data['EMA_20'].to_numpy().flatten()

                # Stock Overview Metrics
                last_close = data['Close'].iloc[-1]
                prev_close = data['Close'].iloc[0]
                change = last_close - prev_close
                pct_change = (change / prev_close) * 100

                # Display Stock Metrics
                tab3.metric("Last Price", f"${last_close:.2f}", delta=f"{change:.2f} ({pct_change:.2f}%)")

                # Plotly Chart
                fig = go.Figure()
                if chart_type == "Candlestick":
                    fig.add_trace(go.Candlestick(
                        x=data['Datetime'],
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name="Candlestick"))
                else:
                    fig.add_trace(go.Scatter(x=data['Datetime'], y=data['Close'], mode='lines', name="Close Price"))

                # Add Technical Indicators
                for indicator in indicators:
                    if indicator == "SMA 20":
                        fig.add_trace(go.Scatter(x=data['Datetime'], y=data['SMA_20'], name="SMA 20"))
                    elif indicator == "EMA 20":
                        fig.add_trace(go.Scatter(x=data['Datetime'], y=data['EMA_20'], name="EMA 20"))

                # Update Chart Layout
                fig.update_layout(
                    title=f"{ticker_name} Stock Price",
                    xaxis_title="Time",
                    yaxis_title="Price (USD)",
                    height=600
                )

                # Display Chart
                tab3.plotly_chart(fig, use_container_width=True)

                # Historical Data Table
                tab3.markdown("### Historical Data")
                tab3.dataframe(data[["Datetime", "Open", "High", "Low", "Close", "Volume"]])

        except Exception as e:
            tab3.error(f"An error occurred: {str(e)}")

# Footer
tab3.markdown("---")
tab3.markdown(':rainbow[Project developed by] :blue-background[Sevilay Munire Girgin]')
tab3.warning("This dashboard is for research purposes only and does not provide investment advice.", icon="❗")



# Footer
tab3.markdown("---")
tab3.markdown(':rainbow[Project developed by] :blue-background[Sevilay Munire Girgin]')
tab3.warning("This dashboard is for research purposes only and does not provide investment advice.", icon="❗")
