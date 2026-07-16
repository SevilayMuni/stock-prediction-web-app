import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from keras.models import load_model
from keras.layers import LSTM

# Page Layout
st.set_page_config(page_title ="SafeStock AI Web Application",  page_icon = "🔮", layout = "wide")
tab1, tab2 = st.tabs(["Forecast", "Dashboard"])
info_multi = '''🔮 SafeStock AI is your go-to platform for exploring AI-powered stock forecasting and analysis using real-time stock values via Yahoo Finance.      
Whether you're a data science enthusiast or a market observer, this app blends cutting-edge deep learning with intuitive tools to bring you actionable insights.'''

with tab1: 
    st.header('SafeStock AI Web Application')
    st.info(info_multi)
    st.write(' ')

class CustomLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)  # Remove unrecognized argument
        super().__init__(*args, **kwargs)

# Load Models
apple_model = load_model('./models/Apple_Model.h5', custom_objects={"LSTM": CustomLSTM}, compile=False)
google_model = load_model('./models/Google_Model.h5', custom_objects={"LSTM": CustomLSTM}, compile=False)
tesla_model = load_model('./models/3rd-Tesla-LSTM-Model.h5', custom_objects={"LSTM": CustomLSTM}, compile=False)
amazon_model = load_model('./models/Amazon-LSTM-Model.h5', custom_objects={"LSTM": CustomLSTM}, compile=False)
intel_model = load_model('./models/Intel-2nd-LSTM-Model.h5', custom_objects={"LSTM": CustomLSTM}, compile=False)
meta_model = load_model('./models/Meta-LSTM-Model.h5', custom_objects={"LSTM": CustomLSTM}, compile=False)
microsoft_model = load_model('./models/Microsoft-LSTM-Model.h5', custom_objects={"LSTM": CustomLSTM}, compile=False)

# Ticker List
ticker_list = ['AAPL', 'AMZN', 'GOOG', 'INTC', 'META', 'MSFT', 'TSLA']

# Define function to calculate 'On Balance Volume (OBV)'
def On_Balance_Volume(Close, Volume):
    change = Close.diff()
    OBV = np.cumsum(np.where(change > 0, Volume, np.where(change < 0, -Volume, 0)))
    return OBV

def _download_with_retry(ticker, start, end, attempts=4):
    last_err = None
    for i in range(attempts):
        try:
            df = yf.download(ticker, start=start, end=end,
                             progress=False, threads=False)
            if not df.empty:
                return df
            last_err = "empty response"
        except Exception as e:
            last_err = e
        time.sleep(2 ** i)  # 1s, 2s, 4s, 8s backoff
    raise RuntimeError(f"yfinance returned no data for {ticker}: {last_err}")

# Define function to retrieve data from Yahoo Finance API and conduct feature engineering
@st.cache_data(ttl=60 * 60, show_spinner=False)  # refresh hourly; don't cache failures
def df_process(ticker):
    # Determine end and start dates for dataset download
    end = datetime.now()
    start = end - relativedelta(months = 3)  

    # Download data between start and end dates (with retry)
    df = _download_with_retry(ticker, start, end)
    # Fix column structure
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.columns.name = None  # Remove index name

    # Rename columns of the DataFrame
    column_dict = {'Open': 'open', 'High': 'high', 'Low': 'low',
                   'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'}
    df = df.rename(columns = column_dict)

    # Add technical indicators via feature engineering:
    # Create 'garman_klass_volatility'
    df['garman_klass_volatility'] = ((np.log(df['high']) - np.log(df['low'])) ** 2) / 2 - (2 * np.log(2) - 1) * ((np.log(df['close']) - np.log(df['open'])) ** 2)

    # Create 'dollar_volume'
    df['dollar_volume'] = (df['close'] * df['volume']) / 1e6

    # Create 'ema' column
    df['ema'] = df['close'].ewm(span=14, adjust=False).mean()

    # Create 'obv'
    df['obv'] = On_Balance_Volume(df['close'], df['volume'])

    # Create 'macd'
    df['macd'] = df['close'].ewm(span = 12, adjust = False).mean() - df['close'].ewm(span = 26, adjust = False).mean()

    # Create 'ma_3_days'
    df['ma_3_days'] = df['close'].rolling(3).mean()
    return df


# Call to fetch and engineer data
try:
    apple_df_processed = df_process(ticker_list[0])
    amazon_df_processed = df_process(ticker_list[1])
    google_df_processed = df_process(ticker_list[2])
    intel_df_processed = df_process(ticker_list[3])
    meta_df_processed = df_process(ticker_list[4])
    microsoft_df_processed = df_process(ticker_list[5])
    tesla_df_processed = df_process(ticker_list[6])
except RuntimeError as e:
    st.error(f"Market data is temporarily unavailable ({e}). Please refresh in a minute.")
    st.stop()

# List selected features
apple_features = ['close', 'garman_klass_volatility', 'dollar_volume', 'obv', 'ma_3_days']
amazon_features = ['close', 'volume', 'dollar_volume', 'obv', 'ema']
google_features = ['close', 'volume', 'dollar_volume', 'obv', 'ma_3_days', 'macd']
intel_features = ['close', 'garman_klass_volatility', 'dollar_volume', 'obv', 'ma_3_days']
meta_features = ['close', 'volume', 'dollar_volume', 'obv', 'ema']
microsoft_features = ['close', 'volume', 'garman_klass_volatility', 'dollar_volume', 'obv', 'ma_3_days']
tesla_features = ['close', 'dollar_volume', 'obv', 'ema', 'ma_3_days']


# Define function scale data
def create_feed_dset(df_processed, feature_list, n_past, model):
    dset = df_processed.filter(feature_list)
    dset.dropna(axis = 0, inplace = True)

    # Guard: need more rows than the lookback window
    if len(dset) <= n_past:
        raise RuntimeError(
            f"Only {len(dset)} usable rows after cleaning; need more than {n_past}.")

    # Scale the datasets
    scaler = MinMaxScaler(feature_range = (0, 1))
    df_scaled = scaler.fit_transform(dset)

    # Create X from the dataset
    dataX = []
    dataY = []
    for i in range(n_past, len(df_scaled)):
        dataX.append(df_scaled[i - n_past:i, 0:df_scaled.shape[1]])
        dataY.append(df_scaled[i,0])
    dataX = np.array(dataX)
    
    # Make predictions using the model
    prediction = model.predict(dataX)
    return prediction, scaler

# Call to get prediction
apple_prediction_init, scaler1 = create_feed_dset(apple_df_processed, apple_features, 21, apple_model)
amazon_prediction_init, scaler2 = create_feed_dset(amazon_df_processed, amazon_features, 15, amazon_model)
google_prediction_init, scaler3 = create_feed_dset(google_df_processed, google_features, 21, google_model)
intel_prediction_init, scaler4 = create_feed_dset(intel_df_processed, intel_features, 25, intel_model)
meta_prediction_init, scaler5 = create_feed_dset(meta_df_processed, meta_features, 20, meta_model)
microsoft_prediction_init, scaler6 = create_feed_dset(microsoft_df_processed, microsoft_features, 20, microsoft_model)
tesla_prediction_init, scaler7 = create_feed_dset(tesla_df_processed, tesla_features, 15, tesla_model)

# Inverse transformation for 5 features
def inverse_transform_predictions1(prediction_init, scaler):
    # Prepare the dataset for inverse transform
    prediction_array = np.repeat(prediction_init, 5, axis = -1)  # Repeat values along the last axis

    # Perform the inverse transform and extract the first column
    pred = scaler.inverse_transform(np.reshape(prediction_array, (len(prediction_init), 5)))[:5, 0]
    return pred

# Inverse transformation for 6 features
def inverse_transform_predictions2(prediction_init, scaler):
    # Prepare the dataset for inverse transform
    prediction_array = np.repeat(prediction_init, 6, axis = -1)  # Repeat values along the last axis

    # Perform the inverse transform and extract the first column
    pred = scaler.inverse_transform(np.reshape(prediction_array, (len(prediction_init), 6)))[:5, 0]
    return pred

# Get prediction list
apple_pred_list = inverse_transform_predictions1(apple_prediction_init, scaler1).tolist()
amazon_pred_list = inverse_transform_predictions1(amazon_prediction_init, scaler2).tolist()
intel_pred_list = inverse_transform_predictions1(intel_prediction_init, scaler4).tolist()
meta_pred_list = inverse_transform_predictions1(meta_prediction_init, scaler5).tolist()
tesla_pred_list = inverse_transform_predictions1(tesla_prediction_init, scaler7).tolist()
google_pred_list = inverse_transform_predictions2(google_prediction_init, scaler3).tolist()
microsoft_pred_list = inverse_transform_predictions2(microsoft_prediction_init, scaler6).tolist()

# Function to create prediction dataframe
def prediction_table(pred_list):
    pred_df = pd.DataFrame({'Predicted Day': ['Tomorrow', '2nd Day', '3rd Day', '4th Day', '5th Day'], 
    'Adj. Closing Price($)': [ '%.2f' % elem for elem in pred_list]})

    # Set df index to the 'name' column
    pred_df.set_index('Predicted Day', inplace=True)
    return pred_df

# Function to generate prediction insight
def generate_insight(df_processed, pred_list):
    """
    Generates and displays stock price insight based on actual and predicted values.

    Parameters:
    df (pd.DataFrame): DataFrame containing the actual stock prices.
    prediction (list): List containing the next predicted price.
    container (streamlit.DeltaGenerator): Streamlit container for displaying the insight.
    """
    # Extract actual values from the DataFrame
    actual_values = df_processed['close'].values.tolist()

    # Ensure there is data to process
    if actual_values and pred_list:
        last_actual_price = actual_values[-1]  # Access the last actual price
        next_predicted_price = pred_list[0]  # Predicted next price

        # Calculate percentage change
        percent_change = (next_predicted_price - last_actual_price) / last_actual_price * 100

        # Generate the HTML insight
        insight = f"""
        <div style="font-family: Arial, sans-serif; font-size: 16px; line-height: 1.6;">
            <strong>The next predicted stock price is:</strong> 
            <span style="color: #4CAF50;">${next_predicted_price:.2f}</span><br>
            <strong>Last actual price:</strong> 
            <span style="color: #FF5722;">${last_actual_price:.2f}</span><br>
            <strong>Change:</strong> 
            <span style="color: {'#4CAF50' if percent_change >= 0 else '#FF5722'};">
                {percent_change:+.2f}%
            </span>
        </div>
        """
    else:
        # Fallback message for insufficient data
        insight = "<div style='font-family: Arial, sans-serif;'>Not enough data to generate insights.</div>"
    return insight

# Stock selection button
stock_selection = tab1.selectbox("Select Stock to Get Predictions:", options=["Apple", "Amazon", "Google", "Intel", "Meta", "Microsoft", "Tesla"])

# Update data based on selection
if stock_selection == "Apple":
    selected_pred_list = apple_pred_list
    selected_df_processed = apple_df_processed
elif stock_selection == "Amazon":
    selected_pred_list = amazon_pred_list
    selected_df_processed = amazon_df_processed
elif stock_selection == "Google":
    selected_pred_list = google_pred_list
    selected_df_processed = google_df_processed
elif stock_selection == "Intel":
    selected_pred_list = intel_pred_list
    selected_df_processed = intel_df_processed
elif stock_selection == "Meta":
    selected_pred_list = meta_pred_list
    selected_df_processed = meta_df_processed
elif stock_selection == "Microsoft":
    selected_pred_list = microsoft_pred_list
    selected_df_processed = microsoft_df_processed
elif stock_selection == "Tesla":
    selected_pred_list = tesla_pred_list
    selected_df_processed = tesla_df_processed

# Generate prediction table and insights
pred_df = prediction_table(selected_pred_list)
insight = generate_insight(selected_df_processed, selected_pred_list)


tab1.col1, tab1.col2 = tab1.columns(2)
with tab1.col1:
    st.markdown(f"""<div style="font-family: Arial, sans-serif; font-size: 18px; line-height: 1.6;"> 
    <strong>{stock_selection}</strong><be> 
    <strong>Predictions for the Next 5 Days</strong>
    </div>""", unsafe_allow_html=True)
    st.dataframe(pred_df)

with tab1.col2:
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.markdown(insight, unsafe_allow_html=True)


dedication = """<div style="font-family: Arial, sans-serif; font-size: 16px; line-height: 1.6;"><i>The StockSense AI is dedicated to my dearest, Ceyhun Utku Girgin.</i>"""
with tab1.container(border = True):
    st.markdown(dedication, unsafe_allow_html=True)
    st.markdown(''':rainbow[End-to-end project is done by] :blue-background[Sevilay Munire Girgin]''')

tab1.warning('Disclaimer: This project is for research and educational purposes only and is not intended for financial or investment advice.', icon="❗")

#--------TAB2----------
import plotly.graph_objects as go
from datetime import date

with tab2: 
    st.header('SafeStock AI: Interactive Stock Dashboard')
    st.markdown(''':blue-background[📊 Technical Analysis: Explore trends with indicators like SMA, EMA, RSI, and OBV using interactive charts.]''')

obv_text = '''Tracks the flow of volume to predict price changes.  
Purpose: Identifies buying/selling pressure based on volume. A rising OBV suggests accumulation (buying), while a falling OBV suggests distribution (selling).  
Use Case: Combine with price trends to confirm breakout patterns or reversals.'''

ma_text = '''Moving averages smooth out price data to identify trends over a period.  
Simple Moving Average (SMA): Average of closing prices over a fixed period.   
Exponential Moving Average (EMA): Similar to SMA but gives more weight to recent prices for faster responsiveness.  
Purpose: SMA -- Tracks long-term trends (e.g., 50-day and 200-day SMA).   
EMA -- Tracks short-term momentum (e.g., 12-day and 26-day EMA).    
Use Case: Bullish signal -- Short-term MA crosses above long-term MA ("Golden Cross").  
Bearish signal -- Short-term MA crosses below long-term MA ("Death Cross").'''

rsi_text = '''RSI measures price momentum to identify overbought/oversold conditions.  
Compares average gains and losses over 14 days to generate a score between 0-100.  
RSI > 70: Overbought (may signal a sell opportunity).  
RSI < 30: Oversold (may signal a buy opportunity).   
Purpose: Indicates potential reversals or continuation in price trends.  
Use Case: Combine with other indicators to confirm breakout or correction signals.'''

tab2.col1, tab2.col2, tab2.col3 = tab2.columns(3)
with tab2.col1:
    with st.popover("On-Balance Volume(OBV)"):
        st.markdown(obv_text)

with tab2.col2:
    with st.popover("Moving Averages(SMA/EMA)"):
        st.markdown(ma_text)
      
with tab2.col3:
    with st.popover("Relative Strength Index(RSI)"):
        st.markdown(rsi_text)

# Fetch and process data
@st.cache_data(ttl=60 * 60, show_spinner=False)
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
        template="plotly_white")
    return fig

# Inputs
START_DATE = "2015-01-01"
ticker_list = ['AAPL', 'AMZN', 'AMD', 'GOOGL', 'INTC', 'META', 'MSFT', 'NVDA', 'TSLA']
selected_stock = tab2.selectbox('Select stock:', ticker_list)

technical_indicator = tab2.selectbox(
    'Select Technical Indicator:',
    [
        'Open-High', 
        'Low-Close', 
        'Stock Volume', 
        'OBV (On-Balance Volume)', 
        'SMA/EMA', 
        'RSI (Relative Strength Index)'])

# Fetch data
data = load_data(selected_stock, START_DATE)
data = calculate_indicators(data)

# Display selected chart
if technical_indicator == 'Open-High':
    fig = plot_line_chart(data, 'Date', ['Open', 'High'], f"Opening versus Highest Prices for {selected_stock}")
elif technical_indicator == 'Low-Close':
    fig = plot_line_chart(data, 'Date', ['Low', 'Close'], f"Lowest versus Closing Prices for {selected_stock}")
elif technical_indicator == 'Stock Volume':
    fig = plot_line_chart(data, 'Date', ['Volume'], f"Stock Volume for {selected_stock}")
elif technical_indicator == 'OBV (On-Balance Volume)':
    fig = plot_line_chart(data, 'Date', ['OBV'], f"OBV for {selected_stock}")
elif technical_indicator == 'SMA/EMA':
    fig = plot_line_chart(data, 'Date', ['Close', 'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200'], f"SMA/EMA for {selected_stock}")
elif technical_indicator == 'RSI (Relative Strength Index)':
    fig = plot_line_chart(data, 'Date', ['RSI'], f"RSI for {selected_stock}")

with tab2: 
    st.plotly_chart(fig)
