from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import joblib
import requests

import os
from datetime import datetime, timedelta

# utilities.py
import requests
import pandas as pd
import time

from datetime import datetime, timedelta
from apis import *

import os
import pandas as pd

def preprocess_corpus(corpus_folder='corpus'):
    for filename in os.listdir(corpus_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(corpus_folder, filename)
            df = pd.read_csv(file_path)
            
            # Check if 'Signal' column exists, and if not, add it with initial value 0
            if 'Signal' not in df.columns:
                df['Signal'] = 0
                
                # Save the modified DataFrame back to the CSV file
                df.to_csv(file_path, index=False)
            
            if 'TakeProfit' not in df.columns:
                df['TakeProfit'] = ''
                df['StopLoss'] = ''
                
                # Save the modified DataFrame back to the CSV file
                df.to_csv(file_path, index=False)
                
            if 'Entry' not in df.columns:
                df['Entry'] = ''
                
                # Save the modified DataFrame back to the CSV file
                df.to_csv(file_path, index=False)

def get_polygon_data(symbol, start_date=datetime(2023, 1, 2), end_date=datetime(2023, 11, 4)):
    """Define the start and end dates for the data collection period
        current_date = datetime.now()
        end_date = current_date.strftime('%Y-%m-%d')
        start_date = (current_date - timedelta(days=365)).strftime('%Y-%m-%d')
    """

    current_date = start_date
    
    corpus = pd.DataFrame()  # Initialize an empty DataFrame

    query_count = 0
    while current_date <= end_date:
        data, is_cached = get_historical(symbol, current_date)
        
        if not is_cached:
            query_count += 1
            
        if data is not None:
            # Concatenate the data to the corpus DataFrame
            corpus = pd.concat([corpus, data])
        
        # Check if you have made 5 calls and then sleep for 61 seconds
        if query_count == 5:
            print("Sleeping for about a minute, free version constraints!")
            query_count = 0
            time.sleep(61)  # Sleep for 61 seconds

        current_date += timedelta(days=1)
        
    return corpus

def get_historical(symbol, date):
    """Convert the date to the required format."""
    date_str = date.strftime('%Y-%m-%d')

    # Check if a CSV file exists for the date
    csv_filename = f"corpus/{symbol}-{date_str}.csv"
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename, index_col='Date', parse_dates=True)
        return df, True

    base_url = f'https://api.polygon.io/v2/aggs/ticker/C:{symbol}/range/5/minute/{date_str}/{date_str}'

    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 5000,
        'apiKey': POLIGON_API_KEY,
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if 'results' in data:
        results = data['results']
        data_list = [{'Date': pd.Timestamp(result['t'], unit='ms'), 'Open': result['o'], 'Close': result['c'], 'High': result['h'], 'Volume': result['v'], 'Low': result['l'], 'Signal': 0, 'Entry': '', 'TakeProfit': '', 'StopLoss': '', } for result in results]

        df = pd.DataFrame(data_list)
        df.set_index('Date', inplace=True)

        # Save the data to a CSV file
        df.to_csv(csv_filename)
        
        return df, False

    return None, False

def load_from_corpus(count, symbol, prediction=False):
    """Loop through CSV files in the "corpus" folder and compile data into an array"""
    data_array = []
    corpus_folder = "corpus"
    counter = 0
    for filename in os.listdir(corpus_folder):
        if counter >= count:
            break
        if filename.startswith(symbol):
            # Read the CSV file into a DataFrame
            file_path = os.path.join(corpus_folder, filename)
            df = pd.read_csv(file_path, parse_dates=['Date'])  # Specify 'Date' column as a date
            df.set_index('Date', inplace=True)  # Set 'Date' as the index

            # Append the data to the array
            data_array.append(df)
            counter += 1
            
            
    if prediction:
        return data_array[-1]
    else: 
        return data_array

def calculate_sma(data, window):
    """Function to calculate SMA for a given DataFrame and window size"""
    return data['Close'].rolling(window=window).mean()

def get_data(symbol):
    # Replace 'YOUR_API_KEY' with your Alpha Vantage API key
    api_key = 'DEIMSC74QZZPUZY8'

    # Define the API endpoint and parameters for historical data
    base_url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_INTRADAY',
        'symbol': symbol,
        'interval': '30min',  # Hourly data
        'apikey': api_key,
        'outputsize': 'full',  # Retrieve the full historical data
    }

    # Make the API request to Alpha Vantage
    response = requests.get(base_url, params=params)
    data = response.json()
    
    return data

def save_model(model, name='trained_model'):
    """Saves the model to disk."""
    # Save the trained model to a file
    model_filename = f'models/{name}.pkl'
    joblib.dump(model, model_filename)
    print(f"Model saved as {model_filename}")

def calculate_ema(data, window):
    return data['Close'].ewm(span=window, adjust=False).mean()

def calculate_stochastic_oscillator(data, k_window, d_window):
    low_min = data['Low'].rolling(window=k_window).min()
    high_max = data['High'].rolling(window=k_window).max()
    k_percent = 100 * (data['Close'] - low_min) / (high_max - low_min)
    d_percent = k_percent.rolling(window=d_window).mean()
    return k_percent, d_percent

def calculate_bollinger_bands(data, window, num_std_dev=2):
    sma = calculate_sma(data, window)
    rolling_std = data['Close'].rolling(window=window).std()
    upper_band = sma + num_std_dev * rolling_std
    lower_band = sma - num_std_dev * rolling_std
    return upper_band, lower_band

def calculate_macd(data, short_window, long_window, signal_window):
    short_ema = calculate_ema(data, short_window)
    long_ema = calculate_ema(data, long_window)
    macd = short_ema - long_ema
    signal_line = macd.rolling(signal_window).mean()
    return macd, signal_line
    
def calculate_rsi(data, window):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
    
def load_model(name='trained_model'):
    """Loads the model from disk."""
    # Load the saved model
    try:
        model_filename = f'models/{name}.pkl'
        model = joblib.load(model_filename)
        return model
    except Exception as ex:
        print(f"Nothing to load here!")
        return None