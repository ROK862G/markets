import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from utilities import *

# Function to calculate SMA for a given DataFrame and window size
def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()

# Function to generate trading signals based on a moving average crossover strategy
def generate_ma_crossover_signals(data):
    data['SMA50'] = calculate_sma(data, window=50)
    data['SMA200'] = calculate_sma(data, window=200)
    data['Signal'] = 0  # 0 for no signal
    data.loc[data['SMA50'] > data['SMA200'], 'Signal'] = 1  # Buy signal (Golden Cross)
    data.loc[data['SMA50'] < data['SMA200'], 'Signal'] = -1  # Sell signal (Death Cross)
    data['Signal'] = data['Signal'].shift(-1)  # Shift to align with the next time step
    return data

# Parse the command-line arguments
parser = argparse.ArgumentParser(description='Visualize data including SMA for a given symbol.')
parser.add_argument('symbol', type=str, help='The symbol to analyze (e.g., USDZAR)')
parser.add_argument('count', type=int, help='The counter to analyze (e.g., 10)')
args = parser.parse_args()

# Define the symbol and the path to the "corpus" folder
symbol = args.symbol.upper()  # Convert the symbol to uppercase
count = args.count

# Initialize lists to store SMA50 and SMA200 values
sma50_values = []
sma200_values = []

# Create a figure and axis for the plot
fig, ax = plt.subplots()

# Loop through CSV files in the "corpus" folder
counter = 0
for filename in os.listdir(corpus_folder):
    if counter >= count:
        break
    if filename.startswith(symbol):
        counter += 1
        # Read the CSV file into a DataFrame
        file_path = os.path.join(corpus_folder, filename)
        df = pd.read_csv(file_path, parse_dates=['Date'])

        # Generate trading signals using the moving average crossover strategy
        df = generate_ma_crossover_signals(df)

        # Plot the closing prices and SMAs for each file
        ax.plot(df['Date'], df['Close'], label='Close Price', color='black')
        ax.plot(df['Date'], df['SMA50'], label='SMA50', color='blue', linestyle='--')
        ax.plot(df['Date'], df['SMA200'], label='SMA200', color='red', linestyle='--')

# Set labels and legend
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title(f'Price and SMAs for {symbol}')
ax.legend()

# Display the plot
plt.show()