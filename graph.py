import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Function to calculate SMA for a given DataFrame and window size
def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()

# Parse the command-line arguments
parser = argparse.ArgumentParser(description='Visualize data including SMA for a given symbol.')
parser.add_argument('symbol', type=str, help='The symbol to analyze (e.g., USDZAR)')
parser.add_argument('count', type=int, help='The number of files to include (e.g., 10)')
args = parser.parse_args()

# Define the symbol and the path to the "corpus" folder
symbol = args.symbol.upper()  # Convert the symbol to uppercase
count = args.count  # Number of files to include
corpus_folder = "corpus"

# Initialize lists to store data
data_array = []

# Loop through CSV files in the "corpus" folder and compile data into an array
counter = 0
for filename in os.listdir(corpus_folder):
    if counter >= count:
        break
    if filename.startswith(symbol):
        # Read the CSV file into a DataFrame
        file_path = os.path.join(corpus_folder, filename)
        df = pd.read_csv(file_path, parse_dates=['Date'])

        # Calculate SMA50 and SMA200
        df['SMA50'] = calculate_sma(df, window=50)
        df['SMA200'] = calculate_sma(df, window=200)

        # Append the data to the array
        data_array.append(df)
        counter += 1

# Combine the data from the array into a single DataFrame
combined_df = pd.concat(data_array)

# Create a figure and axis for the plot
fig, ax = plt.subplots()

# Plot the closing prices and SMAs for all files
ax.plot(combined_df['Date'], combined_df['Close'], label='Close')
ax.plot(combined_df['Date'], combined_df['SMA50'], label='SMA50')
ax.plot(combined_df['Date'], combined_df['SMA200'], label='SMA200')

# Set labels and legend
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title(f'Price and SMAs for {symbol}')
ax.legend()

# Display the plot
plt.show()
