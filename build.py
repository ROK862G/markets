import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from utilities import *
from strategies import *

# Parse the command-line arguments
parser = argparse.ArgumentParser(description='Visualize data including SMA for a given symbol.')
parser.add_argument('symbol', type=str, help='The symbol to analyze (e.g., USDZAR)')
parser.add_argument('count', type=int, help='The number of files to include (e.g., 10)')
args = parser.parse_args()

preprocess_corpus()

# Define the symbol and the path to the "corpus" folder
symbol = args.symbol.upper()  # Convert the symbol to uppercase
count = args.count  # Number of files to include

# Initialize lists to store data
data_array = load_from_corpus(count, symbol)

# Load the model if it exists, or create a new one
model = load_model(symbol)


# Combine the data from the array into a single DataFrame
combined_df = pd.concat(data_array)

# Generate trading signals using the moving average crossover strategy
combined_df = generate_ma_crossover_signals(combined_df)

try:
    combined_df.to_csv(f'build/{symbol}-live-processed.csv')
except Exception as ex:
    print(ex)


# If there is sufficient data, perform the analysis and trading strategy
if len(combined_df) > 0:
    # df['Signal'] = signals
    # df['Entry'] = [entry_price if signal == 1 else None for signal in signals]
    # df['TakeProfit'] = [take_profit if signal == -1 else None for signal in signals]
    # df['StopLoss'] = [stop_loss if signal == -1 else None for signal in signals]
    # df['TradeSuccessful'] = success_rates
    # Prepare input data for machine learning    
    features = combined_df[['EMA8', 'EMA13', 'EMA21', 'EMA55', 'BollingerUpper', 'BollingerUpper', 'FilteredSignal', 'CombinedSignal', 'High', 'Low', 'Entry' ]].dropna()  # Features
    labels = combined_df['Signal'].loc[features.index]  # Labels

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    if len(X_train) > 0:
        if model is None:
            model, accuracy, X_train, X_test, y_train, y_test = train_pattern_recognition_model(features, labels)
            
            print(f"Model accuracy: {accuracy * 100}%")
            
            save_model(model, symbol)

        # Make predictions on the testing set
        y_pred = model.predict(X_test)

        # Implement a basic trading strategy
        combined_df['Position'] = 0
        combined_df.loc[features.index[-len(y_test):], 'Position'] = y_pred
        combined_df['Position'] = combined_df['Position'].shift(1)  # Shift to avoid look-ahead bias

        # Calculate returns based on trading decisions
        combined_df['Returns'] = combined_df['Close'] * combined_df['Position']
        combined_df['Cumulative_Returns'] = combined_df['Returns'].cumsum()

        # Backtest the trading strategy
        combined_df['Strategy_Returns'] = combined_df['Position'].shift(1) * combined_df['Returns']
        combined_df['Cumulative_Strategy_Returns'] = combined_df['Strategy_Returns'].cumsum()

        # Determine trade success
        combined_df['TradeSuccess'] = 0
        for i in range(len(combined_df)):
            if combined_df['Position'][i] == 0:
                continue
            if combined_df['Position'][i] == 1:
                if combined_df['Close'][i] >= combined_df['TakeProfit'][i]:
                    combined_df['TradeSuccess'][i] = 1  # Trade was successful
                elif combined_df['Close'][i] <= combined_df['StopLoss'][i]:
                    combined_df['TradeSuccess'][i] = -1  # Trade was not successful
            elif combined_df['Position'][i] == -1:
                if combined_df['Close'][i] <= combined_df['TakeProfit'][i]:
                    combined_df['TradeSuccess'][i] = 1  # Trade was successful
                elif combined_df['Close'][i] >= combined_df['StopLoss'][i]:
                    combined_df['TradeSuccess'][i] = -1  # Trade was not successful

        # Evaluate performance
        total_returns = combined_df['Cumulative_Returns'].iloc[-1]
        strategy_returns = combined_df['Cumulative_Strategy_Returns'].iloc[-1]
        strategy_returns_percent = strategy_returns / combined_df['Cumulative_Returns'].iloc[0] * 100
        total_trades = len(combined_df[combined_df['Position'] != 0])
        successful_trades = len(combined_df[combined_df['TradeSuccess'] == 1])
        success_rate = (successful_trades / total_trades) * 100


        print(f"Total Returns: {total_returns:.2f}")
        print(f"Strategy Returns: {strategy_returns:.2f}")
        print(f"Strategy Returns (%) : {strategy_returns_percent:.2f}%")

        # Visualize the results, backtest, and evaluate performance
        # (You can use libraries like Matplotlib and Pandas for this)

        # Visualize the results
        plt.figure(figsize=(12, 6))
        plt.plot(combined_df.index, combined_df['Close'], label='Close Price', color='black')
        plt.plot(combined_df.index, combined_df['SMA50'], label='SMA50', color='blue', linestyle='--')
        plt.plot(combined_df.index, combined_df['SMA200'], label='SMA200', color='red', linestyle='--')
        plt.legend(loc='upper left')
        plt.title(f'{symbol} Price and Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()



        # # Plot the strategy returns
        plt.figure(figsize=(12, 6))
        plt.plot(combined_df.index, combined_df['Cumulative_Returns'], label='Buy and Hold', color='black')
        plt.plot(combined_df.index, combined_df['Cumulative_Strategy_Returns'], label='Strategy', color='blue')
        plt.legend(loc='upper left')
        plt.title('Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Returns')
        plt.show()
        
        last = None
        count = 0

        while last == None:
            count += 1
            # Analyze the last signal generated by the strategy
            last_signal = combined_df['Signal'].iloc[-1 * count]

            # Define trading opportunity parameters
            if last_signal == 1:
                entry_price = combined_df['Close'].iloc[-1 * count]  # Current closing price for entry
                exit_price = combined_df['TakeProfit'].iloc[-1 * count]  # Previous closing price for exit
                stop_loss = combined_df['StopLoss'].iloc[-1 * count]  # Set a stop-loss 2% below the entry price
                take_profit = combined_df['TakeProfit'].iloc[-1 * count]  # Set a take-profit 2% above the entry price

                # Provide a trading recommendation
                print("#########################")
                print(f"Trading Opportunity for {symbol}: Buy")
                print(f"Entry Price: {entry_price:.2f}")
                print(f"Exit Price (Previous Close): {exit_price:.2f}")
                print(f"Stop-loss: {stop_loss:.2f}")
                print(f"Take Profit: {take_profit:.2f}")
                
                last = combined_df['Close'].iloc[-1 * count]

            elif last_signal == -1:
                entry_price = combined_df['Close'].iloc[-1 * count]  # Current closing price for entry
                exit_price = combined_df['TakeProfit'].iloc[-1 * count]  # Previous closing price for exit
                stop_loss = combined_df['StopLoss'].iloc[-1 * count]  # Set a stop-loss 2% below the entry price
                take_profit = combined_df['TakeProfit'].iloc[-1 * count]  # Set a take-profit 2% above the entry price

                # Provide a trading recommendation
                print("#########################")
                print(f"Trading Opportunity for {symbol}: Sell")
                print(f"Entry Price: {entry_price:.2f}")
                print(f"Exit Price (Previous Close): {exit_price:.2f}")
                print(f"Stop-loss: {stop_loss:.2f}")
                print(f"Take Profit: {take_profit:.2f}")
                
                last = combined_df['Close'].iloc[-1 * count]

    else:
        print(f"Not enough data for analysis in {symbol}.")
else:
    print(f"Not enough data for analysis in {symbol}.")
