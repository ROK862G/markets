import os
import argparse
import pandas as pd
from utilities import *
from strategies import *
from markets import *
from sklearn.ensemble import RandomForestClassifier

# Parse the command-line arguments
parser = argparse.ArgumentParser(description='Predict trading signals, Entry, TakeProfit, and StopLoss for a given symbol.')
parser.add_argument('symbol', type=str, help='The symbol to analyze (e.g., USDZAR)')
parser.add_argument('count', type=int, help='The number of files to include (e.g., 10)')
args = parser.parse_args()

# Preprocess the corpus and load data for the given symbol
symbol = args.symbol.upper()
count = args.count

preprocess_corpus()

start_date = datetime.now().date() - timedelta(days=2)
end_date = datetime.now().date()
combined_df = get_polygon_data(symbol, start_date, end_date)

combined_df = load_from_corpus(count, symbol, True)

combined_df = generate_ma_crossover_signals(combined_df)

# Ensure there is data for analysis
if len(combined_df) > 0:
    # Load the Random Forest Classifier model
    model = load_model(symbol)

    if model:
        # Prepare input data for prediction
        features = combined_df[['EMA8', 'EMA13', 'EMA21', 'EMA55', 'BollingerUpper', 'BollingerLower', 'FilteredSignal', 'CombinedSignal', 'High', 'Low', 'Entry', 'TakeProfit', 'StopLoss', 'Open', 'Close']].dropna()

        # Predict trading signals using the model
        predicted_signals = model.predict(features)

        # Get the last data point for prediction
        last_data_point = features.iloc[-1]

        # Filter for signals indicating a future position (not 0)
        filtered_signals = [signal for signal in predicted_signals if signal != 0]

        if filtered_signals:
            # Get the last signal
            last_signal = filtered_signals[-1]

            # Define trading opportunity parameters
            entry_price = last_data_point['Entry']
            take_profit = last_data_point['TakeProfit']
            stop_loss = last_data_point['StopLoss']
            open = last_data_point['Open']
            close = last_data_point['Close']

            # Create a dictionary with the trading recommendations
            trading_recommendation = {
                'Signal': last_signal,
                'Entry': entry_price,
                'TakeProfit': take_profit,
                'StopLoss': stop_loss,
                'Open': open,
                'Close': close,
                'High': last_data_point['High'],
                'Low': last_data_point['Low'],
            }

            print(trading_recommendation)
            
            side = None
            if (last_signal == -1.0): 
                side = 'sell' 
            elif (last_signal == 1.0): 
                side = 'buy'
            
            if side is not None:
                is_aproved = 0
                try:
                    is_aproved = int(input("# Would you like to execute the order above? \n 1: Yes. \n 0: No."))
                except Exception as ex:
                    is_aproved = 0
                    
                if (is_aproved == 1):
                    execute_market_order(symbol=symbol, trade_units=10000, take_profit=take_profit, stop_loss=stop_loss, side='buy')
        else:
            print("No actionable signals found.")
    else:
        print("Model not found. Please train the model first.")
else:
    print(f"Not enough data for analysis in {symbol}.")
