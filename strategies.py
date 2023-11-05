import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utilities import *

from tqdm import tqdm

def train_pattern_recognition_model(features, labels):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model performance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy, X_train, X_test, y_train, y_test


def generate_ma_crossover_signals(data, short_window=50, long_window=200):
    """
    Generate moving average crossover signals.
    
    :param data: DataFrame with price data.
    :param short_window: Short-term moving average window.
    :param long_window: Long-term moving average window.
    :return: DataFrame with signals.
    """
    data = data.copy()
    
    data['SMA50'] = data['Close'].rolling(window=short_window).mean()
    data['SMA200'] = data['Close'].rolling(window=long_window).mean()

    # Initialize a new column for trading signals
    # data['Signal'] = 0  # 0 for no signal, 1 for buy, -1 for sell

    # Calculate signal crossovers
    # data['Signal'][short_window:] = np.where(data['SMA50'][short_window:] > data['SMA200'][short_window:], 1, 0)
    # data['Signal'][short_window:] = np.where(data['SMA50'][short_window:] < data['SMA200'][short_window:], -1, data['Signal'][short_window:])
    
    
    # Calculate and add SMA and EMA features to the DataFrame
    data['SMA50'] = calculate_sma(data, 50)
    data['SMA200'] = calculate_sma(data, 200)
    data['EMA8'] = calculate_ema(data, 8)
    data['EMA13'] = calculate_ema(data, 13)
    data['EMA21'] = calculate_ema(data, 21)
    data['EMA55'] = calculate_ema(data, 55)
    
    # Calculate and add RSI feature to the DataFrame
    data['RSI14'] = calculate_rsi(data, 14)
    data['MACD'], data['MACDSignalLine'] = calculate_macd(data, 12, 26, 9)
    
    # Calculate and add Bollinger Bands features to the DataFrame
    data['BollingerUpper'], data['BollingerLower'] = calculate_bollinger_bands(data, 20)
    
    # Calculate and add %K and %D features to the DataFrame
    data['%K'], data['%D'] = calculate_stochastic_oscillator(data, 14, 3)

    return generate_signals(data)


def generate_signals(df):
    signals = []
    entry_prices = []
    take_profits = []
    stop_losses = []
    in_position = False
    
    for i in tqdm(range(len(df)), desc="Processing Signals..."):
        if i < 55:
            signals.append(0)
            entry_prices.append(0)
            take_profits.append(0)
            stop_losses.append(0)
            continue

        ema_8 = df['Close'].rolling(window=8).mean()
        ema_13 = df['Close'].rolling(window=13).mean()
        ema_21 = df['Close'].rolling(window=21).mean()
        ema_55 = df['Close'].rolling(window=55).mean()

        if not in_position and ema_55.iloc[i] < ema_13.iloc[i] and ema_55.iloc[i] < ema_21.iloc[i] and ema_55.iloc[i] < ema_8.iloc[i]:
            # Buy Signal
            signals.append(1)
            entry_prices.append(df['Low'].iloc[i])
            has_target = False
            for n in range(len(df)):
                if n <= i:
                    continue
                # Find the take profit and stop loss
                if ema_55.iloc[n] > ema_13.iloc[n] and ema_55.iloc[n] > ema_21.iloc[n] and ema_55.iloc[n] > ema_8.iloc[n]:
                    take_profits.append(df['Low'].iloc[n])
                    take_profit = df['Low'].iloc[n]
                    has_target = True
                    break
            
            if not has_target:
                stop_losses.append(df['High'].iloc[i])
                take_profits.append(df['Low'].iloc[i])
            else:
                stop_losses.append(df['Low'].iloc[i] - ((df['Low'].iloc[i] - take_profit) / 2))
            in_position = True
        elif in_position and ema_55.iloc[i] > ema_13.iloc[i] and ema_55.iloc[i] > ema_21.iloc[i] and ema_55.iloc[i] > ema_8.iloc[i]:
            # Sell Signal
            signals.append(-1)
            entry_prices.append(df['High'].iloc[i])
            take_profit = 0
            has_target = False
            for n in range(len(df)):
                if n <= i:
                    continue
                # Find the take profit and stop loss
                if ema_55.iloc[n] < ema_13.iloc[n] and ema_55.iloc[n] < ema_21.iloc[n] and ema_55.iloc[n] < ema_8.iloc[n]:
                    take_profits.append(df['High'].iloc[n])
                    take_profit = df['High'].iloc[n]
                    has_target = True
                    break
            
            if not has_target:
                stop_losses.append(df['Low'].iloc[i])
                take_profits.append(df['High'].iloc[i])
            else:
                stop_losses.append(df['High'].iloc[i] + ((df['High'].iloc[i] - take_profit) / 2))
                
            in_position = False
        else:
            signals.append(0)
            entry_prices.append(0)
            take_profits.append(0)
            stop_losses.append(0)

    df['Signal'] = signals
    df['Entry'] = entry_prices
    df['TakeProfit'] = take_profits
    df['StopLoss'] = stop_losses
    
    # Apply a filter to reduce false signals (e.g., signal persistence)
    df['FilteredSignal'] = 0  # Initialize a new column for filtered signals
    filter_window = 5  # Adjust the filter window to your preference

    for i in tqdm(range(len(df)), desc="Filtering Signals..."):
        if i >= filter_window:
            if all(df['Signal'][i - filter_window + 1:i + 1] == 1):
                df.iloc[i, df.columns.get_loc('FilteredSignal')] = 1
            elif all(df['Signal'][i - filter_window + 1:i + 1] == -1):
                df.iloc[i, df.columns.get_loc('FilteredSignal')] = -1
                
    # Combine both signals (raw and filtered) to reduce false positives
    df['CombinedSignal'] = df['Signal'] * df['FilteredSignal']

    return optimize_strategy(df)

def backtest_strategy(data):
    """
    Backtest the trading strategy and calculate performance metrics.
    
    :param data: DataFrame with signals.
    :return: Performance metrics and backtest results.
    """
    # Implement backtesting logic and calculate performance metrics
    # This can include tracking trades, returns, risk management, etc.
    # Return the backtest results

def optimize_strategy(data):
    """
    Optimize the strategy by searching for the best parameters.
    
    :param data: DataFrame with signals.
    :return: Optimized parameters for the strategy.
    """
    # Use optimization techniques to find the best parameter combination
    # For example, optimize the short and long moving average windows
    
    for i in tqdm(range(len(data)), desc="Isolating Faulse Signals..."):
        if data['Signal'].iloc[i] == 1:
            if data['TakeProfit'].iloc[i] < data['Entry'].iloc[i]:
                data['Signal'].iloc[i] = 0
        if data['Signal'].iloc[i] == -1:
            if data['TakeProfit'].iloc[i] > data['Entry'].iloc[i]:
                data['Signal'].iloc[i] = 0
                
    return data

def apply_risk_management(data):
    """
    Apply risk management rules to the strategy.
    
    :param data: DataFrame with signals.
    :return: Data with risk management applied (e.g., stop-loss and take-profit levels).
    """
    # Implement risk management rules based on volatility, ATR, or other factors

def portfolio_management(data):
    """
    Apply portfolio management strategies to diversify risk.
    
    :param data: DataFrame with signals.
    :return: Portfolio performance metrics and results.
    """
    # Manage a portfolio of assets or strategies to diversify risk

# Additional functions for other aspects as needed