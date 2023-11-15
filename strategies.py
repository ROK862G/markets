import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utilities import *

from tqdm import tqdm


NOISE_SCALAR = 2

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
    
    # Step 1: Calculate Features for Simple Moving Average, and Exponential Moving Average
    data['SMA50'] = calculate_sma(data, 50 * NOISE_SCALAR)
    data['SMA200'] = calculate_sma(data, 200 * NOISE_SCALAR)
    data['EMA8'] = calculate_ema(data, 8 * NOISE_SCALAR)
    data['EMA13'] = calculate_ema(data, 13 * NOISE_SCALAR)
    data['EMA21'] = calculate_ema(data, 21 * NOISE_SCALAR)
    data['EMA55'] = calculate_ema(data, 55 * NOISE_SCALAR)
    
    # Step 2: Calculate Features for Relative Strength Index, and Moving Average Convergence
    data['RSI14'] = calculate_rsi(data, 14)
    data['MACD'], data['MACDSignalLine'] = calculate_macd(data, 12, 26, 9)
    
    # Step 3: Calculate and add Bollinger Bands features to the DataFrame
    data['BollingerUpper'], data['BollingerLower'] = calculate_bollinger_bands(data, 20)
    
    # Step 4: Calculate and add %K and %D features to the DataFrame
    data['%K'], data['%D'] = calculate_stochastic_oscillator(data, 14, 3)

    # Continue with Generating Labels.
    return generate_labels(data)


def generate_labels(df):
    signals = []
    entry_prices = []
    take_profits = []
    stop_losses = []
    in_position = False
    last_crossover_index = 0
    
    for i in tqdm(range(len(df)), desc="Processing Features..."):
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

        if not in_position:
            if (
                ema_55.iloc[i] < ema_13.iloc[i]
                and ema_55.iloc[i] < ema_21.iloc[i]
                and ema_55.iloc[i] < ema_8.iloc[i]
            ):
                # Step 1: Buy Signal
                signals.append(1)
                entry_prices.append(df['Low'].iloc[i])
                take_profit = 0
                has_target = False
                
                # Step 2: Find the nearest point where we can exit the trade.
                for n in range(i + 1, len(df)):
                    if (
                        ema_55.iloc[n] > ema_13.iloc[n]
                        and ema_55.iloc[n] > ema_21.iloc[n]
                        and ema_55.iloc[n] > ema_8.iloc[n]
                    ):
                        take_profit = df['High'].iloc[n]
                        take_profits.append(take_profit)
                        stop_losses.append(df['Low'].iloc[i] - ((df['Low'].iloc[i] - take_profit) / 2))
                        last_crossover_index = n
                        in_position = True
                        has_target = True
                        break
                
                # Step 3: Do we have a target exit for this trade?
                if not has_target:
                    stop_losses.append(df['Low'].iloc[i])
                    take_profits.append(df['High'].iloc[i])
            else:
                
                # Step 4: If we are here, we might want to avoid this trade!
                signals.append(0)
                entry_prices.append(0)
                take_profits.append(0)
                stop_losses.append(0)
        else:
            if (
                ema_55.iloc[i] > ema_13.iloc[i]
                and ema_55.iloc[i] > ema_21.iloc[i]
                and ema_55.iloc[i] > ema_8.iloc[i]
            ):
                # Step 1: Sell Signal
                signals.append(-1)
                entry_prices.append(df['High'].iloc[i])
                take_profit = 0
                has_target = False
                
                # Step 2: Find the nearest point where we can exit the trade.
                for n in range(i + 1, len(df)):
                    if (
                        ema_55.iloc[n] < ema_13.iloc[n]
                        and ema_55.iloc[n] < ema_21.iloc[n]
                        and ema_55.iloc[n] < ema_8.iloc[n]
                    ):
                        take_profit = df['Low'].iloc[n]
                        take_profits.append(take_profit)
                        stop_losses.append(df['High'].iloc[i] + ((take_profit - df['High'].iloc[i]) / 2))
                        last_crossover_index = n
                        in_position = False
                        has_target = True
                        break
                
                # Step 3: Do we have a target exit for this trade?
                if not has_target:
                    stop_losses.append(df['High'].iloc[i])
                    take_profits.append(df['Low'].iloc[i])
            else:
                
                # Step 4: If we are here, we might want to avoid this trade!
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
    # Create a boolean mask for the conditions where 'Signal' should be set to 0
    mask = ((data['Signal'] == 1) & (data['Entry'] >= data['TakeProfit'])) | ((data['Signal'] == -1) & (data['Entry'] <= data['TakeProfit']))
    
    # Set 'Signal' to 0 for the rows that meet the conditions
    data.loc[mask, 'Signal'] = 0
    
    return filter_stop_loss_before_take_profit(data)


def filter_stop_loss_before_take_profit(data):
    """
    Filter out trades where StopLoss is hit before TakeProfit.

    :param data: DataFrame with trading signals, TakeProfit, StopLoss, Low, and High.
    :return: Data with trades that hit TakeProfit before StopLoss filtered out.
    """
    data = data.copy()  # Create a copy of the original data to avoid modifying it directly

    trade_in_progress = False
    current_trade_signal = 0

    for i in tqdm(range(len(data)), desc="Simulating Trading Strategy..."):
        if trade_in_progress:
            # Check if the trade is a Buy signal
            if current_trade_signal == 1:
                if data['Low'].iloc[i] < data['StopLoss'].iloc[i]:
                    data.at[i, 'Signal'] = 0  # Set Signal to 0 if StopLoss is hit
                    trade_in_progress = False
            # Check if the trade is a Sell signal
            elif current_trade_signal == -1:
                if data['High'].iloc[i] > data['StopLoss'].iloc[i]:
                    data.at[i, 'Signal'] = 0  # Set Signal to 0 if StopLoss is hit
                    trade_in_progress = False
        # Start tracking a new trade
        if data['Signal'].iloc[i] != 0:
            trade_in_progress = True
            current_trade_signal = data['Signal'].iloc[i]

    return filter_low_earners(data)

def filter_low_earners(data, threshold=0.3):
    """
    Filter out low earners by number of pips.

    :param data: DataFrame with trading signals and price data.
    :param threshold: The threshold for the percentage of low earners to be filtered (default: 0.7).
    :return: Data with low earners filtered out (Signal = 0).
    """
    data = data.copy()  # Create a copy of the original data to avoid modifying it directly

    # Calculate the number of pips earned for each trade
    data['PipsEarned'] = data['Signal'] * (data['TakeProfit'] - data['Entry'])

    # Calculate the threshold for low earners based on the specified percentage
    total_trades = len(data[data['Signal'] != 0])
    num_trades_to_filter = int(total_trades * threshold)

    # Find the 'num_trades_to_filter' lowest-earning trades and set their 'Signal' to 0
    lowest_earning_trades = data.nsmallest(num_trades_to_filter, 'PipsEarned', keep='all')
    for index, row in lowest_earning_trades.iterrows():
        data.loc[index, 'Signal'] = 0

    return data


def simulate_earnings(data, leverage=1000, lot_size=1):
    """
    Simulate earnings in USD based on trades executed with specified leverage and lot size.

    :param data: DataFrame with PipsEarned (Pips earned per trade).
    :param leverage: Leverage used for trading (e.g., 1000).
    :param lot_size: Lot size used for trading (e.g., 1).
    :return: Data with simulated earnings in USD.
    """
    data = data.copy()  # Create a copy of the original data to avoid modifying it directly

    for i in tqdm(range(len(data)), desc="Simulating Earnings..."):
        if data['Signal'].iloc[i] == 0:
            # Set earnings to 0 for trades with Signal equal to 0
            data.at[i, 'EarningsUSD'] = 0
        else:
            # Calculate the earnings in USD based on PipsEarned, leverage, and lot size
            earnings_usd = (data['PipsEarned'].iloc[i] / 10000) * leverage * lot_size
            data.at[i, 'EarningsUSD'] = earnings_usd

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