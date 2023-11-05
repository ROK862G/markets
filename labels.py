import pandas as pd

symbol = input("Enter the currency pair (ex.. USDZAR): ").capitalize()

while True:
    entry_date = pd.to_datetime(input("Enter the entry date (YYYY-MM-DD HH:MM:SS): "))
    
    # Load the CSV file into a DataFrame
    file_path = f'corpus/{symbol}-{entry_date.date()}.csv'  # Replace with the path to your CSV file
    df = pd.read_csv(file_path, parse_dates=['Date'])

    # Input from the user
    order_type = int(input("Enter the order type (Buy=1, Sell=-1): "))
    entry_price = float(input("Enter the Entry level: "))
    take_profit = float(input("Enter the Take Profit level: "))
    stop_loss = float(input("Enter the Stop Loss level: "))

    # Find the closest Close price to the entry
    closest_idx = (df['Date'] - entry_date).abs().idxmin()

    # Add the Signal, Take Profit, and Stop Loss values to the closest row
    df.at[closest_idx, 'Signal'] = order_type
    df.at[closest_idx, 'Entry'] = entry_price
    df.at[closest_idx, 'TakeProfit'] = take_profit
    df.at[closest_idx, 'StopLoss'] = stop_loss

    # Save the updated DataFrame to the same CSV file
    df.to_csv(file_path, index=False)

    print(f"Signal information added to '{file_path}' for the closest entry.")
