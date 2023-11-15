import alpaca_trade_api as tradeapi



def execute_market_order(symbol='EURUSD', trade_units=10000, take_profit=None, stop_loss=None, side='buy'):
    # Replace with your Alpaca API credentials
    api_key_id = 'XXX'
    api_secret_key = 'XXX'
    
    # symbol = symbol[:3] + '/' + symbol[3:]

    # Initialize the Alpaca API client
    api = tradeapi.REST(api_key_id, api_secret_key, base_url='https://paper-api.alpaca.markets')

    # Place a market order based on your prediction
    order_type = 'market'
    if side.lower() == 'buy':
        side = 'buy'
    else:
        side = 'sell'

    order_params = {
        'symbol': symbol,
        'qty': trade_units,
        'type': order_type,
        'time_in_force': 'gtc',  # Good 'til Cancelled
        'side': side,
    }

    if take_profit is not None:
        order_params['take_profit'] = {
            'limit_price': take_profit,
        }

    if stop_loss is not None:
        order_params['stop_loss'] = {
            'stop_price': stop_loss,
        }

    try:
        # Place the order
        order = api.submit_order(**order_params)

        # Print order details
        print(f"Order placed successfully - Order ID: {order.id}")
    except Exception as e:
        print(f"Error placing order: {e}")
