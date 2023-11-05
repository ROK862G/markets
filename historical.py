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
parser.add_argument('start', type=str, help='The start date of data (e.g., 2021-01-01)')
parser.add_argument('end', type=str, help='The end date of data extraction (e.g., 2021-12-31)')
args = parser.parse_args()

# Define the symbol and the path to the "corpus" folder
symbol = args.symbol.upper()  # Convert the symbol to uppercase
start = args.start 
end = args.end 

try:
    start_date = datetime(int(start.split('-')[0]), int(start.split('-')[1]), int(start.split('-')[2]))
    end_date = datetime(int(end.split('-')[0]), int(end.split('-')[1]), int(end.split('-')[2]))
    get_polygon_data(symbol, start_date, end_date)
except Exception as ex:
    print(ex)


data = get_historical()