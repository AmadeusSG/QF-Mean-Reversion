######################################
##  Implementation of the 4-Factor	##
##    Model for Overnight Returns  	##
##									##
##  		 Version 2				##
######################################

import sys
import os
import datetime as dt
import math
import csv
import statistics
import numpy as np
from sklearn import linear_model
import pandas as pd

##################################
## Variable Settings            ##
##################################

# Regression Parameters
file_limit = 10000  # Maximum number of files to read
min_quotes = 20  # Only ticker with greater than this will be used
volatility_days = min_quotes - 1  # Days used to compute volatility
sampling_count = 2000  # Select x stock files by liquidity for regression each day.

# Backtesting Parameters
investment = 10000000  # Initial investment
portfolio_size = 50  # Maximum number of stocks in the portfolio
fee_value = 0.0000218  # Transaction fee based on value sold
fee_quantity = 0.000119  # Transaction fee based on quantity sold
fee_min = 1  # Minimum transaction fee per transaction (applied on selling only)
# risk_free_rate = 0.05       # Any unused funds will gain risk-free rate. (Not implemented)
# capital_gain_tax = 0        # Tax rate applied on profits (Not implemented)
spread = 0.01  # Spread expressed as absolute value
allow_short = True  # Shortselling stocks is allowed
# margin_requirement = 1      # Margin requirement for shortselling expressed as % (Not implemented)
divisible_stocks = False  # Allow shares to be infinitely divisible

# Other variables
quotesdir = "quotes_s"  # Quotes directory

##################################
## Service Variables            ##
##################################

_cwd = os.path.dirname(os.path.realpath(__file__))  # Current working directory
_files = []  # Store name of quotes files
_data = [] #{}  # Store parsed quote files (grouped by ticker)
_daily_quotes = {}  # Store processed quote files (grouped by date)
_coef = {}  # Store daily factor coefficients (grouped by date)
_reference_gain = {}  # Store reference index gain (simple average over return for each day)
_daily_coeffs = {}
_daily_stats = {}

##################################
## Scan Ticker                  ##
##################################

# Scan the _cwd folder and record all csv files inside into _files.

for file in os.listdir(_cwd + "/" + quotesdir + "/"):
    if file.endswith(".csv"):
        _files.append(file)
    if (len(_files) >= file_limit):
        break
print("%d files found." % (len(_files)))

##################################
## Load and Process Data to use ##
##################################

# Load the files listed in _files into _data.
# Generate a new key for each stock.
# Store stock data as a list of tuple in ascending order of date.
# Column names: Ticker, Date, Open, High, Low, Close, Volume, Adj_Close

quotes_count = 0
counter = 0
discarded = []

def df_sum(dataframe):
    npmatrix = dataframe.as_matrix()
    nplist = []
    for i in range(1, len(npmatrix)-volatility_days+1):
        nplist.append(np.sum(npmatrix[i:i+volatility_days+1]))
    return pd.DataFrame(nplist)

for file in _files:
    df = pd.read_csv(quotesdir + "/" + file)
    if df.shape[0] >= min_quotes:
        df["Ticker"] = file.strip(".csv")
        # Overnight returns
        df.sort_values(by=['Date'], ascending=False)
        df['Ris'] = df['Close'].shift(-1)
        df['Ris'] = df.apply(lambda row: np.log(row['Open'] / row['Ris']), axis=1)
        # Intercept
        df['itc'] = 1
        # Size
        df['prc'] = df.apply(lambda row: np.log(row['Close']), axis=1).shift(-1)
        # Momentum
        df['mom'] = df.apply(lambda row: np.log(row['Close'] / row['Open']), axis=1).shift(-1)
        # Intraday Volatility
        df['hlv'] = df.apply(lambda row: ((row['High'] - row['Low']) / row['Close']) ** 2, axis=1)
        df['hlv'] = df_sum(df['hlv'])
        df['hlv'] = df.apply(lambda row: 0.5 * np.log(row['hlv'] / (volatility_days + 1)), axis=1)
        # Volume
        df['vol'] = df_sum(df['Volume'])
        df['vol'] = df.apply(lambda row: np.log(row['vol'] / (volatility_days + 1)), axis=1)
        # Liquidity
        df['liq'] = df.apply(lambda row: row['Close'] * row['Volume'], axis=1).shift(-1)
        # Dividend Payout
        df['div'] = df.apply(lambda row: row['Close'] / row['Adj Close'], axis=1).shift(-1)
        df['div'] = df.apply(lambda row: (row['Adj Close'] / row['Close'] * row['div']) - 1, axis=1)
        df.ix[(df['div'] > 0.15), 'div'] = 0
        df.ix[(df['div'] < 0.00001), 'div'] = 0
        # Append to data list
        _data.append(df)
    else: discarded.append(file.strip(".csv"))
    counter += 1
    print("Parsing and processing raw data ... %d / %d (%1.0f%%)" % (
        counter, len(_files), 100 * counter / len(_files)), end="\r")

sys.stdout.write("\n")
datacounter = len(_data)
print("Concaternating data...", end="\r")
_data = pd.concat(_data)
_data['Date'] = pd.to_datetime(_data['Date'].map(str), format='%Y-%m-%d')
_data = _data[_data['Date'] >= '2010-01-01']
print("Concaternation done!")

sys.stdout.write("\n")
print("%d files loaded into memory (%d quotes recorded)." % (datacounter, _data.shape[0]))
print("%d files discarded for having too few data: " % (len(discarded)), end="")
print(discarded)

##################################
## Normalise Factors by Date    ##
##################################

print(_data.Date.shape)
# hlv and vol will be normalized
newdata2 = _data.groupby(['Date'])['hlv'].mean()
newdata = _data.Date.unique()

# print(_data['Date'])
for date in newdata:
    df = _data[_data.Date == date]
# print(newdata)
# print(newdata.shape)