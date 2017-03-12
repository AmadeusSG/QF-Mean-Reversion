import pandas as pd, numpy as np

def make_stock(stock_dict, csv, stock_name, timestamp_header='Timestamp', header=0, col_name=[]):
    """
    Changes CSV into desired Pandas format.
    If CSV has no header, specify header=None and
    timestamp_header to the column number.
    Also, add col_name (i.e. OHLC, exclude Timestamp) 
    if header=None. Index name will change to 'Timestamp'
    regardless.
    
    Parameters: (stock_dict, csv, stock_name, timestamp_header=
                 'Timestamp', header=True, col_name=[])
    """
    df = pd.read_csv(csv, index_col=timestamp_header, header=header)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index(ascending=False)
    
    if header==None:
        df.columns = col_name
        
    df.index = df.index.rename('Timestamp')
    
    stock_dict[stock_name] = df

# Market Beta
def B_0(df, label='mkt_beta', value=1): 
    """
    Adds the market beta (value=1) into a DataFrame.
    
    Parameters: (df, label='mkt_beta', value=1)
    """
    df[label] = value
    return df
    
itc = B_0

# Size
def B_1(df, close, label='size'):
    """
    Adds size using the closing price into a DataFrame.
    Specify the sliced data (not the label) of the closing price.
    
    Parameters: (df, close, label='size')
    """
    df[label] = np.log(close).shift(-1)
    return df

prc = B_1

# High Minus Low
def B_2(df, open, close, label='hml'):
    """
    Adds HML using the opening and
    closing prices into a DataFrame.
    Specify the sliced data (not the label)
    of the prices needed.
    
    Parameters: (df, open, close, label='hml')
    """
    df[label] = np.log(close/open).shift(-1)
    return df

mom = B_2

# Intraday Volatility
def B_3(df, high, low, close, label='int_vol', sma=21):
    """
    Adds intraday volatility using the high, low and
    closing prices into a DataFrame.
    Specify the sliced data (not the label)
    of the prices needed.
    Use 'sma' to change the mean parameter.
    
    Parameters: (df, high, low, close, label='int_vol', sma=21)
    """
    new_df = ((high-low)/close)**2
    new_df = new_df.rolling(sma).mean().shift(-sma)
    df[label] = np.log(new_df**0.5)
    return df
    
hlv = B_3

# Volume Moving Average
def B_4(df, vol, label='norm_vol', sma=21):
    """
    Adds volume SMA using the volume 
    into a DataFrame.
    Specify the sliced data (not the label)
    of the prices needed.
    Use 'sma' to change the mean parameter.
    
    Parameters: (df, vol, label='norm_vol', sma=21)
    """
    df[label] = np.log(vol.rolling(sma).mean().shift(-sma))
    return df

vol = B_4

# Overnight Returns
def ris(df, open, close, label='ris'):
    """
    Adds overnight returns using the opening and
    closing prices into a DataFrame.
    Specify the sliced data (not the label)
    of the prices needed.
    
    Parameters: (df, open, close, label='ris')
    """
    df[label] = np.log(open/close).shift(-1)
    return df

# Liquidity 
def liq(df, close, vol, label='liq'):
    """
    Adds liquidity using the volume and 
    closing price into a DataFrame.
    Specify the sliced data (not the label)
    of the prices needed.
    
    Parameters: (df, close, vol, label='liq')
    """
    df[label] = (close * vol).shift(-1)
    return df

# Intraday Return
def ir(df, open, close, label='ir'):
    """
    Adds intraday returns using the opening and
    closing prices into a DataFrame.
    Specify the sliced data (not the label)
    of the prices needed.
    
    Parameters: (df, open, close, label='ir')
    """
    df[label] = close/open - 1
    return df
