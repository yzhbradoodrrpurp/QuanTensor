# -*- coding = utf-8 -*-
# @Time: 2025/4/13 14:22
# @Author: Zhihang Yi
# @File: update.py
# @Software: PyCharm

import yfinance as yf
import Data
import pandas as pd
from datetime import datetime

def update():
    """
    get BTC-USD within a defined time period and interval,
    preprocess the data,
    divide the whole dataset into training set and validation set
    and then store them as 'train.csv' and 'ticker_eval.csv'

    :return: None
    """
    ticker = yf.download('BTC-USD', start=Data.start_str, end=Data.end_str, interval=Data.interval_str)

    preprocess(ticker)

    total_len = len(ticker)

    val_len = int(total_len / Data.proportion)
    train_len = int(total_len - val_len)

    ticker_train = ticker.iloc[0:train_len]
    ticker_val = ticker.iloc[train_len:]

    ticker_train.to_csv('train.csv')
    ticker_val.to_csv('val.csv')

def add_signal(ticker):
    extent = ticker.loc[:, 'Close'].diff() / ticker.loc[:, 'Close']
    ticker.loc[:, 'Extent'] = extent

    ticker.loc[:, 'Signal'] = 0
    ticker.loc[ticker.loc[:, 'Extent'] >= Data.extent, 'Signal'] = 1
    ticker.loc[ticker.loc[:, 'Extent'] <= -1 * Data.extent, 'Signal'] = 2

    ticker.drop(columns=['Extent'], inplace=True)

def add_sma8(ticker):
    sma8 = ticker.loc[:, 'Close'].rolling(window=8).mean()
    sma8.fillna(ticker.iloc[:8, :].loc[:, 'Close'], inplace=True)
    ticker.loc[:, 'SMA8'] = sma8

def add_sma24(ticker):
    sma24 = ticker.loc[:, 'Close'].rolling(window=24).mean()
    sma24.fillna(ticker.iloc[:24, :].loc[:, 'Close'], inplace=True)
    ticker.loc[:, 'SMA24'] = sma24

def add_sma48(ticker):
    sma48 = ticker.loc[:, 'Close'].rolling(window=48).mean()
    sma48.fillna(ticker.iloc[:48, :].loc[:, 'Close'], inplace=True)
    ticker.loc[:, 'SMA48'] = sma48

def add_macd(ticker):
    ema9 = ticker.loc[:, 'Close'].ewm(span=9, adjust=False).mean()
    ema12 = ticker.loc[:, 'Close'].ewm(span=12, adjust=False).mean()
    ema26 = ticker.loc[:, 'Close'].ewm(span=26, adjust=False).mean()

    macd = ema12 - ema26
    signal = ema9

    ticker.loc[:, 'MACD'] = macd

def add_rsi(ticker):
    diff = ticker.loc[:, 'Close'].diff()
    diff.fillna(0, inplace=True)

    gain = diff.where(diff > 0, 0)
    loss = diff.where(diff < 0, 0)

    gain_avg = gain.rolling(window=14).mean()
    loss_avg = loss.rolling(window=14).mean()

    gain_avg.fillna(gain_avg.iloc[14:28].mean(), inplace=True)
    loss_avg.fillna(loss_avg.iloc[14:28].mean(), inplace=True)

    rs = gain_avg / loss_avg
    rsi = 100 - (100 / (1 + rs))
    ticker.loc[:, 'RSI'] = rsi

def parse_time(ticker):
    datetime_str = ticker.index
    datetime_obj = pd.to_datetime(datetime_str)

    ticker.loc[:, 'Year'] = datetime_obj.year
    ticker.loc[:, 'Month'] = datetime_obj.month
    ticker.loc[:, 'Day'] = datetime_obj.day
    ticker.loc[:, 'Hour'] = datetime_obj.hour
    ticker.loc[:, 'Weekday'] = datetime_obj.weekday

def delete_volume(ticker):
    ticker.drop(columns=['Volume'], inplace=True)

def preprocess(ticker):
    """

    :return: None
    """
    delete_volume(ticker)

    add_sma8(ticker)
    add_sma24(ticker)
    add_sma48(ticker)

    add_macd(ticker)
    add_rsi(ticker)

    parse_time(ticker)

    add_signal(ticker)
    

if __name__ == '__main__':
    update()
