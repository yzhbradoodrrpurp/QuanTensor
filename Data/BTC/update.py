# -*- coding = utf-8 -*-
# @Time: 2025/4/13 14:22
# @Author: Zhihang Yi
# @File: update.py
# @Software: PyCharm

import yfinance as yf
import Data
import pandas as pd
from datetime import datetime
from datasets import load_dataset
import shutil

def update():
    """
    get BTC-USD within a defined time period and interval,
    preprocess the data,
    divide the whole dataset into training set and validation set
    and then store them as 'train.csv' and 'ticker_eval.csv'

    :return: None
    """
    ticker, source = get_data()
    preprocess(ticker, source)
    save(ticker, source)

def get_data():
    source = input('Type in the source of data:\nA. Yahoo Finace\tB. Hugging Face\tC. Local directory\n')

    if source == 'A':
        ticker = yf.download('BTC-USD', start=Data.start_str, end=Data.end_str, interval=Data.interval_str)
    elif source == 'B':
        info = load_dataset('WinkingFace/CryptoLM-Bitcoin-BTC-USDT')
        ticker = info['train']
        ticker.set_format('pandas')
        ticker = ticker[:]
    elif source == 'C':
        path = input('Type in the path to the .csv file: ')
        ticker = pd.read_csv(path)
    else:
        raise ValueError(f'Invalid option: {source}')

    return ticker, source

def add_signal(ticker):

    task = input('Specify the task:\nA. Classification\tB. Prediction\n')

    if task == 'A':
        extent = ticker.loc[:, 'close'].diff() / ticker.loc[:, 'close']
        ticker.loc[:, 'Extent'] = extent

        ticker.loc[:, 'buy_sell'] = 0
        ticker.loc[ticker.loc[:, 'Extent'] >= Data.extent, 'buy_sell'] = 1
        ticker.loc[ticker.loc[:, 'Extent'] <= -1 * Data.extent, 'buy_sell'] = 2

        ticker.drop(columns=['Extent'], inplace=True)
    elif task == 'B':
        next_close_price = ticker.loc[:, 'close'].shift(-1)
        next_close_price.fillna(ticker.loc[:, 'close'], inplace=True)
        ticker.loc[:, 'next_close_price'] = next_close_price
    else:
        raise ValueError(f'Invalid option: {task}')

def add_sma8(ticker):
    sma8 = ticker.loc[:, 'Close'].rolling(window=8).mean()
    sma8.fillna(ticker.iloc[:8, :].loc[:, 'Close'], inplace=True)
    ticker.loc[:, 'SMA8'] = sma8


def add_sma24(ticker):
    sma24 = ticker.loc[:, 'Close'].rolling(window=24).mean()
    sma24.fillna(ticker.iloc[:24, :].loc[:, 'Close'], inplace=True)
    ticker.loc[:, 'SMA24'] = sma24


def add_sma72(ticker):
    sma72 = ticker.loc[:, 'Close'].rolling(window=72).mean()
    sma72.fillna(ticker.iloc[:72, :].loc[:, 'Close'], inplace=True)
    ticker.loc[:, 'SMA72'] = sma72


def add_sma168(ticker):
    sma168 = ticker.loc[:, 'Close'].rolling(window=168).mean()
    sma168.fillna(ticker.iloc[:168, :].loc[:, 'Close'], inplace=True)
    ticker.loc[:, 'SMA168'] = sma168


def add_ema8(ticker):
    ema8 = ticker.loc[:, 'Close'].ewm(span=8).mean()
    ticker.loc[:, 'EMA8'] = ema8


def add_ema24(ticker):
    ema24 = ticker.loc[:, 'Close'].ewm(span=24).mean()
    ticker.loc[:, 'EMA24'] = ema24


def add_ema72(ticker):
    ema72 = ticker.loc[:, 'Close'].ewm(span=72).mean()
    ticker.loc[:, 'EMA72'] = ema72


def add_ema168(ticker):
    ema168 = ticker.loc[:, 'Close'].ewm(span=168).mean()
    ticker.loc[:, 'EMA168'] = ema168


def add_macd(ticker):
    ema9 = ticker.loc[:, 'Close'].ewm(span=9, adjust=False).mean()
    ema12 = ticker.loc[:, 'Close'].ewm(span=12, adjust=False).mean()
    ema26 = ticker.loc[:, 'Close'].ewm(span=26, adjust=False).mean()

    macd = ema12 - ema26
    signal = ema9

    ticker.loc[:, 'MACD'] = macd
    ticker.loc[:, 'MACDSignal'] = signal


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


def add_atr(ticker):
    high_low = ticker.loc[:, 'High'] - ticker.loc[:, 'Low']
    high_close = ticker.loc[:, 'High'] - ticker.loc[:, 'Close']
    low_close = ticker.loc[:, 'Low'] - ticker.loc[:, 'Close']

    true_range = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = true_range.max(axis=1)

    atr = true_range.rolling(window=14).mean()
    atr.fillna(true_range.iloc[:14], inplace=True)

    ticker.loc[:, 'ATR'] = atr


def add_so(ticker):
    low14 = ticker.loc[:, 'Low'].rolling(window=14).min()
    low14.fillna(ticker.iloc[:14, :].loc[:, 'Low'], inplace=True)

    high14 = ticker.loc[:, 'High'].rolling(window=14).max()
    high14.fillna(ticker.iloc[:14, :].loc[:, 'High'], inplace=True)

    so = (ticker.loc[:, 'Close'] - low14) / (high14 - low14) * 100
    ticker.loc[:, 'SO'] = so


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


def preprocess(ticker, source='A'):
    """

    :return: None
    """
    if source == 'A':
        delete_volume(ticker)

        add_sma8(ticker)
        add_sma24(ticker)
        add_sma72(ticker)
        add_sma168(ticker)

        add_ema8(ticker)
        add_ema24(ticker)
        add_ema72(ticker)
        add_ema168(ticker)

        add_macd(ticker)
        add_rsi(ticker)
        add_atr(ticker)
        add_so(ticker)

        parse_time(ticker)

        add_signal(ticker)
    elif source == 'B':

        ticker.loc[:, 'timestamp'] = pd.to_datetime(ticker.loc[:, 'timestamp'])

        ticker.loc[:, 'Year'] = ticker.loc[:, 'timestamp'].dt.year
        ticker.loc[:, 'Month'] = ticker.loc[:, 'timestamp'].dt.month
        ticker.loc[:, 'Day'] = ticker.loc[:, 'timestamp'].dt.day
        ticker.loc[:, 'Hour'] = ticker.loc[:, 'timestamp'].dt.hour
        ticker.loc[:, 'minute'] = ticker.loc[:, 'timestamp'].dt.minute
        ticker.loc[:, 'second'] = ticker.loc[:, 'timestamp'].dt.second
        ticker.loc[:, 'Weekday'] = ticker.loc[:, 'timestamp'].dt.weekday

        ticker.drop(columns=['timestamp', 'month'], inplace=True)

        ticker.loc[:, 'open'].astype('float')
        ticker.loc[:, 'volume'].astype('float')
    elif source == 'C':
        add_signal(ticker)
    else:
        raise ValueError(f'Invalid option: {source}')


def save(ticker, source='A'):

    total_len = len(ticker)

    val_len = int(total_len / Data.proportion)
    train_len = int(total_len - val_len)

    ticker_train = ticker.iloc[0:train_len]
    ticker_val = ticker.iloc[train_len:]

    ticker_train.to_csv('train.csv', index=False)
    ticker_val.to_csv('val.csv', index=False)


if __name__ == '__main__':
    update()
