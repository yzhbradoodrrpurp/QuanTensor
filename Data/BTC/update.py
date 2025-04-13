# -*- coding = utf-8 -*-
# @Time: 2025/4/13 14:22
# @Author: Zhihang Yi
# @File: update.py
# @Software: PyCharm

import yfinance as yf
import Data
import pandas as pd

def update():
    """
    get BTC-USD within a defined time period and interval,
    preprocess the data,
    divide the whole dataset into training set and validation set
    and then store them as 'btc_train.csv' and 'btc_eval.csv'

    :return: None
    """
    btc = yf.download('BTC-USD', start=Data.start_str, end=Data.end_str, interval=Data.interval_str)

    preprocess(btc)

    total_len = len(btc)

    val_len = int(total_len / Data.proportion)
    train_len = int(total_len - val_len)

    btc_train = btc.iloc[0:train_len]
    btc_val = btc.iloc[train_len:]

    btc_train.to_csv('btc_train.csv')
    btc_val.to_csv('btc_val.csv')


def preprocess(btc):
    """
    add 'Signal' column

    :return: None
    """
    signal = btc.shift(-1).loc[:, 'Close'] > btc.loc[:, 'Close']

    btc.loc[:, 'Signal'] = signal.astype(int)


if __name__ == '__main__':
    update()



