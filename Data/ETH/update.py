# -*- coding = utf-8 -*-
# @Time: 2025/4/14 10:22
# @Author: Zhihang Yi
# @File: update.py
# @Software: PyCharm

import yfinance as yf
import Data
import pandas as pd

def update():
    """
    get ETH-USD within a defined time period and interval,
    preprocess the data,
    divide the whole dataset into training set and validation set
    and then store them as 'train.csv' and 'eth_eval.csv'

    :return: None
    """
    eth = yf.download('ETH-USD', start=Data.start_str, end=Data.end_str, interval=Data.interval_str)

    preprocess(eth)

    total_len = len(eth)

    val_len = int(total_len / Data.proportion)
    train_len = int(total_len - val_len)

    eth_train = eth.iloc[0:train_len]
    eth_val = eth.iloc[train_len:]

    eth_train.to_csv('train.csv')
    eth_val.to_csv('val.csv')


def preprocess(eth):
    """
    add 'Signal' column

    :return: None
    """
    signal = eth.shift(-1).loc[:, 'Close'] > eth.loc[:, 'Close']

    eth.loc[:, 'Signal'] = signal.astype(int)


if __name__ == '__main__':
    update()



