# -*- coding = utf-8 -*-
# @Time: 2025/4/14 10:24
# @Author: Zhihang Yi
# @File: update.py
# @Software: PyCharm

import yfinance as yf
import Data
import pandas as pd

def update():
    """
    get XRP-USD within a defined time period and interval,
    preprocess the data,
    divide the whole dataset into training set and validation set
    and then store them as 'train.csv' and 'xrp_eval.csv'

    :return: None
    """
    xrp = yf.download('XRP-USD', start=Data.start_str, end=Data.end_str, interval=Data.interval_str)

    preprocess(xrp)

    total_len = len(xrp)

    val_len = int(total_len / Data.proportion)
    train_len = int(total_len - val_len)

    xrp_train = xrp.iloc[0:train_len]
    xrp_val = xrp.iloc[train_len:]

    xrp_train.to_csv('train.csv')
    xrp_val.to_csv('val.csv')


def preprocess(xrp):
    """
    add 'Signal' column

    :return: None
    """
    signal = xrp.shift(-1).loc[:, 'Close'] > xrp.loc[:, 'Close']

    xrp.loc[:, 'Signal'] = signal.astype(int)


if __name__ == '__main__':
    update()



