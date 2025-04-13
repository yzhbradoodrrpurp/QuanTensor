# -*- coding = utf-8 -*-
# @Time: 2025/4/13 14:45
# @Author: Zhihang Yi
# @File: GetETHPrice.py
# @Software: PyCharm

import yfinance as yf
import Data

eth = yf.download('ETH-USD', start=Data.start_str, end=Data.end_str, interval=Data.interval_str)
total_len = len(eth)

val_len = int(total_len / Data.proportion)
train_len = int(total_len - val_len)

eth_train = eth.iloc[0:train_len]
eth_val = eth.iloc[train_len:]

eth_train.to_csv('eth_train.csv')
eth_val.to_csv('eth_val.csv')
