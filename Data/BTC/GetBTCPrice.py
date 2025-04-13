# -*- coding = utf-8 -*-
# @Time: 2025/4/13 14:22
# @Author: Zhihang Yi
# @File: GetBTCPrice.py
# @Software: PyCharm

import yfinance as yf
import Data

btc = yf.download('BTC-USD', start=Data.start_str, end=Data.end_str, interval=Data.interval_str)
total_len = len(btc)

val_len = int(total_len / Data.proportion)
train_len = int(total_len - val_len)

btc_train = btc.iloc[0:train_len]
btc_val = btc.iloc[train_len:]

btc_train.to_csv('btc_train.csv')
btc_val.to_csv('btc_val.csv')
