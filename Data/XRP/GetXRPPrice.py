# -*- coding = utf-8 -*-
# @Time: 2025/4/13 14:47
# @Author: Zhihang Yi
# @File: GetXRPPrice.py
# @Software: PyCharm

import yfinance as yf
import Data

xrp = yf.download('XRP-USD', start=Data.start_str, end=Data.end_str, interval=Data.interval_str)
total_len = len(xrp)

val_len = int(total_len / Data.proportion)
train_len = int(total_len - val_len)

xrp_train = xrp.iloc[0:train_len]
xrp_val = xrp.iloc[train_len:]

xrp_train.to_csv('xrp_train.csv')
xrp_val.to_csv('xrp_val.csv')
