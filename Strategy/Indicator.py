# -*- coding = utf-8 -*-
# @Time: 2025/3/28 18:57
# @Author: Zhihang Yi
# @File: Indicator.py
# @Software: PyCharm

import pandas as pd

class SMA21:

    def __init__(self, candlesticks):
        self.candlesticks = candlesticks

    def signal(self):
        candlesticks = pd.Series(self.candlesticks)

        current_price = candlesticks[-1]
        last_price = candlesticks[-2]

        # return recent 100 SMA21 values
        return candlesticks.iloc[::-1].rolling(window=21).mean().iloc[::-1][-100:]
