# -*- coding = utf-8 -*-
# @Time: 2025/3/28 17:06
# @Author: Zhihang Yi
# @File: okxData.py
# @Software: PyCharm

import okxDataAPI
import numpy as np

def funding_rate(tick):
    """
    get the SWAP funding rate of a tick
    NOTE: only support a limited number of ticks

    :param tick: str, 'BTC-USDT', 'ETH-USDT', ...
    :return: float, funding rate of the tick
    """
    public_data_api = okxDataAPI.public_data_api()
    info = public_data_api.get_funding_rate(instId=tick+'-SWAP')

    return float(info['data'][0]['fundingRate'])

def premium(tick):
    """
    get the SWAP premium of a tick

    :param tick: str, 'BTC-USDT', 'ETH-USDT', ...
    :return:
    """
    public_data_api = okxDataAPI.public_data_api()
    info = public_data_api.get_funding_rate(instId=tick+'-SWAP')

    return float(info['data'][0]['premium'])

def prices(tick):
    """
    get the close prices of a tick during a recent period (limit=60)

    :param tick: str, 'BTC-USDT', 'ETH-USDT', ...
    :return:
    """
    market_data_api = okxDataAPI.market_data_api()
    info = market_data_api.get_candlesticks(instId=tick, bar='1m', limit='60')['data']
    info = np.array(info, dtype=np.float32)
    close_prices = info[:, 4]

    return close_prices
