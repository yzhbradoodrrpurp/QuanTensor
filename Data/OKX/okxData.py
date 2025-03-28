# -*- coding = utf-8 -*-
# @Time: 2025/3/28 17:06
# @Author: Zhihang Yi
# @File: okxData.py
# @Software: PyCharm

import okxDataAPI

def funding_rate(tick):
    """
    get the SWAP funding rate of a tick
    NOTE: only support a limited number of ticks

    :param tick: str, 'BTC-USDT', 'ETH-USDT', ...
    :return: float, funding rate of the tick
    """
    public_data_api = okxDataAPI.public_data_api()
    info = public_data_api.get_funding_rate(instId=tick+'SWAP')

    return float(info['data'][0]['fundingRate'])


