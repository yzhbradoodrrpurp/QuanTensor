# -*- coding = utf-8 -*-
# @Time: 2025/3/28 16:22
# @Author: Zhihang Yi
# @File: okxDataAPI.py
# @Software: PyCharm

"""
it's recommended that you store your
API-related information in your environment variables
"""

import os

import okx.MarketData
import okx.PublicData
import okx.Account


api_key = os.getenv('okxAPIKeyData')
api_secret = os.getenv('okxAPISecretData')
passphrase = os.getenv('okxPassphraseData')


def market_data_api():
    return okx.MarketData.MarketAPI(api_key, api_secret, passphrase)

def public_data_api():
    return okx.PublicData.PublicAPI(api_key, api_secret, passphrase)

def account_data_api():
    return okx.Account.AccountAPI(api_key, api_secret, passphrase)
