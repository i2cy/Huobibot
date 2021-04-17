#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: i2cy(i2cy@outlook.com)
# Filename: app_GetMarketData_service
# Created on: 2021/4/16

import json
from i2cylib.utils.path import path_fixer

from huobi.client.market import MarketClient
from huobi.constant import *
from huobi.utils import *
from api.market_db import *

MARKET_CONFIG = "../configs/market.json"

GMD_THREADS = {}

class Updater:

    def __init__(self):
        with open(MARKET_CONFIG) as conf:
            config = json.load(conf)
        try:
            self.monitoring = config["monitoring_markets"]
        except Exception as err:
            raise KeyError("failed to load market config, {}".format(err))


def init():
    pass


def main():
    pass


if __name__ == '__main__':
    init()
    main()
else:
    init()