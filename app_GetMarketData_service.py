#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: i2cy(i2cy@outlook.com)
# Filename: app_GetMarketData_service
# Created on: 2021/4/16

import sys
import time
import json
import threading

from i2cylib.utils.path import path_fixer
from i2cylib.utils.stdout.echo import *
from huobi.client.market import MarketClient
from huobi.constant import *
from huobi.utils import *
from api.market_db import *

MARKET_CONFIG = "../configs/market.json"
HUOBI_CONFIG = "../configs/huobi.json"

GMD_THREADS = {}

ECHO = None
STATICS = {""}
DATABASE = None

class Updater:

    def __init__(self, market_config=MARKET_CONFIG,
                 huobi_config=HUOBI_CONFIG, watchdog_threshold=10,
                 db_api = None):
        path_fixer(market_config)
        path_fixer(huobi_config)

        with open(market_config) as conf:
            config = json.load(conf)
        try:
            self.monitoring = config["monitoring_markets"]
        except Exception as err:
            raise KeyError("failed to load market config, {}".format(err))

        with open(huobi_config) as conf:
            config = json.load(conf)
        try:
            self.url = config["api_url"]
        except Exception as err:
            raise KeyError("failed to load market config, {}".format(err))

        self.live = False
        self.food = {}
        self.watchdog_threshold = watchdog_threshold
        self.watchdog_int_flag = {}
        self.db_api = db_api

        self.market_buff = {"timestamp": 0,
                            "kline_excahngecount": 0,
                            "kline_open": 0,
                            "kline_close": 0,
                            "kline_low": 0,
                            "kline_high": 0,
                            "kline_amount": 0,
                            "kline_volume": 0,
                            "buy_count": 0,
                            "buy_amount": 0,
                            "sell_count": 0,
                            "sell_amount": 0,
                            "depth_buy_0": None,
                            "depth_sell_0": None,
                            "depth_buy_5": None,
                            "depth_sell_5": None
                            }

    def __safety_check__(self):
        if self.db_api is None:
            raise Exception("database api has not connected yet")

    def __watchdog_int__(self):
        ECHO.print("[watchdog] warning: watchdog timeout, interrupting...")

    def __watchdog_thread__(self, index):
        self.watchdog_int_flag.update({index: False})
        while self.live:
            if self.food <= 0:
                if not self.watchdog_int_flag[index]:
                    self.__watchdog_int__(index)
                self.watchdog_int_flag[index] = True
            else:
                self.watchdog_int_flag[index] = False
            time.sleep(0.1)
            self.food -= 0.1

    def __updater_thread__(self, trade_name):
        tick = 0
        while self.live:
            self.watchdog_block(trade_name)
            if tick >= 5:
                huobi_market = MarketClient(url=self.url)
                kline = huobi_market.get_candlestick(trade_name,
                                                     CandlestickInterval.MIN1,
                                                     2)


            tick += 1

    def watchdog_block(self, index):
        while self.watchdog_int_flag[index]:
            time.sleep(0.01)

    def watchdog_feed(self, index):
        self.food.update({index: self.watchdog_threshold})

    def bind_dbapi(self, db_api):
        self.db_api = db_api

    def start(self):
        self.live = True
        for name in self.monitoring:
            watchdog = threading.Thread(target=self.__watchdog_thread__,
                                        args=(name,))
            watchdog.start()

    def stop(self):
        self.live = False


def init():
    global ECHO, DATABASE
    header = "[init]"
    ECHO = Echo()
    ECHO.print("initializing...")
    try:
        DATABASE = MarketDB()
    except Exception as err:
        ECHO.print("{} {}, exiting".format(header, err))
        sys.exit(1)
    ECHO.print("{} market database fully connected".format(header))

    ECHO.print("{} initialized".format(header))


def main():
    header = "[main]"
    ECHO.print("{} ")


if __name__ == '__main__':
    init()
    main()
else:
    init()