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
from huobi.client.generic import GenericClient
from huobi.constant import *
from huobi.utils import *
from api.market_db import *

MARKET_CONFIG = "configs/market.json"
HUOBI_CONFIG = "configs/huobi.json"

GMD_THREADS = {}

ECHO = None
DATABASE = None

class Updater:

    def __init__(self, market_config=MARKET_CONFIG,
                 huobi_config=HUOBI_CONFIG, watchdog_threshold=15,
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
        self.statics = {}

        for ele in self.monitoring:
            self.statics.update({ele.upper: {"price": 0}})

    def __safety_check__(self):
        if self.db_api is None:
            raise Exception("database api has not connected yet")

    def __watchdog_int__(self, index, timeout):
        timestamp = int(time.time()*1000)
        ECHO.print("[watchdog] [{}] [{}] warning: watchdog timeout ({} second(s)),"
                   " interrupting...".format(index, timestamp, timeout))
        self.watchdog_feed(index)

    def __watchdog_thread__(self, index):
        self.watchdog_int_flag.update({index: False})
        self.food.update({index: self.watchdog_threshold})
        while self.live:
            if self.food[index] <= 0:
                if not self.watchdog_int_flag[index]:
                    self.__watchdog_int__(index, self.watchdog_threshold - self.food[index])
                self.watchdog_int_flag[index] = True
            else:
                self.watchdog_int_flag[index] = False
            time.sleep(0.1)
            self.food[index] -= 0.1

    def __decode_trades__(self, trade, tradename, last_tradeid):
        trade_id = trade[0].trade_id
        timestamp = trade[0].ts
        buy_count = 0
        sell_count = 0
        buy_amount = 0
        sell_amount = 0
        buy_max = -1
        sell_max = -1
        buy_min = -1
        sell_min = -1
        if trade_id - last_tradeid > 2000 or last_tradeid - trade_id > 2000:
            if last_tradeid != 0:
                ECHO.print("[updater] [{}] [{}] warning: too many trade during this period (>2000),"
                           " trade id now: {}, the last recorded trade id: {}".format(tradename,
                                                                                      timestamp,
                                                                                      trade_id,
                                                                                      last_tradeid))
        if trade[0].direction == "buy":
            buy_count += 1
            buy_amount += trade[0].amount
            buy_max = trade[0].amount
            buy_min = trade[0].amount
        else:
            sell_count += 1
            sell_amount += trade[0].amount
            sell_max = trade[0].amount
            sell_min = trade[0].amount

        for ele in trade[1:]:
            if ele.trade_id == last_tradeid:
                break
            if ele.direction == "buy":
                buy_count += 1
                buy_amount += ele.amount
                if ele.amount > buy_max:
                    buy_max = ele.amount
                if ele.amount < buy_min:
                    buy_min = ele.amount
            else:
                sell_count += 1
                sell_amount += ele.amount
                if ele.amount > sell_max:
                    sell_max = ele.amount
                if ele.amount < sell_min:
                    sell_min = ele.amount
        return [buy_count, buy_amount, buy_max, buy_min,
                sell_count, sell_amount, sell_max, sell_min], trade_id


    def __updater_thread__(self, trade_name):
        tick = 0
        last_tradeid = 0
        huobi_market = None
        huobi_generic = None
        while self.live:
            if huobi_market is None:
                huobi_market = MarketClient(url=self.url)
            if huobi_generic is None:
                huobi_generic = GenericClient(url=self.url)
            t = time.time()
            timestamp = int(time.time() * 1000)
            try:
                timestamp = huobi_generic.get_exchange_timestamp()
                kline = huobi_market.get_candlestick(trade_name,
                                                     CandlestickInterval.MIN1,
                                                     2)
                depth0 = huobi_market.get_pricedepth(trade_name, DepthStep.STEP0)
                depth5 = huobi_market.get_pricedepth(trade_name, DepthStep.STEP5)
                depth0_buy = json.dumps([[entry.price, entry.amount] for entry in depth0.bids])
                depth0_sell = json.dumps([[entry.price, entry.amount] for entry in depth0.asks])
                depth5_buy = json.dumps([[entry.price, entry.amount] for entry in depth5.bids])
                depth5_sell = json.dumps([[entry.price, entry.amount] for entry in depth5.asks])
                trades = huobi_market.get_history_trade(trade_name, 2000)
                trades, last_tradeid = self.__decode_trades__(trades, trade_name, last_tradeid)

                database_name = trade_name.upper()
                self.db_api.update(database_name, [timestamp,
                                                kline[0].count,
                                                kline[0].open,
                                                kline[0].close,
                                                kline[0].low,
                                                kline[0].high,
                                                kline[0].amount,
                                                kline[0].vol,
                                                trades[0],
                                                trades[1],
                                                trades[2],
                                                trades[3],
                                                trades[4],
                                                trades[5],
                                                trades[6],
                                                trades[7],
                                                depth0_buy,
                                                depth0_sell,
                                                depth5_buy,
                                                depth5_sell])

                self.watchdog_feed(trade_name)

                # debug test
                ECHO.print("[updater] [{}] [{}] debug: update cost {} seconds".format(trade_name,
                                                                                   timestamp,
                                                                                   time.time()-t))
            except Exception as err:
                huobi_market = None
                ECHO.print("[updater] [{}] [{}] error: {}".format(trade_name,
                                                                  timestamp,
                                                                  err))
            self.watchdog_block(trade_name)

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
            updater = threading.Thread(target=self.__updater_thread__,
                                       args=(name,))
            watchdog = threading.Thread(target=self.__watchdog_thread__,
                                        args=(name,))
            watchdog.start()
            updater.start()

    def stop(self):
        for ele in self.monitoring:
            self.watchdog_feed(ele)
        self.live = False


def init():
    global ECHO, DATABASE
    header = "[init]"
    ECHO = Echo()
    ECHO.print("initializing...")
    try:
        DATABASE = MarketDB(echo=ECHO.print)
        DATABASE.start()
    except Exception as err:
        ECHO.print("{} {}, exiting".format(header, err))
        sys.exit(1)
    ECHO.print("{} market database fully connected".format(header))

    ECHO.print("{} initialized".format(header))


def main():
    header = "[main]"
    ECHO.print("{} starting updater service...".format(header))
    updater = Updater(db_api=DATABASE)
    updater.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ECHO.print("{} keyboard interrupt received, stopping...".format(header))
        updater.stop()
        DATABASE.close()
        sys.exit(0)


if __name__ == '__main__':
    init()
    main()
else:
    init()