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


class HuobiGetData:

    def __init__(self, upper, marketclient, trade_name):
        self.upper = upper
        self.buffer = [0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0,
                       "", "", "", ""]
        self.clt = marketclient
        self.trade_name = trade_name
        self.last_tradeid = 0
        self.finished_flags = [False, False, False, False]
        self.exception = None

    def reset(self):
        self.buffer = [0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0,
                       "", "", "", ""]
        self.finished_flags = [False, False, False, False]
        self.exception = None

    def kline_thread(self):
        try:
            kline = self.clt.get_candlestick(self.trade_name,
                                             CandlestickInterval.MIN1,
                                             2)
            self.buffer[1] = kline[0].count
            self.buffer[2] = kline[0].open
            self.buffer[3] = kline[0].close
            self.buffer[4] = kline[0].low
            self.buffer[5] = kline[0].high
            self.buffer[6] = kline[0].amount
            self.buffer[7] = kline[0].vol
        except Exception as err:
            self.exception = err
        self.finished_flags[0] = True

    def depth0_thread(self):
        try:
            depth0 = self.clt.get_pricedepth(self.trade_name, DepthStep.STEP0)
            depth0_buy = json.dumps([[entry.price, entry.amount] for entry in depth0.bids])
            depth0_sell = json.dumps([[entry.price, entry.amount] for entry in depth0.asks])
            self.buffer[16] = depth0_buy
            self.buffer[17] = depth0_sell
        except Exception as err:
            self.exception = err
        self.finished_flags[1] = True

    def depth5_thread(self):
        try:
            depth5 = self.clt.get_pricedepth(self.trade_name, DepthStep.STEP5)
            depth5_buy = json.dumps([[entry.price, entry.amount] for entry in depth5.bids])
            depth5_sell = json.dumps([[entry.price, entry.amount] for entry in depth5.asks])
            self.buffer[18] = depth5_buy
            self.buffer[19] = depth5_sell
        except Exception as err:
            self.exception = err
        self.finished_flags[2] = True

    def trade_thread(self):
        try:
            trades = self.clt.get_history_trade(self.trade_name, 2000)
            trades, self.last_tradeid = self.upper.__decode_trades__(trades,
                                                                     self.trade_name,
                                                                     self.last_tradeid)
            self.buffer[8] = trades[0]
            self.buffer[9] = trades[1]
            self.buffer[10] = trades[2]
            self.buffer[11] = trades[3]
            self.buffer[12] = trades[4]
            self.buffer[13] = trades[5]
            self.buffer[14] = trades[6]
            self.buffer[15] = trades[7]
        except Exception as err:
            self.exception = err
        self.finished_flags[3] = True

    def get(self):
        t = time.time()
        threads = [self.kline_thread,
                   self.trade_thread,
                   self.depth0_thread,
                   self.depth5_thread]
        self.reset()
        self.buffer[0] = self.upper.get_timestamp()
        for ele in threads:
            threading.Thread(target=ele).start()
        finished = self.finished_flags[0] + self.finished_flags[1] \
                   + self.finished_flags[2] + self.finished_flags[3]
        while finished < 4:
            finished = self.finished_flags[0] + self.finished_flags[1] \
                       + self.finished_flags[2] + self.finished_flags[3]
            time.sleep(0.01)
        if not self.exception is None:
            raise self.exception

        return int((time.time() - t) * 1000), self.buffer


class Updater:

    def __init__(self, market_config=MARKET_CONFIG,
                 huobi_config=HUOBI_CONFIG, watchdog_threshold=10,
                 db_api=None):
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
            self.timeout = config["timeout"]
            self.url = config["api"]["url"]
            self.fallback_url = config["fallback_api"]["url"]
            if config["api"]["proxies"]["http"] == "" and config["api"]["proxies"]["https"] == "":
                self.proxies = None
            else:
                if config["api"]["proxies"]["http"] == "":
                    config["api"]["proxies"]["http"] = config["api"]["proxies"]["https"]
                if config["api"]["proxies"]["https"] == "":
                    config["api"]["proxies"]["https"] = config["api"]["proxies"]["http"]
                self.proxies = config["api"]["proxies"]
            if config["fallback_api"]["proxies"]["http"] == "" and config["fallback_api"]["proxies"]["https"] == "":
                self.fallback_proxies = None
            else:
                if config["fallback_api"]["proxies"]["http"] == "":
                    config["fallback_api"]["proxies"]["http"] = config["fallback_api"]["proxies"]["https"]
                if config["fallback_api"]["proxies"]["https"] == "":
                    config["fallback_api"]["proxies"]["https"] = config["fallback_api"]["proxies"]["http"]
                self.fallback_proxies = config["fallback_api"]["proxies"]
        except Exception as err:
            raise KeyError("failed to load huobi api config, {}".format(err))

        self.live = False
        self.food = {}
        self.watchdog_threshold = watchdog_threshold
        self.watchdog_int_flag = {}
        self.db_api = db_api
        self.statics = {}
        self.timestamp_offset = 0

        try:
            if self.proxies is None:
                gen_clt = GenericClient(url=self.url, timeout=self.timeout)
            else:
                gen_clt = GenericClient(url=self.url, timeout=self.timeout, proxies=self.proxies)
            for i in range(20):
                cloud_ts = gen_clt.get_exchange_timestamp()
                self.timestamp_offset -= (self.get_timestamp() - cloud_ts) * ((20 - i) / 20)
        except:
            if self.fallback_proxies is None:
                gen_clt = GenericClient(url=self.fallback_url, timeout=self.timeout)
            else:
                gen_clt = GenericClient(url=self.fallback_url, timeout=self.timeout, proxies=self.fallback_proxies)
                for i in range(20):
                    cloud_ts = gen_clt.get_exchange_timestamp()
                    self.timestamp_offset -= (self.get_timestamp() - cloud_ts) * ((20 - i) / 20)
        ECHO.print("[updater] [init] info: timestamp offset fixing: {}".format(self.timestamp_offset))
        cloud_ts = gen_clt.get_exchange_timestamp()
        fixed_ts = self.get_timestamp()
        ECHO.print("[updater] [init] debug: huobi cloud timestamp: {}, fixed timestamp: {}".format(
            cloud_ts, self.get_timestamp()
        ))

        for ele in self.monitoring:
            self.statics.update({ele.upper(): {"price": -1,
                                               "avg_cost_1min": 0.0,
                                               "ping": 0}})

        self.fallbacked = False

    def __safety_check__(self):
        if self.db_api is None:
            raise Exception("database api has not connected yet")

    def __fallback_thread__(self):
        if self.proxies is None:
            test_clt = MarketClient(url=self.url, timeout=5)
        else:
            test_clt = MarketClient(url=self.url, timeout=5, proxies=self.proxies)
        testing = True
        while testing:
            if not self.live:
                return
            try:
                test_clt.get_candlestick("btcusdt",
                                         CandlestickInterval.MIN1,
                                         2)
                testing = False
            except:
                time.sleep(0.1)
                continue
        ECHO.print("[watchdog] [fallbacker] [{}] the original api url {} has recovered".format(
            time.strftime("%y-%m-%d %H:%M:%S",
                          time.localtime(self.get_timestamp() / 1000)),
            self.url
        ))
        self.fallbacked = False

    def __watchdog_int__(self, index, timeout):
        ECHO.print("[watchdog] [{}] [{}] warning: watchdog timeout ({} second(s)),"
                   " interrupting...".format(index, time.strftime("%y-%m-%d %H:%M:%S",
                                                                  time.localtime(self.get_timestamp() / 1000)),
                                             timeout))
        if self.fallbacked:
            self.watchdog_feed(index)
            pass
        else:
            self.fallbacked = True
            self.watchdog_feed(index)
            ECHO.print("[watchdog] [{}] fallback using url: {}".format(time.strftime("%y-%m-%d %H:%M:%S",
                                                                                     time.localtime(
                                                                                         self.get_timestamp() / 1000)),
                                                                       self.fallback_url))
            threading.Thread(target=self.__fallback_thread__).start()

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
                                                                                      time.strftime("%y-%m-%d %H:%M:%S",
                                                                                        time.localtime(
                                                                                        self.get_timestamp() / 1000)),
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
                if ele.amount < buy_min or buy_min == -1:
                    buy_min = ele.amount
            else:
                sell_count += 1
                sell_amount += ele.amount
                if ele.amount > sell_max:
                    sell_max = ele.amount
                if ele.amount < sell_min or sell_min == -1:
                    sell_min = ele.amount
        return [buy_count, buy_amount, buy_max, buy_min,
                sell_count, sell_amount, sell_max, sell_min], trade_id

    def __updater_thread__(self, trade_name):
        huobi_market = None
        huobi_updater = None
        fallback_status = self.fallbacked
        while self.live:
            if fallback_status != self.fallbacked:
                fallback_status = self.fallbacked
                huobi_market = None
                huobi_updater = None
            if huobi_market is None:
                if self.fallbacked:
                    if self.fallback_proxies is None:
                        huobi_market = MarketClient(url=self.fallback_url,
                                                    timeout=self.timeout)
                    else:
                        huobi_market = MarketClient(url=self.fallback_url,
                                                    proxies=self.fallback_proxies,
                                                    timeout=self.timeout)
                else:
                    if self.proxies is None:
                        huobi_market = MarketClient(url=self.url,
                                                    timeout=self.timeout)
                    else:
                        huobi_market = MarketClient(url=self.url,
                                                    proxies=self.proxies,
                                                    timeout=self.timeout)
            if huobi_updater is None:
                huobi_updater = HuobiGetData(self, huobi_market, trade_name)
            t = time.time()
            timestamp = self.get_timestamp()
            try:
                ping, api_callback = huobi_updater.get()

                database_name = trade_name.upper()
                self.statics[database_name]["price"] = api_callback[3]
                self.db_api.update(database_name, api_callback)

                self.watchdog_feed(trade_name)

                # debug test
                if self.statics[database_name]["avg_cost_1min"] == 0:
                    self.statics[database_name]["avg_cost_1min"] = time.time() - t
                else:
                    self.statics[database_name]["avg_cost_1min"] = (self.statics[database_name]["avg_cost_1min"] * 59 + \
                                                                    (time.time() - t)) / 60
                self.statics[database_name]["ping"] = ping
                # ECHO.print("[updater] [{}] [{}] debug: update cost {} seconds".format(trade_name,
                #                                                                   timestamp,
                #                                                                   time.time()-t))
            except Exception as err:
                time.sleep(1)
                huobi_market = None
                huobi_updater = None
                ECHO.print("[updater] [{}] [{}] error: {}".format(trade_name,
                                                                  time.strftime("%y-%m-%d %H:%M:%S",
                                                                                time.localtime(
                                                                                    self.get_timestamp() / 1000)),
                                                                  err))
            self.watchdog_block(trade_name)

    def get_timestamp(self):
        return int(time.time() * 1000 + self.timestamp_offset)

    def watchdog_block(self, index):
        blocked = False
        while self.watchdog_int_flag[index]:
            if not blocked:
                blocked = True
                ECHO.print("[watchdog] [{}] [{}] interrupted".format(index, self.get_timestamp()))
            time.sleep(0.01)
        if blocked:
            ECHO.print("[watchdog] [{}] [{}] released".format(index, self.get_timestamp()))

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
    tick = 0
    cursor = 0
    try:
        while True:
            msg = "{}".format(time.strftime("%H:%M:%S"))
            if tick % 8 == 0:
                cursor += 1
                if cursor >= len(updater.monitoring):
                    cursor = 0
            msg += "|{}   price: {} USDT   ping: {}ms".format(
                updater.monitoring[cursor].upper(),
                updater.statics[updater.monitoring[cursor].upper()]["price"],
                updater.statics[updater.monitoring[cursor].upper()]["ping"])
            if tick > 60:
                msg += "   average cost: {:.2f}s".format(updater.statics[updater.monitoring[cursor].upper()]
                                                          ["avg_cost_1min"])
            ECHO.buttom_print(msg)
            time.sleep(1)
            tick += 1
    except KeyboardInterrupt:
        ECHO.print("{} keyboard interrupt received, stopping...".format(header))
        updater.stop()
        DATABASE.close()
        print("")
        sys.exit(0)


if __name__ == '__main__':
    init()
    main()
else:
    init()
