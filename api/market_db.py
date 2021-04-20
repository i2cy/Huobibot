#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: i2cy(i2cy@outlook.com)
# Filename: market_db
# Created on: 2021/4/16

import i2cylib.database.sqlite as sql
from i2cylib.utils.path.path_fixer import *
import json

MARKET_CONFIG = "configs/market.json"

DB_GLOB = None

class MarketDB:

    def __init__(self, config=MARKET_CONFIG):
        path_fixer(config)
        with open(config) as conf:
            config = json.load(conf)
        try:
            self.db_file = config["database"]
            path_fixer(self.db_file)
            self.monitoring = config["monitoring_markets"]
            for i in range(len(self.monitoring)):
                self.monitoring[i] = self.monitoring[i].upper()
        except Exception as err:
            raise KeyError("failed to load market config, {}".format(err))
        self.db = sql.SqliteDB(self.db_file)
        self.db.connect()
        self.db.switch_autocommit()
        self.markets = []
        self.contracts = []
        self.all_tables = {}
        self.all_tablenames = []

        self.get_db_info()

        for ele in self.monitoring:
            if not ele in self.markets:
                self.create_market_db(ele)
            if not ele in self.contracts:
                self.create_contract_db(ele)

        self.get_db_info()

        self.offsets = {}

        for ele in self.all_tablenames:
            table = self.db.select_table(ele)
            self.all_tables.update({ele: table})
            self.offsets.update({ele: self.get_offset(table)})

    def get_offset(self, table_object):
        id = 0
        try:
            id = table_object[-1][0]
        except:
            pass
        return int(id)

    def get_db_info(self):
        info = self.db.list_all_tables()
        self.all_tablenames = info
        for ele in info:
            if ele.split("_")[0] == "MARKET":
                self.markets.append(ele.split("_")[1])
            elif ele.split("_")[0] == "CONTRACT":
                self.contracts.append(ele.split("_")[1])

        return {"markets": self.markets, "contracts": self.contracts}

    def create_market_db(self, name):
        table = sql.SqlTable("MARKET_{}".format(name))
        table.add_column("ID", sql.SqlDtype.INTEGER)
        table.add_column("TIMESTAMP", sql.SqlDtype.INTEGER)
        table.add_column("KLINE_EXCHANGECOUNT", sql.SqlDtype.INTEGER)
        table.add_column("KLINE_OPEN", sql.SqlDtype.REAL)
        table.add_column("KLINE_CLOSE", sql.SqlDtype.REAL)
        table.add_column("KLINE_LOW", sql.SqlDtype.REAL)
        table.add_column("KLINE_HIGH", sql.SqlDtype.REAL)
        table.add_column("KLINE_AMOUNT", sql.SqlDtype.REAL)
        table.add_column("KLINE_VOLUME", sql.SqlDtype.REAL)
        table.add_column("BUY_COUNT", sql.SqlDtype.INTEGER)
        table.add_column("BUY_AMOUNT", sql.SqlDtype.REAL)
        table.add_column("SELL_COUNT", sql.SqlDtype.INTEGER)
        table.add_column("SELL_AMOUNT", sql.SqlDtype.REAL)
        table.add_column("DEPTH_BUY0", sql.SqlDtype.BLOB)
        table.add_column("DEPTH_SELL0", sql.SqlDtype.BLOB)
        table.add_column("DEPTH_BUY5", sql.SqlDtype.BLOB)
        table.add_column("DEPTH_SELL5", sql.SqlDtype.BLOB)
        table.add_limit(0, sql.Sqlimit.PRIMARY_KEY)
        self.db.create_table(table)

    def create_contract_db(self, name):
        table = sql.SqlTable("CONTRACT_{}".format(name))
        table.add_column("ID", sql.SqlDtype.INTEGER)
        table.add_column("TIMESTAMP", sql.SqlDtype.INTEGER)
        table.add_column("KLINE_EXCHANGECOUNT", sql.SqlDtype.INTEGER)
        table.add_column("KLINE_OPEN", sql.SqlDtype.REAL)
        table.add_column("KLINE_CLOSE", sql.SqlDtype.REAL)
        table.add_column("KLINE_LOW", sql.SqlDtype.REAL)
        table.add_column("KLINE_HIGH", sql.SqlDtype.REAL)
        table.add_column("KLINE_AMOUNT", sql.SqlDtype.REAL)
        table.add_column("KLINE_VOLUME", sql.SqlDtype.REAL)
        table.add_column("BUY_COUNT", sql.SqlDtype.INTEGER)
        table.add_column("BUY_AMOUNT", sql.SqlDtype.REAL)
        table.add_column("SELL_COUNT", sql.SqlDtype.INTEGER)
        table.add_column("SELL_AMOUNT", sql.SqlDtype.REAL)
        table.add_column("DEPTH_BUY0", sql.SqlDtype.BLOB)
        table.add_column("DEPTH_SELL0", sql.SqlDtype.BLOB)
        table.add_column("DEPTH_BUY5", sql.SqlDtype.BLOB)
        table.add_column("DEPTH_SELL5", sql.SqlDtype.BLOB)
        table.add_limit(0, sql.Sqlimit.PRIMARY_KEY)
        self.db.create_table(table)

    def fetch(self, start_ms, stop_ms=None):
        start_ms = int(start_ms*1000)
        if stop_ms is None:
            stop_ms = start_ms + 1000
        else:
            stop_ms = int(stop_ms*1000)
        ret = {}
        for ele in self.monitoring:
            market_tab = self.all_tables["MARKET_{}".format(ele)]
            contract_tab = self.all_tables["CONTRACT_{}".format(ele)]
            ret.update({ele: {"market": market_tab.get((start_ms, stop_ms), primary_index_column="TIMESTAMP"),
                              "contract": contract_tab.get((start_ms, stop_ms), primary_index_column="TIMESTAMP")}})
        return ret

    def update(self, db_name, data):
        if isinstance(data, tuple):
            data = list(data)
        if len(data) < 16:
            offset = self.offsets[db_name] + 1
            data.insert(0, offset)
        else:
            offset = data[0]
        tab = self.all_tables[db_name]

        try:
            if offset > self.offsets[db_name]:
                tab.append(data)
                self.offsets[db_name] += 1
            else:
                tab.update(data, offset)

        except Exception as err:
            raise Exception("failed to update data in database, {}".format(err))

    def close(self):
        self.db.close()


def init():
    global DB_GLOB
    DB_GLOB = MarketDB()