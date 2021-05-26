#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: i2cy(i2cy@outlook.com)
# Filename: index_dataset
# Created on: 2021/5/11
import time

from rnn_eth_predict import DatasetBase, SAMPLE_SIZE, SAMPLE_TIME_MS, DATASET_INDEX
from api.market_db import *
from i2cylib.utils.stdout.echo import *


def main():
    ECHO = Echo()
    ECHO.buttom_print("NOW: initializing database")
    ECHO.print("[{}] [main] initializing database, it may takes a while...".format(
        time.strftime("%y-%m-%d %H:%M:%S")))
    db = MarketDB()
    ECHO.print("[{}] [main] [database] initialized".format(time.strftime("%y-%m-%d %H:%M:%S")))
    ECHO.print("[{}] [main] sample_time = {} ms".format(time.strftime("%y-%m-%d %H:%M:%S"),
                                                        SAMPLE_TIME_MS))
    ECHO.print("[{}] [main] sample_size = {} samples".format(time.strftime("%y-%m-%d %H:%M:%S"),
                                                             SAMPLE_SIZE))
    ECHO.print("[{}] [main] initializing dataset, it may takes longer time...".format(
        time.strftime("%y-%m-%d %H:%M:%S")))
    ECHO.buttom_print("NOW: initializing dataset")
    dset = DatasetBase(db, sample_time_ms=SAMPLE_TIME_MS, set_size=SAMPLE_SIZE,
                       index_json=DATASET_INDEX, echo=ECHO, use_index_json=False)
    dset.init_dataset()
    ECHO.buttom_print("NOW: dumping index data to {}".format(DATASET_INDEX))
    ECHO.print("[{}] [main] [dataset] initialized".format(time.strftime("%y-%m-%d %H:%M:%S")))
    dset.dump_index()
    ECHO.print("[{}] [main] [dataset] index data has been dumped into json file".format(
        time.strftime("%y-%m-%d %H:%M:%S")))


if __name__ == '__main__':
    main()
