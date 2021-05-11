#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: i2cy(i2cy@outlook.com)
# Filename: index_dataset
# Created on: 2021/5/11


from rnn.eth_predict import DatasetBase, SAMPLE_SIZE, SAMPLE_TIME_MS, DATASET_INDEX
from api.market_db import *
from i2cylib.utils.stdout.echo import *


def main():
    ECHO = Echo()
    ECHO.buttom_print("NOW: initializing database")
    ECHO.print("[main] initializing database, it may takes a while...")
    db = MarketDB()
    ECHO.print("[main] [database] initialized")
    ECHO.print("[main] sample_time = {} ms".format(SAMPLE_TIME_MS))
    ECHO.print("[main] sample_size = {} samples".format(SAMPLE_SIZE))
    ECHO.print("[main] initializing dataset, it may takes longer time...")
    ECHO.buttom_print("NOW: initializing dataset")
    dset = DatasetBase(db, sample_time_ms=SAMPLE_TIME_MS, set_size=SAMPLE_SIZE, index_json=DATASET_INDEX, echo=ECHO)
    dset.init_dataset()
    ECHO.buttom_print("NOW: dumping index data to {}".format(DATASET_INDEX))
    ECHO.print("[main] [dataset] initialized")
    dset.dump_index()
    ECHO.print("[main] [dataset] index data has been dumped into json file")


if __name__ == '__main__':
    main()
