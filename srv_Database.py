#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: i2cy(i2cy@outlook.com)
# Filename: srv_database
# Created on: 2021/4/28

from api.market_db import *
from i2cylib.network.I2TCP_protocol.I2TCP_server import *
from i2cylib.crypto.iccode import *
from i2cylib.utils.stdout import *
from i2cylib.utils.logger import *
from i2cylib.utils.path.path_fixer import *
import time
import json


CONFIG = "configs/database_srv.json"

DATABASE = None
ECHO = Echo()
LOGGER = None
MODLOGGER = None

class ModLogger:

    def __init__(self, logger, echo):
        self.logger = logger
        self.echo = echo

    def DEBUG(self, msg):
        ret = self.logger.DEBUG(msg)
        self.echo(ret)

        return ret

    def INFO(self, msg):
        ret = self.logger.INFO(msg)
        self.echo(ret)

        return ret

    def WARNING(self, msg):
        ret = self.logger.WARNING(msg)
        self.echo(ret)

        return ret

    def ERROR(self, msg):
        ret = self.logger.ERROR(msg)
        self.echo(ret)

        return ret

    def CRITICAL(self, msg):
        ret = self.logger.CRITICAL(msg)
        self.echo(ret)

        return ret

def main():
    global LOGGER, MODLOGGER
    head = "main"
    ECHO.print("[{}] [INFO] [{}] initializing environment".format(time.strftime("%y-%m-%d %H:%M:%S"), head))
    ECHO.buttom_print("initializing environment...")
    path_fixer(CONFIG)
    with open(CONFIG, "r") as f:
        conf = json.load(f)
        f.close()
    log_file = conf["log_filename"]
    log_level = conf["log_level"]
    path_fixer(log_file)
    LOGGER = logger(filename=log_file, echo=False, level=log_level)
    LOGGER.INFO("[{}] initializing".format(head))

    MODLOGGER = ModLogger(logger=LOGGER, echo=ECHO.print)

    MODLOGGER.DEBUG("[{}] building database connection...".format(head))
    ECHO.buttom_print("initializing database connection...")
    db_api = MarketDB()
    MODLOGGER.INFO("[{}] successfully built connection with \"{}\"".format(head, db_api.db_file))

    MODLOGGER.DEBUG("[{}] starting server...".format(head))
    port = conf["server_port"]
    key = conf["dyn_key"]
    server = I2TCPserver(key=key, port=port, logger=MODLOGGER)
    server.start()


if __name__ == '__main__':
    main()