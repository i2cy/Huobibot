from huobi.client.market import MarketClient
from huobi.utils import *
import time

market_client = MarketClient(url="https://api.huobi.li")

while True:
    list_obj = market_client.get_market_trade(symbol="ethusdt")
    LogInfo.output_list(list_obj)












