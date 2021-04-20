from huobi.client.market import MarketClient
from huobi.utils import *

market_client = MarketClient(url="https://api.huobi.li")

while True:
    list_obj = market_client.get_history_trade("btcusdt", 200)
    LogInfo.output_list(list_obj)
