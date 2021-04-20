from huobi.client.market import MarketClient
from huobi.constant import *

market_client = MarketClient(url="https://api.huobi.li")

while True:
    obj = market_client.get_market_detail_merged("btcusdt")
    obj.print_object()



