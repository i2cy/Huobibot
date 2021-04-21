from huobi.client.market import MarketClient
from huobi.utils import *

market_client = MarketClient(url="https://api.huobi.li")

list_obj = market_client.get_history_trade("dogeusdt", 2000)
LogInfo.output_list(list_obj)
print(list_obj[0].trade_id)
print(list_obj[0].direction, type(list_obj[0].direction))