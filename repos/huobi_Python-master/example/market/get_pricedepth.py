
from huobi.client.market import MarketClient
from huobi.utils import *


market_client = MarketClient(url="https://api.huobi.li")
symbol = "btcusdt"
depth = market_client.get_pricedepth(symbol, DepthStep.STEP0)
LogInfo.output("---- Top {size} bids ----".format(size=len(depth.bids)))
i = 0
print([[entry.price, entry.amount] for entry in depth.bids])
for entry in depth.bids:
    i = i + 1
    LogInfo.output(str(i) + ": price: " + str(entry.price) + ", amount: " + str(entry.amount))

LogInfo.output("---- Top {size} asks ----".format(size=len(depth.asks)))

i = 0
for entry in depth.asks:
    i = i + 1
    LogInfo.output(str(i) + ": price: " + str(entry.price) + ", amount: " + str(entry.amount))