from huobi.client.market import MarketClient
import time


def callback(trade_req: 'TradeDetailReq'):
    print("---- trade_event:  ----")
    print(trade_req[1])
    print()



market_client = MarketClient(url="https://api.huobi.li")
while True:
    market_client.req_trade_detail("enjusdt", callback)



