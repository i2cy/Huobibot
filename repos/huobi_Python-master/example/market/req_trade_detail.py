from huobi.client.market import MarketClient



def callback(trade_req: 'TradeDetailReq'):
    print("---- trade_event:  ----")
    trade_req.print_object()
    print()



market_client = MarketClient(url="https://api.huobi.li")
while True:
    market_client.req_trade_detail("hbcusdt", callback)



