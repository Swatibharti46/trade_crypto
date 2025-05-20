  
def fetch_price_and_volatility(symbol='BTC-USDT', interval='1m', window=100):
    import requests
    import numpy as np

    url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}&bar={interval}&limit={window}"
    response = requests.get(url)
    data = response.json()['data']
    closes = np.array([float(candle[4]) for candle in data][::-1])

    volatility = np.std(np.diff(np.log(closes)))
    price = closes[-1]
    return price, volatility


