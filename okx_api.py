import requests
import numpy as np

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CryptoSimBot/1.0)"
}

def get_spot_assets():
    """
    Fetches all spot trading instruments from OKX.

    Returns:
        list of str: Sorted list of instrument IDs like 'BTC-USDT'.
                     Returns ['BTC-USDT'] if API call fails.
    """
    url = "https://www.okx.com/api/v5/public/instruments"
    params = {"instType": "SPOT"}
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        assets = sorted(set(item["instId"] for item in data))
        return assets
    except requests.RequestException as e:
        print(f"Failed to fetch spot assets: {e}")
        return ["BTC-USDT"]


def fetch_and_compute_volatility(instId="BTC-USDT", minutes=100):
    """
    Fetches recent 1-minute candles and calculates annualized volatility.

    Args:
        instId (str): Trading instrument ID, e.g., 'BTC-USDT'.
        minutes (int): Number of 1-minute bars to fetch (max 100).

    Returns:
        float or None: Annualized volatility in percentage, rounded to 4 decimals.
                       Returns None if data fetch or calculation fails.
    """
    url = "https://www.okx.com/api/v5/market/candles"
    params = {
        "instId": instId,
        "bar": "1m",
        "limit": minutes
    }
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
        data = response.json().get("data", [])
        if len(data) < 2:
            return None

        # Extract close prices (index 4 in each candle)
        prices = [float(candle[4]) for candle in data]

        # Calculate log returns
        log_returns = np.diff(np.log(prices))

        # Volatility: standard deviation of log returns
        # Annualize assuming 365 days * 24 hours * 60 minutes = 525600 minutes per year
        minutes_per_year = 525600
        vol = np.std(log_returns) * np.sqrt(minutes_per_year) * 100  # percentage

        return round(vol, 4)
    except requests.RequestException as e:
        print(f"Failed to fetch candles or compute volatility: {e}")
        return None
    except Exception as e:
        print(f"Error during volatility calculation: {e}")
        return None
