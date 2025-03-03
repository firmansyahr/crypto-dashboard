import requests
import pandas as pd
import fetch_crypto_data
import ping_api

# Fungsi untuk mengambil data pasar cryptocurrency
def fetch_crypto_data(vs_currency='usd', per_page=10, page=1):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': vs_currency,
        'order': 'market_cap_desc',
        'per_page': per_page,
        'page': page,
        'sparkline': 'false',
        'x_cg_demo_api_key': 'CG-DZxcYDnVPqT3L2gqeP4PaTuN'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        return df
    else:
        print("Error fetching data:", response.status_code)
        return pd.DataFrame()

# Fungsi untuk menguji koneksi API
def ping_api():
    url = "https://api.coingecko.com/api/v3/ping"
    params = {
        'x_cg_demo_api_key': 'CG-DZxcYDnVPqT3L2gqeP4PaTuN'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code}
