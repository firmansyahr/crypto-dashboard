import requests
import pandas as pd

# Konfigurasi dasar
BASE_URL = "https://api.coingecko.com/api/v3"
API_KEY = "CG-DZxcYDnVPqT3L2gqeP4PaTuN"  # API key demo (opsional untuk beberapa endpoint)

# 1. Ping API
def ping_api():
    """
    Mengirim request ke endpoint /ping untuk mengecek status API.
    Mengembalikan JSON status jika berhasil, atau dict error jika gagal.
    """
    url = f"{BASE_URL}/ping"
    params = {'x_cg_demo_api_key': API_KEY}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error pinging API:", response.status_code)
        return {"error": response.status_code}

# 2. Simple Price
def fetch_simple_price(ids, vs_currencies):
    """
    Mengambil harga sederhana untuk coin yang diberikan.
    
    Parameter:
      - ids: string, misalnya "bitcoin,ethereum"
      - vs_currencies: string, misalnya "usd,eur"
    
    Mengembalikan data harga dalam format JSON atau None jika gagal.
    """
    url = f"{BASE_URL}/simple/price"
    params = {
        'ids': ids,
        'vs_currencies': vs_currencies,
        'x_cg_demo_api_key': API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error fetching simple price:", response.status_code)
        return None

# 3. Coin Markets
def fetch_coin_markets(vs_currency='usd', per_page=10, page=1):
    """
    Mengambil data pasar untuk koin-koin.
    Data mencakup harga terkini, market cap, volume, dan data sparkline (historis ringkas).
    
    Mengembalikan DataFrame dengan data pasar, atau DataFrame kosong jika gagal.
    """
    url = f"{BASE_URL}/coins/markets"
    params = {
        'vs_currency': vs_currency,
        'order': 'market_cap_desc',
        'per_page': per_page,
        'page': page,
        'sparkline': 'true',  # Aktifkan sparkline untuk data historis ringkas
        'x_cg_demo_api_key': API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return pd.DataFrame(data)
    else:
        print("Error fetching coin markets:", response.status_code)
        return pd.DataFrame()

# 4. Market Chart Data (historical data)
def fetch_market_chart(coin_id, vs_currency='usd', days=30):
    """
    Mengambil data historis market chart untuk coin tertentu.
    
    Parameter:
      - coin_id: ID koin (misalnya "bitcoin")
      - vs_currency: mata uang (misalnya "usd")
      - days: jumlah hari data historis (misalnya 30 atau 'max')
    
    Mengembalikan data dalam format JSON atau None jika gagal.
    """
    url = f"{BASE_URL}/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': vs_currency,
        'days': days,
        'x_cg_demo_api_key': API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error fetching market chart:", response.status_code)
        return None

# 5. Historical Data on a Specific Date
def fetch_history(coin_id, date, localization='false'):
    """
    Mengambil snapshot data historis untuk coin tertentu pada tanggal spesifik.
    
    Format tanggal: 'dd-mm-yyyy'
    Mengembalikan data dalam format JSON atau None jika gagal.
    """
    url = f"{BASE_URL}/coins/{coin_id}/history"
    params = {
        'date': date,
        'localization': localization,
        'x_cg_demo_api_key': API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error fetching history data:", response.status_code)
        return None

# 6. Contract Market Chart (untuk token berbasis kontrak)
def fetch_contract_market_chart(coin_id, contract_address, vs_currency='usd', days=30):
    """
    Mengambil data historis market chart untuk token berdasarkan contract address.
    
    Parameter:
      - coin_id: ID koin (misalnya "ethereum")
      - contract_address: alamat kontrak token (misalnya token ERC-20)
      - vs_currency: mata uang (misalnya "usd")
      - days: jumlah hari data historis (misalnya 30)
    
    Mengembalikan data dalam format JSON atau None jika gagal.
    """
    url = f"{BASE_URL}/coins/{coin_id}/contract/{contract_address}/market_chart"
    params = {
        'vs_currency': vs_currency,
        'days': days,
        'x_cg_demo_api_key': API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error fetching contract market chart:", response.status_code)
        return None

# 7. Daftar Semua Coins
def fetch_coins_list():
    """
    Mengambil daftar semua coins yang tersedia di CoinGecko.
    
    Mengembalikan data dalam format JSON atau None jika gagal.
    """
    url = f"{BASE_URL}/coins/list"
    params = {'x_cg_demo_api_key': API_KEY}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error fetching coins list:", response.status_code)
        return None
