import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data_fetch import fetch_crypto_data, ping_api

st.title("Cryptocurrency Dashboard")

# Uji koneksi API dengan endpoint ping
ping_result = ping_api()
st.write("Ping API Result:", ping_result)

# Ambil data pasar cryptocurrency
df = fetch_crypto_data(per_page=10)
st.subheader("Data Pasar Cryptocurrency")
st.dataframe(df[['name', 'current_price', 'market_cap', 'total_volume']])

# Visualisasi: Harga Cryptocurrency
st.subheader("Grafik Harga Cryptocurrency")
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(df['name'], df['current_price'], color='skyblue')
ax.set_xlabel('Cryptocurrency')
ax.set_ylabel('Harga (USD)')
ax.set_title('Harga Terkini Cryptocurrency')
plt.xticks(rotation=45)
st.pyplot(fig)

st.markdown("Data diambil dari [CoinGecko](https://www.coingecko.com/).")