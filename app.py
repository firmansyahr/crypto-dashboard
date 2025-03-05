import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pmdarima import auto_arima

# Import fungsi dari data_fetch
from data_fetch import (
    fetch_coins_list,
    fetch_simple_price,
    fetch_coin_markets,
    fetch_market_chart,
)

##################################
# Fungsi ARIMA dengan opsi log, musiman, dan drift
##################################
def predict_arima(prices, forecast_period=7, use_log=True, seasonal=True, m=7):
    # Transformasi data dengan log (selalu true)
    prices_transformed = np.log(prices)

    # Membangun model ARIMA/SARIMA dengan tambahan trend (drift)
    model = auto_arima(
        prices_transformed,
        start_p=1, start_q=1,
        max_p=5, max_q=5,
        seasonal=seasonal,
        m=m if seasonal else 1,
        trend='c',  # Mengasumsikan adanya konstanta (drift) jika d=1
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    forecast_transformed = model.predict(n_periods=forecast_period)

    # Kembalikan ke skala asli dengan eksponensial
    forecast = np.exp(forecast_transformed)

    model_summary = model.summary().as_text()
    return forecast, model_summary

def generate_recommendation(current_price, predicted_price):
    if predicted_price > current_price * 1.02:
        return "Rekomendasi: Beli (harga diprediksi naik signifikan)"
    elif predicted_price < current_price * 0.98:
        return "Rekomendasi: Jual (harga diprediksi turun signifikan)"
    else:
        return "Rekomendasi: Tahan (perubahan harga tidak signifikan)"

##################################
# Tampilan Dashboard dengan Streamlit
##################################
st.title("Analytics Cryptocurrency Using (ARIMA/SARIMA)")

# 1. Pengaturan: Pilihan Coin
st.sidebar.header("Pengaturan")
allowed_coins = [
    "bitcoin", "ethereum", "tether", "ripple", "binancecoin", "solana",
    "usd-coin", "cardano", "dogecoin", "tron", "pi", "hedera-hashgraph",
    "chainlink", "leo-token", "stellar", "avalanche-2", "sui", "litecoin",
    "shiba-inu", "pepe", "trump", "ton-crystal", "hype", "monero"
]

coins_data = fetch_coins_list()
if coins_data:
    df_coins = pd.DataFrame(coins_data)
    df_coins_filtered = df_coins[df_coins["id"].isin(allowed_coins)]
    coin_options = df_coins_filtered["id"].tolist() if not df_coins_filtered.empty else allowed_coins
else:
    st.error("Gagal memuat daftar coins. Pastikan koneksi API berjalan.")
    coin_options = allowed_coins

selected_coin = st.sidebar.selectbox("Pilih Coin:", coin_options, index=0)
st.write(f"Coin yang dipilih: {selected_coin.capitalize()}")

# Opsi tambahan tidak ditampilkan di sidebar karena log, seasonal, dan resampling harian selalu aktif
use_log_transform = True
use_seasonal = True
seasonal_period = 7
resample_daily = True

# 3. Harga Saat Ini (Simple Price)
st.header(f"Harga Saat Ini - {selected_coin.capitalize()}")
price_data = fetch_simple_price(selected_coin, "usd")
if price_data and selected_coin in price_data:
    current_price = price_data[selected_coin]["usd"]
    st.metric(label=f"Harga {selected_coin.capitalize()} (USD)", value=current_price)
else:
    st.warning("Gagal mengambil data harga sederhana.")

# 4. Data Pasar (Coin Markets)
st.header("Data Pasar Cryptocurrency")
df_markets = fetch_coin_markets(per_page=200)
if not df_markets.empty:
    df_selected = df_markets[df_markets["id"] == selected_coin]
    if df_selected.empty:
        st.warning(f"Tidak menemukan data pasar untuk {selected_coin}.")
    else:
        st.dataframe(df_selected[[
            "name", "current_price", "market_cap", "total_volume", "high_24h", "low_24h"
        ]])
else:
    st.error("Gagal mengambil data pasar.")

# 5. Data Historis & Visualisasi
st.header("Data Historis & Prediksi Harga")
days_option = st.selectbox("Pilih Rentang Hari Historis:", [7, 14, 30, 90, 180, 365], index=2)
market_chart_data = fetch_market_chart(selected_coin, days=days_option)
if market_chart_data and "prices" in market_chart_data:
    prices_list = market_chart_data["prices"]  # list of [timestamp, price]
    df_prices = pd.DataFrame(prices_list, columns=["timestamp", "price"])
    df_prices["timestamp"] = pd.to_datetime(df_prices["timestamp"], unit="ms")
    df_prices.set_index("timestamp", inplace=True)

    st.subheader(f"Grafik Harga {selected_coin.capitalize()} (USD) - {days_option} Hari Terakhir")
    st.line_chart(df_prices["price"])

    # --- Persiapan Data untuk Prediksi ---
    df_prices = df_prices.sort_index()
    if resample_daily:
        # Resample ke frekuensi harian
        df_prices_resampled = df_prices.resample('D').mean()
    else:
        df_prices_resampled = df_prices

    prices_series = df_prices_resampled["price"]

    st.subheader("Prediksi Harga dengan auto_arima")
    forecast_period = st.selectbox("Pilih jumlah hari prediksi ke depan:", [7, 14, 30, 90, 180], index=0)

    if len(prices_series) > 10:  # Pastikan data cukup untuk prediksi
        forecast, model_summary = predict_arima(
            prices_series,
            forecast_period=forecast_period,
            use_log=use_log_transform,
            seasonal=use_seasonal,
            m=seasonal_period
        )
        st.write(f"Prediksi harga untuk {forecast_period} hari ke depan:")
        st.write(forecast)

        with st.expander("Ringkasan Model ARIMA/SARIMA"):
            st.text(model_summary)

        # --- Visualisasi dengan Indeks Waktu yang Konsisten ---
        fig, ax = plt.subplots(figsize=(10, 6))
        if resample_daily:
            # Data sudah dalam frekuensi harian
            last_date = prices_series.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_period)
            ax.plot(prices_series.index, prices_series, label="Data Historis")
            ax.plot(forecast_dates, forecast, label="Prediksi ARIMA", color="red", marker="o")
            ax.set_xlabel("Tanggal")
        else:
            # Jika tidak disample, hitung frekuensi berdasarkan selisih antara dua titik terakhir
            last_date = prices_series.index[-1]
            if len(prices_series.index) >= 2:
                freq = prices_series.index[-1] - prices_series.index[-2]
            else:
                freq = pd.Timedelta(days=1)  # fallback jika hanya ada satu titik
            forecast_dates = [last_date + freq * (i + 1) for i in range(forecast_period)]
            ax.plot(prices_series.index, prices_series, label="Data Historis")
            ax.plot(forecast_dates, forecast, label="Prediksi ARIMA", color="red", marker="o")
            ax.set_xlabel("Tanggal")
        ax.set_ylabel("Harga (USD)")
        ax.set_title(f"Prediksi ARIMA untuk {selected_coin.capitalize()}")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

        # --- Rekomendasi Analitik ---
        current_price_chart = prices_series.iloc[-1]
        if len(forecast) > 0:
            if isinstance(forecast, pd.Series):
                first_pred = forecast.iloc[0]
            else:
                first_pred = forecast[0]
        else:
            first_pred = current_price_chart
        recommendation = generate_recommendation(current_price_chart, first_pred)
        st.write(f"Harga terakhir: {current_price_chart:.2f} USD")
        st.write(f"Prediksi harga periode pertama: {first_pred:.2f} USD")
        st.write(recommendation)
    else:
        st.info("Data historis terlalu sedikit untuk prediksi ARIMA.")
else:
    st.warning("Gagal mengambil data historis untuk coin yang dipilih.")

# 8. Daftar Semua Coins (Opsional)
st.header("Daftar Semua Coins")
coins_list = fetch_coins_list()
if coins_list:
    st.write(f"Total coins: {len(coins_list)}")
    st.dataframe(pd.DataFrame(coins_list).head(10))
else:
    st.error("Gagal mengambil daftar coins.")

st.markdown("Data diambil dari [CoinGecko](https://www.coingecko.com/).")
