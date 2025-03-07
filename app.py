import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import io

# Import fungsi untuk mengambil data dari CoinGecko
from data_fetch import (
    fetch_coins_list,
    fetch_simple_price,
    fetch_coin_markets,
    fetch_market_chart,
)

##################################
# Fungsi Prediksi dengan ARIMA/SARIMA
##################################
def predict_arima(prices, forecast_period=7, use_log=True, seasonal=True, m=7):
    # Lakukan transformasi log (selalu aktif)
    prices_transformed = np.log(prices)
    
    # Bangun model SARIMA dengan komponen non-musiman dan musiman
    model = auto_arima(
        prices_transformed,
        start_p=1, start_q=1,
        max_p=5, max_q=5,
        seasonal=seasonal,
        m=m if seasonal else 1,
        trend='c',  # Menambahkan drift (konstanta)
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

##################################
# Fungsi Prediksi dengan Exponential Smoothing
##################################
def predict_expsmoothing(prices, forecast_period=7, seasonal_periods=7):
    model = ExponentialSmoothing(prices, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
    model_fit = model.fit(optimized=True)
    forecast = model_fit.forecast(forecast_period)
    return forecast, model_fit.summary()

##################################
# Fungsi untuk Membuat Sequences untuk LSTM
##################################
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

##################################
# Fungsi untuk Membangun Model LSTM
##################################
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

##################################
# Fungsi Prediksi dengan LSTM
##################################
def predict_lstm(prices, forecast_period=7, n_steps=10, epochs=50):
    # Ubah data ke array dan scaling
    data = np.array(prices).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Buat sequence untuk pelatihan
    X, y = create_sequences(scaled_data, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    model = build_lstm_model((n_steps, 1))
    model.fit(X, y, epochs=epochs, verbose=0)
    
    # Inisialisasi forecast_input dengan n_steps data terakhir
    forecast_input = scaled_data[-n_steps:].reshape(1, n_steps, 1)
    forecast_scaled = []
    
    for _ in range(forecast_period):
        pred = model.predict(forecast_input, verbose=0)
        forecast_scaled.append(pred[0, 0])
        # Update forecast_input: hapus data pertama dan tambahkan pred baru
        forecast_input = np.concatenate((forecast_input[:, 1:, :], pred.reshape(1, 1, 1)), axis=1)
    
    forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
    
    summary_io = io.StringIO()
    model.summary(print_fn=lambda x: summary_io.write(x + "\n"))
    lstm_summary = summary_io.getvalue()
    
    return forecast, lstm_summary

##################################
# Fungsi Rekomendasi Preskriptif
##################################
def generate_recommendation(current_price, last_forecast, threshold=0.02):
    """
    Jika perbedaan relatif kurang dari threshold, rekomendasi Tahan.
    Jika nilai prediksi periode terakhir > harga saat ini: Beli.
    Jika nilai prediksi periode terakhir < harga saat ini: Jual.
    """
    if abs(last_forecast - current_price) / current_price < threshold:
        return "Rekomendasi: Tahan (perubahan harga tidak signifikan)"
    elif last_forecast > current_price:
        return "Rekomendasi: Beli (harga diprediksi naik)"
    else:
        return "Rekomendasi: Jual (harga diprediksi turun)"

##################################
# Tampilan Dashboard dengan Streamlit
##################################
st.title("Analytics Cryptocurrency: ARIMA, Exponential Smoothing & LSTM")

# 1. Pilihan Coin (Sidebar)
st.sidebar.header("Pengaturan")
allowed_coins = [
    "bitcoin", "ethereum", "tether", "ripple", "binancecoin", "solana",
    "usd-coin", "cardano", "dogecoin", "tron", "pi", "hedera-hashgraph",
    "chainlink", "leo-token", "stellar", "avalanche-2", "sui", "litecoin",
    "shiba-inu", "pepe", "trump", "ton-crystal", "hype", "monero"
]
coins_data = fetch_coins_list()
if coins_data is not None:
    df_coins = pd.DataFrame(coins_data)
    df_coins_filtered = df_coins[df_coins["id"].isin(allowed_coins)]
    coin_options = df_coins_filtered["id"].tolist() if not df_coins_filtered.empty else allowed_coins
else:
    st.error("Gagal memuat daftar coins. Pastikan koneksi API berjalan.")
    coin_options = allowed_coins

selected_coin = st.sidebar.selectbox("Pilih Coin:", coin_options, index=0)
st.write(f"Coin yang dipilih: {selected_coin.capitalize()}")

# Opsi forecasting sudah ditetapkan secara default:
use_log_transform = True
use_seasonal = True
seasonal_period = 7
resample_daily = True

# 2. Harga Saat Ini (Simple Price)
st.header(f"Harga Saat Ini - {selected_coin.capitalize()}")
price_data = fetch_simple_price(selected_coin, "usd")
if price_data and selected_coin in price_data:
    current_price = price_data[selected_coin]["usd"]
    st.metric(label=f"Harga {selected_coin.capitalize()} (USD)", value=current_price)
else:
    st.warning("Gagal mengambil data harga sederhana.")

# 3. Data Pasar (Coin Markets)
st.header("Data Pasar Cryptocurrency")
df_markets = fetch_coin_markets(per_page=200)
if df_markets is not None and not df_markets.empty:
    df_selected = df_markets[df_markets["id"] == selected_coin]
    if df_selected.empty:
        st.warning(f"Tidak menemukan data pasar untuk {selected_coin}.")
    else:
        st.dataframe(df_selected[[
            "name", "current_price", "market_cap", "total_volume", "high_24h", "low_24h"
        ]])
else:
    st.error("Gagal mengambil data pasar.")

# 4. Data Historis & Visualisasi
st.header("Data Historis & Prediksi Harga")
days_option = st.selectbox("Pilih Rentang Hari Historis:", [7, 14, 30, 90, 180, 365], index=2)
market_chart_data = fetch_market_chart(selected_coin, days=days_option)
if market_chart_data and "prices" in market_chart_data:
    prices_list = market_chart_data["prices"]
    df_prices = pd.DataFrame(prices_list, columns=["timestamp", "price"])
    df_prices["timestamp"] = pd.to_datetime(df_prices["timestamp"], unit="ms")
    df_prices.set_index("timestamp", inplace=True)
    
    st.subheader(f"Grafik Harga {selected_coin.capitalize()} (USD) - {days_option} Hari Terakhir")
    st.line_chart(df_prices["price"])
    
    # Resample ke harian
    df_prices = df_prices.sort_index()
    if resample_daily:
        df_prices_resampled = df_prices.resample('D').mean()
    else:
        df_prices_resampled = df_prices
    prices_series = df_prices_resampled["price"]
    
    forecast_period = st.selectbox("Pilih jumlah hari prediksi ke depan:", [7, 14, 30, 90, 180], index=0)
    
    if len(prices_series) > 10:
        # Prediksi dengan ARIMA
        forecast_arima, model_summary_arima = predict_arima(
            prices_series,
            forecast_period=forecast_period,
            use_log=use_log_transform,
            seasonal=use_seasonal,
            m=seasonal_period
        )
        st.subheader("Prediksi dengan ARIMA")
        st.write(f"Prediksi untuk {forecast_period} hari ke depan:")
        st.write(forecast_arima)
        with st.expander("Ringkasan Model ARIMA/SARIMA"):
            st.text(model_summary_arima)
        
        # Prediksi dengan Exponential Smoothing
        forecast_es, model_summary_es = predict_expsmoothing(
            prices_series,
            forecast_period=forecast_period,
            seasonal_periods=seasonal_period
        )
        st.subheader("Prediksi dengan Exponential Smoothing")
        st.write(f"Prediksi untuk {forecast_period} hari ke depan:")
        st.write(forecast_es)
        with st.expander("Ringkasan Model Exponential Smoothing"):
            st.text(model_summary_es)
        
        # Prediksi dengan LSTM
        forecast_lstm, lstm_summary = predict_lstm(
            prices_series,
            forecast_period=forecast_period,
            n_steps=10,
            epochs=50
        )
        st.subheader("Prediksi dengan LSTM")
        st.write(f"Prediksi untuk {forecast_period} hari ke depan:")
        st.write(forecast_lstm)
        with st.expander("Ringkasan Model LSTM"):
            st.text(lstm_summary)
        
        # Visualisasi Prediksi (menggunakan ARIMA sebagai contoh)
        fig, ax = plt.subplots(figsize=(10,6))
        last_date = prices_series.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_period)
        ax.plot(prices_series.index, prices_series, label="Data Historis")
        ax.plot(forecast_dates, forecast_arima, label="Prediksi ARIMA", color="red", marker="o")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Harga (USD)")
        ax.set_title(f"Prediksi ARIMA untuk {selected_coin.capitalize()}")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        
        fig2, ax2 = plt.subplots(figsize=(10,6))
        ax2.plot(prices_series.index, prices_series, label="Data Historis")
        ax2.plot(forecast_dates, forecast_es, label="Prediksi Exponential Smoothing", color="green", marker="o")
        ax2.set_xlabel("Tanggal")
        ax2.set_ylabel("Harga (USD)")
        ax2.set_title(f"Prediksi Exponential Smoothing untuk {selected_coin.capitalize()}")
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)
        
        fig3, ax3 = plt.subplots(figsize=(10,6))
        ax3.plot(prices_series.index, prices_series, label="Data Historis")
        ax3.plot(forecast_dates, forecast_lstm, label="Prediksi LSTM", color="orange", marker="o")
        ax3.set_xlabel("Tanggal")
        ax3.set_ylabel("Harga (USD)")
        ax3.set_title(f"Prediksi LSTM untuk {selected_coin.capitalize()}")
        ax3.legend()
        plt.tight_layout()
        st.pyplot(fig3)
        
        # --- Rekomendasi Preskriptif ---
        last_forecast_value = forecast_arima[-1]
        recommendation = generate_recommendation(current_price, last_forecast_value)
        st.write(f"Harga saat ini: {current_price:.2f} USD")
        st.write(f"Nilai prediksi periode terakhir (ARIMA): {last_forecast_value:.2f} USD")
        st.write(recommendation)
    else:
        st.info("Data historis terlalu sedikit untuk prediksi.")
else:
    st.warning("Gagal mengambil data historis untuk coin yang dipilih.")


st.markdown("Data diambil dari [CoinGecko](https://www.coingecko.com/).")
