import streamlit as st
import yfinance as yf
from google import genai
import pandas as pd

# --- PAGE SETUP ---
st.set_page_config(page_title="SPY Buddy PRO", page_icon="📈", layout="wide")
st.title("📈 SPY Buddy Pro (Live Edition)")
st.markdown("Your AI-powered technical assistant for S&P 500 market trends.")

# --- TIMEFRAME & REFRESH ---
# FIXED: Added so Streamlit knows how wide to make the columns!
col_tf, col_ref = st.columns() 
with col_tf:
    timeframe = st.selectbox(
        "Select Chart Timeframe (Live Data)",
        ["1 Day", "1 Hour", "15 Min", "5 Min", "1 Min"],
        index=0
    )
with col_ref:
    st.write("") # Spacing to align button with dropdown
    st.write("")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear() # Clears cache to force a fresh download

# Map user selection to yfinance parameters
tf_map = {
    "1 Day": {"period": "1y", "interval": "1d"},
    "1 Hour": {"period": "730d", "interval": "1h"},
    "15 Min": {"period": "60d", "interval": "15m"},
    "5 Min": {"period": "60d", "interval": "5m"},
    "1 Min": {"period": "7d", "interval": "1m"}
}

period = tf_map[timeframe]["period"]
interval = tf_map[timeframe]["interval"]

# --- FETCH MARKET DATA ---
@st.cache_data(ttl=60) 
def get_market_data(p, i):
    spy_hist = yf.Ticker("SPY").history(period=p, interval=i)
    vix_hist = yf.Ticker("^VIX").history(period="1d")
    return spy_hist, vix_hist

try:
    spy_hist, vix_hist = get_market_data(period, interval)

    # --- TECHNICAL INDICATOR CALCULATIONS ---
    spy_hist['SMA_20'] = spy_hist['Close'].rolling(window=20).mean()
    spy_hist['SMA_200'] = spy_hist['Close'].rolling(window=200).mean()

    # RSI Calculation
    delta = spy_hist['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    spy_hist['RSI'] = 100 - (100 / (1 + rs))

    # --- CURRENT METRICS ---
    curr_p = round(spy_hist['Close'].iloc[-1], 2)
    open_p = round(spy_hist['Open'].iloc[-1], 2) 
    prev_close = round(spy_hist['Close'].iloc[-2], 2)
    vix_p = round(vix_hist['Close'].iloc[-1], 2)
    
    sma20_p = round(spy_hist['SMA_20'].iloc[-1], 2)
