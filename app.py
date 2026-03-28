import streamlit as st
import yfinance as yf
from google import genai
import pandas as pd

st.set_page_config(page_title="SPY Buddy PRO", page_icon="📈", layout="wide")
st.title("📈 SPY Buddy Pro (Live Edition)")

col_tf, col_ref = st.columns() 
with col_tf:
    timeframe = st.selectbox("Chart Timeframe", ["1 Day", "1 Hour", "15 Min", "5 Min", "1 Min"], index=0)
with col_ref:
    st.write("") 
    st.write("")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()

tf_map = {"1 Day": {"p": "1y", "i": "1d"}, "1 Hour": {"p": "730d", "i": "1h"}, "15 Min": {"p": "60d", "i": "15m"}, "5 Min": {"p": "60d", "i": "5m"}, "1 Min": {"p": "7d", "i": "1m"}}
period = tf_map[timeframe]["p"]
interval = tf_map[timeframe]["i"]

@st.cache_data(ttl=60) 
def get_market_data(p, i):
    return yf.Ticker("SPY").history(period=p, interval=i), yf.Ticker("^VIX").history(period="1d")
