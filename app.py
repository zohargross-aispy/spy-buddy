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

try:
    spy_hist, vix_hist = get_market_data(period, interval)
    spy_hist['SMA_20'] = spy_hist['Close'].rolling(window=20).mean()
    spy_hist['SMA_200'] = spy_hist['Close'].rolling(window=200).mean()
    
    delta = spy_hist['Close'].diff()
    up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
    rs = up.ewm(com=13, adjust=False).mean() / down.ewm(com=13, adjust=False).mean()
    spy_hist['RSI'] = 100 - (100 / (1 + rs))

    curr_p = round(spy_hist['Close'].iloc[-1], 2)
    open_p = round(spy_hist['Open'].iloc[-1], 2) 
    prev_close = round(spy_hist['Close'].iloc[-2], 2)
    vix_p = round(vix_hist['Close'].iloc[-1], 2)
    
    sma20_p = round(spy_hist['SMA_20'].iloc[-1], 2) if not pd.isna(spy_hist['SMA_20'].iloc[-1]) else "N/A"
    sma200_p = round(spy_hist['SMA_200'].iloc[-1], 2) if not pd.isna(spy_hist['SMA_200'].iloc[-1]) else "N/A"
    rsi_p = round(spy_hist['RSI'].iloc[-1], 2) if not pd.isna(spy_hist['RSI'].iloc[-1]) else 50.0
    vol_p = int(spy_hist['Volume'].iloc[-1])
    trend = "Bullish 🐂" if curr_p > open_p else "Bearish 🐻"

    st.subheader(f"Market Overview ({timeframe})")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("SPY Price", f"${curr_p}", f"{round(curr_p - prev_close, 2)}", delta_color="normal")
    col2.metric("RSI (14)", f"{rsi_p}", "Overbought" if rsi_p > 70 else "Oversold" if rsi_p < 30 else "Neutral", delta_color="off" if 30 <= rsi_p <= 70 else "inverse")
    col3.metric("VIX", f"{vix_p}")
    col4.metric("Volume", f"{vol_p:,}")

    st.markdown(f"**Trend:** {trend} &nbsp;|&nbsp; **20-SMA:** ${sma20_p} &nbsp;|&nbsp; **200-SMA:** ${sma200_p}")
    
    st.line_chart(spy_hist[['Close', 'SMA_20', 'SMA_200']].tail(90))
    st.bar_chart(spy_hist[['Volume']].tail(90))

    st.subheader("🤖 AI Analysis")
    if "GOOGLE_API_KEY" in st.secrets:
        if st.button("Generate Expert Analysis"):
            prompt = f"SPY at ${curr_p}, Open ${open_p} ({trend}). 20SMA: ${sma20_p}, 200SMA: ${sma200_p}. RSI: {rsi_p}. Vol: {vol_p}. VIX: {vix_p}. 3-sentence technical analysis for {timeframe} timeframe."
            with st.spinner("Analyzing..."):
                st.info(genai.Client(api_key=st.secrets["GOOGLE_API_KEY"]).models.generate_content(model='gemini-2.5-flash', contents=prompt).text)
    else:
        st.warning("⚠️ Add GOOGLE_API_KEY to secrets.")
except Exception as e:
    st.error(f"Error: {e}")
