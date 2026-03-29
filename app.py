import streamlit as st
import yfinance as yf
from google import genai
import pandas as pd

st.set_page_config(page_title="SPY Buddy PRO", page_icon="📈", layout="wide")
st.title("📈 SPY Buddy Pro (Algo Edition)")

# FIX: st.columns needs a spec
col_tf, col_ref = st.columns(2)

with col_tf:
    timeframe = st.selectbox(
        "Chart Timeframe",
        ["1 Day", "1 Hour", "15 Min", "5 Min", "1 Min"],
        index=0
    )

with col_ref:
    st.write("")
    st.write("")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

tf_map = {
    "1 Day": {"p": "1y", "i": "1d"},
    "1 Hour": {"p": "730d", "i": "1h"},
    "15 Min": {"p": "60d", "i": "15m"},
    "5 Min": {"p": "60d", "i": "5m"},
    "1 Min": {"p": "7d", "i": "1m"},
}

period = tf_map[timeframe]["p"]
interval = tf_map[timeframe]["i"]


@st.cache_data(ttl=60)
def get_market_data(p, i):
    spy = yf.Ticker("SPY").history(period=p, interval=i)
    vix = yf.Ticker("^VIX").history(period="5d", interval="1d")
    return spy, vix


try:
    spy_hist, vix_hist = get_market_data(period, interval)

    if spy_hist.empty:
        st.error("No SPY data returned. Try refreshing or changing the timeframe.")
        st.stop()

    if vix_hist.empty:
        st.error("No VIX data returned.")
        st.stop()

    # Indicators
    spy_hist["SMA_20"] = spy_hist["Close"].rolling(window=20).mean()
    spy_hist["SMA_200"] = spy_hist["Close"].rolling(window=200).mean()

    delta = spy_hist["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    avg_gain = up.ewm(com=13, adjust=False).mean()
    avg_loss = down.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    spy_hist["RSI"] = 100 - (100 / (1 + rs))

    # Make sure we have enough rows
    if len(spy_hist) < 2:
        st.error("Not enough SPY data to calculate metrics.")
        st.stop()

    curr_p = round(float(spy_hist["Close"].iloc[-1]), 2)
    prev_close = round(float(spy_hist["Close"].iloc[-2]), 2)
    vix_p = round(float(vix_hist["Close"].iloc[-1]), 2)
    sma20_p = float(spy_hist["SMA_20"].iloc[-1]) if pd.notna(spy_hist["SMA_20"].iloc[-1]) else None
    sma200_p = float(spy_hist["SMA_200"].iloc[-1]) if pd.notna(spy_hist["SMA_200"].iloc[-1]) else None
    rsi_p = float(spy_hist["RSI"].iloc[-1]) if pd.notna(spy_hist["RSI"].iloc[-1]) else None

    if sma20_p is None or rsi_p is None:
        st.warning("Not enough data yet to calculate SMA/RSI for this timeframe.")
        st.stop()

    # --- ALGO SIGNAL LOGIC ---
    if curr_p > sma20_p and rsi_p < 35:
        signal, color = "STRONG BUY 🚀", "green"
    elif curr_p > sma20_p:
        signal, color = "BUY 📈", "green"
    elif curr_p < sma20_p and rsi_p > 65:
        signal, color = "STRONG SELL ⚠️", "red"
    else:
        signal, color = "SELL 📉", "red"

    st.subheader(f"Algo Signal: :{color}[{signal}]")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("SPY Price", f"${curr_p}", f"{round(curr_p - prev_close, 2)}")
    col2.metric("RSI (14)", f"{round(rsi_p, 2)}")
    col3.metric("VIX", f"{vix_p}")
    col4.metric("20-SMA", f"${round(sma20_p, 2)}")

    chart_cols = ["Close", "SMA_20"]
    if sma200_p is not None:
        chart_cols.append("SMA_200")

    st.line_chart(spy_hist[chart_cols].tail(90))

    st.subheader("🤖 AI Technical Verdict")

    if "GOOGLE_API_KEY" in st.secrets:
        if st.button("Run Deep Analysis"):
            prompt = (
                f"SPY is at ${curr_p}, 20SMA is ${round(sma20_p, 2)}, "
                f"RSI is {round(rsi_p, 2)}, and VIX is {vix_p}. "
                f"Act as a hedge fund lead and give a 3-sentence "
                f"high-conviction technical verdict for {timeframe} traders."
            )

            with st.spinner("Processing..."):
                client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                st.info(response.text)
    else:
        st.caption("Add GOOGLE_API_KEY to Streamlit secrets to enable AI analysis.")

except Exception as e:
    st.error(f"Error: {e}")
