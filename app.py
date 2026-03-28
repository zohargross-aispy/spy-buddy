import streamlit as st
import yfinance as yf
from google import genai
import pandas as pd

# --- PAGE SETUP ---
st.set_page_config(page_title="SPY Buddy PRO", page_icon="📈", layout="wide")
st.title("📈 SPY Buddy Pro (Live Edition)")
st.markdown("Your AI-powered technical assistant for S&P 500 market trends.")

# --- TIMEFRAME & REFRESH ---
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
# Cache lowered to 60 seconds so you get near real-time updates
@st.cache_data(ttl=60) 
def get_market_data(p, i):
    spy_hist = yf.Ticker("SPY").history(period=p, interval=i)
    vix_hist = yf.Ticker("^VIX").history(period="1d")
    return spy_hist, vix_hist

try:
    spy_hist, vix_hist = get_market_data(period, interval)

    # --- TECHNICAL INDICATOR CALCULATIONS ---
    # Calculates Moving Averages based on the timeframe selected
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
    
    # Safely get Technicals in case there isn't enough data (like 1 min charts)
    sma20_p = round(spy_hist['SMA_20'].iloc[-1], 2) if not pd.isna(spy_hist['SMA_20'].iloc[-1]) else "N/A"
    sma200_p = round(spy_hist['SMA_200'].iloc[-1], 2) if not pd.isna(spy_hist['SMA_200'].iloc[-1]) else "N/A"
    rsi_p = round(spy_hist['RSI'].iloc[-1], 2) if not pd.isna(spy_hist['RSI'].iloc[-1]) else 50.0
    vol_p = int(spy_hist['Volume'].iloc[-1])

    trend = "Bullish 🐂" if curr_p > open_p else "Bearish 🐻"

    # --- DISPLAY TOP METRICS ---
    st.subheader(f"Market Overview ({timeframe} Chart)")
    col1, col2, col3, col4 = st.columns(4)
    spy_delta = round(curr_p - prev_close, 2)
    
    # Adjust labels dynamically
    tf_label = timeframe.split().lower()
    if tf_label == "day": tf_label = "day"
    elif tf_label == "hour": tf_label = "hour"
    else: tf_label = "period"

    col1.metric("SPY Price", f"${curr_p}", f"{spy_delta} (vs prev {tf_label})", delta_color="normal")
    col2.metric("RSI (14-period)", f"{rsi_p}", "Overbought (>70)" if rsi_p > 70 else "Oversold (<30)" if rsi_p < 30 else "Neutral", delta_color="off" if 30 <= rsi_p <= 70 else "inverse")
    col3.metric("VIX (Volatility)", f"{vix_p}")
    col4.metric(f"Volume (This {tf_label})", f"{vol_p:,}")

    st.markdown(f"**Current Trend:** {trend} &nbsp;|&nbsp; **20-Period SMA:** ${sma20_p} &nbsp;|&nbsp; **200-Period SMA:** ${sma200_p}")

    # --- CHARTS ---
    # Slice the last 90 periods so the chart stays zoomed in and readable
    st.subheader(f"📊 Price & Moving Averages (Last 90 {tf_label}s)")
    chart_data = spy_hist[['Close', 'SMA_20', 'SMA_200']].tail(90)
    st.line_chart(chart_data)
    
    st.subheader(f"📉 Volume (Last 90 {tf_label}s)")
    vol_data = spy_hist[['Volume']].tail(90)
    st.bar_chart(vol_data)

    # --- AI INSIGHTS ---
    st.subheader("🤖 AI Technical Analysis")
    
    if "GOOGLE_API_KEY" in st.secrets:
        client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
        
        if st.button("Generate Expert Analysis"):
            # Update the AI prompt to know exactly what timeframe it is looking at!
            prompt = f"""
            Act as an expert day trader. The SPY ETF is currently at ${curr_p}. 
            We are looking at a {timeframe} chart timeframe.
            Here is the current technical setup for this timeframe:
            - Current Candle Open: ${open_p} ({trend})
            - 20-Period SMA: ${sma20_p}
            - 200-Period SMA: ${sma200_p}
            - RSI (14): {rsi_p} (Overbought > 70, Oversold < 30)
            - Volume for this {tf_label}: {vol_p:,} shares
            - VIX (Volatility): {vix_p}

            Based strictly on these technical indicators for a {timeframe} chart, give a concise, 3-sentence technical analysis of the current momentum and what a day trader should watch out for over the next few candles.
            """
            
            with st.spinner("Analyzing intraday technicals..."):
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                st.info(response.text)
    else:
        st.warning("⚠️ Google API Key not found. Please add it to your Streamlit secrets to use AI features.")

except Exception as e:
    st.error(f"Error fetching market data: {e}")
