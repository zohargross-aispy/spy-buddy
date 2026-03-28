import streamlit as st
import yfinance as yf
from google import genai

# --- PAGE SETUP ---
# Changed to "wide" layout to give our charts more room
st.set_page_config(page_title="SPY Buddy PRO", page_icon="📈", layout="wide")
st.title("📈 SPY Buddy Pro")
st.markdown("Your AI-powered technical assistant for S&P 500 market trends.")

# --- FETCH MARKET DATA ---
@st.cache_data(ttl=300) 
def get_market_data():
    # We now fetch 1 year of data so we can calculate the 200-Day Moving Average
    spy_hist = yf.Ticker("SPY").history(period="1y")
    vix_hist = yf.Ticker("^VIX").history(period="1d")
    return spy_hist, vix_hist

try:
    spy_hist, vix_hist = get_market_data()

    # --- TECHNICAL INDICATOR CALCULATIONS ---
    # 1. Moving Averages (20-day and 200-day)
    spy_hist['SMA_20'] = spy_hist['Close'].rolling(window=20).mean()
    spy_hist['SMA_200'] = spy_hist['Close'].rolling(window=200).mean()

    # 2. RSI (14-day Relative Strength Index)
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
    sma200_p = round(spy_hist['SMA_200'].iloc[-1], 2)
    rsi_p = round(spy_hist['RSI'].iloc[-1], 2)
    vol_p = int(spy_hist['Volume'].iloc[-1])

    trend = "Bullish 🐂" if curr_p > open_p else "Bearish 🐻"

    # --- DISPLAY TOP METRICS ---
    st.subheader("Market Overview")
    col1, col2, col3, col4 = st.columns(4)
    spy_delta = round(curr_p - prev_close, 2)
    
    col1.metric("SPY Price", f"${curr_p}", f"{spy_delta}", delta_color="normal")
    col2.metric("RSI (14-day)", f"{rsi_p}", "Overbought (>70)" if rsi_p > 70 else "Oversold (<30)" if rsi_p < 30 else "Neutral", delta_color="off" if 30 <= rsi_p <= 70 else "inverse")
    col3.metric("VIX (Volatility)", f"{vix_p}")
    col4.metric("Today's Volume", f"{vol_p:,}")

    st.markdown(f"**Intraday Trend:** {trend} &nbsp;|&nbsp; **20-Day SMA:** ${sma20_p} &nbsp;|&nbsp; **200-Day SMA:** ${sma200_p}")

    # --- CHARTS ---
    # We slice the last 90 days for the charts so they are easy to read
    st.subheader("📊 Price & Moving Averages (Past 3 Months)")
    chart_data = spy_hist[['Close', 'SMA_20', 'SMA_200']].tail(90)
    st.line_chart(chart_data)
    
    st.subheader("📉 Volume (Past 3 Months)")
    vol_data = spy_hist[['Volume']].tail(90)
    st.bar_chart(vol_data)

    # --- AI INSIGHTS ---
    st.subheader("🤖 AI Technical Analysis")
    
    if "GOOGLE_API_KEY" in st.secrets:
        client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
        
        if st.button("Generate Expert Analysis"):
            # The prompt is now vastly upgraded to include all our new technical data
            prompt = f"""
            Act as an expert technical stock analyst. The SPY ETF is currently at ${curr_p}. 
            Here is the current technical setup:
            - Today's Open: ${open_p} ({trend} intraday)
            - 20-Day SMA: ${sma20_p}
            - 200-Day SMA: ${sma200_p}
            - RSI (14): {rsi_p} (Overbought is >70, Oversold is <30)
            - Today's Volume: {vol_p:,} shares
            - VIX (Volatility): {vix_p}

            Based strictly on these technical indicators, give a concise, 3-sentence technical analysis of the current market momentum and what a day trader should watch out for.
            """
            
            with st.spinner("Analyzing technical indicators..."):
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                st.info(response.text)
    else:
        st.warning("⚠️ Google API Key not found. Please add it to your Streamlit secrets to use AI features.")

except Exception as e:
    st.error(f"Error fetching market data: {e}")
