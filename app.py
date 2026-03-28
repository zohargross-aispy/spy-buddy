import streamlit as st
import yfinance as yf
from google import genai

# --- PAGE SETUP ---
st.set_page_config(page_title="SPY Buddy", page_icon="📈", layout="centered")
st.title("📈 SPY Buddy")
st.markdown("Your AI-powered assistant for S&P 500 market trends.")

# --- FETCH MARKET DATA ---
@st.cache_data(ttl=300) 
def get_market_data():
    spy_hist = yf.Ticker("SPY").history(period="5d")
    vix_hist = yf.Ticker("^VIX").history(period="1d")
    return spy_hist, vix_hist

try:
    spy_hist, vix_hist = get_market_data()

    # --- PRICE LOGIC & CALCULATIONS ---
    curr_p = round(spy_hist['Close'].iloc[-1], 2)
    open_p = round(spy_hist['Open'].iloc[-1], 2) 
    
    prev_close = round(spy_hist['Close'].iloc[-2], 2) if len(spy_hist) > 1 else open_p
    vix_p = round(vix_hist['Close'].iloc[-1], 2)

    trend = "Bullish 🐂" if curr_p > open_p else "Bearish 🐻"

    # --- DISPLAY METRICS ---
    col1, col2, col3 = st.columns(3)
    spy_delta = round(curr_p - prev_close, 2)
    
    # delta_color="normal" handles the red/green for the price change
    col1.metric("SPY Price", f"${curr_p}", f"{spy_delta}", delta_color="normal")
    col2.metric("SPY Open", f"${open_p}")
    col3.metric("VIX (Volatility)", f"{vix_p}")

    # This displays a Green (success) or Red (error) alert box based on trend
    if "Bullish" in trend:
        st.success(f"**Current Trend:** {trend}")
    else:
        st.error(f"**Current Trend:** {trend}")

    # --- AI INSIGHTS ---
    st.subheader("🤖 AI Market Insight")
    
    if "GOOGLE_API_KEY" in st.secrets:
        client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
        
        if st.button("Generate Quick Analysis"):
            prompt = f"The SPY ETF is currently at ${curr_p}, opened at ${open_p} ({trend}). VIX is at {vix_p}. Give a brief, 2-sentence expert observation on this setup."
            
            with st.spinner("Analyzing market conditions..."):
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                st.info(response.text)
    else:
        st.warning("⚠️ Google API Key not found. Please add it to your Streamlit secrets to use AI features.")

except Exception as e:
    st.error(f"Error fetching market data: {e}")
