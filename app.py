import streamlit as st
from google import genai
import yfinance as yf
import feedparser
from datetime import datetime

# --- 1. SETUP ---
st.set_page_config(page_title="SPY Morning Buddy", page_icon="☀️", layout="centered")

# Your verified AI Studio Key
API_KEY = "AIzaSyBy5egrvlIOZ-BRYoCNdUkJDG01VZXFsbo"
client = genai.Client(api_key=API_KEY)

# --- 2. THE APP INTERFACE ---
st.title("☀️ SPY Morning Buddy V3.0")
st.write("2026 Institutional Macro Engine")

if st.button('🚀 Generate My Game Plan'):
    with st.spinner('Accessing Gemini 3.1 Pro...'):
        try:
            # Market Data Fetch
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="5d")
            vix = yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]
            
            # --- THE FULLY FIXED PRICE LOGIC ---
            curr_p = round(spy_hist['Close'].iloc[-1], 2)
            open_p = round(spy_hist['Open'].iloc, 2)
            trend = "Bullish" if curr_p > open_p else "Bearish"
            
            # Dashboard
            c1, c2, c3 = st.columns(3)
            c1.metric("SPY", f"${curr_p}")
            c2.metric("VIX", f"{round(vix, 2)}")
            c3.metric("Trend", trend)
            st.line_chart(spy_hist['Close'])

            # --- THE GENERATION (Using the New 3.1 Model) ---
            prompt = f"Today is {datetime.now().strftime('%A, %B %d')}. SPY: ${curr_p}, VIX: {vix}. Analyze and provide a 0DTE trade plan with Bias, Levels, and Strategy."
            
            # Note: Using the official 2026 model ID
            response = client.models.generate_content(
                model="gemini-3.1-pro",
                contents=prompt
            )
            
            st.markdown("### 📈 YOUR DAILY GAME PLAN")
            st.success(response.text)
            
        except Exception as e:
            st.error(f"Error: {e}")
