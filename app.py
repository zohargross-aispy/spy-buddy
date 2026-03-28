import streamlit as st
import google.generativeai as genai
import yfinance as yf
import feedparser
from datetime import datetime

# --- 1. SETUP ---
st.set_page_config(page_title="SPY Morning Buddy", page_icon="☀️", layout="centered")

# Using your verified 'SPY BUDDY' key
API_KEY = "AIzaSyBy5egrvlIOZ-BRYoCNdUkJDG01VZXFsbo"
genai.configure(api_key=API_KEY)

# --- 2. THE ADVANCED UI ---
st.title("☀️ SPY Morning Buddy V2.0")
st.write("Institutional Macro Game Plan")

if st.button('🚀 Generate My Game Plan'):
    with st.spinner('Syncing with live market data & Gemini 3.1 Pro...'):
        try:
            # Fetch SPY Data (5 days for trend)
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="5d")
            current_price = round(spy_hist['Close'].iloc[-1], 2)
            five_day_open = round(spy_hist['Open'].iloc, 2)
            trend = "Bullish/Up" if current_price > five_day_open else "Bearish/Down"
            
            # Fetch VIX (Volatility/Fear Gauge)
            vix = yf.Ticker("^VIX")
            vix_price = round(vix.history(period="1d")['Close'].iloc[-1], 2)
            
            # Fetch News
            feed = feedparser.parse("https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY")
            news = " | ".join([entry.title for entry in feed.entries[:5]])
            
            # Get Today's Date
            today = datetime.now().strftime("%A, %B %d, %Y")
            
            # --- DISPLAY DASHBOARD ---
            col1, col2, col3 = st.columns(3)
            col1.metric("SPY Price", f"${current_price}")
            col2.metric("VIX (Volatility)", f"{vix_price}")
            col3.metric("5-Day Trend", trend)
            
            # Add a sleek mini-chart to the app
            st.write("**SPY 5-Day Price Action**")
            st.line_chart(spy_hist['Close'])
            
            # --- CONSULT THE AI ---
            model = genai.GenerativeModel('gemini-3.1-pro-preview')
            prompt = (
                f"System: You are an elite Macro Strategist. Today is {today}. "
                f"SPY Price: ${current_price}. 5-Day Trend is {trend}. "
                f"The VIX is currently at {vix_price}. "
                f"Recent News: {news}. "
                f"Generate a highly accurate 0DTE trade plan. Include: 1. Daily Bias, 2. Macro Reasoning (factor in the VIX), 3. Key Levels, 4. Specific 0DTE Play."
            )
            
            response = model.generate_content(prompt)
            
            st.markdown("### 📈 YOUR DAILY GAME PLAN")
            st.success(response.text)
            
            # Show the raw news at the bottom for the user
            with st.expander("📰 Read Today's Raw Headlines"):
                for entry in feed.entries[:5]:
                    st.write(f"- {entry.title}")
            
        except Exception as e:
            st.error(f"Error: {e}")
