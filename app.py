import streamlit as st
import google.generativeai as genai
import yfinance as yf
import feedparser

# --- 1. SETUP ---
st.set_page_config(page_title="SPY Morning Buddy", page_icon="☀️")

# Using your verified 'SPY BUDDY' key (...Fsbo)
API_KEY = "AIzaSyBy5egrvlIOZ-BRYoCNdUkJDG01VZXFsbo"
genai.configure(api_key=API_KEY)

# --- 2. THE APP INTERFACE ---
st.title("☀️ SPY Morning Buddy")
st.write("Click below to get your 2026 Macro Game Plan.")

if st.button('🚀 Generate My Game Plan'):
    with st.spinner('Syncing with Gemini 3.1 Pro...'):
        try:
            # Fetch Market Data
            spy = yf.Ticker("SPY")
            price = round(spy.history(period="1d")['Close'].iloc[-1], 2)
            feed = feedparser.parse("https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY")
            news = " | ".join([entry.title for entry in feed.entries[:5]])
            
            # --- UPDATED MODEL NAME FOR 2026 ---
            # Using the Gemini 3.1 Pro Preview seen in your AI Studio dashboard
            model = genai.GenerativeModel('gemini-3.1-pro-preview')
            
            prompt = (
                f"System: You are an elite Macro Strategist. "
                f"Current SPY Price: ${price}. Recent News: {news}. "
                f"Generate a 0DTE trade plan including Bias, Reasoning, Levels, and the specific Play."
            )
            
            response = model.generate_content(prompt)
            
            st.metric("Current SPY Price", f"${price}")
            st.markdown("### 📈 YOUR DAILY GAME PLAN")
            st.success(response.text)
            
        except Exception as e:
            st.error(f"Error: {e}")
            st.write("Tip: If you still see a 404, check the model name in your AI Studio dropdown.")
