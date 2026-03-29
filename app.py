import streamlit as st
    import yfinance as yf
    from google import genai
    import pandas as pd
    
    st.set_page_config(page_title="SPY Buddy PRO", page_icon="📈", layout="wide")
    st.title("📈 SPY Buddy Pro (Algo Edition)")
    
    col_tf, col_ref = st.columns([3, 1]) 
    with col_tf:
        timeframe = st.selectbox("Chart Timeframe", ["1 Day", "1 Hour", "15 Min", "5 Min", "1 Min"], index=0)
    with col_ref:
        st.write(""); st.write("")
        if st.button("🔄 Refresh Data"): st.cache_data.clear()
    
    tf_map = {"1 Day": {"p": "1y", "i": "1d"}, "1 Hour": {"p": "730d", "i": "1h"}, "15 Min": {"p": "60d", "i": "15m"}, "5 Min": {"p": "60d", "i": "5m"}, "1 Min": {"p": "7d", "i": "1m"}}
    period, interval = tf_map[timeframe]["p"], tf_map[timeframe]["i"]
    
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
        prev_close = round(spy_hist['Close'].iloc[-2], 2)
        vix_p = round(vix_hist['Close'].iloc[-1], 2)
        sma20_p = spy_hist['SMA_20'].iloc[-1]
        rsi_p = spy_hist['RSI'].iloc[-1]
    
        # --- ALGO SIGNAL LOGIC ---
        if curr_p > sma20_p and rsi_p < 35: signal, color = "STRONG BUY 🚀", "green"
        elif curr_p > sma20_p: signal, color = "BUY 📈", "green"
        elif curr_p < sma20_p and rsi_p > 65: signal, color = "STRONG SELL ⚠️", "red"
        else: signal, color = "SELL 📉", "red"
    
        st.subheader(f"Algo Signal: :{color}[{signal}]")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("SPY Price", f"${curr_p}", f"{round(curr_p - prev_close, 2)}")
        col2.metric("RSI (14)", f"{round(rsi_p, 2)}")
        col3.metric("VIX", f"{vix_p}")
        col4.metric("20-SMA", f"${round(sma20_p, 2)}")
    
        st.line_chart(spy_hist[['Close', 'SMA_20', 'SMA_200']].tail(90))
    
        st.subheader("🤖 AI Technical Verdict")
        if "GOOGLE_API_KEY" in st.secrets:
            if st.button("Run Deep Analysis"):
                prompt = f"SPY ${curr_p}, 20SMA ${sma20_p}, RSI {rsi_p}, VIX {vix_p}. Act as a hedge fund lead. Give a 3-sentence high-conviction verdict for {timeframe} traders."
                with st.spinner("Processing..."):
                    st.info(genai.Client(api_key=st.secrets["GOOGLE_API_KEY"]).models.generate_content(model='gemini-2.5-flash', contents=prompt).text)
    except Exception as e:
        st.error(f"Error: {e}")
    
