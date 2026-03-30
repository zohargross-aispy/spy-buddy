import json
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from websocket import WebSocketApp

st.set_page_config(page_title="SPY Buddy Alpaca Live", page_icon="📡", layout="wide")
st.title("📡 SPY Buddy Alpaca Live")
st.caption("Real-time stock data from Alpaca. Research / education only.")

# ----------------------------
# SECRETS
# ----------------------------
ALPACA_KEY = st.secrets.get("ALPACA_API_KEY", "")
ALPACA_SECRET = st.secrets.get("ALPACA_SECRET_KEY", "")

# ----------------------------
# CONFIG
# ----------------------------
TIMEFRAME_TO_ALPACA = {
    "1 Min": "1Min",
    "5 Min": "5Min",
    "15 Min": "15Min",
    "1 Hour": "1Hour",
    "1 Day": "1Day",
}

CHART_BAR_COUNT = {
    "1 Min": 180,
    "5 Min": 180,
    "15 Min": 180,
    "1 Hour": 220,
    "1 Day": 220,
}

DEFAULT_SYMBOL = "SPY"

# ----------------------------
# GLOBAL LIVE STORE
# ----------------------------
@st.cache_resource
def get_live_store():
    return {
        "streams": {},   # key -> thread/meta
        "data": {}       # key -> live data
    }


def stream_key(symbol: str, feed: str) -> str:
    return f"{symbol.upper()}|{feed.lower()}"


def ensure_data_slot(store: dict, key: str):
    if key not in store["data"]:
        store["data"][key] = {
            "status": "Not connected",
            "last_error": None,
            "last_message_time": None,
            "trade": None,
            "quote": None,
            "bars": deque(maxlen=600),
        }


def start_alpaca_stream(symbol: str, feed: str):
    symbol = symbol.upper()
    feed = feed.lower()
    key = stream_key(symbol, feed)
    store = get_live_store()
    ensure_data_slot(store, key)

    if key in store["streams"]:
        meta = store["streams"][key]
        if meta.get("running"):
            return

    url = f"wss://stream.data.alpaca.markets/v2/{feed}"

    def on_open(ws):
        store["data"][key]["status"] = "Connected"
        auth_msg = {
            "action": "auth",
            "key": ALPACA_KEY,
            "secret": ALPACA_SECRET,
        }
        ws.send(json.dumps(auth_msg))
        sub_msg = {
            "action": "subscribe",
            "trades": [symbol],
            "quotes": [symbol],
            "bars": [symbol],
        }
        ws.send(json.dumps(sub_msg))

    def on_message(ws, message):
        try:
            payload = json.loads(message)
            if isinstance(payload, dict):
                payload = [payload]

            for item in payload:
                msg_type = item.get("T")
                store["data"][key]["last_message_time"] = datetime.now().isoformat()

                if msg_type == "success":
                    msg = item.get("msg", "")
                    store["data"][key]["status"] = f"Connected: {msg}"

                elif msg_type == "error":
                    store["data"][key]["last_error"] = item.get("msg", "Unknown stream error")
                    store["data"][key]["status"] = "Stream error"

                elif msg_type == "t":
                    store["data"][key]["trade"] = {
                        "price": item.get("p"),
                        "size": item.get("s"),
                        "time": item.get("t"),
                    }

                elif msg_type == "q":
                    store["data"][key]["quote"] = {
                        "bid": item.get("bp"),
                        "ask": item.get("ap"),
                        "bid_size": item.get("bs"),
                        "ask_size": item.get("as"),
                        "time": item.get("t"),
                    }

                elif msg_type == "b":
                    bar = {
                        "time": item.get("t"),
                        "open": item.get("o"),
                        "high": item.get("h"),
                        "low": item.get("l"),
                        "close": item.get("c"),
                        "volume": item.get("v"),
                    }
                    bars = store["data"][key]["bars"]
                    if bars and bars[-1]["time"] == bar["time"]:
                        bars[-1] = bar
                    else:
                        bars.append(bar)

        except Exception as e:
            store["data"][key]["last_error"] = str(e)
            store["data"][key]["status"] = "Message parse error"

    def on_error(ws, error):
        store["data"][key]["last_error"] = str(error)
        store["data"][key]["status"] = "Connection error"

    def on_close(ws, close_status_code, close_msg):
        store["data"][key]["status"] = f"Closed ({close_status_code})"

    def run():
        store["streams"][key]["running"] = True
        while store["streams"][key]["running"]:
            try:
                ws = WebSocketApp(
                    url,
                    on_open=on_open,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close,
                )
                store["streams"][key]["ws"] = ws
                ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                store["data"][key]["last_error"] = str(e)
                store["data"][key]["status"] = "Reconnect error"

            if store["streams"][key]["running"]:
                time.sleep(3)

    thread = threading.Thread(target=run, daemon=True)
    store["streams"][key] = {"thread": thread, "running": True, "ws": None}
    thread.start()


def fetch_alpaca_bars(symbol: str, timeframe: str, feed: str, limit: int = 300) -> pd.DataFrame:
    if not ALPACA_KEY or not ALPACA_SECRET:
        return pd.DataFrame()

    headers = {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
    }

    url = "https://data.alpaca.markets/v2/stocks/bars"
    params = {
        "symbols": symbol.upper(),
        "timeframe": timeframe,
        "limit": limit,
        "feed": feed.lower(),
        "adjustment": "raw",
        "sort": "asc",
    }

    try:
        r = requests.get(url, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        payload = r.json()
        bars_map = payload.get("bars", {})
        bars = bars_map.get(symbol.upper(), [])
        if not bars:
            return pd.DataFrame()

        df = pd.DataFrame(bars)
        df["Time"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert("America/New_York")
        df = df.rename(columns={
            "o": "Open",
            "h": "High",
            "l": "Low",
            "c": "Close",
            "v": "Volume",
            "vw": "VWAP",
            "n": "Trades",
        })
        return df[["Time", "Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        return pd.DataFrame()


def merge_live_bars(hist_df: pd.DataFrame, symbol: str, feed: str) -> pd.DataFrame:
    store = get_live_store()
    key = stream_key(symbol, feed)
    ensure_data_slot(store, key)

    live_bars = list(store["data"][key]["bars"])
    if not live_bars:
        return hist_df

    live_df = pd.DataFrame(live_bars)
    if live_df.empty:
        return hist_df

    live_df["Time"] = pd.to_datetime(live_df["time"], utc=True).dt.tz_convert("America/New_York")
    live_df = live_df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })
    live_df = live_df[["Time", "Open", "High", "Low", "Close", "Volume"]]

    if hist_df.empty:
        return live_df.sort_values("Time").drop_duplicates(subset=["Time"], keep="last")

    merged = pd.concat([hist_df, live_df], ignore_index=True)
    merged = merged.sort_values("Time").drop_duplicates(subset=["Time"], keep="last")
    return merged


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    out["EMA_8"] = out["Close"].ewm(span=8, adjust=False).mean()
    out["EMA_21"] = out["Close"].ewm(span=21, adjust=False).mean()
    out["EMA_50"] = out["Close"].ewm(span=50, adjust=False).mean()

    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    out["RSI"] = 100 - (100 / (1 + rs))

    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]

    return out


def latest_signal(df: pd.DataFrame) -> str:
    if df.empty or len(df) < 30:
        return "NO TRADE"
    row = df.iloc[-1]
    score = 0
    score += 1 if row["Close"] > row["EMA_8"] else -1
    score += 1 if row["EMA_8"] > row["EMA_21"] else -1
    score += 1 if row["EMA_21"] > row["EMA_50"] else -1
    score += 1 if row["MACD_HIST"] > 0 else -1
    if pd.notna(row["RSI"]):
        score += 1 if row["RSI"] > 52 else -1 if row["RSI"] < 45 else 0

    if score >= 4:
        return "BUY"
    if score <= -4:
        return "SELL"
    return "NO TRADE"


def make_chart(df: pd.DataFrame, symbol: str):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["Time"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name=symbol,
    ))
    for col in ["EMA_8", "EMA_21", "EMA_50"]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["Time"], y=df[col], mode="lines", name=col))

    fig.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        dragmode="pan",
    )
    return fig


# ----------------------------
# SIDEBAR
# ----------------------------
with st.sidebar:
    st.header("Live Feed")
    symbol = st.text_input("Ticker", value=DEFAULT_SYMBOL).upper().strip()
    timeframe_label = st.selectbox("Timeframe", list(TIMEFRAME_TO_ALPACA.keys()), index=0)
    feed = st.selectbox("Alpaca feed", ["iex", "sip"], index=0, help="Free Alpaca accounts typically use IEX. SIP requires a paid data plan.")
    auto_sec = st.slider("Refresh every N seconds", min_value=1, max_value=10, value=2)

    st.divider()
    st.subheader("Alpaca Secrets")
    st.caption("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in Streamlit secrets.")

    if st.button("Start / Restart Stream", use_container_width=True):
        start_alpaca_stream(symbol, feed)
        st.cache_data.clear()

# Auto-refresh UI so live data paints
st_autorefresh(interval=auto_sec * 1000, key=f"live_refresh_{symbol}_{feed}_{timeframe_label}")

# Start stream automatically if possible
if ALPACA_KEY and ALPACA_SECRET:
    start_alpaca_stream(symbol, feed)

# ----------------------------
# MAIN
# ----------------------------
if not ALPACA_KEY or not ALPACA_SECRET:
    st.error("Missing Alpaca secrets. Add ALPACA_API_KEY and ALPACA_SECRET_KEY in Streamlit secrets.")
    st.stop()

store = get_live_store()
key = stream_key(symbol, feed)
ensure_data_slot(store, key)
slot = store["data"][key]

alpaca_tf = TIMEFRAME_TO_ALPACA[timeframe_label]
hist = fetch_alpaca_bars(symbol, alpaca_tf, feed, limit=CHART_BAR_COUNT[timeframe_label])
df = merge_live_bars(hist, symbol, feed)
df = add_indicators(df)

signal = latest_signal(df)

trade = slot.get("trade") or {}
quote = slot.get("quote") or {}

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Signal", signal)
c2.metric("Last Trade", "N/A" if trade.get("price") is None else f"${trade['price']:.2f}")
spread = None
if quote.get("bid") is not None and quote.get("ask") is not None:
    spread = quote["ask"] - quote["bid"]
c3.metric("Bid / Ask", "N/A" if quote.get("bid") is None else f"{quote['bid']:.2f} / {quote['ask']:.2f}")
c4.metric("Spread", "N/A" if spread is None else f"{spread:.3f}")
c5.metric("Stream", slot.get("status", "Unknown"))

if slot.get("last_error"):
    st.warning(f"Last stream error: {slot['last_error']}")

if df.empty:
    st.info("No bars loaded yet.")
else:
    st.plotly_chart(make_chart(df, symbol), use_container_width=True, config={
        "scrollZoom": True,
        "displaylogo": False,
    })

    with st.expander("Latest live packet"):
        packet = {
            "trade": trade,
            "quote": quote,
            "last_message_time": slot.get("last_message_time"),
        }
        st.json(packet)

st.caption("This version uses Alpaca's real-time market data stream plus Alpaca historical bars. It is a better foundation for true live updates than yfinance.")
