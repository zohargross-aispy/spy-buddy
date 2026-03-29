import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Optional AI import
try:
    from google import genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False


# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="SPY Buddy Pro 2.0", page_icon="📈", layout="wide")
st.title("📈 SPY Buddy Pro 2.0")
st.caption("Educational / research dashboard. Not financial advice.")


# ----------------------------
# CONFIG
# ----------------------------
TF_MAP = {
    "1 Day": {"period": "2y", "interval": "1d"},
    "1 Hour": {"period": "730d", "interval": "1h"},
    "15 Min": {"period": "60d", "interval": "15m"},
    "5 Min": {"period": "60d", "interval": "5m"},
    "1 Min": {"period": "7d", "interval": "1m"},
}

DEFAULT_SYMBOL = "SPY"


# ----------------------------
# HELPERS
# ----------------------------
def safe_round(x, digits=2):
    if pd.isna(x):
        return None
    return round(float(x), digits)


def signal_color(signal: str) -> str:
    return {
        "BUY": "green",
        "HOLD": "blue",
        "SELL": "red",
        "NO TRADE": "orange",
    }.get(signal, "gray")


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if df.empty:
        return df

    # EMAs
    df["EMA_8"] = df["Close"].ewm(span=8, adjust=False).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()

    # RSI (14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    # ATR (14)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # Volume average
    if "Volume" in df.columns:
        df["VOL_AVG_20"] = df["Volume"].rolling(20).mean()
    else:
        df["VOL_AVG_20"] = np.nan

    return df


def market_regime(daily_df: pd.DataFrame, vix_value: float) -> str:
    if daily_df.empty or len(daily_df) < 200:
        return "Insufficient Data"

    row = daily_df.iloc[-1]

    bullish = (
        row["Close"] > row["EMA_50"]
        and row["EMA_50"] > row["EMA_200"]
        and row["RSI"] > 52
    )
    bearish = (
        row["Close"] < row["EMA_50"]
        and row["EMA_50"] < row["EMA_200"]
        and row["RSI"] < 48
    )

    if bullish and vix_value < 20:
        return "Bull Trend"
    if bullish and vix_value >= 20:
        return "Bull Trend / High Vol"
    if bearish and vix_value >= 20:
        return "Bear Trend"
    if bearish and vix_value < 20:
        return "Bear Trend / Low Vol"
    return "Range / Transition"


def timeframe_alignment(df: pd.DataFrame) -> int:
    if df.empty or len(df) < 50:
        return 0

    row = df.iloc[-1]
    score = 0

    if row["Close"] > row["EMA_21"]:
        score += 1
    else:
        score -= 1

    if row["EMA_21"] > row["EMA_50"]:
        score += 1
    else:
        score -= 1

    if row["RSI"] > 52:
        score += 1
    elif row["RSI"] < 45:
        score -= 1

    if row["MACD_HIST"] > 0:
        score += 1
    else:
        score -= 1

    return score


def build_signal(entry_df: pd.DataFrame, hourly_df: pd.DataFrame, daily_df: pd.DataFrame, vix_value: float):
    if entry_df.empty or len(entry_df) < 50:
        return {
            "signal": "NO TRADE",
            "score": 0,
            "confidence": 0,
            "reasons": ["Not enough entry timeframe data."],
            "risk": "Unknown",
            "stop": None,
            "target": None,
            "regime": "Insufficient Data",
        }

    row = entry_df.iloc[-1]
    score = 0
    reasons = []

    # ----------------------------
    # Trend
    # ----------------------------
    if row["Close"] > row["EMA_21"]:
        score += 1
        reasons.append("Price is above EMA21.")
    else:
        score -= 1
        reasons.append("Price is below EMA21.")

    if row["EMA_21"] > row["EMA_50"]:
        score += 1
        reasons.append("EMA21 is above EMA50.")
    else:
        score -= 1
        reasons.append("EMA21 is below EMA50.")

    if not pd.isna(row["EMA_200"]):
        if row["EMA_50"] > row["EMA_200"]:
            score += 1
            reasons.append("EMA50 is above EMA200.")
        else:
            score -= 1
            reasons.append("EMA50 is below EMA200.")

    # ----------------------------
    # Momentum
    # ----------------------------
    if 52 <= row["RSI"] <= 68:
        score += 1
        reasons.append("RSI is in a bullish momentum zone.")
    elif row["RSI"] < 45:
        score -= 1
        reasons.append("RSI is weak.")
    elif row["RSI"] > 72:
        score -= 1
        reasons.append("RSI is overheated / extended.")

    if row["MACD_HIST"] > 0:
        score += 1
        reasons.append("MACD histogram is positive.")
    else:
        score -= 1
        reasons.append("MACD histogram is negative.")

    # ----------------------------
    # Volume
    # ----------------------------
    if "Volume" in entry_df.columns and not pd.isna(row["VOL_AVG_20"]):
        if row["Volume"] > row["VOL_AVG_20"]:
            score += 1
            reasons.append("Volume is above the 20-bar average.")
        else:
            reasons.append("Volume is not confirming strongly.")

    # ----------------------------
    # VIX filter
    # ----------------------------
    if vix_value < 18:
        score += 1
        reasons.append("VIX is supportive for trend continuation.")
    elif vix_value > 24:
        score -= 2
        reasons.append("VIX is elevated, which raises risk.")
    else:
        reasons.append("VIX is neutral.")

    # ----------------------------
    # Multi-timeframe alignment
    # ----------------------------
    htf_score = timeframe_alignment(hourly_df)
    dtf_score = timeframe_alignment(daily_df)

    if htf_score >= 2:
        score += 1
        reasons.append("Hourly timeframe confirms the setup.")
    elif htf_score <= -2:
        score -= 1
        reasons.append("Hourly timeframe disagrees.")

    if dtf_score >= 2:
        score += 2
        reasons.append("Daily timeframe confirms the bigger trend.")
    elif dtf_score <= -2:
        score -= 2
        reasons.append("Daily timeframe disagrees with the trade direction.")

    # ----------------------------
    # Extension filter
    # ----------------------------
    extended = False
    if not pd.isna(row["ATR"]) and row["ATR"] > 0:
        distance_from_ema21 = abs(row["Close"] - row["EMA_21"]) / row["ATR"]
        if distance_from_ema21 > 1.8:
            extended = True
            score -= 1
            reasons.append("Price is extended relative to ATR and EMA21.")

    regime = market_regime(daily_df, vix_value)

    # ----------------------------
    # Final signal
    # ----------------------------
    if score >= 6 and "Bear" not in regime and not extended:
        signal = "BUY"
    elif score <= -4:
        signal = "SELL"
    elif 2 <= score <= 5:
        signal = "HOLD"
    else:
        signal = "NO TRADE"

    # ----------------------------
    # Risk / confidence / levels
    # ----------------------------
    atr = row["ATR"] if not pd.isna(row["ATR"]) else None
    close = row["Close"]

    stop = None
    target = None
    risk = "Medium"

    if atr is not None and atr > 0:
        if signal == "BUY":
            stop = close - 1.2 * atr
            target = close + 2.0 * atr
        elif signal == "SELL":
            stop = close + 1.2 * atr
            target = close - 2.0 * atr

    if vix_value > 24:
        risk = "High"
    elif vix_value < 18 and "Bull" in regime:
        risk = "Low"

    confidence = min(95, max(5, 50 + score * 6))

    return {
        "signal": signal,
        "score": score,
        "confidence": confidence,
        "reasons": reasons,
        "risk": risk,
        "stop": stop,
        "target": target,
        "regime": regime,
    }


@st.cache_data(ttl=60)
def get_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df = df[~df.index.duplicated(keep="last")]
    return df


@st.cache_data(ttl=60)
def get_all_data(symbol: str, timeframe_label: str):
    cfg = TF_MAP[timeframe_label]

    entry_df = get_history(symbol, cfg["period"], cfg["interval"])
    hourly_df = get_history(symbol, "60d", "1h")
    daily_df = get_history(symbol, "2y", "1d")
    vix_df = get_history("^VIX", "3mo", "1d")

    return entry_df, hourly_df, daily_df, vix_df


def render_signal_banner(signal: str):
    color = signal_color(signal)
    emoji = {
        "BUY": "🚀",
        "HOLD": "🛡️",
        "SELL": "⚠️",
        "NO TRADE": "⏸️",
    }.get(signal, "📊")
    st.subheader(f"Signal: :{color}[{signal} {emoji}]")


def format_price(x):
    if x is None or pd.isna(x):
        return "N/A"
    return f"${x:,.2f}"


# ----------------------------
# UI CONTROLS
# ----------------------------
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    symbol = st.text_input("Ticker", value=DEFAULT_SYMBOL).upper().strip()

with col2:
    timeframe = st.selectbox(
        "Chart Timeframe",
        list(TF_MAP.keys()),
        index=0
    )

with col3:
    st.write("")
    st.write("")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ----------------------------
# LOAD DATA
# ----------------------------
try:
    entry_raw, hourly_raw, daily_raw, vix_raw = get_all_data(symbol, timeframe)

    if entry_raw.empty:
        st.error(f"No data returned for {symbol}. Try another ticker or timeframe.")
        st.stop()

    if vix_raw.empty:
        st.error("No VIX data returned.")
        st.stop()

    entry_df = add_indicators(entry_raw)
    hourly_df = add_indicators(hourly_raw)
    daily_df = add_indicators(daily_raw)

    if len(entry_df) < 20:
        st.error("Not enough data to calculate indicators on the selected timeframe.")
        st.stop()

    last = entry_df.iloc[-1]
    prev = entry_df.iloc[-2] if len(entry_df) > 1 else last

    curr_price = safe_round(last["Close"], 2)
    prev_price = safe_round(prev["Close"], 2)
    day_change = None if prev_price is None else round(curr_price - prev_price, 2)

    vix_value = safe_round(vix_raw["Close"].iloc[-1], 2)

    signal_data = build_signal(entry_df, hourly_df, daily_df, vix_value)

    # ----------------------------
    # TOP OUTPUT
    # ----------------------------
    render_signal_banner(signal_data["signal"])

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Price", format_price(curr_price), None if day_change is None else f"{day_change:+.2f}")
    m2.metric("Confidence", f"{signal_data['confidence']}%")
    m3.metric("Score", f"{signal_data['score']}")
    m4.metric("VIX", f"{vix_value}")
    m5.metric("Risk", signal_data["risk"])
    m6.metric("Regime", signal_data["regime"])

    n1, n2, n3, n4 = st.columns(4)
    n1.metric("RSI", f"{safe_round(last['RSI'], 2)}")
    n2.metric("ATR", format_price(safe_round(last["ATR"], 2)))
    n3.metric("Stop", format_price(safe_round(signal_data["stop"], 2)))
    n4.metric("Target", format_price(safe_round(signal_data["target"], 2)))

    # ----------------------------
    # BUY / HOLD / SELL RULES
    # ----------------------------
    with st.expander("How the engine thinks"):
        st.markdown(
            """
**BUY**
- Price above EMA21
- EMA21 above EMA50
- Daily trend agrees
- RSI healthy, MACD positive
- VIX not too hot
- Price not overly extended

**HOLD**
- Trend is still okay, but setup is not strong enough for a fresh buy
- Mixed signals, but not broken

**SELL**
- Price below EMA21 / EMA50
- Momentum weakens
- Daily trend disagrees or breaks
- VIX risk rises

**NO TRADE**
- Signals are mixed
- Price is too stretched
- Volatility is hostile
            """
        )

    # ----------------------------
    # CHART
    # ----------------------------
    chart_cols = ["Close", "EMA_8", "EMA_21", "EMA_50"]
    if "EMA_200" in entry_df.columns and entry_df["EMA_200"].notna().sum() > 0:
        chart_cols.append("EMA_200")

    st.subheader(f"{symbol} Chart")
    st.line_chart(entry_df[chart_cols].tail(150))

    # ----------------------------
    # REASONS / DIAGNOSTICS
    # ----------------------------
    left, right = st.columns([1.4, 1])

    with left:
        st.subheader("Why this signal")
        for reason in signal_data["reasons"]:
            st.write(f"- {reason}")

    with right:
        st.subheader("Latest Snapshot")
        latest_view = pd.DataFrame({
            "Close": [safe_round(last["Close"], 2)],
            "EMA_8": [safe_round(last["EMA_8"], 2)],
            "EMA_21": [safe_round(last["EMA_21"], 2)],
            "EMA_50": [safe_round(last["EMA_50"], 2)],
            "EMA_200": [safe_round(last["EMA_200"], 2)],
            "RSI": [safe_round(last["RSI"], 2)],
            "MACD": [safe_round(last["MACD"], 3)],
            "MACD_HIST": [safe_round(last["MACD_HIST"], 3)],
            "ATR": [safe_round(last["ATR"], 2)],
            "Volume": [int(last["Volume"]) if not pd.isna(last["Volume"]) else None],
        })
        st.dataframe(latest_view, use_container_width=True)

    # ----------------------------
    # OPTIONAL AI PANEL
    # ----------------------------
    st.subheader("🤖 AI Technical Verdict")

    api_key = None
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    elif "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]

    if api_key and GENAI_AVAILABLE:
        if st.button("Run Deep Analysis"):
            prompt = f"""
You are a disciplined institutional SPY market strategist.

Analyze this setup and respond in 5 short bullet points:
1. Market regime
2. What favors bulls
3. What favors bears
4. Best action right now: BUY / HOLD / SELL / NO TRADE
5. Risk to invalidation

Data:
Ticker: {symbol}
Chart timeframe: {timeframe}
Current price: {curr_price}
Signal: {signal_data['signal']}
Confidence: {signal_data['confidence']}%
Score: {signal_data['score']}
Risk: {signal_data['risk']}
Regime: {signal_data['regime']}
RSI: {safe_round(last['RSI'], 2)}
ATR: {safe_round(last['ATR'], 2)}
EMA8: {safe_round(last['EMA_8'], 2)}
EMA21: {safe_round(last['EMA_21'], 2)}
EMA50: {safe_round(last['EMA_50'], 2)}
EMA200: {safe_round(last['EMA_200'], 2)}
MACD: {safe_round(last['MACD'], 3)}
MACD Histogram: {safe_round(last['MACD_HIST'], 3)}
VIX: {vix_value}
Suggested stop: {safe_round(signal_data['stop'], 2)}
Suggested target: {safe_round(signal_data['target'], 2)}
"""

            with st.spinner("Running AI analysis..."):
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                st.info(response.text)

    else:
        if not GENAI_AVAILABLE:
            st.caption("Optional AI panel unavailable because `google-genai` is not installed.")
        else:
            st.caption("Add `GEMINI_API_KEY` or `GOOGLE_API_KEY` to Streamlit secrets to enable AI analysis.")

    # ----------------------------
    # RAW DATA
    # ----------------------------
    with st.expander("Raw price data"):
        st.dataframe(entry_df.tail(50), use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
