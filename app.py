import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional AI
try:
    from google import genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False


# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="SPY Buddy Pro 3.0",
    page_icon="📈",
    layout="wide"
)

st.title("📈 SPY Buddy Pro 3.0")
st.caption("Research / education dashboard. Not financial advice.")


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
    if x is None or pd.isna(x):
        return None
    return round(float(x), digits)


def fmt_price(x):
    if x is None or pd.isna(x):
        return "N/A"
    return f"${float(x):,.2f}"


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


def normalize_index_to_date(idx):
    ts = pd.to_datetime(idx)
    if getattr(ts, "tz", None) is not None:
        ts = ts.tz_convert(None)
    return pd.Series(ts).dt.normalize().values


def attach_daily_vix(df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty or vix_df.empty:
        df["VIX_CLOSE"] = np.nan
        return df

    vix = vix_df.copy()
    vix.index = pd.to_datetime(vix.index)
    if getattr(vix.index, "tz", None) is not None:
        vix.index = vix.index.tz_convert(None)
    vix["VIX_DATE"] = vix.index.normalize()

    temp = df.copy()
    temp.index = pd.to_datetime(temp.index)
    if getattr(temp.index, "tz", None) is not None:
        temp.index = temp.index.tz_convert(None)
    temp["BAR_DATE"] = temp.index.normalize()

    vix_map = vix[["Close", "VIX_DATE"]].drop_duplicates(subset="VIX_DATE").set_index("VIX_DATE")["Close"]
    temp["VIX_CLOSE"] = temp["BAR_DATE"].map(vix_map).ffill()
    temp.drop(columns=["BAR_DATE"], inplace=True)
    return temp


def timeframe_bias(df: pd.DataFrame) -> int:
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


def detect_market_regime(daily_df: pd.DataFrame, vix_value: float) -> str:
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

    if bullish and vix_value < 18:
        return "Bull Trend"
    if bullish and vix_value >= 18:
        return "Bull Trend / High Vol"
    if bearish and vix_value >= 20:
        return "Bear Trend"
    if bearish and vix_value < 20:
        return "Bear Trend / Low Vol"
    return "Range / Transition"


def current_signal(entry_df: pd.DataFrame, hourly_df: pd.DataFrame, daily_df: pd.DataFrame, vix_value: float):
    if entry_df.empty or len(entry_df) < 50:
        return {
            "signal": "NO TRADE",
            "score": 0,
            "confidence": 0,
            "risk": "Unknown",
            "stop": None,
            "target": None,
            "regime": "Insufficient Data",
            "reasons": ["Not enough data."],
        }

    row = entry_df.iloc[-1]
    score = 0
    reasons = []

    # Trend
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

    # Momentum
    if 52 <= row["RSI"] <= 68:
        score += 1
        reasons.append("RSI is in a healthy bullish zone.")
    elif row["RSI"] < 45:
        score -= 1
        reasons.append("RSI is weak.")
    elif row["RSI"] > 72:
        score -= 1
        reasons.append("RSI is stretched / overheated.")

    if row["MACD_HIST"] > 0:
        score += 1
        reasons.append("MACD histogram is positive.")
    else:
        score -= 1
        reasons.append("MACD histogram is negative.")

    # Volume
    if "Volume" in entry_df.columns and not pd.isna(row["VOL_AVG_20"]):
        if row["Volume"] > row["VOL_AVG_20"]:
            score += 1
            reasons.append("Volume is above the 20-bar average.")
        else:
            reasons.append("Volume is not strongly confirming.")

    # VIX
    if vix_value < 18:
        score += 1
        reasons.append("VIX is supportive.")
    elif vix_value > 24:
        score -= 2
        reasons.append("VIX is elevated and raises risk.")
    else:
        reasons.append("VIX is neutral.")

    # Higher timeframe alignment
    htf = timeframe_bias(hourly_df)
    dtf = timeframe_bias(daily_df)

    if htf >= 2:
        score += 1
        reasons.append("Hourly trend confirms.")
    elif htf <= -2:
        score -= 1
        reasons.append("Hourly trend disagrees.")

    if dtf >= 2:
        score += 2
        reasons.append("Daily trend confirms.")
    elif dtf <= -2:
        score -= 2
        reasons.append("Daily trend disagrees.")

    # Extension filter
    extended = False
    if not pd.isna(row["ATR"]) and row["ATR"] > 0:
        dist = abs(row["Close"] - row["EMA_21"]) / row["ATR"]
        if dist > 1.8:
            extended = True
            score -= 1
            reasons.append("Price is extended versus ATR and EMA21.")

    regime = detect_market_regime(daily_df, vix_value)

    if score >= 6 and "Bear" not in regime and not extended:
        signal = "BUY"
    elif score <= -4:
        signal = "SELL"
    elif 2 <= score <= 5:
        signal = "HOLD"
    else:
        signal = "NO TRADE"

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
        "score": int(score),
        "confidence": int(confidence),
        "risk": risk,
        "stop": stop,
        "target": target,
        "regime": regime,
        "reasons": reasons,
    }


def vector_signal_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["score"] = 0

    # Trend
    out["score"] += np.where(out["Close"] > out["EMA_21"], 1, -1)
    out["score"] += np.where(out["EMA_21"] > out["EMA_50"], 1, -1)
    out["score"] += np.where(out["EMA_50"] > out["EMA_200"], 1, -1)

    # RSI
    out["score"] += np.where((out["RSI"] >= 52) & (out["RSI"] <= 68), 1, 0)
    out["score"] += np.where(out["RSI"] < 45, -1, 0)
    out["score"] += np.where(out["RSI"] > 72, -1, 0)

    # MACD
    out["score"] += np.where(out["MACD_HIST"] > 0, 1, -1)

    # Volume
    out["score"] += np.where(out["Volume"] > out["VOL_AVG_20"], 1, 0)

    # VIX
    if "VIX_CLOSE" in out.columns:
        out["score"] += np.where(out["VIX_CLOSE"] < 18, 1, 0)
        out["score"] += np.where(out["VIX_CLOSE"] > 24, -2, 0)

    # Extended filter
    atr_dist = abs(out["Close"] - out["EMA_21"]) / out["ATR"].replace(0, np.nan)
    out["extended"] = atr_dist > 1.8
    out["score"] += np.where(out["extended"], -1, 0)

    # Labels
    out["signal_label"] = np.select(
        [
            out["score"] >= 6,
            out["score"] <= -4,
            (out["score"] >= 2) & (out["score"] <= 5),
        ],
        [
            "BUY",
            "SELL",
            "HOLD",
        ],
        default="NO TRADE"
    )

    return out


def run_backtest(df: pd.DataFrame):
    bt = df.copy()
    bt = vector_signal_score(bt)
    bt = bt.dropna(subset=["EMA_21", "EMA_50", "EMA_200", "RSI", "MACD_HIST", "ATR"]).copy()

    if bt.empty or len(bt) < 50:
        return None, None

    # Long-only state machine
    position = []
    in_pos = 0

    for _, row in bt.iterrows():
        enter_long = row["score"] >= 6 and not row["extended"]
        hold_long = row["score"] >= 2 and row["Close"] > row["EMA_50"]
        exit_long = row["score"] <= 1 or row["Close"] < row["EMA_21"]

        if in_pos == 0 and enter_long:
            in_pos = 1
        elif in_pos == 1 and exit_long and not hold_long:
            in_pos = 0

        position.append(in_pos)

    bt["position"] = position

    # Use previous bar position to avoid lookahead
    bt["ret"] = bt["Close"].pct_change().fillna(0)
    bt["strategy_ret"] = bt["ret"] * bt["position"].shift(1).fillna(0)
    bt["equity_curve"] = (1 + bt["strategy_ret"]).cumprod()
    bt["buy_hold_curve"] = (1 + bt["ret"]).cumprod()

    # Drawdown
    bt["equity_peak"] = bt["equity_curve"].cummax()
    bt["drawdown"] = bt["equity_curve"] / bt["equity_peak"] - 1

    # Trades
    bt["position_change"] = bt["position"].diff().fillna(0)
    entries = bt.index[bt["position_change"] == 1].tolist()
    exits = bt.index[bt["position_change"] == -1].tolist()

    if len(exits) < len(entries):
        exits.append(bt.index[-1])

    trades = []
    for entry_time, exit_time in zip(entries, exits):
        entry_price = bt.loc[entry_time, "Close"]
        exit_price = bt.loc[exit_time, "Close"]
        trade_return = (exit_price / entry_price) - 1
        trades.append({
            "Entry Time": entry_time,
            "Exit Time": exit_time,
            "Entry Price": round(float(entry_price), 2),
            "Exit Price": round(float(exit_price), 2),
            "Return %": round(trade_return * 100, 2),
        })

    trades_df = pd.DataFrame(trades)

    total_return = (bt["equity_curve"].iloc[-1] - 1) * 100
    buy_hold_return = (bt["buy_hold_curve"].iloc[-1] - 1) * 100
    max_drawdown = bt["drawdown"].min() * 100
    total_trades = len(trades_df)

    if total_trades > 0:
        win_rate = (trades_df["Return %"] > 0).mean() * 100
        avg_trade = trades_df["Return %"].mean()
    else:
        win_rate = 0.0
        avg_trade = 0.0

    stats = {
        "Strategy Return %": round(total_return, 2),
        "Buy & Hold %": round(buy_hold_return, 2),
        "Max Drawdown %": round(max_drawdown, 2),
        "Trades": int(total_trades),
        "Win Rate %": round(win_rate, 2),
        "Avg Trade %": round(avg_trade, 2),
    }

    return bt, {"stats": stats, "trades": trades_df}


def make_candlestick_chart(df: pd.DataFrame, symbol: str):
    chart_df = df.tail(180).copy()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.60, 0.20, 0.20],
        specs=[[{"secondary_y": True}], [{}], [{}]]
    )

    # Candles
    fig.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df["Open"],
            high=chart_df["High"],
            low=chart_df["Low"],
            close=chart_df["Close"],
            name="Candles"
        ),
        row=1, col=1, secondary_y=False
    )

    # EMAs
    for col in ["EMA_8", "EMA_21", "EMA_50", "EMA_200"]:
        if col in chart_df.columns and chart_df[col].notna().sum() > 0:
            fig.add_trace(
                go.Scatter(
                    x=chart_df.index,
                    y=chart_df[col],
                    mode="lines",
                    name=col
                ),
                row=1, col=1, secondary_y=False
            )

    # Volume
    fig.add_trace(
        go.Bar(
            x=chart_df.index,
            y=chart_df["Volume"],
            name="Volume",
            opacity=0.25
        ),
        row=1, col=1, secondary_y=True
    )

    # RSI
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["RSI"],
            mode="lines",
            name="RSI"
        ),
        row=2, col=1
    )
    fig.add_hline(y=70, row=2, col=1, line_dash="dot")
    fig.add_hline(y=30, row=2, col=1, line_dash="dot")

    # MACD
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["MACD"],
            mode="lines",
            name="MACD"
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["MACD_SIGNAL"],
            mode="lines",
            name="MACD Signal"
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Bar(
            x=chart_df.index,
            y=chart_df["MACD_HIST"],
            name="MACD Hist",
            opacity=0.4
        ),
        row=3, col=1
    )

    fig.update_layout(
        title=f"{symbol} Candlestick + EMA / RSI / MACD",
        xaxis_rangeslider_visible=False,
        height=900,
        legend_orientation="h"
    )

    fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)

    return fig


def make_backtest_chart(bt_df: pd.DataFrame):
    chart_df = bt_df.copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["equity_curve"],
            mode="lines",
            name="Strategy"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["buy_hold_curve"],
            mode="lines",
            name="Buy & Hold"
        )
    )
    fig.update_layout(
        title="Backtest Equity Curve",
        height=450
    )
    return fig


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
    entry = get_history(symbol, cfg["period"], cfg["interval"])
    hourly = get_history(symbol, "60d", "1h")
    daily = get_history(symbol, "2y", "1d")
    vix = get_history("^VIX", "6mo", "1d")
    return entry, hourly, daily, vix


# ----------------------------
# SIDEBAR
# ----------------------------
with st.sidebar:
    st.header("Controls")
    symbol = st.text_input("Ticker", value=DEFAULT_SYMBOL).upper().strip()
    timeframe = st.selectbox("Chart Timeframe", list(TF_MAP.keys()), index=0)

    st.divider()
    st.subheader("Backtest")
    backtest_bars = st.slider("Bars to backtest", min_value=120, max_value=2000, value=500, step=20)

    st.divider()
    show_ai = st.checkbox("Enable AI panel", value=True)

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ----------------------------
# LOAD DATA
# ----------------------------
try:
    entry_raw, hourly_raw, daily_raw, vix_raw = get_all_data(symbol, timeframe)

    if entry_raw.empty:
        st.error(f"No data returned for {symbol}.")
        st.stop()

    if vix_raw.empty:
        st.error("No VIX data returned.")
        st.stop()

    entry_df = add_indicators(entry_raw)
    hourly_df = add_indicators(hourly_raw)
    daily_df = add_indicators(daily_raw)
    entry_df = attach_daily_vix(entry_df, vix_raw)

    if len(entry_df) < 30:
        st.error("Not enough data for indicators on this timeframe.")
        st.stop()

    last = entry_df.iloc[-1]
    prev = entry_df.iloc[-2] if len(entry_df) > 1 else last

    curr_price = safe_round(last["Close"], 2)
    prev_price = safe_round(prev["Close"], 2)
    change = None if prev_price is None else round(curr_price - prev_price, 2)
    vix_value = safe_round(vix_raw["Close"].iloc[-1], 2)

    sig = current_signal(entry_df, hourly_df, daily_df, vix_value)

    # ----------------------------
    # ALERT LOGIC
    # ----------------------------
    alert_key = f"last_signal_{symbol}_{timeframe}"
    history_key = f"alert_history_{symbol}_{timeframe}"

    if history_key not in st.session_state:
        st.session_state[history_key] = []

    prev_signal = st.session_state.get(alert_key)

    if prev_signal is None:
        st.session_state[alert_key] = sig["signal"]
    elif prev_signal != sig["signal"]:
        st.session_state[history_key].insert(0, {
            "Time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Old Signal": prev_signal,
            "New Signal": sig["signal"],
            "Price": curr_price,
        })
        st.session_state[alert_key] = sig["signal"]
        st.warning(f"Signal changed: {prev_signal} → {sig['signal']} at {fmt_price(curr_price)}")

    # ----------------------------
    # HEADER METRICS
    # ----------------------------
    st.subheader(f"Signal: :{signal_color(sig['signal'])}[{sig['signal']}]")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Price", fmt_price(curr_price), None if change is None else f"{change:+.2f}")
    c2.metric("Confidence", f"{sig['confidence']}%")
    c3.metric("Score", f"{sig['score']}")
    c4.metric("VIX", f"{vix_value}")
    c5.metric("Risk", sig["risk"])
    c6.metric("Regime", sig["regime"])

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("RSI", f"{safe_round(last['RSI'], 2)}")
    d2.metric("ATR", fmt_price(safe_round(last["ATR"], 2)))
    d3.metric("Stop", fmt_price(safe_round(sig["stop"], 2)))
    d4.metric("Target", fmt_price(safe_round(sig["target"], 2)))

    # ----------------------------
    # TABS
    # ----------------------------
    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Backtest", "Alerts", "Raw Data"])

    with tab1:
        with st.expander("How the engine decides"):
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
- Trend is okay, but setup is not strong enough for a fresh buy

**SELL**
- Trend and momentum break down
- Higher timeframe disagrees
- Risk rises

**NO TRADE**
- Signals are mixed
- Price is too stretched
- Volatility is hostile
                """
            )

        fig = make_candlestick_chart(entry_df, symbol)
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns([1.2, 1])

        with left:
            st.subheader("Why this signal")
            for reason in sig["reasons"]:
                st.write(f"- {reason}")

        with right:
            st.subheader("Latest Snapshot")
            snap = pd.DataFrame({
                "Close": [safe_round(last["Close"], 2)],
                "EMA_8": [safe_round(last["EMA_8"], 2)],
                "EMA_21": [safe_round(last["EMA_21"], 2)],
                "EMA_50": [safe_round(last["EMA_50"], 2)],
                "EMA_200": [safe_round(last["EMA_200"], 2)],
                "RSI": [safe_round(last["RSI"], 2)],
                "MACD": [safe_round(last["MACD"], 3)],
                "MACD_SIGNAL": [safe_round(last["MACD_SIGNAL"], 3)],
                "MACD_HIST": [safe_round(last["MACD_HIST"], 3)],
                "ATR": [safe_round(last["ATR"], 2)],
                "VIX_CLOSE": [safe_round(last["VIX_CLOSE"], 2)],
                "Volume": [int(last["Volume"]) if not pd.isna(last["Volume"]) else None],
            })
            st.dataframe(snap, use_container_width=True)

        if show_ai:
            st.subheader("🤖 AI Technical Verdict")

            api_key = None
            if "GEMINI_API_KEY" in st.secrets:
                api_key = st.secrets["GEMINI_API_KEY"]
            elif "GOOGLE_API_KEY" in st.secrets:
                api_key = st.secrets["GOOGLE_API_KEY"]

            if api_key and GENAI_AVAILABLE:
                if st.button("Run Deep Analysis"):
                    prompt = f"""
You are a disciplined institutional market strategist.

Analyze this setup in 6 short bullet points:
1. Market regime
2. What favors bulls
3. What favors bears
4. Best action now: BUY / HOLD / SELL / NO TRADE
5. Key invalidation
6. Short tactical note

Data:
Ticker: {symbol}
Timeframe: {timeframe}
Price: {curr_price}
Signal: {sig['signal']}
Confidence: {sig['confidence']}%
Score: {sig['score']}
Risk: {sig['risk']}
Regime: {sig['regime']}
RSI: {safe_round(last['RSI'], 2)}
ATR: {safe_round(last['ATR'], 2)}
EMA8: {safe_round(last['EMA_8'], 2)}
EMA21: {safe_round(last['EMA_21'], 2)}
EMA50: {safe_round(last['EMA_50'], 2)}
EMA200: {safe_round(last['EMA_200'], 2)}
MACD: {safe_round(last['MACD'], 3)}
MACD Signal: {safe_round(last['MACD_SIGNAL'], 3)}
MACD Histogram: {safe_round(last['MACD_HIST'], 3)}
VIX: {vix_value}
Suggested stop: {safe_round(sig['stop'], 2)}
Suggested target: {safe_round(sig['target'], 2)}
"""

                    with st.spinner("Running AI analysis..."):
                        client = genai.Client(api_key=api_key)
                        response = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=prompt
                        )
                        st.info(response.text)
            else:
                st.caption("To enable AI, install `google-genai` and add `GEMINI_API_KEY` or `GOOGLE_API_KEY` to Streamlit secrets.")

    with tab2:
        st.subheader("Simple Backtest")

        bt_source = entry_df.tail(backtest_bars).copy()
        bt_df, bt_result = run_backtest(bt_source)

        if bt_df is None:
            st.info("Not enough data for backtest.")
        else:
            s = bt_result["stats"]

            b1, b2, b3, b4, b5, b6 = st.columns(6)
            b1.metric("Strategy Return", f"{s['Strategy Return %']}%")
            b2.metric("Buy & Hold", f"{s['Buy & Hold %']}%")
            b3.metric("Max Drawdown", f"{s['Max Drawdown %']}%")
            b4.metric("Trades", f"{s['Trades']}")
            b5.metric("Win Rate", f"{s['Win Rate %']}%")
            b6.metric("Avg Trade", f"{s['Avg Trade %']}%")

            st.plotly_chart(make_backtest_chart(bt_df), use_container_width=True)

            with st.expander("Backtest trade log"):
                trades_df = bt_result["trades"]
                if trades_df.empty:
                    st.write("No completed trades in this window.")
                else:
                    st.dataframe(trades_df, use_container_width=True)

            st.caption("This backtest is intentionally simple. It is a fast sanity check, not execution-grade research.")

    with tab3:
        st.subheader("Signal Alerts")
        st.caption("Alerts are logged when the signal changes on refresh / rerun.")

        alerts = st.session_state[history_key]
        if alerts:
            st.dataframe(pd.DataFrame(alerts), use_container_width=True)
        else:
            st.write("No signal changes logged yet.")

    with tab4:
        st.subheader("Raw Data")
        st.dataframe(entry_df.tail(100), use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
