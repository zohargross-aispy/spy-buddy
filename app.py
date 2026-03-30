"""
SPY Buddy Pro Elite
====================
Real-time trading signal dashboard powered by Alpaca Markets data.
Set ALPACA_API_KEY and ALPACA_SECRET_KEY in Streamlit secrets or as
environment variables before running.

Not financial advice.
"""

import json
import os
import urllib.request
import urllib.error
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    ALPACA_SDK_AVAILABLE = True
except ImportError:
    ALPACA_SDK_AVAILABLE = False

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# PAGE CONFIG  (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SPY Buddy Pro Elite",
    page_icon="📈",
    layout="wide",
)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"

TF_MAP: Dict[str, Dict[str, Any]] = {
    "1 Day":  {"amount": 1,  "unit": "Day",    "limit": 500,  "period_days": 730},
    "1 Hour": {"amount": 1,  "unit": "Hour",   "limit": 1000, "period_days": 60},
    "15 Min": {"amount": 15, "unit": "Minute", "limit": 1000, "period_days": 30},
    "5 Min":  {"amount": 5,  "unit": "Minute", "limit": 1000, "period_days": 15},
    "1 Min":  {"amount": 1,  "unit": "Minute", "limit": 1000, "period_days": 7},
}

DEFAULT_SYMBOL = "SPY"


# ---------------------------------------------------------------------------
# CREDENTIAL HELPERS
# ---------------------------------------------------------------------------
def get_alpaca_keys() -> Tuple[Optional[str], Optional[str]]:
    """Return (api_key, secret_key) from Streamlit secrets or env vars."""
    api_key = None
    secret_key = None

    # Try Streamlit secrets first
    try:
        api_key = st.secrets.get("ALPACA_API_KEY") or st.secrets.get("alpaca_api_key")
        secret_key = st.secrets.get("ALPACA_SECRET_KEY") or st.secrets.get("alpaca_secret_key")
    except Exception:
        pass

    # Fall back to environment variables
    if not api_key:
        api_key = os.environ.get("ALPACA_API_KEY")
    if not secret_key:
        secret_key = os.environ.get("ALPACA_SECRET_KEY")

    return api_key, secret_key


def get_gemini_key() -> Optional[str]:
    """Return Gemini API key from Streamlit secrets or env vars."""
    try:
        key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


# ---------------------------------------------------------------------------
# DATA FETCHING  (Alpaca primary, yfinance fallback)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_alpaca_bars(
    symbol: str,
    timeframe_label: str,
    api_key: str,
    secret_key: str,
) -> pd.DataFrame:
    """
    Fetch OHLCV bars from Alpaca using the official SDK.
    Falls back to yfinance if the SDK is unavailable or the request fails.
    """
    cfg = TF_MAP[timeframe_label]

    if ALPACA_SDK_AVAILABLE and api_key and secret_key:
        try:
            client = StockHistoricalDataClient(api_key, secret_key)

            unit_map = {
                "Day":    TimeFrame(1, TimeFrameUnit.Day),
                "Hour":   TimeFrame(1, TimeFrameUnit.Hour),
                "Minute": TimeFrame(cfg["amount"], TimeFrameUnit.Minute),
            }
            tf = unit_map[cfg["unit"]]

            start = datetime.now(timezone.utc) - timedelta(days=cfg["period_days"])

            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                limit=cfg["limit"],
                feed="iex",          # free IEX feed; change to "sip" for paid
            )

            bars = client.get_stock_bars(request)
            df = bars.df

            if df is None or df.empty:
                raise ValueError("Empty response from Alpaca")

            # Alpaca SDK returns a MultiIndex (symbol, timestamp) — flatten it
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(symbol, level="symbol")

            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_convert("America/New_York")

            df.rename(columns={
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume",
            }, inplace=True)

            df = df[~df.index.duplicated(keep="last")]
            return df[["Open", "High", "Low", "Close", "Volume"]]

        except Exception as e:
            st.warning(f"Alpaca fetch failed ({e}). Falling back to yfinance.")

    # ---- yfinance fallback ------------------------------------------------
    try:
        import yfinance as yf

        yf_interval_map = {
            "1 Day":  ("2y",  "1d"),
            "1 Hour": ("60d", "1h"),
            "15 Min": ("60d", "15m"),
            "5 Min":  ("60d", "5m"),
            "1 Min":  ("7d",  "1m"),
        }
        period, interval = yf_interval_map[timeframe_label]
        df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)

        if df is None or df.empty:
            return pd.DataFrame()

        df = df[~df.index.duplicated(keep="last")]
        return df[["Open", "High", "Low", "Close", "Volume"]]

    except Exception as e:
        st.error(f"Both Alpaca and yfinance failed: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def fetch_vix(api_key: str, secret_key: str) -> pd.DataFrame:
    """Fetch VIX daily bars (always uses yfinance — Alpaca doesn't carry VIX)."""
    try:
        import yfinance as yf
        df = yf.Ticker("^VIX").history(period="6mo", interval="1d", auto_adjust=False)
        if df is not None and not df.empty:
            return df[~df.index.duplicated(keep="last")]
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=60)
def fetch_latest_quote(symbol: str, api_key: str, secret_key: str) -> Optional[float]:
    """
    Fetch the latest trade price from Alpaca REST directly.
    Returns None if unavailable.
    """
    if not (api_key and secret_key):
        return None
    try:
        url = f"https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest?feed=iex"
        req = urllib.request.Request(
            url,
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": secret_key,
            },
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        return float(data["trade"]["p"])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# TECHNICAL INDICATORS
# ---------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate EMA, RSI, MACD, ATR, and volume average."""
    df = df.copy()
    if df.empty:
        return df

    # EMAs
    for span in [8, 21, 50, 200]:
        df[f"EMA_{span}"] = df["Close"].ewm(span=span, adjust=False).mean()

    # RSI (14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD (12/26/9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    # ATR (14)
    hl  = df["High"] - df["Low"]
    hpc = (df["High"] - df["Close"].shift()).abs()
    lpc = (df["Low"]  - df["Close"].shift()).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # Volume average (20 bars)
    df["VOL_AVG_20"] = df["Volume"].rolling(20).mean() if "Volume" in df.columns else np.nan

    return df


def attach_daily_vix(df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """Merge daily VIX close onto any timeframe dataframe."""
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

    vix_map = (
        vix[["Close", "VIX_DATE"]]
        .drop_duplicates(subset="VIX_DATE")
        .set_index("VIX_DATE")["Close"]
    )
    temp["VIX_CLOSE"] = temp["BAR_DATE"].map(vix_map).ffill()
    temp.drop(columns=["BAR_DATE"], inplace=True)
    return temp


# ---------------------------------------------------------------------------
# SIGNAL ENGINE
# ---------------------------------------------------------------------------
def safe_round(x: Any, digits: int = 2) -> Optional[float]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        return round(float(x), digits)
    except (ValueError, TypeError):
        return None


def fmt_price(x: Any) -> str:
    v = safe_round(x, 2)
    return f"${v:,.2f}" if v is not None else "N/A"


def signal_color(signal: str) -> str:
    return {"BUY": "green", "HOLD": "blue", "SELL": "red", "NO TRADE": "orange"}.get(signal, "gray")


def timeframe_bias(df: pd.DataFrame) -> int:
    if df.empty or len(df) < 50:
        return 0
    row = df.iloc[-1]
    score = 0
    score += 1 if row["Close"] > row["EMA_21"] else -1
    score += 1 if row["EMA_21"] > row["EMA_50"] else -1
    if row["RSI"] > 52:
        score += 1
    elif row["RSI"] < 45:
        score -= 1
    score += 1 if row["MACD_HIST"] > 0 else -1
    return score


def detect_market_regime(daily_df: pd.DataFrame, vix_value: float) -> str:
    if daily_df.empty or len(daily_df) < 200:
        return "Insufficient Data"
    row = daily_df.iloc[-1]
    bullish = (row["Close"] > row["EMA_50"] and row["EMA_50"] > row["EMA_200"] and row["RSI"] > 52)
    bearish = (row["Close"] < row["EMA_50"] and row["EMA_50"] < row["EMA_200"] and row["RSI"] < 48)
    if bullish and vix_value < 18:
        return "Bull Trend"
    if bullish and vix_value >= 18:
        return "Bull Trend / High Vol"
    if bearish and vix_value >= 20:
        return "Bear Trend"
    if bearish and vix_value < 20:
        return "Bear Trend / Low Vol"
    return "Range / Transition"


def current_signal(
    entry_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    vix_value: float,
) -> Dict[str, Any]:
    """Score the current bar and return a full signal dict."""
    if entry_df.empty or len(entry_df) < 50:
        return {
            "signal": "NO TRADE", "score": 0, "confidence": 0,
            "risk": "Unknown", "stop": None, "target": None,
            "regime": "Insufficient Data", "reasons": ["Not enough data."],
        }

    row = entry_df.iloc[-1]
    score = 0
    reasons = []

    # --- Trend ---
    if row["Close"] > row["EMA_21"]:
        score += 1; reasons.append("Price is above EMA21.")
    else:
        score -= 1; reasons.append("Price is below EMA21.")

    if row["EMA_21"] > row["EMA_50"]:
        score += 1; reasons.append("EMA21 is above EMA50.")
    else:
        score -= 1; reasons.append("EMA21 is below EMA50.")

    if not pd.isna(row["EMA_200"]):
        if row["EMA_50"] > row["EMA_200"]:
            score += 1; reasons.append("EMA50 is above EMA200.")
        else:
            score -= 1; reasons.append("EMA50 is below EMA200.")

    # --- Momentum ---
    if 52 <= row["RSI"] <= 68:
        score += 1; reasons.append("RSI is in a healthy bullish zone.")
    elif row["RSI"] < 45:
        score -= 1; reasons.append("RSI is weak.")
    elif row["RSI"] > 72:
        score -= 1; reasons.append("RSI is stretched / overheated.")

    if row["MACD_HIST"] > 0:
        score += 1; reasons.append("MACD histogram is positive.")
    else:
        score -= 1; reasons.append("MACD histogram is negative.")

    # --- Volume ---
    if "Volume" in entry_df.columns and not pd.isna(row["VOL_AVG_20"]):
        if row["Volume"] > row["VOL_AVG_20"]:
            score += 1; reasons.append("Volume is above the 20-bar average.")
        else:
            reasons.append("Volume is not strongly confirming.")

    # --- VIX ---
    if vix_value < 18:
        score += 1; reasons.append("VIX is supportive (< 18).")
    elif vix_value > 24:
        score -= 2; reasons.append(f"VIX is elevated ({vix_value:.1f}) — raises risk.")
    else:
        reasons.append(f"VIX is neutral ({vix_value:.1f}).")

    # --- Higher timeframe alignment ---
    htf = timeframe_bias(hourly_df)
    dtf = timeframe_bias(daily_df)

    if htf >= 2:
        score += 1; reasons.append("Hourly trend confirms.")
    elif htf <= -2:
        score -= 1; reasons.append("Hourly trend disagrees.")

    if dtf >= 2:
        score += 2; reasons.append("Daily trend confirms.")
    elif dtf <= -2:
        score -= 2; reasons.append("Daily trend disagrees.")

    # --- Extension filter ---
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

    atr   = row["ATR"] if not pd.isna(row["ATR"]) else None
    close = row["Close"]
    stop  = target = None
    risk  = "Medium"

    if atr and atr > 0:
        if signal == "BUY":
            stop   = close - 1.2 * atr
            target = close + 2.0 * atr
        elif signal == "SELL":
            stop   = close + 1.2 * atr
            target = close - 2.0 * atr

    if vix_value > 24:
        risk = "High"
    elif vix_value < 18 and "Bull" in regime:
        risk = "Low"

    confidence = min(95, max(5, 50 + score * 6))

    return {
        "signal": signal, "score": int(score), "confidence": int(confidence),
        "risk": risk, "stop": stop, "target": target,
        "regime": regime, "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# VECTORISED SIGNAL SCORING  (for backtest + chart arrows)
# ---------------------------------------------------------------------------
def vector_signal_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["score"] = 0

    out["score"] += np.where(out["Close"] > out["EMA_21"], 1, -1)
    out["score"] += np.where(out["EMA_21"] > out["EMA_50"], 1, -1)
    out["score"] += np.where(out["EMA_50"] > out["EMA_200"], 1, -1)

    out["score"] += np.where((out["RSI"] >= 52) & (out["RSI"] <= 68), 1, 0)
    out["score"] += np.where(out["RSI"] < 45, -1, 0)
    out["score"] += np.where(out["RSI"] > 72, -1, 0)

    out["score"] += np.where(out["MACD_HIST"] > 0, 1, -1)

    if "Volume" in out.columns and "VOL_AVG_20" in out.columns:
        out["score"] += np.where(out["Volume"] > out["VOL_AVG_20"], 1, 0)

    if "VIX_CLOSE" in out.columns:
        out["score"] += np.where(out["VIX_CLOSE"] < 18, 1, 0)
        out["score"] += np.where(out["VIX_CLOSE"] > 24, -2, 0)

    atr_dist = abs(out["Close"] - out["EMA_21"]) / out["ATR"].replace(0, np.nan)
    out["extended"] = atr_dist > 1.8
    out["score"] += np.where(out["extended"], -1, 0)

    out["signal_label"] = np.select(
        [out["score"] >= 6, out["score"] <= -4, (out["score"] >= 2) & (out["score"] <= 5)],
        ["BUY", "SELL", "HOLD"],
        default="NO TRADE",
    )
    return out


def find_chart_signals(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    marked = vector_signal_score(df.copy())
    marked["prev_signal"] = marked["signal_label"].shift(1)
    marked["fresh_buy"]  = (marked["signal_label"] == "BUY")  & (marked["prev_signal"] != "BUY")
    marked["fresh_sell"] = (marked["signal_label"] == "SELL") & (marked["prev_signal"] != "SELL")

    buy_rows, sell_rows = [], []
    open_trade = None

    for idx, row in marked.iterrows():
        if row["fresh_buy"]:
            buy_rows.append({
                "index": idx, "Low": row["Low"], "High": row["High"],
                "ATR": row["ATR"], "Close": row["Close"],
                "label": f"BUY<br>{float(row['Close']):.2f}",
            })
            open_trade = {"price": float(row["Close"])}

        elif row["fresh_sell"]:
            label = f"SELL<br>{float(row['Close']):.2f}"
            if open_trade is not None:
                pnl = ((float(row["Close"]) / open_trade["price"]) - 1.0) * 100.0
                label = f"SELL<br>{float(row['Close']):.2f}<br>{pnl:+.2f}%"
                open_trade = None
            sell_rows.append({
                "index": idx, "Low": row["Low"], "High": row["High"],
                "ATR": row["ATR"], "Close": row["Close"], "label": label,
            })

    return pd.DataFrame(buy_rows), pd.DataFrame(sell_rows)


# ---------------------------------------------------------------------------
# BACKTEST
# ---------------------------------------------------------------------------
def run_backtest(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    bt = vector_signal_score(df.copy())
    bt = bt.dropna(subset=["EMA_21", "EMA_50", "EMA_200", "RSI", "MACD_HIST", "ATR"]).copy()

    if bt.empty or len(bt) < 50:
        return None, None

    position, in_pos = [], 0
    for _, row in bt.iterrows():
        enter = row["score"] >= 6 and not row["extended"]
        hold  = row["score"] >= 2 and row["Close"] > row["EMA_50"]
        exit_ = row["score"] <= 1 or row["Close"] < row["EMA_21"]

        if in_pos == 0 and enter:
            in_pos = 1
        elif in_pos == 1 and exit_ and not hold:
            in_pos = 0
        position.append(in_pos)

    bt["position"]      = position
    bt["ret"]           = bt["Close"].pct_change().fillna(0)
    bt["strategy_ret"]  = bt["ret"] * bt["position"].shift(1).fillna(0)
    bt["equity_curve"]  = (1 + bt["strategy_ret"]).cumprod()
    bt["buy_hold_curve"]= (1 + bt["ret"]).cumprod()
    bt["equity_peak"]   = bt["equity_curve"].cummax()
    bt["drawdown"]      = bt["equity_curve"] / bt["equity_peak"] - 1

    bt["pos_change"] = bt["position"].diff().fillna(0)
    entries = bt.index[bt["pos_change"] == 1].tolist()
    exits   = bt.index[bt["pos_change"] == -1].tolist()
    if len(exits) < len(entries):
        exits.append(bt.index[-1])

    trades = []
    for en, ex in zip(entries, exits):
        ep = bt.loc[en, "Close"]
        xp = bt.loc[ex, "Close"]
        trades.append({
            "Entry Time":  en,
            "Exit Time":   ex,
            "Entry Price": round(float(ep), 2),
            "Exit Price":  round(float(xp), 2),
            "Return %":    round(((xp / ep) - 1) * 100, 2),
        })

    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    win_rate  = (trades_df["Return %"] > 0).mean() * 100 if total_trades else 0.0
    avg_trade = trades_df["Return %"].mean()             if total_trades else 0.0

    stats = {
        "Strategy Return %": round((bt["equity_curve"].iloc[-1] - 1) * 100, 2),
        "Buy & Hold %":      round((bt["buy_hold_curve"].iloc[-1] - 1) * 100, 2),
        "Max Drawdown %":    round(bt["drawdown"].min() * 100, 2),
        "Trades":            int(total_trades),
        "Win Rate %":        round(win_rate, 2),
        "Avg Trade %":       round(avg_trade, 2),
    }
    return bt, {"stats": stats, "trades": trades_df}


# ---------------------------------------------------------------------------
# WEBHOOK
# ---------------------------------------------------------------------------
def send_webhook_alert(webhook_url: str, payload: dict) -> Tuple[bool, str]:
    if not webhook_url:
        return False, "Missing webhook URL"
    try:
        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            webhook_url, data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return True, f"HTTP {getattr(resp, 'status', 200)}"
    except urllib.error.HTTPError as e:
        return False, f"HTTPError {e.code}"
    except urllib.error.URLError as e:
        return False, f"URLError {e.reason}"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# CHARTS
# ---------------------------------------------------------------------------
def make_candlestick_chart(df: pd.DataFrame, symbol: str):
    chart_df = df.tail(180).copy()

    # Green/red volume bars matching candle direction
    vol_colors = [
        "rgba(0,180,90,0.30)" if c >= o else "rgba(220,50,50,0.30)"
        for c, o in zip(chart_df["Close"], chart_df["Open"])
    ]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.60, 0.20, 0.20],
        specs=[[{"secondary_y": True}], [{}], [{}]],
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df["Open"], high=chart_df["High"],
        low=chart_df["Low"],   close=chart_df["Close"],
        name="Candles",
    ), row=1, col=1, secondary_y=False)

    # EMAs
    ema_colors = {"EMA_8": "#f59e0b", "EMA_21": "#3b82f6", "EMA_50": "#a855f7", "EMA_200": "#ef4444"}
    for col, color in ema_colors.items():
        if col in chart_df.columns and chart_df[col].notna().sum() > 0:
            fig.add_trace(go.Scatter(
                x=chart_df.index, y=chart_df[col],
                mode="lines", name=col,
                line=dict(color=color, width=1.2),
            ), row=1, col=1, secondary_y=False)

    # Volume — right axis
    fig.add_trace(go.Bar(
        x=chart_df.index, y=chart_df["Volume"],
        name="Volume", marker_color=vol_colors,
    ), row=1, col=1, secondary_y=True)

    # RSI
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["RSI"],
        mode="lines", name="RSI",
        line=dict(color="#06b6d4", width=1.5),
    ), row=2, col=1)
    for level in [70, 30]:
        fig.add_hline(y=level, row=2, col=1, line_dash="dot",
                      line_color="rgba(150,150,150,0.5)")

    # MACD
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["MACD"],
        mode="lines", name="MACD",
        line=dict(color="#3b82f6", width=1.5),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["MACD_SIGNAL"],
        mode="lines", name="Signal",
        line=dict(color="#f59e0b", width=1.2),
    ), row=3, col=1)
    hist_colors = [
        "rgba(0,180,90,0.55)" if v >= 0 else "rgba(220,50,50,0.55)"
        for v in chart_df["MACD_HIST"].fillna(0)
    ]
    fig.add_trace(go.Bar(
        x=chart_df.index, y=chart_df["MACD_HIST"],
        name="Histogram", marker_color=hist_colors,
    ), row=3, col=1)

    # Buy / Sell arrows
    buy_pts, sell_pts = find_chart_signals(chart_df)
    if not buy_pts.empty:
        buy_pts = buy_pts.tail(20)
    if not sell_pts.empty:
        sell_pts = sell_pts.tail(20)

    for _, r in buy_pts.iterrows():
        y_val = r["Low"] - (r["ATR"] * 0.25 if pd.notna(r["ATR"]) else r["Low"] * 0.003)
        fig.add_annotation(
            x=r["index"], y=y_val, xref="x", yref="y",
            text=r["label"], showarrow=True,
            arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor="#00b45a",
            ax=0, ay=40, font=dict(color="#00b45a", size=9), align="center",
        )

    for _, r in sell_pts.iterrows():
        y_val = r["High"] + (r["ATR"] * 0.25 if pd.notna(r["ATR"]) else r["High"] * 0.003)
        fig.add_annotation(
            x=r["index"], y=y_val, xref="x", yref="y",
            text=r["label"], showarrow=True,
            arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor="#e03030",
            ax=0, ay=-44, font=dict(color="#e03030", size=9), align="center",
        )

    fig.update_layout(
        title=f"{symbol} — Price · RSI · MACD",
        xaxis_rangeslider_visible=False,
        height=900,
        legend_orientation="h",
        dragmode="pan",
        hovermode="x unified",
        margin=dict(l=60, r=80, t=50, b=40),
        xaxis=dict(fixedrange=False, showspikes=True, spikemode="across",
                   spikesnap="cursor", spikedash="dot"),
        xaxis2=dict(fixedrange=False, showspikes=True, spikemode="across", spikesnap="cursor"),
        xaxis3=dict(fixedrange=False, showspikes=True, spikemode="across", spikesnap="cursor"),
    )

    # Price — LEFT axis (y fixed so scroll only zooms x)
    fig.update_yaxes(title_text="Price", side="left",  fixedrange=True,
                     row=1, col=1, secondary_y=False)
    # Volume — RIGHT axis, capped at 25 % of pane height
    fig.update_yaxes(title_text="Volume", side="right", fixedrange=True,
                     showgrid=False,
                     range=[0, chart_df["Volume"].max() * 4],
                     row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="RSI",  fixedrange=True, row=2, col=1)
    fig.update_yaxes(title_text="MACD", fixedrange=True, row=3, col=1)

    return fig


def make_backtest_chart(bt_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df["equity_curve"],
                             mode="lines", name="Strategy",
                             line=dict(color="#3b82f6", width=2)))
    fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df["buy_hold_curve"],
                             mode="lines", name="Buy & Hold",
                             line=dict(color="#9ca3af", width=1.5, dash="dot")))
    fig.update_layout(title="Backtest Equity Curve", height=450,
                      hovermode="x unified", dragmode="pan")
    return fig


# ---------------------------------------------------------------------------
# AI ANALYSIS
# ---------------------------------------------------------------------------
def build_ai_prompt(symbol, timeframe, curr_price, sig, last, vix_value):
    return f"""You are a disciplined institutional market strategist.

Analyze this setup in 6 concise bullet points:
1. Market regime summary
2. What favors bulls right now
3. What favors bears right now
4. Recommended action: BUY / HOLD / SELL / NO TRADE — and why
5. Key level that would invalidate the current thesis
6. One tactical note for the next session

Market data:
Ticker: {symbol}
Timeframe: {timeframe}
Price: {curr_price}
Signal: {sig['signal']}  |  Confidence: {sig['confidence']}%  |  Score: {sig['score']}
Risk: {sig['risk']}  |  Regime: {sig['regime']}
RSI: {safe_round(last['RSI'], 2)}  |  ATR: {safe_round(last['ATR'], 2)}
EMA8: {safe_round(last['EMA_8'], 2)}  EMA21: {safe_round(last['EMA_21'], 2)}
EMA50: {safe_round(last['EMA_50'], 2)}  EMA200: {safe_round(last['EMA_200'], 2)}
MACD: {safe_round(last['MACD'], 3)}  Signal: {safe_round(last['MACD_SIGNAL'], 3)}  Hist: {safe_round(last['MACD_HIST'], 3)}
VIX: {vix_value}
Stop: {safe_round(sig['stop'], 2)}  Target: {safe_round(sig['target'], 2)}
"""


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    st.title("📈 SPY Buddy Pro Elite")
    st.caption("Real-time signal dashboard powered by Alpaca Markets. Not financial advice.")

    # ---- Sidebar ----
    with st.sidebar:
        st.header("Controls")

        # Alpaca credentials input (if not in secrets)
        api_key, secret_key = get_alpaca_keys()
        if not api_key or not secret_key:
            st.warning("Alpaca keys not found in secrets. Enter them below:")
            api_key    = st.text_input("Alpaca API Key",    type="password", key="ak")
            secret_key = st.text_input("Alpaca Secret Key", type="password", key="sk")

        symbol    = st.text_input("Ticker", value=DEFAULT_SYMBOL).upper().strip()
        timeframe = st.selectbox("Chart Timeframe", list(TF_MAP.keys()), index=0)

        st.divider()
        st.subheader("Backtest")
        backtest_bars = st.slider("Bars to backtest", 120, 2000, 500, 20)

        st.divider()
        st.subheader("Alerts")
        enable_webhook = st.checkbox("Enable webhook alerts", value=False)
        webhook_url    = st.text_input(
            "Webhook URL", value="", type="password",
            help="Discord, Slack-compatible, or any automation webhook.",
        )

        st.divider()
        show_ai = st.checkbox("Enable AI analysis", value=True)

        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ---- Data fetch ----
    with st.spinner("Fetching market data from Alpaca…"):
        entry_raw  = fetch_alpaca_bars(symbol,  timeframe, api_key or "", secret_key or "")
        hourly_raw = fetch_alpaca_bars(symbol,  "1 Hour",  api_key or "", secret_key or "")
        daily_raw  = fetch_alpaca_bars(symbol,  "1 Day",   api_key or "", secret_key or "")
        vix_raw    = fetch_vix(api_key or "", secret_key or "")

    if entry_raw.empty:
        st.error(f"No data returned for **{symbol}**. Check the ticker or your Alpaca credentials.")
        st.stop()

    if vix_raw.empty:
        st.warning("VIX data unavailable — using default value of 20.")
        vix_raw = pd.DataFrame({"Close": [20.0]}, index=[pd.Timestamp.now()])

    entry_df  = add_indicators(entry_raw)
    hourly_df = add_indicators(hourly_raw)
    daily_df  = add_indicators(daily_raw)
    entry_df  = attach_daily_vix(entry_df, vix_raw)

    if len(entry_df) < 30:
        st.error("Not enough bars to compute indicators on this timeframe.")
        st.stop()

    last  = entry_df.iloc[-1]
    prev  = entry_df.iloc[-2] if len(entry_df) > 1 else last

    # Try to get a fresher quote from Alpaca
    live_price = fetch_latest_quote(symbol, api_key or "", secret_key or "")
    curr_price = safe_round(live_price or last["Close"], 2)
    prev_price = safe_round(prev["Close"], 2)
    change     = round(curr_price - prev_price, 2) if curr_price and prev_price else None
    vix_value  = safe_round(vix_raw["Close"].iloc[-1], 2) or 20.0

    sig = current_signal(entry_df, hourly_df, daily_df, vix_value)

    # ---- Alert logic ----
    alert_key   = f"last_signal_{symbol}_{timeframe}"
    history_key = f"alert_history_{symbol}_{timeframe}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    prev_signal = st.session_state.get(alert_key)
    if prev_signal is None:
        st.session_state[alert_key] = sig["signal"]
    elif prev_signal != sig["signal"]:
        event_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_row  = {
            "Time": event_time, "Old": prev_signal, "New": sig["signal"],
            "Price": curr_price, "Ticker": symbol, "Timeframe": timeframe,
            "Confidence": sig["confidence"], "Risk": sig["risk"],
        }
        st.session_state[history_key].insert(0, alert_row)
        st.session_state[alert_key] = sig["signal"]

        toast_icon = {"BUY": "🟢", "HOLD": "🔵", "SELL": "🔴", "NO TRADE": "🟠"}.get(sig["signal"], "📈")
        st.warning(f"Signal changed: {prev_signal} → {sig['signal']} at {fmt_price(curr_price)}")
        st.toast(f"{symbol} {timeframe}: {prev_signal} → {sig['signal']} at {fmt_price(curr_price)}", icon=toast_icon)

        if enable_webhook and webhook_url:
            payload = {"text": (
                f"{symbol} {timeframe} signal: {prev_signal} → {sig['signal']} | "
                f"Price: {curr_price} | Confidence: {sig['confidence']}% | "
                f"Risk: {sig['risk']} | Regime: {sig['regime']}"
            )}
            ok, msg = send_webhook_alert(webhook_url, payload)
            st.toast("Webhook sent ✅" if ok else f"Webhook failed: {msg}", icon="✅" if ok else "❌")

    # ---- Header metrics ----
    st.subheader(f"Signal: :{signal_color(sig['signal'])}[{sig['signal']}]")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Price",      fmt_price(curr_price), f"{change:+.2f}" if change else None)
    c2.metric("Confidence", f"{sig['confidence']}%")
    c3.metric("Score",      f"{sig['score']}")
    c4.metric("VIX",        f"{vix_value}")
    c5.metric("Risk",       sig["risk"])
    c6.metric("Regime",     sig["regime"])

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("RSI",    f"{safe_round(last['RSI'], 2)}")
    d2.metric("ATR",    fmt_price(safe_round(last["ATR"], 2)))
    d3.metric("Stop",   fmt_price(safe_round(sig["stop"], 2)))
    d4.metric("Target", fmt_price(safe_round(sig["target"], 2)))

    # ---- Tabs ----
    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Backtest", "Alerts", "Raw Data"])

    # ======================== TAB 1 — DASHBOARD ========================
    with tab1:
        with st.expander("How the engine decides"):
            st.markdown("""
**BUY** — Score ≥ 6, regime not bearish, price not over-extended.
**HOLD** — Score 2–5: trend is okay but not strong enough for a fresh entry.
**SELL** — Score ≤ −4: trend and momentum have broken down.
**NO TRADE** — Mixed signals, stretched price, or hostile volatility.
            """)

        st.subheader(f"{symbol} Chart")
        if PLOTLY_AVAILABLE:
            fig = make_candlestick_chart(entry_df, symbol)
            st.plotly_chart(fig, use_container_width=True, config={
                "scrollZoom": True, "displaylogo": False,
                "modeBarButtonsToRemove": ["select2d", "lasso2d"],
            })
            st.caption("Scroll to zoom the time axis · Click-drag to pan · Double-click to reset.")
        else:
            st.warning("Plotly not installed — showing simplified chart.")
            cols = ["Close", "EMA_8", "EMA_21", "EMA_50"]
            if entry_df["EMA_200"].notna().sum() > 0:
                cols.append("EMA_200")
            st.line_chart(entry_df[cols].tail(150))

        left, right = st.columns([1.2, 1])

        with left:
            st.subheader("Why this signal")
            for reason in sig["reasons"]:
                st.write(f"- {reason}")

        with right:
            st.subheader("Latest Snapshot")
            snap = pd.DataFrame({
                "Metric": ["Close", "EMA_8", "EMA_21", "EMA_50", "EMA_200",
                            "RSI", "MACD", "MACD_SIG", "MACD_HIST", "ATR", "VIX", "Volume"],
                "Value": [
                    safe_round(last["Close"], 2), safe_round(last["EMA_8"], 2),
                    safe_round(last["EMA_21"], 2), safe_round(last["EMA_50"], 2),
                    safe_round(last["EMA_200"], 2), safe_round(last["RSI"], 2),
                    safe_round(last["MACD"], 3), safe_round(last["MACD_SIGNAL"], 3),
                    safe_round(last["MACD_HIST"], 3), safe_round(last["ATR"], 2),
                    safe_round(last.get("VIX_CLOSE"), 2),
                    int(last["Volume"]) if not pd.isna(last["Volume"]) else None,
                ],
            })
            st.dataframe(snap, use_container_width=True, hide_index=True)

        # ---- AI Panel ----
        if show_ai:
            st.subheader("🤖 AI Technical Verdict")
            gemini_key = get_gemini_key()

            if gemini_key and GENAI_AVAILABLE:
                if st.button("▶ Run Deep Analysis"):
                    prompt = build_ai_prompt(symbol, timeframe, curr_price, sig, last, vix_value)
                    with st.spinner("Running AI analysis…"):
                        try:
                            client   = genai.Client(api_key=gemini_key)
                            response = client.models.generate_content(
                                model="gemini-2.5-flash", contents=prompt,
                            )
                            st.info(response.text)
                        except Exception as e:
                            st.error(f"AI analysis failed: {e}")
            else:
                st.caption(
                    "To enable AI analysis, add `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) "
                    "to your Streamlit secrets and install `google-genai`."
                )

    # ======================== TAB 2 — BACKTEST ========================
    with tab2:
        st.subheader("Simple Backtest")
        bt_df, bt_result = run_backtest(entry_df.tail(backtest_bars).copy())

        if bt_df is None:
            st.info("Not enough data for backtest on this timeframe / bar count.")
        else:
            s = bt_result["stats"]
            b1, b2, b3, b4, b5, b6 = st.columns(6)
            b1.metric("Strategy Return", f"{s['Strategy Return %']}%")
            b2.metric("Buy & Hold",      f"{s['Buy & Hold %']}%")
            b3.metric("Max Drawdown",    f"{s['Max Drawdown %']}%")
            b4.metric("Trades",          f"{s['Trades']}")
            b5.metric("Win Rate",        f"{s['Win Rate %']}%")
            b6.metric("Avg Trade",       f"{s['Avg Trade %']}%")

            if PLOTLY_AVAILABLE:
                st.plotly_chart(make_backtest_chart(bt_df), use_container_width=True,
                                config={"scrollZoom": True, "displaylogo": False})
            else:
                st.line_chart(bt_df[["equity_curve", "buy_hold_curve"]])

            with st.expander("Trade log"):
                trades_df = bt_result["trades"]
                if trades_df.empty:
                    st.write("No completed trades in this window.")
                else:
                    st.dataframe(trades_df, use_container_width=True)

            st.caption("Simplified backtest — a quick sanity check, not execution-grade research.")

    # ======================== TAB 3 — ALERTS ========================
    with tab3:
        st.subheader("Signal Change Alerts")
        st.caption("Logged whenever the signal changes on refresh.")
        alerts = st.session_state[history_key]
        if alerts:
            st.dataframe(pd.DataFrame(alerts), use_container_width=True)
        else:
            st.write("No signal changes logged yet.")

    # ======================== TAB 4 — RAW DATA ========================
    with tab4:
        st.subheader("Raw Data (last 100 bars)")
        st.dataframe(entry_df.tail(100), use_container_width=True)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Unexpected error: {e}")
