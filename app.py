"""
SPY Buddy Pro Elite  v2
=======================
A polished upgrade of the original dashboard.

What changed vs the original:
  - Dark professional theme via custom CSS (no external dependency)
  - Colour-coded signal badge with glow effect
  - EMA lines each have a distinct colour on the chart
  - Volume bars are green/red matching the candle direction
  - MACD histogram bars are green/red
  - Chart: price axis LEFT, volume axis RIGHT
  - Chart: click-drag = pan  |  scroll = zoom time axis only (no vertical stretch)
  - Crosshair / unified hover across all three panels
  - Backtest equity chart styled to match the dark theme
  - "Why this signal" reasons are colour-coded (bullish green / bearish red / neutral)
  - Latest Snapshot shown as a clean vertical metric table (Metric | Value)
  - VIX fallback: warning instead of hard stop so the rest of the app still loads
  - ImportError caught correctly (not bare Exception) for optional imports
  - `get_history` has a 3-attempt retry with a 1-second back-off
  - AI analysis wrapped in try/except so an API error never crashes the tab
  - `if __name__ == "__main__"` guard added

Not changed:
  - All indicator maths (EMA, RSI, MACD, ATR, VOL_AVG_20)
  - Signal scoring logic and thresholds
  - Backtest engine
  - Webhook logic
  - Tab structure (Dashboard / Backtest / Alerts / Raw Data)
"""

import json
import time
import urllib.request
import urllib.error
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ── Optional Plotly ────────────────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ── Optional Gemini ────────────────────────────────────────────────────────────
try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be the very first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SPY Buddy Pro Elite",
    page_icon="📈",
    layout="wide",
)


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS  – dark trading-terminal theme
# ══════════════════════════════════════════════════════════════════════════════
SIGNAL_GLOW = {
    "BUY":      "0 0 12px 3px rgba(0,200,100,0.55)",
    "HOLD":     "0 0 12px 3px rgba(59,130,246,0.55)",
    "SELL":     "0 0 12px 3px rgba(239,68,68,0.55)",
    "NO TRADE": "0 0 12px 3px rgba(251,146,60,0.55)",
}
SIGNAL_BG = {
    "BUY":      "#052e16",
    "HOLD":     "#1e3a5f",
    "SELL":     "#450a0a",
    "NO TRADE": "#431407",
}
SIGNAL_FG = {
    "BUY":      "#4ade80",
    "HOLD":     "#93c5fd",
    "SELL":     "#f87171",
    "NO TRADE": "#fb923c",
}

st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d1117;
    color: #e6edf3;
}
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"]  { color: #8b949e !important; font-size: 0.78rem; }
[data-testid="stMetricValue"]  { color: #e6edf3 !important; font-size: 1.35rem; font-weight: 700; }
[data-testid="stMetricDelta"]  { font-size: 0.82rem; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
    color: #8b949e;
    border-bottom: 2px solid transparent;
    font-weight: 600;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #58a6ff;
    border-bottom: 2px solid #58a6ff;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: #21262d;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 8px;
    font-weight: 600;
}
[data-testid="stButton"] > button:hover {
    background: #30363d;
    border-color: #58a6ff;
    color: #58a6ff;
}

/* ── Divider ── */
hr { border-color: #30363d; }

/* ── Caption / small text ── */
small, .stCaption { color: #8b949e !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
TF_MAP: Dict[str, Dict[str, str]] = {
    "1 Day":  {"period": "2y",   "interval": "1d"},
    "1 Hour": {"period": "730d", "interval": "1h"},
    "15 Min": {"period": "60d",  "interval": "15m"},
    "5 Min":  {"period": "60d",  "interval": "5m"},
    "1 Min":  {"period": "7d",   "interval": "1m"},
}

DEFAULT_SYMBOL = "SPY"

# Chart colours
EMA_COLORS = {
    "EMA_8":   "#f59e0b",   # amber
    "EMA_21":  "#3b82f6",   # blue
    "EMA_50":  "#a855f7",   # purple
    "EMA_200": "#ef4444",   # red
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def safe_round(x: Any, digits: int = 2) -> Optional[float]:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return round(float(x), digits)
    except (TypeError, ValueError):
        return None


def fmt_price(x: Any) -> str:
    v = safe_round(x, 2)
    return f"${v:,.2f}" if v is not None else "N/A"


def signal_color(signal: str) -> str:
    return {"BUY": "green", "HOLD": "blue", "SELL": "red", "NO TRADE": "orange"}.get(signal, "gray")


def signal_badge(signal: str) -> str:
    """Return an HTML badge with glow for the current signal."""
    bg  = SIGNAL_BG.get(signal, "#1c1c1c")
    fg  = SIGNAL_FG.get(signal, "#e6edf3")
    glow = SIGNAL_GLOW.get(signal, "none")
    return (
        f'<span style="background:{bg};color:{fg};padding:6px 20px;'
        f'border-radius:8px;font-size:1.5rem;font-weight:800;'
        f'box-shadow:{glow};letter-spacing:0.05em;">{signal}</span>'
    )


def reason_icon(reason: str) -> str:
    """Prefix a reason string with a coloured icon based on sentiment."""
    bull = ("above", "confirms", "healthy", "positive", "supportive", "volume is above")
    bear = ("below", "disagrees", "weak", "negative", "elevated", "stretched", "extended")
    lo   = reason.lower()
    if any(k in lo for k in bull):
        return f"🟢 {reason}"
    if any(k in lo for k in bear):
        return f"🔴 {reason}"
    return f"⚪ {reason}"


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


# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════════════════
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df

    for span in (8, 21, 50, 200):
        df[f"EMA_{span}"] = df["Close"].ewm(span=span, adjust=False).mean()

    delta    = df["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"]   = df["MACD"] - df["MACD_SIGNAL"]

    hl  = df["High"] - df["Low"]
    hpc = (df["High"] - df["Close"].shift()).abs()
    lpc = (df["Low"]  - df["Close"].shift()).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    df["VOL_AVG_20"] = df["Volume"].rolling(20).mean() if "Volume" in df.columns else np.nan

    return df


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

    vix_map = (
        vix[["Close", "VIX_DATE"]]
        .drop_duplicates(subset="VIX_DATE")
        .set_index("VIX_DATE")["Close"]
    )
    temp["VIX_CLOSE"] = temp["BAR_DATE"].map(vix_map).ffill()
    temp.drop(columns=["BAR_DATE"], inplace=True)
    return temp


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL ENGINE  (unchanged logic from original)
# ══════════════════════════════════════════════════════════════════════════════
def timeframe_bias(df: pd.DataFrame) -> int:
    if df.empty or len(df) < 50:
        return 0
    row   = df.iloc[-1]
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
    row     = daily_df.iloc[-1]
    bullish = row["Close"] > row["EMA_50"] and row["EMA_50"] > row["EMA_200"] and row["RSI"] > 52
    bearish = row["Close"] < row["EMA_50"] and row["EMA_50"] < row["EMA_200"] and row["RSI"] < 48
    if bullish and vix_value < 18:  return "Bull Trend"
    if bullish:                     return "Bull Trend / High Vol"
    if bearish and vix_value >= 20: return "Bear Trend"
    if bearish:                     return "Bear Trend / Low Vol"
    return "Range / Transition"


def current_signal(
    entry_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    vix_value: float,
) -> Dict[str, Any]:
    if entry_df.empty or len(entry_df) < 50:
        return {
            "signal": "NO TRADE", "score": 0, "confidence": 0,
            "risk": "Unknown", "stop": None, "target": None,
            "regime": "Insufficient Data", "reasons": ["Not enough data."],
        }

    row     = entry_df.iloc[-1]
    score   = 0
    reasons = []

    # Trend
    if row["Close"] > row["EMA_21"]:
        score += 1;  reasons.append("Price is above EMA21.")
    else:
        score -= 1;  reasons.append("Price is below EMA21.")

    if row["EMA_21"] > row["EMA_50"]:
        score += 1;  reasons.append("EMA21 is above EMA50.")
    else:
        score -= 1;  reasons.append("EMA21 is below EMA50.")

    if not pd.isna(row["EMA_200"]):
        if row["EMA_50"] > row["EMA_200"]:
            score += 1;  reasons.append("EMA50 is above EMA200.")
        else:
            score -= 1;  reasons.append("EMA50 is below EMA200.")

    # Momentum
    if 52 <= row["RSI"] <= 68:
        score += 1;  reasons.append("RSI is in a healthy bullish zone.")
    elif row["RSI"] < 45:
        score -= 1;  reasons.append("RSI is weak.")
    elif row["RSI"] > 72:
        score -= 1;  reasons.append("RSI is stretched / overheated.")

    if row["MACD_HIST"] > 0:
        score += 1;  reasons.append("MACD histogram is positive.")
    else:
        score -= 1;  reasons.append("MACD histogram is negative.")

    # Volume
    if "Volume" in entry_df.columns and not pd.isna(row["VOL_AVG_20"]):
        if row["Volume"] > row["VOL_AVG_20"]:
            score += 1;  reasons.append("Volume is above the 20-bar average.")
        else:
            reasons.append("Volume is not strongly confirming.")

    # VIX
    if vix_value < 18:
        score += 1;  reasons.append(f"VIX is supportive ({vix_value:.1f}).")
    elif vix_value > 24:
        score -= 2;  reasons.append(f"VIX is elevated ({vix_value:.1f}) — raises risk.")
    else:
        reasons.append(f"VIX is neutral ({vix_value:.1f}).")

    # Higher timeframe alignment
    htf = timeframe_bias(hourly_df)
    dtf = timeframe_bias(daily_df)

    if htf >= 2:
        score += 1;  reasons.append("Hourly trend confirms.")
    elif htf <= -2:
        score -= 1;  reasons.append("Hourly trend disagrees.")

    if dtf >= 2:
        score += 2;  reasons.append("Daily trend confirms.")
    elif dtf <= -2:
        score -= 2;  reasons.append("Daily trend disagrees.")

    # Extension filter
    extended = False
    if not pd.isna(row["ATR"]) and row["ATR"] > 0:
        if abs(row["Close"] - row["EMA_21"]) / row["ATR"] > 1.8:
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


def vector_signal_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["score"] = 0

    out["score"] += np.where(out["Close"] > out["EMA_21"],  1, -1)
    out["score"] += np.where(out["EMA_21"] > out["EMA_50"], 1, -1)
    out["score"] += np.where(out["EMA_50"] > out["EMA_200"],1, -1)

    out["score"] += np.where((out["RSI"] >= 52) & (out["RSI"] <= 68), 1, 0)
    out["score"] += np.where(out["RSI"] < 45,  -1, 0)
    out["score"] += np.where(out["RSI"] > 72,  -1, 0)

    out["score"] += np.where(out["MACD_HIST"] > 0, 1, -1)

    if "Volume" in out.columns and "VOL_AVG_20" in out.columns:
        out["score"] += np.where(out["Volume"] > out["VOL_AVG_20"], 1, 0)

    if "VIX_CLOSE" in out.columns:
        out["score"] += np.where(out["VIX_CLOSE"] < 18,  1,  0)
        out["score"] += np.where(out["VIX_CLOSE"] > 24, -2,  0)

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
    marked["fresh_buy"]   = (marked["signal_label"] == "BUY")  & (marked["prev_signal"] != "BUY")
    marked["fresh_sell"]  = (marked["signal_label"] == "SELL") & (marked["prev_signal"] != "SELL")

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
                pnl   = ((float(row["Close"]) / open_trade["price"]) - 1.0) * 100.0
                label = f"SELL<br>{float(row['Close']):.2f}<br>{pnl:+.2f}%"
                open_trade = None
            sell_rows.append({
                "index": idx, "Low": row["Low"], "High": row["High"],
                "ATR": row["ATR"], "Close": row["Close"], "label": label,
            })

    return pd.DataFrame(buy_rows), pd.DataFrame(sell_rows)


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST  (unchanged logic)
# ══════════════════════════════════════════════════════════════════════════════
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

    bt["position"]       = position
    bt["ret"]            = bt["Close"].pct_change().fillna(0)
    bt["strategy_ret"]   = bt["ret"] * bt["position"].shift(1).fillna(0)
    bt["equity_curve"]   = (1 + bt["strategy_ret"]).cumprod()
    bt["buy_hold_curve"] = (1 + bt["ret"]).cumprod()
    bt["equity_peak"]    = bt["equity_curve"].cummax()
    bt["drawdown"]       = bt["equity_curve"] / bt["equity_peak"] - 1

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

    trades_df    = pd.DataFrame(trades)
    total_trades = len(trades_df)
    win_rate     = (trades_df["Return %"] > 0).mean() * 100 if total_trades else 0.0
    avg_trade    = trades_df["Return %"].mean()             if total_trades else 0.0

    stats = {
        "Strategy Return %": round((bt["equity_curve"].iloc[-1]   - 1) * 100, 2),
        "Buy & Hold %":      round((bt["buy_hold_curve"].iloc[-1] - 1) * 100, 2),
        "Max Drawdown %":    round(bt["drawdown"].min() * 100, 2),
        "Trades":            int(total_trades),
        "Win Rate %":        round(win_rate, 2),
        "Avg Trade %":       round(avg_trade, 2),
    }
    return bt, {"stats": stats, "trades": trades_df}


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
_DARK_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(color="#c9d1d9", family="monospace"),
    xaxis=dict(
        gridcolor="#21262d", zerolinecolor="#30363d",
        fixedrange=False,
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikecolor="#58a6ff", spikedash="dot", spikethickness=1,
    ),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
    legend=dict(
        orientation="h", bgcolor="rgba(13,17,23,0.8)",
        bordercolor="#30363d", borderwidth=1,
        font=dict(size=11),
    ),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#161b22", bordercolor="#30363d", font_color="#e6edf3"),
    dragmode="pan",
    margin=dict(l=60, r=80, t=50, b=40),
)


def make_candlestick_chart(df: pd.DataFrame, symbol: str):
    chart_df = df.tail(180).copy()

    vol_colors = [
        "rgba(0,200,100,0.35)" if c >= o else "rgba(239,68,68,0.35)"
        for c, o in zip(chart_df["Close"], chart_df["Open"])
    ]
    hist_colors = [
        "rgba(0,200,100,0.60)" if v >= 0 else "rgba(239,68,68,0.60)"
        for v in chart_df["MACD_HIST"].fillna(0)
    ]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.60, 0.20, 0.20],
        specs=[[{"secondary_y": True}], [{}], [{}]],
    )

    # ── Candlesticks ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df["Open"], high=chart_df["High"],
        low=chart_df["Low"],   close=chart_df["Close"],
        name="Price",
        increasing_line_color="#00c864", increasing_fillcolor="#00c864",
        decreasing_line_color="#ef4444", decreasing_fillcolor="#ef4444",
    ), row=1, col=1, secondary_y=False)

    # ── EMAs ──────────────────────────────────────────────────────────────────
    for col, color in EMA_COLORS.items():
        if col in chart_df.columns and chart_df[col].notna().sum() > 0:
            fig.add_trace(go.Scatter(
                x=chart_df.index, y=chart_df[col],
                mode="lines", name=col,
                line=dict(color=color, width=1.4),
            ), row=1, col=1, secondary_y=False)

    # ── Volume — right axis ───────────────────────────────────────────────────
    fig.add_trace(go.Bar(
        x=chart_df.index, y=chart_df["Volume"],
        name="Volume", marker_color=vol_colors,
    ), row=1, col=1, secondary_y=True)

    # ── RSI ───────────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["RSI"],
        mode="lines", name="RSI",
        line=dict(color="#06b6d4", width=1.6),
    ), row=2, col=1)
    for lvl, clr in [(70, "rgba(239,68,68,0.4)"), (30, "rgba(0,200,100,0.4)")]:
        fig.add_hline(y=lvl, row=2, col=1,
                      line_dash="dot", line_color=clr, line_width=1)

    # ── MACD ──────────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["MACD"],
        mode="lines", name="MACD",
        line=dict(color="#3b82f6", width=1.6),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["MACD_SIGNAL"],
        mode="lines", name="Signal",
        line=dict(color="#f59e0b", width=1.2),
    ), row=3, col=1)
    fig.add_trace(go.Bar(
        x=chart_df.index, y=chart_df["MACD_HIST"],
        name="Histogram", marker_color=hist_colors,
    ), row=3, col=1)

    # ── Buy / Sell arrows ─────────────────────────────────────────────────────
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
            arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor="#00c864",
            ax=0, ay=40, font=dict(color="#00c864", size=9), align="center",
        )

    for _, r in sell_pts.iterrows():
        y_val = r["High"] + (r["ATR"] * 0.25 if pd.notna(r["ATR"]) else r["High"] * 0.003)
        fig.add_annotation(
            x=r["index"], y=y_val, xref="x", yref="y",
            text=r["label"], showarrow=True,
            arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor="#ef4444",
            ax=0, ay=-44, font=dict(color="#ef4444", size=9), align="center",
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    layout = dict(**_DARK_LAYOUT)
    layout.update(
        title=dict(text=f"{symbol} — Price · RSI · MACD", font=dict(size=15, color="#e6edf3")),
        xaxis_rangeslider_visible=False,
        height=900,
        xaxis2=dict(gridcolor="#21262d", fixedrange=False,
                    showspikes=True, spikemode="across", spikesnap="cursor",
                    spikecolor="#58a6ff", spikedash="dot"),
        xaxis3=dict(gridcolor="#21262d", fixedrange=False,
                    showspikes=True, spikemode="across", spikesnap="cursor",
                    spikecolor="#58a6ff", spikedash="dot"),
    )
    fig.update_layout(**layout)

    # Price — LEFT, y locked so scroll only zooms x
    fig.update_yaxes(title_text="Price",  side="left",  fixedrange=True,
                     gridcolor="#21262d", zerolinecolor="#30363d",
                     row=1, col=1, secondary_y=False)
    # Volume — RIGHT, capped at 25 % of pane height
    fig.update_yaxes(title_text="Volume", side="right", fixedrange=True,
                     showgrid=False, range=[0, chart_df["Volume"].max() * 4],
                     row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="RSI",  fixedrange=True,
                     gridcolor="#21262d", row=2, col=1)
    fig.update_yaxes(title_text="MACD", fixedrange=True,
                     gridcolor="#21262d", row=3, col=1)

    return fig


def make_backtest_chart(bt_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bt_df.index, y=bt_df["equity_curve"],
        mode="lines", name="Strategy",
        line=dict(color="#3b82f6", width=2.2),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.07)",
    ))
    fig.add_trace(go.Scatter(
        x=bt_df.index, y=bt_df["buy_hold_curve"],
        mode="lines", name="Buy & Hold",
        line=dict(color="#6b7280", width=1.4, dash="dot"),
    ))
    layout = dict(**_DARK_LAYOUT)
    layout.update(
        title=dict(text="Backtest Equity Curve", font=dict(size=14, color="#e6edf3")),
        height=450,
    )
    fig.update_layout(**layout)
    fig.update_yaxes(gridcolor="#21262d", zerolinecolor="#30363d")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING  (with retry)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=60)
def get_history(symbol: str, period: str, interval: str, retries: int = 3) -> pd.DataFrame:
    last_err = None
    for attempt in range(retries):
        try:
            df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
            if df is not None and not df.empty:
                df = df[~df.index.duplicated(keep="last")]
                return df
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(1)
    return pd.DataFrame()


@st.cache_data(ttl=60)
def get_all_data(symbol: str, timeframe_label: str):
    cfg    = TF_MAP[timeframe_label]
    entry  = get_history(symbol,  cfg["period"], cfg["interval"])
    hourly = get_history(symbol,  "60d",  "1h")
    daily  = get_history(symbol,  "2y",   "1d")
    vix    = get_history("^VIX",  "6mo",  "1d")
    return entry, hourly, daily, vix


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        '<h1 style="color:#e6edf3;margin-bottom:0">📈 SPY Buddy Pro Elite</h1>',
        unsafe_allow_html=True,
    )
    st.caption("Research / education dashboard. Not financial advice.")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Controls")
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
            help="Paste a Discord, Slack-compatible, or automation webhook URL.",
        )

        st.divider()
        show_ai = st.checkbox("Enable AI panel", value=True)

        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ── Data ──────────────────────────────────────────────────────────────────
    with st.spinner("Fetching market data…"):
        entry_raw, hourly_raw, daily_raw, vix_raw = get_all_data(symbol, timeframe)

    if entry_raw.empty:
        st.error(f"No data returned for **{symbol}**. Check the ticker symbol or your network.")
        st.stop()

    if vix_raw.empty:
        st.warning("VIX data unavailable — using a default value of 20.")
        vix_raw = pd.DataFrame({"Close": [20.0]}, index=[pd.Timestamp.now()])

    entry_df  = add_indicators(entry_raw)
    hourly_df = add_indicators(hourly_raw)
    daily_df  = add_indicators(daily_raw)
    entry_df  = attach_daily_vix(entry_df, vix_raw)

    if len(entry_df) < 30:
        st.error("Not enough bars to compute indicators on this timeframe.")
        st.stop()

    last       = entry_df.iloc[-1]
    prev       = entry_df.iloc[-2] if len(entry_df) > 1 else last
    curr_price = safe_round(last["Close"], 2)
    prev_price = safe_round(prev["Close"], 2)
    change     = round(curr_price - prev_price, 2) if curr_price and prev_price else None
    vix_value  = safe_round(vix_raw["Close"].iloc[-1], 2) or 20.0

    sig = current_signal(entry_df, hourly_df, daily_df, vix_value)

    # ── Alert logic ───────────────────────────────────────────────────────────
    alert_key   = f"last_signal_{symbol}_{timeframe}"
    history_key = f"alert_history_{symbol}_{timeframe}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    prev_signal = st.session_state.get(alert_key)
    if prev_signal is None:
        st.session_state[alert_key] = sig["signal"]
    elif prev_signal != sig["signal"]:
        event_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state[history_key].insert(0, {
            "Time": event_time, "Old Signal": prev_signal,
            "New Signal": sig["signal"], "Price": curr_price,
            "Ticker": symbol, "Timeframe": timeframe,
            "Confidence": sig["confidence"], "Risk": sig["risk"],
        })
        st.session_state[alert_key] = sig["signal"]
        toast_icon = {"BUY": "🟢", "HOLD": "🔵", "SELL": "🔴", "NO TRADE": "🟠"}.get(sig["signal"], "📈")
        st.warning(f"Signal changed: {prev_signal} → {sig['signal']} at {fmt_price(curr_price)}")
        st.toast(f"{symbol} {timeframe}: {prev_signal} → {sig['signal']} at {fmt_price(curr_price)}", icon=toast_icon)

        if enable_webhook and webhook_url:
            payload = {"text": (
                f"{symbol} {timeframe} signal changed: {prev_signal} → {sig['signal']} | "
                f"Price: {curr_price} | Confidence: {sig['confidence']}% | "
                f"Risk: {sig['risk']} | Regime: {sig['regime']}"
            )}
            ok, msg = send_webhook_alert(webhook_url, payload)
            st.toast("Webhook alert sent ✅" if ok else f"Webhook failed: {msg}",
                     icon="✅" if ok else "❌")

    # ── Signal badge + top metrics ─────────────────────────────────────────────
    badge_col, _ = st.columns([1, 3])
    with badge_col:
        st.markdown(
            f'<div style="margin:8px 0 16px 0">{signal_badge(sig["signal"])}</div>',
            unsafe_allow_html=True,
        )

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

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔁 Backtest", "🔔 Alerts", "📋 Raw Data"])

    # ════════════════════════ TAB 1 — DASHBOARD ═══════════════════════════════
    with tab1:
        with st.expander("How the engine decides"):
            st.markdown("""
| Signal | Condition |
|--------|-----------|
| **BUY** | Score ≥ 6, regime not bearish, price not over-extended |
| **HOLD** | Score 2–5: trend is okay but not strong enough for a fresh entry |
| **SELL** | Score ≤ −4: trend and momentum have broken down |
| **NO TRADE** | Mixed signals, stretched price, or hostile volatility |
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
            fallback = ["Close", "EMA_8", "EMA_21", "EMA_50"]
            if entry_df["EMA_200"].notna().sum() > 0:
                fallback.append("EMA_200")
            st.line_chart(entry_df[fallback].tail(150))

        left, right = st.columns([1.2, 1])

        with left:
            st.subheader("Why this signal")
            for reason in sig["reasons"]:
                st.markdown(reason_icon(reason))

        with right:
            st.subheader("Latest Snapshot")
            snap_rows = [
                ("Close",     safe_round(last["Close"], 2)),
                ("EMA 8",     safe_round(last["EMA_8"], 2)),
                ("EMA 21",    safe_round(last["EMA_21"], 2)),
                ("EMA 50",    safe_round(last["EMA_50"], 2)),
                ("EMA 200",   safe_round(last["EMA_200"], 2)),
                ("RSI",       safe_round(last["RSI"], 2)),
                ("MACD",      safe_round(last["MACD"], 3)),
                ("MACD Sig",  safe_round(last["MACD_SIGNAL"], 3)),
                ("MACD Hist", safe_round(last["MACD_HIST"], 3)),
                ("ATR",       safe_round(last["ATR"], 2)),
                ("VIX",       safe_round(last.get("VIX_CLOSE"), 2)),
                ("Volume",    int(last["Volume"]) if not pd.isna(last["Volume"]) else None),
            ]
            st.dataframe(
                pd.DataFrame(snap_rows, columns=["Metric", "Value"]),
                use_container_width=True, hide_index=True,
            )

        # ── AI Panel ──────────────────────────────────────────────────────────
        if show_ai:
            st.subheader("🤖 AI Technical Verdict")
            api_key = None
            try:
                api_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
            except Exception:
                pass

            if api_key and GENAI_AVAILABLE:
                if st.button("▶ Run Deep Analysis"):
                    prompt = f"""
You are a disciplined institutional market strategist.

Analyze this setup in 6 short bullet points:
1. Market regime
2. What favors bulls
3. What favors bears
4. Best action now: BUY / HOLD / SELL / NO TRADE
5. Key invalidation level
6. Short tactical note for the next session

Data:
Ticker: {symbol}  |  Timeframe: {timeframe}  |  Price: {curr_price}
Signal: {sig['signal']}  |  Confidence: {sig['confidence']}%  |  Score: {sig['score']}
Risk: {sig['risk']}  |  Regime: {sig['regime']}
RSI: {safe_round(last['RSI'], 2)}  |  ATR: {safe_round(last['ATR'], 2)}
EMA8: {safe_round(last['EMA_8'], 2)}  EMA21: {safe_round(last['EMA_21'], 2)}
EMA50: {safe_round(last['EMA_50'], 2)}  EMA200: {safe_round(last['EMA_200'], 2)}
MACD: {safe_round(last['MACD'], 3)}  Signal: {safe_round(last['MACD_SIGNAL'], 3)}  Hist: {safe_round(last['MACD_HIST'], 3)}
VIX: {vix_value}  |  Stop: {safe_round(sig['stop'], 2)}  |  Target: {safe_round(sig['target'], 2)}
"""
                    with st.spinner("Running AI analysis…"):
                        try:
                            client   = genai.Client(api_key=api_key)
                            response = client.models.generate_content(
                                model="gemini-2.5-flash", contents=prompt,
                            )
                            st.info(response.text)
                        except Exception as e:
                            st.error(f"AI analysis failed: {e}")
            else:
                st.caption(
                    "To enable AI, install `google-genai` and add "
                    "`GEMINI_API_KEY` or `GOOGLE_API_KEY` to Streamlit secrets."
                )

    # ════════════════════════ TAB 2 — BACKTEST ════════════════════════════════
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

            with st.expander("Backtest trade log"):
                trades_df = bt_result["trades"]
                if trades_df.empty:
                    st.write("No completed trades in this window.")
                else:
                    st.dataframe(trades_df, use_container_width=True)

            st.caption("Intentionally simple — a fast sanity check, not execution-grade research.")

    # ════════════════════════ TAB 3 — ALERTS ══════════════════════════════════
    with tab3:
        st.subheader("Signal Alerts")
        st.caption("Logged whenever the signal changes on refresh / rerun.")
        alerts = st.session_state[history_key]
        if alerts:
            st.dataframe(pd.DataFrame(alerts), use_container_width=True)
        else:
            st.info("No signal changes logged yet.")

    # ════════════════════════ TAB 4 — RAW DATA ════════════════════════════════
    with tab4:
        st.subheader("Raw Data (last 100 bars)")
        st.dataframe(entry_df.tail(100), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
