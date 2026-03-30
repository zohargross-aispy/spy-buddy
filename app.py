"""
SPY Buddy Options V2 — Dark Edition
=====================================
100% faithful to the original app(8).py logic.
Changes:
  • Dark terminal theme (no white anywhere)
  • Colour-coded state badge with glow
  • Metric cards styled for dark background
  • Chart: dark bg, bright candles, coloured EMAs, volume bars, RSI panel added
  • Quality score rendered as a coloured progress bar
  • News cards with styled borders instead of plain markdown
  • Underlying bias badge (BULLISH / BEARISH / NEUTRAL) with colour
  • Trade plan section uses a cleaner card layout
  • All original logic, session state, Alpaca calls, and order flow untouched
"""

import math
from typing import Any, Dict, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="SPY Buddy Options V2", page_icon="🎯", layout="wide")

# ── Dark theme CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main { background: #0d1117 !important; color: #e6edf3 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #161b22 !important; border-right: 1px solid #30363d; }
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select { background: #21262d !important; border-color: #30363d !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: .78rem !important; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 1.25rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: .82rem !important; }

/* ── Tabs ── */
[data-testid="stTabs"] button { color: #8b949e !important; border-bottom: 2px solid transparent !important; font-weight: 600 !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #58a6ff !important; border-bottom: 2px solid #58a6ff !important; }

/* ── Expanders ── */
[data-testid="stExpander"] { background: #161b22 !important; border: 1px solid #30363d !important; border-radius: 8px !important; }
[data-testid="stExpander"] summary { color: #c9d1d9 !important; }

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: #21262d !important; color: #c9d1d9 !important;
    border: 1px solid #30363d !important; border-radius: 8px !important; font-weight: 600 !important;
}
[data-testid="stButton"] > button:hover {
    background: #30363d !important; border-color: #58a6ff !important; color: #58a6ff !important;
}

/* ── Inputs / selects ── */
input, select, textarea,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] select {
    background: #21262d !important; color: #e6edf3 !important;
    border: 1px solid #30363d !important; border-radius: 6px !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] { border: 1px solid #30363d !important; border-radius: 8px !important; }

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 8px !important; }

/* ── Dividers ── */
hr { border-color: #30363d !important; }

/* ── Captions / small text ── */
small, .stCaption, [data-testid="stCaptionContainer"] { color: #8b949e !important; }

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div { border-radius: 6px !important; }

/* ── Slider ── */
[data-testid="stSlider"] [data-testid="stTickBar"] { color: #8b949e !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CREDENTIALS
# ══════════════════════════════════════════════════════════════════════════════
ALPACA_KEY    = st.secrets.get("ALPACA_API_KEY",    "")
ALPACA_SECRET = st.secrets.get("ALPACA_SECRET_KEY", "")

PAPER_BASE = "https://paper-api.alpaca.markets"
DATA_BASE  = "https://data.alpaca.markets"

DEFAULT_SYMBOL = "SPY"

# ══════════════════════════════════════════════════════════════════════════════
# COLOUR HELPERS
# ══════════════════════════════════════════════════════════════════════════════
_STATE_BG = {
    "ENTER CALL": "#052e16", "ENTER PUT": "#450a0a",
    "HOLD CALL":  "#1e3a5f", "HOLD PUT":  "#1e3a5f",
    "TP1 HIT":    "#2e1065", "WEAKENING": "#431407",
    "EXIT CALL":  "#431407", "EXIT PUT":  "#431407",
    "NO TRADE":   "#1c1c1c",
}
_STATE_FG = {
    "ENTER CALL": "#4ade80", "ENTER PUT": "#f87171",
    "HOLD CALL":  "#93c5fd", "HOLD PUT":  "#93c5fd",
    "TP1 HIT":    "#c084fc", "WEAKENING": "#fb923c",
    "EXIT CALL":  "#fb923c", "EXIT PUT":  "#fb923c",
    "NO TRADE":   "#6b7280",
}
_STATE_GLOW = {
    "ENTER CALL": "0 0 14px 4px rgba(74,222,128,.55)",
    "ENTER PUT":  "0 0 14px 4px rgba(248,113,113,.55)",
    "HOLD CALL":  "0 0 14px 4px rgba(147,197,253,.4)",
    "HOLD PUT":   "0 0 14px 4px rgba(147,197,253,.4)",
    "TP1 HIT":    "0 0 14px 4px rgba(192,132,252,.55)",
    "WEAKENING":  "0 0 14px 4px rgba(251,146,60,.5)",
    "EXIT CALL":  "0 0 14px 4px rgba(251,146,60,.5)",
    "EXIT PUT":   "0 0 14px 4px rgba(251,146,60,.5)",
    "NO TRADE":   "none",
}
_BIAS_STYLE = {
    "BULLISH": ("🟢", "#4ade80", "#052e16"),
    "BEARISH": ("🔴", "#f87171", "#450a0a"),
    "NEUTRAL": ("⚪", "#9ca3af", "#1c1c1c"),
}
_EMA_COLORS = {"EMA_8": "#f59e0b", "EMA_21": "#3b82f6", "EMA_50": "#a855f7"}


def state_badge(state: str) -> str:
    bg   = _STATE_BG.get(state,   "#1c1c1c")
    fg   = _STATE_FG.get(state,   "#9ca3af")
    glow = _STATE_GLOW.get(state, "none")
    return (f'<span style="background:{bg};color:{fg};padding:10px 28px;'
            f'border-radius:8px;font-size:1.7rem;font-weight:900;'
            f'box-shadow:{glow};letter-spacing:.06em;display:inline-block">'
            f'{state}</span>')


def bias_badge(bias: str) -> str:
    icon, fg, bg = _BIAS_STYLE.get(bias, ("⚪", "#9ca3af", "#1c1c1c"))
    return (f'<span style="background:{bg};color:{fg};padding:6px 18px;'
            f'border-radius:6px;font-size:1.1rem;font-weight:800">'
            f'{icon} {bias}</span>')


def quality_bar(score: int, ok: bool) -> str:
    color = "#4ade80" if ok else "#ef4444"
    label = "PASS" if ok else "FAIL"
    return (f'<div style="background:#21262d;border-radius:6px;height:18px;width:100%;margin-top:4px">'
            f'<div style="background:{color};width:{score}%;height:100%;border-radius:6px;'
            f'transition:width .4s"></div></div>'
            f'<small style="color:{color};font-weight:700">{score}/100 — {label}</small>')


# ══════════════════════════════════════════════════════════════════════════════
# API HELPERS  (original, untouched)
# ══════════════════════════════════════════════════════════════════════════════
def headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID":     ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
        "accept":              "application/json",
    }


def api_get(url: str, params: Optional[dict] = None) -> dict:
    r = requests.get(url, headers=headers(), params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def api_post(url: str, payload: dict) -> dict:
    r = requests.post(
        url,
        headers={**headers(), "content-type": "application/json"},
        json=payload,
        timeout=20,
    )
    r.raise_for_status()
    return r.json()


# ══════════════════════════════════════════════════════════════════════════════
# FORMATTERS  (original, untouched)
# ══════════════════════════════════════════════════════════════════════════════
def fmt_money(x: Any) -> str:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "N/A"
        return f"${float(x):,.2f}"
    except Exception:
        return "N/A"


def fmt_num(x: Any, digits: int = 2) -> str:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "N/A"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "N/A"


def safe_get(d: dict, *path, default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def state_color(state: str) -> str:
    return {
        "ENTER CALL": "green", "ENTER PUT":  "red",
        "HOLD CALL":  "blue",  "HOLD PUT":   "blue",
        "TP1 HIT":    "violet","WEAKENING":  "orange",
        "EXIT CALL":  "orange","EXIT PUT":   "orange",
        "NO TRADE":   "gray",
    }.get(state, "gray")


# ══════════════════════════════════════════════════════════════════════════════
# POSITION MEMORY  (original, untouched)
# ══════════════════════════════════════════════════════════════════════════════
if "active_trade" not in st.session_state:
    st.session_state.active_trade = None


def save_active_trade(trade: dict):
    st.session_state.active_trade = trade


def clear_active_trade():
    st.session_state.active_trade = None


def active_trade_matches(contract_symbol: Optional[str]) -> bool:
    t = st.session_state.active_trade
    return bool(t and contract_symbol and t.get("contract_symbol") == contract_symbol)


# ══════════════════════════════════════════════════════════════════════════════
# ALPACA DATA  (original, untouched)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=20)
def get_stock_snapshot(symbol: str) -> dict:
    return api_get(f"{DATA_BASE}/v2/stocks/{symbol}/snapshot")


@st.cache_data(ttl=30)
def get_stock_bars(symbol: str, timeframe: str = "5Min", limit: int = 120) -> pd.DataFrame:
    payload = api_get(
        f"{DATA_BASE}/v2/stocks/bars",
        params={
            "symbols":    symbol.upper(),
            "timeframe":  timeframe,
            "limit":      limit,
            "adjustment": "raw",
            "feed":       "iex",
            "sort":       "asc",
        },
    )
    bars = payload.get("bars", {}).get(symbol.upper(), [])
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars)
    df["Time"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert("America/New_York")
    df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    return df[["Time", "Open", "High", "Low", "Close", "Volume"]]


@st.cache_data(ttl=60)
def get_option_contracts(symbol: str, expiration_date: Optional[str], option_type: Optional[str]) -> list:
    params = {"underlying_symbols": symbol.upper(), "status": "active", "limit": 1000}
    if expiration_date:
        params["expiration_date"] = expiration_date
    if option_type:
        params["type"] = option_type.lower()
    payload = api_get(f"{PAPER_BASE}/v2/options/contracts", params=params)
    return payload.get("option_contracts", [])


@st.cache_data(ttl=20)
def get_option_snapshot(contract_symbol: str) -> dict:
    payload = api_get(
        f"{DATA_BASE}/v1beta1/options/snapshots",
        params={"symbols": contract_symbol},
    )
    return payload.get("snapshots", {}).get(contract_symbol, {})


@st.cache_data(ttl=30)
def get_news(symbols: str, limit: int = 8) -> list:
    payload = api_get(
        f"{DATA_BASE}/v1beta1/news",
        params={"symbols": symbols, "limit": limit, "sort": "desc"},
    )
    return payload.get("news", [])


def get_open_positions() -> list:
    try:
        return api_get(f"{PAPER_BASE}/v2/positions")
    except Exception:
        return []


def find_position(symbol: str) -> Optional[dict]:
    for p in get_open_positions():
        if p.get("symbol") == symbol:
            return p
    return None


def place_option_order(
    symbol: str, qty: int, side: str,
    order_type: str = "market", limit_price: Optional[float] = None
) -> dict:
    payload: Dict[str, Any] = {
        "symbol":        symbol,
        "qty":           str(qty),
        "side":          side,
        "type":          order_type,
        "time_in_force": "day",
    }
    if order_type == "limit" and limit_price is not None:
        payload["limit_price"] = str(limit_price)
    return api_post(f"{PAPER_BASE}/v2/orders", payload)


# ══════════════════════════════════════════════════════════════════════════════
# SIGNALS / QUALITY  (original, untouched)
# ══════════════════════════════════════════════════════════════════════════════
def add_underlying_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    out["EMA_8"]  = out["Close"].ewm(span=8,  adjust=False).mean()
    out["EMA_21"] = out["Close"].ewm(span=21, adjust=False).mean()
    out["EMA_50"] = out["Close"].ewm(span=50, adjust=False).mean()
    delta    = out["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, pd.NA)
    out["RSI"] = 100 - (100 / (1 + rs))
    return out


def stock_signal(df: pd.DataFrame) -> tuple[str, int, list]:
    if df.empty or len(df) < 30:
        return "NO TRADE", 0, ["Not enough bar data."]
    row = df.iloc[-1]; score = 0; reasons = []
    if row["Close"] > row["EMA_8"]:
        score += 1; reasons.append("Price above EMA 8.")
    else:
        score -= 1; reasons.append("Price below EMA 8.")
    if row["EMA_8"] > row["EMA_21"]:
        score += 1; reasons.append("EMA 8 above EMA 21.")
    else:
        score -= 1; reasons.append("EMA 8 below EMA 21.")
    if row["EMA_21"] > row["EMA_50"]:
        score += 1; reasons.append("EMA 21 above EMA 50.")
    else:
        score -= 1; reasons.append("EMA 21 below EMA 50.")
    if pd.notna(row["RSI"]):
        if row["RSI"] > 55:
            score += 1; reasons.append("RSI supportive.")
        elif row["RSI"] < 45:
            score -= 1; reasons.append("RSI weak.")
    if score >= 3:  return "BULLISH", score, reasons
    if score <= -3: return "BEARISH", score, reasons
    return "NEUTRAL", score, reasons


def state_series(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["state"] = []; return out
    states = []; mode = "FLAT"
    for _, row in out.iterrows():
        score = 0
        score += 1 if row["Close"] > row["EMA_8"]  else -1
        score += 1 if row["EMA_8"]  > row["EMA_21"] else -1
        score += 1 if row["EMA_21"] > row["EMA_50"] else -1
        if pd.notna(row["RSI"]):
            if row["RSI"] > 55:   score += 1
            elif row["RSI"] < 45: score -= 1
        bullish         = score >= 3
        bearish         = score <= -3
        weakening_long  = (row["Close"] < row["EMA_8"]) or (pd.notna(row["RSI"]) and row["RSI"] < 48)
        weakening_short = (row["Close"] > row["EMA_8"]) or (pd.notna(row["RSI"]) and row["RSI"] > 52)
        state = "NO TRADE"
        if mode == "FLAT":
            if bullish:  state = "BUY";       mode = "LONG"
            elif bearish: state = "SELL";     mode = "SHORT"
        elif mode == "LONG":
            if bearish or weakening_long:     state = "EXIT BUY";  mode = "FLAT"
            else:                             state = "HOLD BUY"
        elif mode == "SHORT":
            if bullish or weakening_short:    state = "EXIT SELL"; mode = "FLAT"
            else:                             state = "HOLD SELL"
        states.append(state)
    out["state"]      = states
    out["prev_state"] = out["state"].shift(1)
    return out


def find_chart_signals(df: pd.DataFrame):
    marked = state_series(df)
    buy_rows, sell_rows, exit_buy_rows, exit_sell_rows = [], [], [], []
    for _, row in marked.iterrows():
        base  = {"index": row["Time"], "Low": row["Low"], "High": row["High"],
                 "ATR": 0 if pd.isna(row.get("ATR")) else row.get("ATR", 0),
                 "Close": row["Close"]}
        state = row["state"]; prev_state = row["prev_state"]
        if state == "BUY"      and prev_state != "BUY":
            buy_rows.append({**base,  "label": f"BUY<br>{float(row['Close']):.2f}"})
        elif state == "SELL"   and prev_state != "SELL":
            sell_rows.append({**base, "label": f"SELL<br>{float(row['Close']):.2f}"})
        elif state == "EXIT BUY"  and prev_state != "EXIT BUY":
            exit_buy_rows.append({**base,  "label": f"EXIT<br>{float(row['Close']):.2f}"})
        elif state == "EXIT SELL" and prev_state != "EXIT SELL":
            exit_sell_rows.append({**base, "label": f"EXIT<br>{float(row['Close']):.2f}"})
    return (pd.DataFrame(buy_rows), pd.DataFrame(sell_rows),
            pd.DataFrame(exit_buy_rows), pd.DataFrame(exit_sell_rows))


def contract_quality(
    underlying_price, strike, bid, ask, volume, open_interest, iv, delta
) -> dict:
    score = 100; reasons = []; spread = None; spread_pct = None
    if bid is not None and ask is not None:
        spread = float(ask) - float(bid)
        if ask and ask > 0:
            spread_pct = spread / float(ask)
    if spread_pct is None:
        score -= 25; reasons.append("No usable bid/ask spread.")
    elif spread_pct > 0.20:
        score -= 30; reasons.append("Spread too wide.")
    elif spread_pct > 0.10:
        score -= 15; reasons.append("Spread somewhat wide.")
    else:
        reasons.append("Spread acceptable.")
    if ask is None:
        score -= 15; reasons.append("No ask price.")
    elif ask < 0.10:
        score -= 20; reasons.append("Premium too tiny / noisy.")
    elif ask > 10:
        score -= 10; reasons.append("Premium is expensive.")
    else:
        reasons.append("Premium in a workable range.")
    if underlying_price is not None and strike is not None and underlying_price > 0:
        moneyness = abs(float(strike) - float(underlying_price)) / float(underlying_price)
        if moneyness > 0.05:
            score -= 20; reasons.append("Strike far from underlying.")
        elif moneyness > 0.02:
            score -= 8;  reasons.append("Strike slightly far from underlying.")
        else:
            reasons.append("Strike near the underlying.")
    if volume is not None:
        if float(volume) < 10:
            score -= 15; reasons.append("Low contract volume.")
        else:
            reasons.append("Volume acceptable.")
    if open_interest is not None:
        if float(open_interest) < 50:
            score -= 15; reasons.append("Low open interest.")
        else:
            reasons.append("Open interest acceptable.")
    if iv is not None and float(iv) > 2.0:
        score -= 10; reasons.append("Implied volatility very high.")
    if delta is None:
        reasons.append("Greeks unavailable.")
    else:
        reasons.append("Delta available.")
    score = max(0, min(100, score))
    return {"score": score, "quality_ok": score >= 55,
            "spread": spread, "spread_pct": spread_pct, "reasons": reasons}


def derive_options_state(stock_bias: str, option_side: str, has_position: bool, quality_ok: bool) -> str:
    side = option_side.upper()
    if not quality_ok: return "NO TRADE"
    if side == "CALL":
        if stock_bias == "BULLISH" and not has_position: return "ENTER CALL"
        if stock_bias == "BULLISH" and has_position:     return "HOLD CALL"
        if stock_bias != "BULLISH" and has_position:     return "WEAKENING"
        return "NO TRADE"
    if side == "PUT":
        if stock_bias == "BEARISH" and not has_position: return "ENTER PUT"
        if stock_bias == "BEARISH" and has_position:     return "HOLD PUT"
        if stock_bias != "BEARISH" and has_position:     return "WEAKENING"
        return "NO TRADE"
    return "NO TRADE"


def manage_active_trade(active_trade: Optional[dict], current_premium: Optional[float], stock_bias: str) -> dict:
    if not active_trade: return {"state": None, "notes": []}
    notes = []; side = active_trade["option_side"].upper(); state = f"HOLD {side}"
    stop_hit = tp1_hit = tp2_hit = weakening = False
    if current_premium is not None:
        if current_premium <= active_trade["premium_stop"]:
            stop_hit = True; notes.append("Premium stop hit.")
        elif current_premium >= active_trade["tp2"]:
            tp2_hit  = True; notes.append("TP2 hit.")
        elif current_premium >= active_trade["tp1"]:
            tp1_hit  = True; notes.append("TP1 hit.")
    if side == "CALL" and stock_bias != "BULLISH":
        weakening = True; notes.append("Underlying bias weakened for calls.")
    if side == "PUT"  and stock_bias != "BEARISH":
        weakening = True; notes.append("Underlying bias weakened for puts.")
    if stop_hit or tp2_hit: state = f"EXIT {side}"
    elif tp1_hit:           state = "TP1 HIT"
    elif weakening:         state = "WEAKENING"
    return {"state": state, "notes": notes}


# ══════════════════════════════════════════════════════════════════════════════
# CHART  (dark-themed, RSI panel added, coloured EMAs & volume)
# ══════════════════════════════════════════════════════════════════════════════
def make_underlying_chart(df: pd.DataFrame, symbol: str):
    if df.empty:
        return go.Figure()
    chart_df = df.copy()
    chart_df["Time"] = pd.to_datetime(chart_df["Time"], errors="coerce")
    chart_df = chart_df.dropna(subset=["Time"]).sort_values("Time").tail(140).copy()

    vol_colors = [
        "rgba(0,200,100,.4)" if c >= o else "rgba(239,68,68,.4)"
        for c, o in zip(chart_df["Close"], chart_df["Open"])
    ]
    rsi_available = "RSI" in chart_df.columns and chart_df["RSI"].notna().sum() > 5

    fig = make_subplots(
        rows=2 if rsi_available else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.72, 0.28] if rsi_available else [1.0],
        specs=[[{"secondary_y": True}]] + ([[{}]] if rsi_available else []),
    )

    # ── Candles ───────────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=chart_df["Time"],
        open=chart_df["Open"], high=chart_df["High"],
        low=chart_df["Low"],   close=chart_df["Close"],
        name=symbol,
        increasing_line_color="#00c864", increasing_fillcolor="#00c864",
        decreasing_line_color="#ef4444", decreasing_fillcolor="#ef4444",
    ), row=1, col=1, secondary_y=False)

    # ── EMAs ──────────────────────────────────────────────────────────────────
    for col, color in _EMA_COLORS.items():
        if col in chart_df.columns and chart_df[col].notna().sum() > 0:
            fig.add_trace(go.Scatter(
                x=chart_df["Time"], y=chart_df[col],
                mode="lines", name=col,
                line=dict(color=color, width=1.6),
            ), row=1, col=1, secondary_y=False)

    # ── Volume ────────────────────────────────────────────────────────────────
    fig.add_trace(go.Bar(
        x=chart_df["Time"], y=chart_df["Volume"],
        name="Volume", marker_color=vol_colors,
    ), row=1, col=1, secondary_y=True)

    # ── Buy / Sell / Exit arrows ───────────────────────────────────────────────
    buy_pts, sell_pts, exit_buy_pts, exit_sell_pts = find_chart_signals(chart_df)

    for _, row in (buy_pts.tail(8) if not buy_pts.empty else buy_pts).iterrows():
        fig.add_annotation(x=row["index"],
            y=row["Low"] - max(float(row["ATR"]) * 0.2, 0.05),
            text=row["label"], showarrow=True, arrowhead=2,
            arrowcolor="#00c864", font=dict(color="#00c864", size=10), ax=0, ay=38)

    for _, row in (sell_pts.tail(8) if not sell_pts.empty else sell_pts).iterrows():
        fig.add_annotation(x=row["index"],
            y=row["High"] + max(float(row["ATR"]) * 0.2, 0.05),
            text=row["label"], showarrow=True, arrowhead=2,
            arrowcolor="#ef4444", font=dict(color="#ef4444", size=10), ax=0, ay=-38)

    for _, row in (exit_buy_pts.tail(8) if not exit_buy_pts.empty else exit_buy_pts).iterrows():
        fig.add_annotation(x=row["index"],
            y=row["High"] + max(float(row["ATR"]) * 0.15, 0.05),
            text=row["label"], showarrow=True, arrowhead=2,
            arrowcolor="#fb923c", font=dict(color="#fb923c", size=10), ax=0, ay=-32)

    for _, row in (exit_sell_pts.tail(8) if not exit_sell_pts.empty else exit_sell_pts).iterrows():
        fig.add_annotation(x=row["index"],
            y=row["Low"] - max(float(row["ATR"]) * 0.15, 0.05),
            text=row["label"], showarrow=True, arrowhead=2,
            arrowcolor="#fb923c", font=dict(color="#fb923c", size=10), ax=0, ay=32)

    # ── RSI panel ─────────────────────────────────────────────────────────────
    if rsi_available:
        fig.add_trace(go.Scatter(
            x=chart_df["Time"], y=chart_df["RSI"],
            mode="lines", name="RSI",
            line=dict(color="#06b6d4", width=1.6),
        ), row=2, col=1)
        for lvl, clr in [(70, "rgba(239,68,68,.4)"), (50, "rgba(255,255,255,.15)"), (30, "rgba(0,200,100,.4)")]:
            fig.add_hline(y=lvl, row=2, col=1,
                          line_dash="dot", line_color=clr, line_width=1)

    # ── Layout ────────────────────────────────────────────────────────────────
    _grid = dict(gridcolor="#21262d", zerolinecolor="#30363d")
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9", family="monospace"),
        height=600 if rsi_available else 480,
        xaxis_rangeslider_visible=False,
        dragmode="pan",
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#161b22", bordercolor="#30363d", font_color="#e6edf3"),
        legend=dict(orientation="h", bgcolor="rgba(13,17,23,.85)",
                    bordercolor="#30363d", borderwidth=1, font=dict(size=10)),
        margin=dict(l=60, r=80, t=40, b=40),
        xaxis=dict(type="date", range=[chart_df["Time"].iloc[0], chart_df["Time"].iloc[-1]],
                   showspikes=True, spikemode="across", spikecolor="#58a6ff",
                   spikedash="dot", spikethickness=1, **_grid),
    )
    fig.update_yaxes(title_text="Price",  side="left",  fixedrange=True, **_grid, row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Volume", side="right", fixedrange=True, showgrid=False,
                     range=[0, chart_df["Volume"].max() * 4], row=1, col=1, secondary_y=True)
    if rsi_available:
        fig.update_yaxes(title_text="RSI", fixedrange=True, range=[0, 100], **_grid, row=2, col=1)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# GUARD — credentials
# ══════════════════════════════════════════════════════════════════════════════
if not ALPACA_KEY or not ALPACA_SECRET:
    st.error("🔑 Add **ALPACA_API_KEY** and **ALPACA_SECRET_KEY** to Streamlit secrets first.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<h1 style="color:#e6edf3;margin-bottom:2px">🎯 SPY Buddy Options V2</h1>',
    unsafe_allow_html=True,
)
st.caption("Position-aware options manager with locked entry, TP/SL tracking, chart, live P&L, and news.")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR  (original logic, dark-styled)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Setup")
    symbol       = st.text_input("Underlying", value=DEFAULT_SYMBOL).upper().strip()
    option_side  = st.selectbox("Direction", ["Call", "Put"])
    tf           = st.selectbox("Underlying timeframe", ["1Min", "5Min", "15Min", "1Hour"], index=1)
    qty          = st.number_input("Contracts", min_value=1, max_value=100, value=10, step=1)
    order_style  = st.selectbox("Order type", ["market", "limit"])

# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCH  (original logic, untouched)
# ══════════════════════════════════════════════════════════════════════════════
pre_snapshot        = get_stock_snapshot(symbol)
pre_underlying_price = safe_get(pre_snapshot, "latestTrade", "p")

contracts_raw = get_option_contracts(symbol, None, option_side)
expirations   = sorted({c.get("expiration_date") for c in contracts_raw if c.get("expiration_date")})

colA, colB, colC = st.columns(3)
with colA:
    expiration = st.selectbox("Expiration", expirations, index=0 if expirations else None)
with colB:
    contracts_for_exp = [c for c in contracts_raw if c.get("expiration_date") == expiration] if expiration else []
    strikes = sorted({float(c.get("strike_price")) for c in contracts_for_exp if c.get("strike_price") is not None})
    default_strike_index = 0
    if strikes and pre_underlying_price is not None:
        default_strike_index = min(range(len(strikes)), key=lambda i: abs(strikes[i] - float(pre_underlying_price)))
    strike = st.selectbox("Strike", strikes, index=default_strike_index if strikes else None)
with colC:
    if st.button("🔄 Refresh data", use_container_width=True, key="top_refresh_btn"):
        st.cache_data.clear(); st.rerun()

selected_contract = None
if expiration and strike is not None:
    for c in contracts_for_exp:
        if float(c.get("strike_price")) == float(strike):
            selected_contract = c; break

contract_symbol = selected_contract.get("symbol") if selected_contract else None

underlying_snapshot = get_stock_snapshot(symbol)
bars = add_underlying_indicators(get_stock_bars(symbol, tf, 120))
stock_bias, stock_score, stock_reasons = stock_signal(bars)

last_trade  = safe_get(underlying_snapshot, "latestTrade", "p")
daily_close = safe_get(underlying_snapshot, "dailyBar",    "c")
prev_close  = safe_get(underlying_snapshot, "prevDailyBar","c")

snapshot         = get_option_snapshot(contract_symbol) if contract_symbol else {}
quote_bid        = safe_get(snapshot, "latestQuote", "bp")
quote_ask        = safe_get(snapshot, "latestQuote", "ap")
quote_mid        = None
if quote_bid is not None and quote_ask is not None:
    quote_mid = (float(quote_bid) + float(quote_ask)) / 2.0
last_option_trade = safe_get(snapshot, "latestTrade", "p")
current_premium   = quote_mid if quote_mid is not None else last_option_trade

delta         = safe_get(snapshot, "greeks", "delta")
gamma         = safe_get(snapshot, "greeks", "gamma")
theta         = safe_get(snapshot, "greeks", "theta")
vega          = safe_get(snapshot, "greeks", "vega")
iv            = safe_get(snapshot, "implied_volatility")
day_bar       = safe_get(snapshot, "dailyBar", default={}) or {}
option_volume = day_bar.get("v")
open_interest = selected_contract.get("open_interest") if selected_contract else None

quality = contract_quality(
    underlying_price=last_trade, strike=strike,
    bid=quote_bid, ask=quote_ask,
    volume=option_volume, open_interest=open_interest,
    iv=iv, delta=delta,
)

broker_position = find_position(contract_symbol) if contract_symbol else None
has_position    = broker_position is not None or active_trade_matches(contract_symbol)

base_state = derive_options_state(stock_bias, option_side, has_position, quality["quality_ok"])
managed    = manage_active_trade(
    st.session_state.active_trade if active_trade_matches(contract_symbol) else None,
    current_premium, stock_bias,
)
state = managed["state"] or base_state

# ══════════════════════════════════════════════════════════════════════════════
# PANELS
# ══════════════════════════════════════════════════════════════════════════════

# ── State badge ───────────────────────────────────────────────────────────────
badge_col, _ = st.columns([1, 3])
with badge_col:
    st.markdown(f'<div style="margin:10px 0 18px">{state_badge(state)}</div>',
                unsafe_allow_html=True)

# ── Underlying ────────────────────────────────────────────────────────────────
st.markdown('<h3 style="color:#8b949e;font-size:1rem;margin-bottom:8px;text-transform:uppercase;letter-spacing:.08em">Underlying</h3>', unsafe_allow_html=True)
u1, u2, u3, u4 = st.columns(4)
u1.markdown(f'<div style="padding:4px 0">{bias_badge(stock_bias)}</div>', unsafe_allow_html=True)
u2.metric("Score",       stock_score)
u3.metric("Latest Trade",fmt_money(last_trade))
u4.metric("Daily / Prev",f"{fmt_money(daily_close)} / {fmt_money(prev_close)}")

with st.expander("Why the underlying bias"):
    for r in stock_reasons:
        icon = "🟢" if "above" in r.lower() or "supportive" in r.lower() else "🔴"
        st.markdown(f"{icon} {r}")

st.divider()

# ── Option Contract ────────────────────────────────────────────────────────────
st.markdown('<h3 style="color:#8b949e;font-size:1rem;margin-bottom:8px;text-transform:uppercase;letter-spacing:.08em">Option Contract</h3>', unsafe_allow_html=True)
o1, o2, o3, o4, o5, o6 = st.columns(6)
o1.metric("Contract",  contract_symbol or "N/A")
o2.metric("Bid",       fmt_money(quote_bid))
o3.metric("Ask",       fmt_money(quote_ask))
o4.metric("Mid",       fmt_money(quote_mid))
o5.metric("Last",      fmt_money(last_option_trade))
o6.metric("Spread",    fmt_money(quality["spread"]))

if any(x is not None for x in [delta, gamma, theta, vega, iv]):
    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Delta", fmt_num(delta, 3))
    g2.metric("Gamma", fmt_num(gamma, 4))
    g3.metric("Theta", fmt_num(theta, 4))
    g4.metric("Vega",  fmt_num(vega,  4))
    g5.metric("IV",    fmt_num(iv,    3))

st.divider()

# ── Contract Quality ───────────────────────────────────────────────────────────
st.markdown('<h3 style="color:#8b949e;font-size:1rem;margin-bottom:8px;text-transform:uppercase;letter-spacing:.08em">Contract Quality</h3>', unsafe_allow_html=True)
q1, q2, q3, q4 = st.columns(4)
q1.markdown(f'<div style="padding:8px 0">{quality_bar(quality["score"], quality["quality_ok"])}</div>',
            unsafe_allow_html=True)
q2.metric("Volume",        fmt_num(option_volume, 0))
q3.metric("Open Interest", fmt_num(open_interest, 0))
q4.metric("Spread %",      f"{quality['spread_pct']*100:.1f}%" if quality["spread_pct"] else "N/A")

with st.expander("Why this contract passed / failed"):
    for r in quality["reasons"]:
        icon = "✅" if any(w in r.lower() for w in ("acceptable","near","available")) else "⚠️"
        st.markdown(f"{icon} {r}")

st.divider()

# ── Trade State ────────────────────────────────────────────────────────────────
st.markdown('<h3 style="color:#8b949e;font-size:1rem;margin-bottom:8px;text-transform:uppercase;letter-spacing:.08em">Trade State</h3>', unsafe_allow_html=True)
s1, s2, s3 = st.columns(3)
s1.metric("State",                  state)
s2.metric("Holding this contract?", "Yes" if has_position else "No")
s3.metric("Direction",              option_side)

st.divider()

# ── Trade Plan ─────────────────────────────────────────────────────────────────
st.markdown('<h3 style="color:#8b949e;font-size:1rem;margin-bottom:8px;text-transform:uppercase;letter-spacing:.08em">Trade Plan</h3>', unsafe_allow_html=True)

active_trade = st.session_state.get("active_trade")
is_same_locked_trade = bool(
    active_trade and contract_symbol
    and active_trade.get("contract_symbol") == contract_symbol
)

if is_same_locked_trade:
    locked_entry = float(active_trade["entry_premium"])
    locked_stop  = float(active_trade["premium_stop"])
    locked_tp1   = float(active_trade["tp1"])
    locked_tp2   = float(active_trade["tp2"])
else:
    locked_entry = locked_stop = locked_tp1 = locked_tp2 = None

default_entry = locked_entry if locked_entry is not None else (
    quote_ask if quote_ask is not None else (last_option_trade if last_option_trade is not None else 0.0)
)

if not is_same_locked_trade:
    st.session_state["entry_premium_input"] = float(default_entry or 0.0)

p1, p2, p3, p4, p5 = st.columns(5)
with p1:
    entry_premium = st.number_input(
        "Entry premium", min_value=0.0, value=float(default_entry or 0.0),
        step=0.05, key="entry_premium_input", disabled=is_same_locked_trade)
with p2:
    stop_pct = st.slider("Stop %",  5,  50,  20, 1, key="stop_pct_input",  disabled=is_same_locked_trade)
with p3:
    tp1_pct  = st.slider("TP1 %",  10, 100,  30, 1, key="tp1_pct_input",   disabled=is_same_locked_trade)
with p4:
    tp2_pct  = st.slider("TP2 %",  20, 200,  50, 1, key="tp2_pct_input",   disabled=is_same_locked_trade)
with p5:
    min_rr   = st.slider("Min R/R", 1.0, 3.0, 2.0, 0.1, key="min_rr_input", disabled=is_same_locked_trade)

if is_same_locked_trade:
    premium_stop = locked_stop; tp1 = locked_tp1; tp2 = locked_tp2
    rr1 = ((tp1 - locked_entry) / max(0.0001, locked_entry - premium_stop)) if locked_entry > premium_stop else None
    rr2 = ((tp2 - locked_entry) / max(0.0001, locked_entry - premium_stop)) if locked_entry > premium_stop else None
else:
    premium_stop = tp1 = tp2 = rr1 = rr2 = None
    if entry_premium > 0:
        risk_amt     = entry_premium * (stop_pct / 100.0)
        premium_stop = max(0.01, entry_premium - risk_amt)
        tp1          = entry_premium * (1 + tp1_pct / 100.0)
        tp2          = entry_premium * (1 + tp2_pct / 100.0)
        rr1          = (tp1 - entry_premium) / max(0.0001, entry_premium - premium_stop)
        rr2          = (tp2 - entry_premium) / max(0.0001, entry_premium - premium_stop)

r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("Premium Stop", fmt_money(premium_stop))
r2.metric("TP1",          fmt_money(tp1))
r3.metric("TP2",          fmt_money(tp2))
r4.metric("R/R to TP1",   "N/A" if rr1 is None else f"{rr1:.2f}")
r5.metric("R/R to TP2",   "N/A" if rr2 is None else f"{rr2:.2f}")

if is_same_locked_trade:
    st.success("🔒 Locked trade active. Trade Plan is frozen until cleared or closed.")

if base_state in ["ENTER CALL", "ENTER PUT"] and rr1 is not None and rr1 < min_rr:
    st.warning("Direction is good, but reward/risk is below your minimum. No trade.")
elif state == "NO TRADE":
    st.info("No trade right now.")
elif state in ["HOLD CALL", "HOLD PUT"]:
    st.info(f"{state}. Manage the open position.")
elif state == "TP1 HIT":
    st.success("🎯 TP1 HIT. Consider taking partial profit and tightening your stop.")
elif state in ["EXIT CALL", "EXIT PUT"]:
    st.warning(f"⚠️ {state}. Hard exit condition triggered.")
elif state == "WEAKENING":
    st.warning("⚠️ WEAKENING. Soft warning only. Momentum or bias has weakened.")
else:
    st.success(state)

if managed["notes"]:
    with st.expander("Trade manager notes"):
        for n in managed["notes"]:
            st.markdown(f"• {n}")

st.divider()

# ── Live P&L ───────────────────────────────────────────────────────────────────
st.markdown('<h3 style="color:#8b949e;font-size:1rem;margin-bottom:8px;text-transform:uppercase;letter-spacing:.08em">Live P&L</h3>', unsafe_allow_html=True)
pos = broker_position if broker_position else None
if is_same_locked_trade and current_premium is not None:
    custom_pl     = (float(current_premium) - float(st.session_state.active_trade["entry_premium"])) \
                    * int(st.session_state.active_trade["qty"]) * 100
    custom_pl_pct = ((float(current_premium) / float(st.session_state.active_trade["entry_premium"])) - 1.0) * 100
else:
    custom_pl = custom_pl_pct = None

pl_color = "normal"
if custom_pl is not None:
    pl_color = "normal"  # Streamlit handles delta colours automatically

l1, l2, l3, l4, l5, l6 = st.columns(6)
l1.metric("Position?",       "Yes" if has_position else "No")
l2.metric("Qty",             str(st.session_state.active_trade["qty"]) if is_same_locked_trade
          else (pos.get("qty") if pos else "0"))
l3.metric("Locked Entry",    fmt_money(st.session_state.active_trade["entry_premium"]) if is_same_locked_trade
          else fmt_money(pos.get("avg_entry_price") if pos else None))
l4.metric("Current Premium", fmt_money(current_premium))
l5.metric("Custom P/L",      fmt_money(custom_pl),
          delta=f"{custom_pl_pct:.2f}%" if custom_pl_pct is not None else None)
l6.metric("Custom P/L %",    fmt_num(custom_pl_pct, 2))

st.divider()

# ── Actions ────────────────────────────────────────────────────────────────────
st.markdown('<h3 style="color:#8b949e;font-size:1rem;margin-bottom:8px;text-transform:uppercase;letter-spacing:.08em">Actions</h3>', unsafe_allow_html=True)
limit_seed  = quote_ask if quote_ask is not None else entry_premium
limit_price = st.number_input(
    "Limit price (used only for limit orders)",
    min_value=0.0, value=float(limit_seed or 0.0), step=0.05, key="limit_price_input")

a1, a2, a3 = st.columns(3)
can_start = (
    contract_symbol is not None and entry_premium > 0
    and premium_stop is not None and tp1 is not None and tp2 is not None
)

if a1.button("🔒 Start / Lock Trade", use_container_width=True, disabled=not can_start, key="lock_trade_btn"):
    save_active_trade({
        "contract_symbol": contract_symbol,
        "option_side":     option_side.upper(),
        "qty":             int(qty),
        "entry_premium":   float(entry_premium),
        "premium_stop":    float(premium_stop),
        "tp1":             float(tp1),
        "tp2":             float(tp2),
    })
    st.success("Trade locked. Entry, stop, and targets will now stay fixed.")

if a2.button("📈 Paper Buy to Open", use_container_width=True, disabled=contract_symbol is None, key="buy_open_btn"):
    try:
        order = place_option_order(contract_symbol, int(qty), "buy",
                                   order_style, limit_price if order_style == "limit" else None)
        st.success(f"Submitted buy order: {order.get('id', 'ok')}")
    except Exception as e:
        st.error(f"Buy order failed: {e}")

if a3.button("📉 Paper Sell to Close", use_container_width=True, disabled=contract_symbol is None, key="sell_close_btn"):
    try:
        order = place_option_order(contract_symbol, int(qty), "sell",
                                   order_style, limit_price if order_style == "limit" else None)
        st.success(f"Submitted sell order: {order.get('id', 'ok')}")
        if active_trade_matches(contract_symbol):
            clear_active_trade()
    except Exception as e:
        st.error(f"Sell order failed: {e}")

c1, c2 = st.columns(2)
if c1.button("🗑️ Clear Locked Trade", use_container_width=True, key="clear_trade_btn"):
    clear_active_trade(); st.info("Locked trade cleared.")
if c2.button("🔄 Refresh data", use_container_width=True, key="bottom_refresh_btn"):
    st.cache_data.clear(); st.rerun()

st.divider()

# ── Underlying Chart ───────────────────────────────────────────────────────────
st.markdown('<h3 style="color:#8b949e;font-size:1rem;margin-bottom:8px;text-transform:uppercase;letter-spacing:.08em">Underlying Chart</h3>', unsafe_allow_html=True)
if not bars.empty:
    st.plotly_chart(
        make_underlying_chart(bars, symbol),
        use_container_width=True,
        config={"scrollZoom": True, "displaylogo": False,
                "modeBarButtonsToRemove": ["select2d", "lasso2d"]},
    )
    st.caption("Scroll = zoom time axis · Click-drag = pan · Double-click = reset")
else:
    st.info("No underlying chart data returned.")

st.divider()

# ── News ───────────────────────────────────────────────────────────────────────
st.markdown('<h3 style="color:#8b949e;font-size:1rem;margin-bottom:8px;text-transform:uppercase;letter-spacing:.08em">News</h3>', unsafe_allow_html=True)
news = get_news(symbol, limit=8)
if not news:
    st.write("No recent news returned.")
else:
    for item in news[:6]:
        headline = item.get("headline", "No headline")
        source   = item.get("source", "")
        ts       = item.get("updated_at", item.get("created_at", ""))
        summary  = item.get("summary", "")
        st.markdown(
            f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;'
            f'padding:14px 18px;margin-bottom:10px">'
            f'<div style="color:#e6edf3;font-weight:700;font-size:.97rem">{headline}</div>'
            f'<div style="color:#8b949e;font-size:.78rem;margin:4px 0 8px">{source} &nbsp;•&nbsp; {ts}</div>'
            + (f'<div style="color:#c9d1d9;font-size:.88rem">{summary[:220]}{"..." if len(summary)>220 else ""}</div>' if summary else "")
            + "</div>",
            unsafe_allow_html=True,
        )

with st.expander("More contract details"):
    st.json({"contract": selected_contract or {}, "snapshot": snapshot or {}})

st.caption("Keeps strike selection near the underlying price and refreshes the planning entry premium until you lock the trade.")
