"""
SPY Buddy Options — Quant Edition
==================================
Data: Tradier API (real-time, no delay)
Chart: Plotly multi-panel (candlestick + volume + RSI/StochRSI + MACD + Squeeze)
Signals: prev_state tracked inside loop — no duplicates, no stacking.
TDA: 1H/4H/Daily use EMA 21/50/200.
Sidebar: Account Size + Risk % calculator.
"""

import math
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="SPY Buddy — Quant Edition", page_icon="🧠", layout="wide")

# ── Dark theme CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main { background:#0d1117!important; color:#e6edf3!important; }
[data-testid="stSidebar"] { background:#161b22!important; border-right:1px solid #30363d; }
[data-testid="stSidebar"] * { color:#c9d1d9!important; }
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select { background:#21262d!important; border-color:#30363d!important; }
[data-testid="stMetric"] {
    background:#161b22!important; border:1px solid #30363d!important;
    border-radius:10px!important; padding:12px 16px!important;
}
[data-testid="stMetricLabel"] { color:#8b949e!important; font-size:.78rem!important; }
[data-testid="stMetricValue"] { color:#e6edf3!important; font-size:1.2rem!important; font-weight:700!important; }
[data-testid="stTabs"] button { color:#8b949e!important; border-bottom:2px solid transparent!important; font-weight:600!important; }
[data-testid="stTabs"] button[aria-selected="true"] { color:#58a6ff!important; border-bottom:2px solid #58a6ff!important; }
[data-testid="stExpander"] { background:#161b22!important; border:1px solid #30363d!important; border-radius:8px!important; }
[data-testid="stExpander"] summary { color:#c9d1d9!important; }
[data-testid="stButton"] > button {
    background:#21262d!important; color:#c9d1d9!important;
    border:1px solid #30363d!important; border-radius:8px!important; font-weight:600!important;
}
[data-testid="stButton"] > button:hover {
    background:#30363d!important; border-color:#58a6ff!important; color:#58a6ff!important;
}
input, select, textarea,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background:#21262d!important; color:#e6edf3!important;
    border:1px solid #30363d!important; border-radius:6px!important;
}
[data-testid="stDataFrame"] { border:1px solid #30363d!important; border-radius:8px!important; }
hr { border-color:#30363d!important; }
small, .stCaption, [data-testid="stCaptionContainer"] { color:#8b949e!important; }
.risk-box {
    background:#0d2137; border:1px solid #1d4ed8; border-radius:10px;
    padding:14px 16px; margin:8px 0 4px;
}
.risk-label { color:#93c5fd; font-size:.75rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }
.risk-value { color:#e6edf3; font-size:1.45rem; font-weight:900; margin-top:2px; }
.risk-sub   { color:#8b949e; font-size:.75rem; margin-top:2px; }
.tda-card {
    background:#161b22; border:1px solid #30363d; border-radius:10px;
    padding:12px 14px; margin-bottom:4px;
}
.tda-tf { color:#8b949e; font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }
.tda-bias-bull { color:#4ade80; font-size:1.1rem; font-weight:900; }
.tda-bias-bear { color:#f87171; font-size:1.1rem; font-weight:900; }
.tda-bias-neut { color:#9ca3af; font-size:1.1rem; font-weight:900; }
.tda-score { color:#8b949e; font-size:.78rem; }
.tda-reason { color:#c9d1d9; font-size:.75rem; margin-top:4px; line-height:1.4; }
.cascade-ok   { background:#052e16; border:1px solid #4ade80; border-radius:8px; padding:10px 18px; color:#4ade80; font-weight:800; font-size:1rem; }
.cascade-warn { background:#431407; border:1px solid #fb923c; border-radius:8px; padding:10px 18px; color:#fb923c; font-weight:800; font-size:1rem; }
.cascade-neut { background:#1c1c1c; border:1px solid #6b7280; border-radius:8px; padding:10px 18px; color:#9ca3af; font-weight:800; font-size:1rem; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CREDENTIALS  (Tradier)
# ══════════════════════════════════════════════════════════════════════════════
TRADIER_TOKEN      = st.secrets.get("TRADIER_TOKEN", "")
TRADIER_ACCOUNT_ID = st.secrets.get("TRADIER_ACCOUNT_ID", "")
# Use sandbox for paper trading, live for real-time data
TRADIER_BASE       = "https://api.tradier.com"
DEFAULT_SYMBOL     = "SPY"

# ══════════════════════════════════════════════════════════════════════════════
# COLOUR / BADGE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
_STATE_BG  = {"ENTER CALL":"#052e16","ENTER PUT":"#450a0a","HOLD CALL":"#1e3a5f","HOLD PUT":"#1e3a5f",
              "TP1 HIT":"#2e1065","WEAKENING":"#431407","EXIT CALL":"#431407","EXIT PUT":"#431407","NO TRADE":"#1c1c1c"}
_STATE_FG  = {"ENTER CALL":"#4ade80","ENTER PUT":"#f87171","HOLD CALL":"#93c5fd","HOLD PUT":"#93c5fd",
              "TP1 HIT":"#c084fc","WEAKENING":"#fb923c","EXIT CALL":"#fb923c","EXIT PUT":"#fb923c","NO TRADE":"#6b7280"}
_STATE_GLOW= {"ENTER CALL":"0 0 18px 5px rgba(74,222,128,.6)","ENTER PUT":"0 0 18px 5px rgba(248,113,113,.6)",
              "HOLD CALL":"0 0 14px 4px rgba(147,197,253,.4)","HOLD PUT":"0 0 14px 4px rgba(147,197,253,.4)",
              "TP1 HIT":"0 0 14px 4px rgba(192,132,252,.55)","WEAKENING":"0 0 14px 4px rgba(251,146,60,.5)",
              "EXIT CALL":"0 0 14px 4px rgba(251,146,60,.5)","EXIT PUT":"0 0 14px 4px rgba(251,146,60,.5)","NO TRADE":"none"}
_BIAS_STYLE= {"BULLISH":("🟢","#4ade80","#052e16"),"BEARISH":("🔴","#f87171","#450a0a"),"NEUTRAL":("⚪","#9ca3af","#1c1c1c")}

def state_badge(state:str)->str:
    bg=_STATE_BG.get(state,"#1c1c1c"); fg=_STATE_FG.get(state,"#9ca3af"); glow=_STATE_GLOW.get(state,"none")
    return (f'<span style="background:{bg};color:{fg};padding:12px 32px;border-radius:8px;'
            f'font-size:1.9rem;font-weight:900;box-shadow:{glow};letter-spacing:.06em;display:inline-block">{state}</span>')

def bias_badge(bias:str)->str:
    icon,fg,bg=_BIAS_STYLE.get(bias,("⚪","#9ca3af","#1c1c1c"))
    return (f'<span style="background:{bg};color:{fg};padding:6px 18px;border-radius:6px;'
            f'font-size:1.05rem;font-weight:800">{icon} {bias}</span>')

def certainty_bar(score:int)->str:
    if score>=75:   color,label="#4ade80","HIGH"
    elif score>=55: color,label="#f59e0b","MEDIUM"
    else:           color,label="#ef4444","LOW"
    return (f'<div style="background:#21262d;border-radius:6px;height:20px;width:100%;margin:4px 0 2px">'
            f'<div style="background:{color};width:{score}%;height:100%;border-radius:6px;transition:width .4s"></div></div>'
            f'<small style="color:{color};font-weight:700;font-size:.85rem">{score}% certainty — {label}</small>')

def quality_bar(score:int,ok:bool)->str:
    color="#4ade80" if ok else "#ef4444"; label="PASS" if ok else "FAIL"
    return (f'<div style="background:#21262d;border-radius:6px;height:16px;width:100%;margin-top:4px">'
            f'<div style="background:{color};width:{score}%;height:100%;border-radius:6px"></div></div>'
            f'<small style="color:{color};font-weight:700">{score}/100 — {label}</small>')

def section(title:str):
    st.markdown(f'<h3 style="color:#8b949e;font-size:.9rem;margin:18px 0 8px;'
                f'text-transform:uppercase;letter-spacing:.1em">{title}</h3>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TRADIER API HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _tradier_headers()->Dict[str,str]:
    return {
        "Authorization": f"Bearer {TRADIER_TOKEN}",
        "Accept": "application/json",
    }

def _tradier_get(path:str, params:Optional[dict]=None)->dict:
    url = f"{TRADIER_BASE}{path}"
    r = requests.get(url, headers=_tradier_headers(), params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def _tradier_post(path:str, data:dict)->dict:
    url = f"{TRADIER_BASE}{path}"
    r = requests.post(url, headers={**_tradier_headers(), "Content-Type":"application/x-www-form-urlencoded"},
                      data=data, timeout=20)
    r.raise_for_status()
    return r.json()

# ══════════════════════════════════════════════════════════════════════════════
# FORMATTERS
# ══════════════════════════════════════════════════════════════════════════════
def fmt_money(x:Any)->str:
    try:
        if x is None or (isinstance(x,float) and math.isnan(x)): return "N/A"
        return f"${float(x):,.2f}"
    except: return "N/A"

def fmt_num(x:Any,digits:int=2)->str:
    try:
        if x is None or (isinstance(x,float) and math.isnan(x)): return "N/A"
        return f"{float(x):.{digits}f}"
    except: return "N/A"

def fmt_pct(x:Any,digits:int=1)->str:
    try:
        if x is None or (isinstance(x,float) and math.isnan(x)): return "N/A"
        return f"{float(x)*100:.{digits}f}%"
    except: return "N/A"

def safe_get(d:dict,*path,default=None):
    cur=d
    for key in path:
        if not isinstance(cur,dict) or key not in cur: return default
        cur=cur[key]
    return cur

# ══════════════════════════════════════════════════════════════════════════════
# POSITION MEMORY
# ══════════════════════════════════════════════════════════════════════════════
if "active_trade" not in st.session_state: st.session_state.active_trade=None
if "trade_history" not in st.session_state: st.session_state.trade_history=[]

def save_active_trade(trade:dict): st.session_state.active_trade=trade
def clear_active_trade():
    if st.session_state.active_trade:
        st.session_state.trade_history.append({**st.session_state.active_trade,"closed_at":dt.datetime.now().isoformat()})
    st.session_state.active_trade=None
def active_trade_matches(contract_symbol:Optional[str])->bool:
    t=st.session_state.active_trade
    return bool(t and contract_symbol and t.get("contract_symbol")==contract_symbol)

# ══════════════════════════════════════════════════════════════════════════════
# TRADIER DATA FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── Timeframe mapping ──────────────────────────────────────────────────────────
# Tradier has two bar endpoints:
#   timesales: 1min, 5min, 15min  (intraday, up to 40 days)
#   history:   daily, weekly, monthly
# For 1Hour and 4Hour we fetch 5min bars and resample.
_TF_MAP = {
    "1Min":  ("timesales", "1min"),
    "5Min":  ("timesales", "5min"),
    "15Min": ("timesales", "15min"),
    "1Hour": ("timesales", "5min"),   # resample 5min → 1H
    "4Hour": ("timesales", "5min"),   # resample 5min → 4H
    "1Day":  ("history",   "daily"),
    "1Week": ("history",   "weekly"),
}

@st.cache_data(ttl=20)
def get_stock_quote(symbol:str)->dict:
    """Real-time quote for a single symbol. Returns the quote dict."""
    try:
        data = _tradier_get("/v1/markets/quotes", params={"symbols": symbol.upper(), "greeks": "false"})
        q = safe_get(data, "quotes", "quote", default={})
        # When multiple symbols, Tradier returns a list; handle both cases
        if isinstance(q, list):
            q = next((x for x in q if x.get("symbol","").upper()==symbol.upper()), {})
        return q or {}
    except: return {}

@st.cache_data(ttl=30)
def get_stock_bars(symbol:str, timeframe:str="5Min", limit:int=200)->pd.DataFrame:
    """
    Fetch OHLCV bars from Tradier.
    Handles timesales (intraday) and history (daily/weekly).
    For 1Hour and 4Hour, fetches 5min bars and resamples.
    """
    endpoint, interval = _TF_MAP.get(timeframe, ("timesales", "5min"))
    resample_rule = None
    if timeframe == "1Hour":  resample_rule = "1h"
    if timeframe == "4Hour":  resample_rule = "4h"

    try:
        if endpoint == "timesales":
            # Calculate start/end to get enough bars
            # 5min bars: need limit * 5 minutes of data
            # Add extra days to account for weekends/holidays
            minutes_needed = limit * (5 if interval == "5min" else (1 if interval == "1min" else 15))
            if resample_rule:
                # For resampling, fetch more raw bars
                minutes_needed = limit * (60 if timeframe=="1Hour" else 240) + 1440
            days_needed = max(5, math.ceil(minutes_needed / 390) + 3)  # 390 min/trading day
            end_dt   = dt.datetime.now()
            start_dt = end_dt - dt.timedelta(days=days_needed)
            params = {
                "symbol":   symbol.upper(),
                "interval": interval,
                "start":    start_dt.strftime("%Y-%m-%d %H:%M"),
                "end":      end_dt.strftime("%Y-%m-%d %H:%M"),
                "session_filter": "open",
            }
            data = _tradier_get("/v1/markets/timesales", params=params)
            raw = safe_get(data, "series", "data", default=[])
            if not raw: return pd.DataFrame()
            if isinstance(raw, dict): raw = [raw]  # single bar edge case

            df = pd.DataFrame(raw)
            df["Time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")
            df = df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
            for col in ["Open","High","Low","Close","Volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            df = df[["Time","Open","High","Low","Close","Volume"]].sort_values("Time").reset_index(drop=True)

            # Resample to 1H or 4H if needed
            if resample_rule:
                df = df.set_index("Time")
                df = df.resample(resample_rule, label="left", closed="left").agg(
                    Open=("Open","first"), High=("High","max"),
                    Low=("Low","min"),   Close=("Close","last"),
                    Volume=("Volume","sum")
                ).dropna(subset=["Open","Close"]).reset_index()
                df = df.rename(columns={"Time":"Time"})

            # Return last `limit` bars
            return df.tail(limit).reset_index(drop=True)

        else:  # history endpoint (daily / weekly)
            days_needed = limit * (7 if interval=="weekly" else 1) + 30
            end_dt   = dt.date.today()
            start_dt = end_dt - dt.timedelta(days=days_needed)
            params = {
                "symbol":   symbol.upper(),
                "interval": interval,
                "start":    start_dt.strftime("%Y-%m-%d"),
                "end":      end_dt.strftime("%Y-%m-%d"),
            }
            data = _tradier_get("/v1/markets/history", params=params)
            raw = safe_get(data, "history", "day", default=[])
            if not raw: return pd.DataFrame()
            if isinstance(raw, dict): raw = [raw]

            df = pd.DataFrame(raw)
            df["Time"] = pd.to_datetime(df["date"])
            df = df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
            for col in ["Open","High","Low","Close","Volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            df = df[["Time","Open","High","Low","Close","Volume"]].sort_values("Time").reset_index(drop=True)
            return df.tail(limit).reset_index(drop=True)

    except Exception as e:
        st.warning(f"Bar data error ({symbol} {timeframe}): {e}")
        return pd.DataFrame()

@st.cache_data(ttl=120)
def get_vix_spot()->Optional[float]:
    try:
        vix = yf.Ticker("^VIX").history(period="5d", interval="1d", auto_adjust=False)
        if vix is None or vix.empty: return None
        return float(vix["Close"].dropna().iloc[-1])
    except: return None

@st.cache_data(ttl=60)
def get_option_expirations(symbol:str)->List[str]:
    """Fetch all available expiration dates for a symbol."""
    try:
        data = _tradier_get("/v1/markets/options/expirations",
                            params={"symbol": symbol.upper(), "includeAllRoots": "true"})
        exps = safe_get(data, "expirations", "date", default=[])
        if isinstance(exps, str): exps = [exps]
        return sorted(exps) if exps else []
    except: return []

@st.cache_data(ttl=30)
def get_option_chain(symbol:str, expiration:str, option_type:Optional[str]=None)->List[dict]:
    """
    Fetch full options chain for a given expiration with greeks.
    Returns a list of option contract dicts — each has bid, ask, greeks, IV, OI, etc.
    """
    try:
        data = _tradier_get("/v1/markets/options/chains",
                            params={"symbol": symbol.upper(),
                                    "expiration": expiration,
                                    "greeks": "true"})
        contracts = safe_get(data, "options", "option", default=[])
        if not contracts: return []
        if isinstance(contracts, dict): contracts = [contracts]
        if option_type:
            contracts = [c for c in contracts if c.get("option_type","").lower() == option_type.lower()]
        return contracts
    except: return []

def get_open_positions()->list:
    """Fetch open positions from Tradier brokerage account."""
    if not TRADIER_ACCOUNT_ID: return []
    try:
        data = _tradier_get(f"/v1/accounts/{TRADIER_ACCOUNT_ID}/positions")
        pos = safe_get(data, "positions", "position", default=[])
        if isinstance(pos, dict): pos = [pos]
        return pos or []
    except: return []

def find_position(symbol:str)->Optional[dict]:
    for p in get_open_positions():
        if p.get("symbol")==symbol: return p
    return None

def place_option_order(option_symbol:str, qty:int, side:str,
                       order_type:str="market", limit_price:Optional[float]=None)->dict:
    """
    Place an option order via Tradier.
    side: 'buy_to_open' | 'buy_to_close' | 'sell_to_open' | 'sell_to_close'
    """
    if not TRADIER_ACCOUNT_ID:
        raise ValueError("TRADIER_ACCOUNT_ID not set in secrets.")
    data: Dict[str,Any] = {
        "class":         "option",
        "symbol":        option_symbol,   # OCC option symbol e.g. SPY240419C00530000
        "option_symbol": option_symbol,
        "quantity":      str(qty),
        "side":          side,
        "type":          order_type,
        "duration":      "day",
        "preview":       "false",
    }
    if order_type == "limit" and limit_price is not None:
        data["price"] = str(round(limit_price, 2))
    result = _tradier_post(f"/v1/accounts/{TRADIER_ACCOUNT_ID}/orders", data)
    return result.get("order", result)

@st.cache_data(ttl=60)
def get_news(symbol:str, limit:int=8)->list:
    """Fetch news via yfinance as Tradier doesn't have a news endpoint."""
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news or []
        return news[:limit]
    except: return []

# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def add_indicators(df:pd.DataFrame, htf:bool=False)->pd.DataFrame:
    """
    htf=True  → add EMA 21/50/200 (for 1H, 4H, Daily TDA analysis)
    htf=False → add EMA 8/21/50  (for primary chart and lower TFs)
    """
    out=df.copy()
    if out.empty or len(out)<10: return out
    for col in ["Open","High","Low","Close","Volume"]:
        if col in out.columns:
            out[col]=pd.to_numeric(out[col],errors="coerce").astype("float64")
    out=out.dropna(subset=["Close"])
    if out.empty or len(out)<10: return out

    if htf:
        out["EMA_21"] =out["Close"].ewm(span=21, adjust=False).mean()
        out["EMA_50"] =out["Close"].ewm(span=50, adjust=False).mean()
        out["EMA_200"]=out["Close"].ewm(span=200,adjust=False).mean()
        out["EMA_8"]=out["EMA_21"]  # alias for state machine
    else:
        out["EMA_8"] =out["Close"].ewm(span=8, adjust=False).mean()
        out["EMA_21"]=out["Close"].ewm(span=21,adjust=False).mean()
        out["EMA_50"]=out["Close"].ewm(span=50,adjust=False).mean()

    delta=out["Close"].diff()
    gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
    avg_gain=gain.ewm(alpha=1/14,adjust=False).mean()
    avg_loss=loss.ewm(alpha=1/14,adjust=False).mean()
    rs=avg_gain/avg_loss.replace(0,np.nan)
    out["RSI"]=(100-(100/(1+rs))).astype("float64")

    ema12=out["Close"].ewm(span=12,adjust=False).mean()
    ema26=out["Close"].ewm(span=26,adjust=False).mean()
    out["MACD"]=ema12-ema26
    out["MACD_signal"]=out["MACD"].ewm(span=9,adjust=False).mean()
    out["MACD_hist"]=out["MACD"]-out["MACD_signal"]

    rsi_series=out["RSI"].astype("float64")
    rsi_min=rsi_series.rolling(14).min()
    rsi_max=rsi_series.rolling(14).max()
    stoch_rsi=(rsi_series-rsi_min)/(rsi_max-rsi_min+1e-9)
    out["StochRSI_K"]=stoch_rsi.rolling(3).mean()*100
    out["StochRSI_D"]=out["StochRSI_K"].rolling(3).mean()

    hi=out["High"]; lo=out["Low"]; cl=out["Close"]
    tr=pd.concat([hi-lo,(hi-cl.shift()).abs(),(lo-cl.shift()).abs()],axis=1).max(axis=1)
    atr14=tr.ewm(alpha=1/14,adjust=False).mean()
    out["ATR"]=atr14
    dm_plus =(hi-hi.shift()).clip(lower=0)
    dm_minus=(lo.shift()-lo).clip(lower=0)
    dm_plus =dm_plus.where(dm_plus>dm_minus,0)
    dm_minus=dm_minus.where(dm_minus>dm_plus,0)
    di_plus =100*dm_plus.ewm(alpha=1/14,adjust=False).mean()/atr14.replace(0,pd.NA)
    di_minus=100*dm_minus.ewm(alpha=1/14,adjust=False).mean()/atr14.replace(0,pd.NA)
    dx=100*(di_plus-di_minus).abs()/(di_plus+di_minus+1e-9)
    out["ADX"]=dx.ewm(alpha=1/14,adjust=False).mean()
    out["DI_plus"]=di_plus; out["DI_minus"]=di_minus

    if "Time" in out.columns:
        out["_date"]=pd.to_datetime(out["Time"]).dt.date
        out["_tp"]=(out["High"]+out["Low"]+out["Close"])/3
        out["_tpvol"]=out["_tp"]*out["Volume"]
        out["_cumvol"] =out.groupby("_date")["Volume"].transform("cumsum")
        out["_cumtpvol"]=out.groupby("_date")["_tpvol"].transform("cumsum")
        out["VWAP"]=out["_cumtpvol"]/out["_cumvol"].replace(0,np.nan)
        out.drop(columns=["_date","_tp","_tpvol","_cumvol","_cumtpvol"],inplace=True,errors="ignore")

    bb_mid=out["Close"].rolling(20).mean()
    bb_std=out["Close"].rolling(20).std()
    out["BB_upper"]=bb_mid+2*bb_std
    out["BB_lower"]=bb_mid-2*bb_std
    out["BB_mid"]=bb_mid

    kc_mid=out["Close"].ewm(span=20,adjust=False).mean()
    out["KC_upper"]=kc_mid+1.5*atr14
    out["KC_lower"]=kc_mid-1.5*atr14

    out["Squeeze_ON"]=(out["BB_upper"]<out["KC_upper"])&(out["BB_lower"]>out["KC_lower"])
    sq_val=out["Close"]-((out["High"].rolling(20).max()+out["Low"].rolling(20).min())/2+bb_mid)/2
    out["Squeeze_hist"]=sq_val.ewm(span=5,adjust=False).mean()

    out["Vol_avg"]=out["Volume"].rolling(20).mean()
    out["Vol_ratio"]=out["Volume"]/out["Vol_avg"].replace(0,pd.NA)

    return out

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def stock_signal(df:pd.DataFrame)->Tuple[str,int,List[str],int]:
    if df.empty or len(df)<30:
        return "NEUTRAL",0,["Not enough bar data."],0

    row=df.iloc[-1]
    score=0; reasons=[]; certainty_points=0; certainty_max=0

    certainty_max+=3
    if row["Close"]>row["EMA_8"]:
        score+=1; certainty_points+=1; reasons.append("✅ Price above EMA 8.")
    else:
        score-=1; reasons.append("❌ Price below EMA 8.")
    if row["EMA_8"]>row["EMA_21"]:
        score+=1; certainty_points+=1; reasons.append("✅ EMA 8 above EMA 21.")
    else:
        score-=1; reasons.append("❌ EMA 8 below EMA 21.")
    if row["EMA_21"]>row["EMA_50"]:
        score+=1; certainty_points+=1; reasons.append("✅ EMA 21 above EMA 50.")
    else:
        score-=1; reasons.append("❌ EMA 21 below EMA 50.")

    certainty_max+=1
    if pd.notna(row.get("RSI")):
        if row["RSI"]>55:
            score+=1; certainty_points+=1; reasons.append(f"✅ RSI {row['RSI']:.1f} — bullish momentum.")
        elif row["RSI"]<45:
            score-=1; reasons.append(f"❌ RSI {row['RSI']:.1f} — bearish momentum.")
        else:
            reasons.append(f"⚪ RSI {row['RSI']:.1f} — neutral zone.")

    certainty_max+=2
    if pd.notna(row.get("MACD")) and pd.notna(row.get("MACD_signal")):
        if row["MACD"]>row["MACD_signal"]:
            score+=1; certainty_points+=1; reasons.append("✅ MACD above signal line — bullish.")
        else:
            score-=1; reasons.append("❌ MACD below signal line — bearish.")
        if pd.notna(row.get("MACD_hist")):
            if row["MACD_hist"]>0:
                score+=1; certainty_points+=1; reasons.append("✅ MACD histogram positive — momentum building.")
            else:
                score-=1; reasons.append("❌ MACD histogram negative — momentum fading.")

    certainty_max+=1
    if pd.notna(row.get("StochRSI_K")) and pd.notna(row.get("StochRSI_D")):
        if row["StochRSI_K"]>row["StochRSI_D"] and row["StochRSI_K"]<80:
            score+=1; certainty_points+=1; reasons.append(f"✅ StochRSI K({row['StochRSI_K']:.0f}) crossed above D — bullish.")
        elif row["StochRSI_K"]<row["StochRSI_D"] and row["StochRSI_K"]>20:
            score-=1; reasons.append(f"❌ StochRSI K({row['StochRSI_K']:.0f}) crossed below D — bearish.")
        elif row["StochRSI_K"]>=80:
            reasons.append(f"⚠️ StochRSI overbought ({row['StochRSI_K']:.0f}) — caution on calls.")
        elif row["StochRSI_K"]<=20:
            reasons.append(f"⚠️ StochRSI oversold ({row['StochRSI_K']:.0f}) — caution on puts.")

    certainty_max+=2
    if pd.notna(row.get("ADX")):
        adx=row["ADX"]
        if adx>25:
            certainty_points+=1
            if pd.notna(row.get("DI_plus")) and pd.notna(row.get("DI_minus")):
                if row["DI_plus"]>row["DI_minus"]:
                    score+=1; certainty_points+=1; reasons.append(f"✅ ADX {adx:.1f} — strong uptrend confirmed.")
                else:
                    score-=1; reasons.append(f"❌ ADX {adx:.1f} — strong downtrend confirmed.")
        elif adx<18:
            score-=1; reasons.append(f"⚠️ ADX {adx:.1f} — choppy market, low conviction.")
        else:
            reasons.append(f"⚪ ADX {adx:.1f} — moderate trend strength.")

    certainty_max+=1
    if pd.notna(row.get("VWAP")):
        if row["Close"]>row["VWAP"]:
            score+=1; certainty_points+=1; reasons.append(f"✅ Price above VWAP ${row['VWAP']:.2f}.")
        else:
            score-=1; reasons.append(f"❌ Price below VWAP ${row['VWAP']:.2f}.")

    certainty_max+=1
    if pd.notna(row.get("Squeeze_ON")) and pd.notna(row.get("Squeeze_hist")):
        if not row["Squeeze_ON"] and pd.notna(df.iloc[-2].get("Squeeze_ON")) and df.iloc[-2]["Squeeze_ON"]:
            if row["Squeeze_hist"]>0:
                score+=2; certainty_points+=1; reasons.append("🚀 TTM Squeeze FIRED bullish.")
            else:
                score-=2; reasons.append("🚀 TTM Squeeze FIRED bearish.")
        elif row["Squeeze_ON"]:
            reasons.append("⏳ TTM Squeeze ON — coiling.")
        else:
            reasons.append("⚪ TTM Squeeze off — normal volatility.")

    certainty_max+=1
    if pd.notna(row.get("Vol_ratio")):
        if row["Vol_ratio"]>1.5:
            certainty_points+=1; reasons.append(f"✅ Volume {row['Vol_ratio']:.1f}× average.")
        elif row["Vol_ratio"]<0.7:
            score-=1; reasons.append(f"⚠️ Volume {row['Vol_ratio']:.1f}× average — weak conviction.")
        else:
            reasons.append(f"⚪ Volume {row['Vol_ratio']:.1f}× average.")

    if score>=3:   bias="BULLISH"
    elif score<=-3: bias="BEARISH"
    else:           bias="NEUTRAL"
    certainty_pct=int(certainty_points/max(certainty_max,1)*100) if bias!="NEUTRAL" else max(0,int((certainty_points/max(certainty_max,1)*100)-20))
    return bias,score,reasons,certainty_pct

def stock_signal_htf(df:pd.DataFrame)->Tuple[str,int,List[str],int]:
    """Higher-timeframe signal using EMA 21/50/200. Used for 1H, 4H, Daily TDA cards."""
    if df.empty or len(df)<30:
        return "NEUTRAL",0,["Not enough bar data."],0

    row=df.iloc[-1]
    score=0; reasons=[]; certainty_points=0; certainty_max=0

    certainty_max+=3
    if pd.notna(row.get("EMA_21")) and row["Close"]>row["EMA_21"]:
        score+=1; certainty_points+=1; reasons.append("✅ Price above EMA 21.")
    else:
        score-=1; reasons.append("❌ Price below EMA 21.")
    if pd.notna(row.get("EMA_21")) and pd.notna(row.get("EMA_50")) and row["EMA_21"]>row["EMA_50"]:
        score+=1; certainty_points+=1; reasons.append("✅ EMA 21 above EMA 50.")
    else:
        score-=1; reasons.append("❌ EMA 21 below EMA 50.")
    if pd.notna(row.get("EMA_50")) and pd.notna(row.get("EMA_200")) and row["EMA_50"]>row["EMA_200"]:
        score+=1; certainty_points+=1; reasons.append("✅ EMA 50 above EMA 200 — long-term bull.")
    else:
        score-=1; reasons.append("❌ EMA 50 below EMA 200 — long-term bear.")

    certainty_max+=1
    if pd.notna(row.get("EMA_200")):
        if row["Close"]>row["EMA_200"]:
            score+=1; certainty_points+=1; reasons.append(f"✅ Price above EMA 200 (${row['EMA_200']:.2f}).")
        else:
            score-=1; reasons.append(f"❌ Price below EMA 200 (${row['EMA_200']:.2f}) — bearish structure.")

    certainty_max+=1
    if pd.notna(row.get("RSI")):
        if row["RSI"]>55:
            score+=1; certainty_points+=1; reasons.append(f"✅ RSI {row['RSI']:.1f} — bullish momentum.")
        elif row["RSI"]<45:
            score-=1; reasons.append(f"❌ RSI {row['RSI']:.1f} — bearish momentum.")
        else:
            reasons.append(f"⚪ RSI {row['RSI']:.1f} — neutral zone.")

    certainty_max+=2
    if pd.notna(row.get("MACD")) and pd.notna(row.get("MACD_signal")):
        if row["MACD"]>row["MACD_signal"]:
            score+=1; certainty_points+=1; reasons.append("✅ MACD above signal — bullish.")
        else:
            score-=1; reasons.append("❌ MACD below signal — bearish.")
        if pd.notna(row.get("MACD_hist")):
            if row["MACD_hist"]>0:
                score+=1; certainty_points+=1; reasons.append("✅ MACD histogram positive.")
            else:
                score-=1; reasons.append("❌ MACD histogram negative.")

    certainty_max+=2
    if pd.notna(row.get("ADX")):
        adx=row["ADX"]
        if adx>25:
            certainty_points+=1
            if pd.notna(row.get("DI_plus")) and pd.notna(row.get("DI_minus")):
                if row["DI_plus"]>row["DI_minus"]:
                    score+=1; certainty_points+=1; reasons.append(f"✅ ADX {adx:.1f} — strong uptrend.")
                else:
                    score-=1; reasons.append(f"❌ ADX {adx:.1f} — strong downtrend.")
        elif adx<18:
            score-=1; reasons.append(f"⚠️ ADX {adx:.1f} — choppy, low conviction.")

    if score>=3:   bias="BULLISH"
    elif score<=-3: bias="BEARISH"
    else:           bias="NEUTRAL"
    certainty_pct=int(certainty_points/max(certainty_max,1)*100) if bias!="NEUTRAL" else max(0,int((certainty_points/max(certainty_max,1)*100)-20))
    return bias,score,reasons,certainty_pct

# ══════════════════════════════════════════════════════════════════════════════
# MARKET SESSION
# ══════════════════════════════════════════════════════════════════════════════
def market_session()->Tuple[str,str]:
    now=dt.datetime.now(dt.timezone(dt.timedelta(hours=-4)))
    t=now.time()
    if t<dt.time(4,0):   return "Closed","🔴"
    if t<dt.time(9,30):  return "Pre-Market","🟡"
    if t<dt.time(16,0):  return "Market Open","🟢"
    if t<dt.time(20,0):  return "After-Hours","🟡"
    return "Closed","🔴"

def expected_move(price:float,iv:float,dte:int)->Tuple[float,float]:
    move=price*iv*math.sqrt(dte/365)
    return round(price-move,2),round(price+move,2)

# ══════════════════════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME SIGNAL
# ══════════════════════════════════════════════════════════════════════════════
_TF_LIMITS = {"1Min":200,"5Min":200,"15Min":150,"1Hour":120,"4Hour":80,"1Day":100,"1Week":80}

@st.cache_data(ttl=60)
def _get_tf_bias(symbol:str,tf:str)->Tuple[str,int,List[str],int]:
    limit=_TF_LIMITS.get(tf,150)
    try:
        df=add_indicators(get_stock_bars(symbol,tf,limit))
        if df.empty or len(df)<20: return "NEUTRAL",0,["Insufficient data."],0
        return stock_signal(df)
    except Exception as e:
        return "ERROR",0,[str(e)],0

def multi_tf_signal(symbol:str,primary_tf:str)->Tuple[str,int,List[str],int,Dict[str,str]]:
    tfs=["5Min","15Min","1Hour"]
    tf_biases={}
    for tf in tfs:
        bias,_,_,_=_get_tf_bias(symbol,tf)
        tf_biases[tf]=bias

    primary_bias,primary_score,primary_reasons,primary_cert=_get_tf_bias(symbol,primary_tf)

    biases=[b for b in tf_biases.values() if b not in ("NEUTRAL","ERROR")]
    if len(biases)>=2:
        bull=sum(1 for b in biases if b=="BULLISH")
        bear=sum(1 for b in biases if b=="BEARISH")
        if bull>bear:
            primary_cert=min(100,primary_cert+15)
            primary_reasons.insert(0,"✅ Multi-TF alignment: majority BULLISH.")
        elif bear>bull:
            primary_cert=min(100,primary_cert+15)
            primary_reasons.insert(0,"✅ Multi-TF alignment: majority BEARISH.")
        else:
            primary_cert=max(0,primary_cert-25)
            primary_reasons.insert(0,"⚠️ Timeframes CONFLICT — certainty reduced.")

    return primary_bias,primary_score,primary_reasons,primary_cert,tf_biases

# ══════════════════════════════════════════════════════════════════════════════
# TOP-DOWN ANALYSIS  (W / D / 4H / 1H / 15min / 5min)
# ══════════════════════════════════════════════════════════════════════════════
_TDA_TIMEFRAMES = [
    ("Weekly",  "1Week",  80,  False, "Macro trend — big-picture direction."),
    ("Daily",   "1Day",   120, True,  "Intermediate trend — swing direction. EMA 21/50/200."),
    ("4-Hour",  "4Hour",  80,  True,  "Short-term trend — intraday swing bias. EMA 21/50/200."),
    ("1-Hour",  "1Hour",  120, True,  "Intraday momentum — entry zone. EMA 21/50/200."),
    ("15-Min",  "15Min",  150, False, "Execution context — is momentum aligned?"),
    ("5-Min",   "5Min",   200, False, "Entry trigger — fine-tune entry timing."),
]

@st.cache_data(ttl=45)
def get_tda_panel(symbol:str)->List[dict]:
    results=[]
    for label,tf,limit,use_htf,purpose in _TDA_TIMEFRAMES:
        try:
            df=add_indicators(get_stock_bars(symbol,tf,limit), htf=use_htf)
            if df.empty or len(df)<20:
                results.append({"label":label,"tf":tf,"bias":"N/A","score":0,
                                 "reasons":["Not enough data."],"cert":0,"purpose":purpose,"htf":use_htf})
                continue
            if use_htf:
                bias,score,reasons,cert=stock_signal_htf(df)
            else:
                bias,score,reasons,cert=stock_signal(df)
            results.append({"label":label,"tf":tf,"bias":bias,"score":score,
                             "reasons":reasons[:3],"cert":cert,"purpose":purpose,"htf":use_htf})
        except Exception as e:
            results.append({"label":label,"tf":tf,"bias":"ERROR","score":0,
                             "reasons":[str(e)],"cert":0,"purpose":purpose,"htf":use_htf})
    return results

def render_tda_panel(symbol:str):
    section("Top-Down Analysis — W / D / 4H / 1H / 15min / 5min")
    tda=get_tda_panel(symbol)

    valid_biases=[r["bias"] for r in tda if r["bias"] not in ("N/A","ERROR","NEUTRAL")]
    bull_count=sum(1 for b in valid_biases if b=="BULLISH")
    bear_count=sum(1 for b in valid_biases if b=="BEARISH")
    total=len(valid_biases)

    if total==0:
        cascade_html='<div class="cascade-neut">No data available for cascade analysis.</div>'
    elif bull_count==total:
        cascade_html=f'<div class="cascade-ok">✅ FULL BULLISH CASCADE — All {total} timeframes aligned BULLISH. Highest confidence for CALLS.</div>'
    elif bear_count==total:
        cascade_html=f'<div class="cascade-ok" style="background:#450a0a;border-color:#f87171;color:#f87171">✅ FULL BEARISH CASCADE — All {total} timeframes aligned BEARISH. Highest confidence for PUTS.</div>'
    elif bull_count>=4:
        cascade_html=f'<div class="cascade-ok">{bull_count}/{total} timeframes BULLISH — Strong bullish alignment. Good for CALLS.</div>'
    elif bear_count>=4:
        cascade_html=f'<div class="cascade-ok" style="background:#450a0a;border-color:#f87171;color:#f87171">{bear_count}/{total} timeframes BEARISH — Strong bearish alignment. Good for PUTS.</div>'
    elif bull_count>bear_count:
        cascade_html=f'<div class="cascade-warn">⚠️ {bull_count}/{total} BULLISH — Partial alignment. Reduce size or wait.</div>'
    elif bear_count>bull_count:
        cascade_html=f'<div class="cascade-warn">⚠️ {bear_count}/{total} BEARISH — Partial alignment. Reduce size or wait.</div>'
    else:
        cascade_html='<div class="cascade-neut">⚪ Mixed signals — No clear directional edge. Stay flat.</div>'

    st.markdown(cascade_html,unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)

    cols=st.columns(6)
    for i,r in enumerate(tda):
        bias=r["bias"]
        if bias=="BULLISH":   bias_cls="tda-bias-bull"; bias_icon="▲ BULLISH"
        elif bias=="BEARISH": bias_cls="tda-bias-bear"; bias_icon="▼ BEARISH"
        elif bias=="NEUTRAL": bias_cls="tda-bias-neut"; bias_icon="◆ NEUTRAL"
        else:                 bias_cls="tda-bias-neut"; bias_icon=f"— {bias}"

        ema_note=" <small style='color:#4b5563'>(21/50/200)</small>" if r.get("htf") else ""
        cert_color="#4ade80" if r["cert"]>=65 else ("#f59e0b" if r["cert"]>=45 else "#ef4444")
        reasons_html="<br>".join(r["reasons"][:3])

        card=f"""
<div class="tda-card">
  <div class="tda-tf">{r['label']}{ema_note}</div>
  <div class="{bias_cls}">{bias_icon}</div>
  <div class="tda-score" style="color:{cert_color}">Score: {r['score']:+d} &nbsp;|&nbsp; {r['cert']}%</div>
  <div class="tda-reason">{reasons_html}</div>
</div>"""
        cols[i].markdown(card,unsafe_allow_html=True)

    with st.expander("What is Top-Down Analysis?"):
        st.markdown("""
**Top-Down Analysis** starts from the highest timeframe and works down to the execution timeframe.
Trade in the direction of the dominant trend.

| Timeframe | EMAs Used | Role |
|-----------|-----------|------|
| **Weekly** | 8/21/50 | Macro trend — big-picture direction |
| **Daily** | **21/50/200** | Intermediate trend — swing bias |
| **4-Hour** | **21/50/200** | Short-term trend — intraday swing |
| **1-Hour** | **21/50/200** | Intraday momentum — entry zone |
| **15-Min** | 8/21/50 | Execution context |
| **5-Min** | 8/21/50 | Entry trigger |

**Cascade rule:** The more timeframes that agree, the higher the conviction.
Full 6/6 alignment = highest-probability setup. Mixed = stay flat or reduce size.
""")

# ══════════════════════════════════════════════════════════════════════════════
# STATE MACHINE  (FIXED: prev_state tracked inside loop)
# ══════════════════════════════════════════════════════════════════════════════
def state_series(df:pd.DataFrame)->pd.DataFrame:
    out=df.copy()
    if out.empty: out["state"]=[]; out["prev_state"]=[]; return out
    states=[]; prev_states=[]; mode="FLAT"; prev_state="NONE"
    for _,row in out.iterrows():
        score=0
        score+=1 if row["Close"]>row["EMA_8"]  else -1
        score+=1 if row["EMA_8"] >row["EMA_21"] else -1
        score+=1 if row["EMA_21"]>row["EMA_50"] else -1
        if pd.notna(row.get("RSI")):
            if row["RSI"]>55:   score+=1
            elif row["RSI"]<45: score-=1
        bullish=score>=3; bearish=score<=-3
        weakening_long =(row["Close"]<row["EMA_8"])or(pd.notna(row.get("RSI"))and row["RSI"]<48)
        weakening_short=(row["Close"]>row["EMA_8"])or(pd.notna(row.get("RSI"))and row["RSI"]>52)
        state="NO TRADE"
        if mode=="FLAT":
            if bullish:   state="BUY";       mode="LONG"
            elif bearish: state="SELL";      mode="SHORT"
        elif mode=="LONG":
            if bearish or weakening_long:    state="EXIT BUY";  mode="FLAT"
            else:                            state="HOLD BUY"
        elif mode=="SHORT":
            if bullish or weakening_short:   state="EXIT SELL"; mode="FLAT"
            else:                            state="HOLD SELL"
        prev_states.append(prev_state)
        states.append(state)
        prev_state=state   # ← FIXED: track inside loop
    out["state"]=states
    out["prev_state"]=prev_states
    return out

def find_chart_signals(df:pd.DataFrame):
    marked=state_series(df)
    buy_rows,sell_rows,exit_buy_rows,exit_sell_rows=[],[],[],[]
    for _,row in marked.iterrows():
        base={"time":row["Time"],"low":row["Low"],"high":row["High"],
              "atr":0 if pd.isna(row.get("ATR")) else float(row.get("ATR",0)),
              "close":row["Close"]}
        state=row["state"]; prev=row["prev_state"]
        if state=="BUY"       and prev!="BUY":      buy_rows.append(base)
        elif state=="SELL"    and prev!="SELL":     sell_rows.append(base)
        elif state=="EXIT BUY"  and prev!="EXIT BUY":  exit_buy_rows.append(base)
        elif state=="EXIT SELL" and prev!="EXIT SELL": exit_sell_rows.append(base)
    return (pd.DataFrame(buy_rows), pd.DataFrame(sell_rows),
            pd.DataFrame(exit_buy_rows), pd.DataFrame(exit_sell_rows))

# ══════════════════════════════════════════════════════════════════════════════
# CONTRACT QUALITY
# ══════════════════════════════════════════════════════════════════════════════
def contract_quality(underlying_price,strike,bid,ask,volume,open_interest,iv,delta)->dict:
    score=100; reasons=[]; spread=None; spread_pct=None
    if bid is not None and ask is not None:
        spread=float(ask)-float(bid)
        if ask and ask>0: spread_pct=spread/float(ask)
    if spread_pct is None:      score-=25; reasons.append("No usable bid/ask spread.")
    elif spread_pct>0.20:       score-=30; reasons.append("Spread too wide.")
    elif spread_pct>0.10:       score-=15; reasons.append("Spread somewhat wide.")
    else:                       reasons.append("Spread acceptable.")
    if ask is None:             score-=15; reasons.append("No ask price.")
    elif ask<0.10:              score-=20; reasons.append("Premium too tiny / noisy.")
    elif ask>10:                score-=10; reasons.append("Premium is expensive.")
    else:                       reasons.append("Premium in a workable range.")
    if underlying_price is not None and strike is not None and underlying_price>0:
        moneyness=abs(float(strike)-float(underlying_price))/float(underlying_price)
        if moneyness>0.05:      score-=20; reasons.append("Strike far from underlying.")
        elif moneyness>0.02:    score-=8;  reasons.append("Strike slightly far from underlying.")
        else:                   reasons.append("Strike near the underlying.")
    if volume is not None:
        if float(volume)<10:    score-=15; reasons.append("Low contract volume.")
        else:                   reasons.append("Volume acceptable.")
    if open_interest is not None:
        if float(open_interest)<50: score-=15; reasons.append("Low open interest.")
        else:                   reasons.append("Open interest acceptable.")
    if iv is not None and float(iv)>2.0: score-=10; reasons.append("Implied volatility very high.")
    reasons.append("Delta available." if delta is not None else "Greeks unavailable.")
    score=max(0,min(100,score))
    return {"score":score,"quality_ok":score>=55,"spread":spread,"spread_pct":spread_pct,"reasons":reasons}

# ══════════════════════════════════════════════════════════════════════════════
# IV RANK + PUT/CALL RATIO
# ══════════════════════════════════════════════════════════════════════════════
def compute_put_call_ratio(contracts_raw:list)->Tuple[Optional[float],str]:
    call_oi=put_oi=0
    for c in contracts_raw:
        oi=c.get("open_interest") or 0
        if c.get("option_type","").lower()=="call": call_oi+=float(oi)
        elif c.get("option_type","").lower()=="put":  put_oi+=float(oi)
    if call_oi==0: return None,"N/A"
    pcr=put_oi/call_oi
    if pcr<0.7:   sentiment="Bullish (more calls)"
    elif pcr>1.2: sentiment="Bearish (more puts)"
    else:         sentiment="Neutral"
    return round(pcr,2),sentiment

# ══════════════════════════════════════════════════════════════════════════════
# SMART CONTRACT AUTO-PICKER
# ══════════════════════════════════════════════════════════════════════════════
def auto_pick_contract(contracts_raw:list,option_type:str,underlying_price:Optional[float],
                       min_dte:int=21,max_dte:int=45)->Optional[dict]:
    today=dt.date.today()
    candidates=[]
    for c in contracts_raw:
        exp=c.get("expiration_date")
        if not exp: continue
        try: dte=(dt.date.fromisoformat(exp)-today).days
        except: continue
        if not (min_dte<=dte<=max_dte): continue
        if c.get("option_type","").lower() != option_type.lower(): continue
        # Tradier chain already has bid/ask/greeks in the same object
        bid   = c.get("bid")
        ask   = c.get("ask")
        greeks= c.get("greeks") or {}
        delta = greeks.get("delta")
        oi    = c.get("open_interest") or 0
        if delta is None or bid is None or ask is None: continue
        delta_abs=abs(float(delta))
        if not (0.35<=delta_abs<=0.60): continue
        if float(oi)<100: continue
        spread_pct=(float(ask)-float(bid))/float(ask) if float(ask)>0 else 1
        if spread_pct>0.20: continue
        candidates.append({**c,"_delta":float(delta),"_bid":float(bid),"_ask":float(ask),
                            "_dte":dte,"_spread_pct":spread_pct})
    if not candidates: return None
    candidates.sort(key=lambda x:(abs(abs(x["_delta"])-0.50),x["_spread_pct"]))
    return candidates[0]

# ══════════════════════════════════════════════════════════════════════════════
# OPTIONS STATE MACHINE
# ══════════════════════════════════════════════════════════════════════════════
def derive_options_state(stock_bias:str,option_side:str,has_position:bool,quality_ok:bool)->str:
    side=option_side.upper()
    if not quality_ok: return "NO TRADE"
    if not has_position:
        if stock_bias=="BULLISH" and side=="CALL": return "ENTER CALL"
        if stock_bias=="BEARISH" and side=="PUT":  return "ENTER PUT"
        return "NO TRADE"
    else:
        if side=="CALL":
            if stock_bias=="BULLISH": return "HOLD CALL"
            return "EXIT CALL"
        if side=="PUT":
            if stock_bias=="BEARISH": return "HOLD PUT"
            return "EXIT PUT"
    return "NO TRADE"

def manage_active_trade(active_trade:Optional[dict],current_premium:Optional[float],stock_bias:str)->dict:
    if not active_trade: return {"state":None,"notes":[]}
    notes=[]; side=active_trade["option_side"].upper(); state=f"HOLD {side}"
    stop_hit=tp1_hit=tp2_hit=weakening=False
    if current_premium is not None:
        if current_premium<=active_trade["premium_stop"]:   stop_hit=True; notes.append("Premium stop hit.")
        elif current_premium>=active_trade["tp2"]:           tp2_hit=True;  notes.append("TP2 hit.")
        elif current_premium>=active_trade["tp1"]:           tp1_hit=True;  notes.append("TP1 hit.")
    if side=="CALL" and stock_bias!="BULLISH": weakening=True; notes.append("Underlying bias weakened for calls.")
    if side=="PUT"  and stock_bias!="BEARISH": weakening=True; notes.append("Underlying bias weakened for puts.")
    if stop_hit or tp2_hit: state=f"EXIT {side}"
    elif tp1_hit:           state="TP1 HIT"
    elif weakening:         state="WEAKENING"
    return {"state":state,"notes":notes}

# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY MULTI-PANEL CHART
# ══════════════════════════════════════════════════════════════════════════════
_CHART_BG   = "#0d1117"
_CHART_GRID = "#21262d"
_CHART_TEXT = "#c9d1d9"
_CHART_AXIS = "#30363d"

def _plotly_layout(fig:go.Figure, rows:int, height:int=820):
    """Apply consistent dark theme to a Plotly figure."""
    fig.update_layout(
        height=height,
        paper_bgcolor=_CHART_BG,
        plot_bgcolor=_CHART_BG,
        font=dict(color=_CHART_TEXT, family="monospace", size=11),
        margin=dict(l=0, r=60, t=30, b=20),
        legend=dict(
            bgcolor="rgba(22,27,34,0.85)",
            bordercolor=_CHART_AXIS,
            borderwidth=1,
            font=dict(size=10),
            x=0.01, y=0.99,
        ),
        # ── Anti-glitch settings ──────────────────────────────────────────
        uirevision="chart_stable",
        hovermode="x",
        dragmode="pan",
        xaxis_rangeslider_visible=False,
        autosize=True,
    )
    for i in range(1, rows+1):
        fig.update_xaxes(
            gridcolor=_CHART_GRID, gridwidth=1,
            zeroline=False, linecolor=_CHART_AXIS,
            tickfont=dict(color=_CHART_TEXT, size=10),
            showspikes=True, spikecolor="#58a6ff",
            spikethickness=1, spikemode="across",
            row=i, col=1,
        )
        fig.update_yaxes(
            gridcolor=_CHART_GRID, gridwidth=1,
            zeroline=False, linecolor=_CHART_AXIS,
            tickfont=dict(color=_CHART_TEXT, size=10),
            showspikes=True, spikecolor="#58a6ff",
            spikethickness=1, side="right",
            row=i, col=1,
        )
    return fig

def make_chart(df:pd.DataFrame, symbol:str, breakeven:Optional[float]=None):
    """Render a Plotly multi-panel chart: Price / Volume / RSI+StochRSI / MACD / Squeeze."""
    if df.empty:
        st.info("No chart data available.")
        return

    cdf = df.copy()
    cdf["Time"] = pd.to_datetime(cdf["Time"], errors="coerce")
    if cdf["Time"].dt.tz is not None:
        cdf["Time"] = cdf["Time"].dt.tz_convert("America/New_York").dt.tz_localize(None)

    has_vwap    = "VWAP"        in cdf.columns and cdf["VWAP"].notna().sum()>0
    has_bb      = "BB_upper"    in cdf.columns and cdf["BB_upper"].notna().sum()>0
    has_rsi     = "RSI"         in cdf.columns and cdf["RSI"].notna().sum()>0
    has_macd    = "MACD"        in cdf.columns and cdf["MACD"].notna().sum()>0
    has_squeeze = "Squeeze_hist"in cdf.columns and cdf["Squeeze_hist"].notna().sum()>0
    has_volume  = "Volume"      in cdf.columns and cdf["Volume"].notna().sum()>0

    # Count sub-panels
    n_rows = 1
    row_heights = [0.50]
    if has_volume:  n_rows+=1; row_heights.append(0.10)
    if has_rsi:     n_rows+=1; row_heights.append(0.14)
    if has_macd:    n_rows+=1; row_heights.append(0.14)
    if has_squeeze: n_rows+=1; row_heights.append(0.12)
    total = sum(row_heights)
    row_heights = [h/total for h in row_heights]

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
    )

    # ── Row 1: Candlestick ────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=cdf["Time"],
        open=cdf["Open"], high=cdf["High"],
        low=cdf["Low"],   close=cdf["Close"],
        increasing_line_color="#00c864", increasing_fillcolor="#00c864",
        decreasing_line_color="#ef4444", decreasing_fillcolor="#ef4444",
        name=symbol, showlegend=False,
        whiskerwidth=0.3,
    ), row=1, col=1)

    # EMA lines
    for col_name, color, dash, label in [
        ("EMA_8",  "#f59e0b", "solid",  "EMA 8"),
        ("EMA_21", "#3b82f6", "solid",  "EMA 21"),
        ("EMA_50", "#a855f7", "solid",  "EMA 50"),
    ]:
        if col_name in cdf.columns and cdf[col_name].notna().sum() > 0:
            fig.add_trace(go.Scatter(
                x=cdf["Time"], y=cdf[col_name],
                mode="lines", name=label,
                line=dict(color=color, width=1.5, dash=dash),
                hovertemplate=f"{label}: %{{y:.2f}}<extra></extra>",
            ), row=1, col=1)

    # VWAP
    if has_vwap:
        fig.add_trace(go.Scatter(
            x=cdf["Time"], y=cdf["VWAP"],
            mode="lines", name="VWAP",
            line=dict(color="#06b6d4", width=1.2, dash="dot"),
            hovertemplate="VWAP: %{y:.2f}<extra></extra>",
        ), row=1, col=1)

    # Bollinger Bands (filled area)
    if has_bb:
        fig.add_trace(go.Scatter(
            x=cdf["Time"], y=cdf["BB_upper"],
            mode="lines", name="BB+",
            line=dict(color="rgba(139,92,246,0.4)", width=1, dash="dot"),
            hovertemplate="BB+: %{y:.2f}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=cdf["Time"], y=cdf["BB_lower"],
            mode="lines", name="BB-",
            fill="tonexty",
            fillcolor="rgba(139,92,246,0.05)",
            line=dict(color="rgba(139,92,246,0.4)", width=1, dash="dot"),
            hovertemplate="BB-: %{y:.2f}<extra></extra>",
        ), row=1, col=1)

    # Breakeven line
    if breakeven is not None:
        fig.add_hline(
            y=float(breakeven), row=1, col=1,
            line=dict(color="#ef4444", width=1.5, dash="dash"),
            annotation_text=f"BE ${breakeven:.2f}",
            annotation_font=dict(color="#ef4444", size=10),
        )

    # ── Signal markers ────────────────────────────────────────────────────
    buy_df, sell_df, exit_buy_df, exit_sell_df = find_chart_signals(cdf)
    if not buy_df.empty:
        fig.add_trace(go.Scatter(
            x=buy_df["time"], y=buy_df["low"] * 0.9985,
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=12, color="#00c864",
                        line=dict(color="#00c864", width=1)),
            text=[f"BUY<br>{v:.2f}" for v in buy_df["close"]],
            textposition="bottom center",
            textfont=dict(color="#00c864", size=9),
            name="BUY", showlegend=True,
            hovertemplate="BUY @ %{customdata:.2f}<extra></extra>",
            customdata=buy_df["close"],
        ), row=1, col=1)
    if not sell_df.empty:
        fig.add_trace(go.Scatter(
            x=sell_df["time"], y=sell_df["high"] * 1.0015,
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=12, color="#ef4444",
                        line=dict(color="#ef4444", width=1)),
            text=[f"SELL<br>{v:.2f}" for v in sell_df["close"]],
            textposition="top center",
            textfont=dict(color="#ef4444", size=9),
            name="SELL", showlegend=True,
            hovertemplate="SELL @ %{customdata:.2f}<extra></extra>",
            customdata=sell_df["close"],
        ), row=1, col=1)
    if not exit_buy_df.empty:
        fig.add_trace(go.Scatter(
            x=exit_buy_df["time"], y=exit_buy_df["high"] * 1.0015,
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=12, color="#fb923c",
                        line=dict(color="#fb923c", width=1)),
            text=[f"EXIT<br>{v:.2f}" for v in exit_buy_df["close"]],
            textposition="top center",
            textfont=dict(color="#fb923c", size=9),
            name="EXIT BUY", showlegend=True,
            hovertemplate="EXIT BUY @ %{customdata:.2f}<extra></extra>",
            customdata=exit_buy_df["close"],
        ), row=1, col=1)
    if not exit_sell_df.empty:
        fig.add_trace(go.Scatter(
            x=exit_sell_df["time"], y=exit_sell_df["low"] * 0.9985,
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=12, color="#fb923c",
                        line=dict(color="#fb923c", width=1)),
            text=[f"EXIT<br>{v:.2f}" for v in exit_sell_df["close"]],
            textposition="bottom center",
            textfont=dict(color="#fb923c", size=9),
            name="EXIT SELL", showlegend=True,
            hovertemplate="EXIT SELL @ %{customdata:.2f}<extra></extra>",
            customdata=exit_sell_df["close"],
        ), row=1, col=1)

    current_row = 2

    # ── Row 2: Volume ──────────────────────────────────────────────────────
    if has_volume:
        vol_colors = ["rgba(0,200,100,0.4)" if c >= o else "rgba(239,68,68,0.4)"
                      for c, o in zip(cdf["Close"], cdf["Open"])]
        fig.add_trace(go.Bar(
            x=cdf["Time"], y=cdf["Volume"],
            marker_color=vol_colors,
            name="Volume", showlegend=False,
            hovertemplate="Vol: %{y:,.0f}<extra></extra>",
        ), row=current_row, col=1)
        fig.update_yaxes(title_text="Vol", title_font=dict(size=9),
                         row=current_row, col=1)
        current_row += 1

    # ── Row 3: RSI + StochRSI ─────────────────────────────────────────────
    if has_rsi:
        fig.add_trace(go.Scatter(
            x=cdf["Time"], y=cdf["RSI"],
            mode="lines", name="RSI",
            line=dict(color="#06b6d4", width=1.5),
            hovertemplate="RSI: %{y:.1f}<extra></extra>",
        ), row=current_row, col=1)
        if "StochRSI_K" in cdf.columns:
            fig.add_trace(go.Scatter(
                x=cdf["Time"], y=cdf["StochRSI_K"],
                mode="lines", name="StochK",
                line=dict(color="#f59e0b", width=1, dash="dot"),
                hovertemplate="K: %{y:.1f}<extra></extra>",
            ), row=current_row, col=1)
        if "StochRSI_D" in cdf.columns:
            fig.add_trace(go.Scatter(
                x=cdf["Time"], y=cdf["StochRSI_D"],
                mode="lines", name="StochD",
                line=dict(color="#a855f7", width=1, dash="dot"),
                hovertemplate="D: %{y:.1f}<extra></extra>",
            ), row=current_row, col=1)
        for level, color in [(70, "rgba(239,68,68,0.3)"), (30, "rgba(74,222,128,0.3)"),
                             (50, "rgba(139,148,158,0.2)")]:
            fig.add_hline(y=level, row=current_row, col=1,
                          line=dict(color=color, width=1, dash="dot"))
        fig.update_yaxes(title_text="RSI", title_font=dict(size=9),
                         range=[0, 100], row=current_row, col=1)
        current_row += 1

    # ── Row 4: MACD ───────────────────────────────────────────────────────
    if has_macd:
        macd_hist = cdf["MACD_hist"].fillna(0)
        hist_colors = ["rgba(0,200,100,0.7)" if v >= 0 else "rgba(239,68,68,0.7)" for v in macd_hist]
        fig.add_trace(go.Bar(
            x=cdf["Time"], y=macd_hist,
            marker_color=hist_colors,
            name="MACD Hist", showlegend=False,
            hovertemplate="Hist: %{y:.4f}<extra></extra>",
        ), row=current_row, col=1)
        fig.add_trace(go.Scatter(
            x=cdf["Time"], y=cdf["MACD"],
            mode="lines", name="MACD",
            line=dict(color="#3b82f6", width=1.5),
            hovertemplate="MACD: %{y:.4f}<extra></extra>",
        ), row=current_row, col=1)
        fig.add_trace(go.Scatter(
            x=cdf["Time"], y=cdf["MACD_signal"],
            mode="lines", name="Signal",
            line=dict(color="#f87171", width=1.5),
            hovertemplate="Sig: %{y:.4f}<extra></extra>",
        ), row=current_row, col=1)
        fig.add_hline(y=0, row=current_row, col=1,
                      line=dict(color="rgba(139,148,158,0.3)", width=1))
        fig.update_yaxes(title_text="MACD", title_font=dict(size=9),
                         row=current_row, col=1)
        current_row += 1

    # ── Row 5: TTM Squeeze ────────────────────────────────────────────────
    if has_squeeze:
        sq = cdf["Squeeze_hist"].fillna(0)
        sq_colors = ["rgba(0,200,100,0.7)" if v >= 0 else "rgba(239,68,68,0.7)" for v in sq]
        sq_dot_colors = []
        if "Squeeze_ON" in cdf.columns:
            sq_dot_colors = ["#ef4444" if v else "#00c864"
                             for v in cdf["Squeeze_ON"].fillna(False)]
        fig.add_trace(go.Bar(
            x=cdf["Time"], y=sq,
            marker_color=sq_colors,
            name="Squeeze", showlegend=False,
            hovertemplate="Sq: %{y:.4f}<extra></extra>",
        ), row=current_row, col=1)
        if sq_dot_colors:
            fig.add_trace(go.Scatter(
                x=cdf["Time"], y=[0]*len(cdf),
                mode="markers",
                marker=dict(symbol="circle", size=4, color=sq_dot_colors),
                name="Sq ON/OFF", showlegend=False,
                hovertemplate="Sq dot<extra></extra>",
            ), row=current_row, col=1)
        fig.add_hline(y=0, row=current_row, col=1,
                      line=dict(color="rgba(139,148,158,0.3)", width=1))
        fig.update_yaxes(title_text="Squeeze", title_font=dict(size=9),
                         row=current_row, col=1)

    # ── Apply theme ────────────────────────────────────────────────────────
    _plotly_layout(fig, n_rows, height=860)
    for i in range(1, n_rows):
        fig.update_xaxes(showticklabels=False, row=i, col=1)
    fig.update_xaxes(showticklabels=True, row=n_rows, col=1)
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        transition=dict(duration=0, easing="linear"),
        xaxis=dict(matches="x"),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="main_price_chart",
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["autoScale2d","lasso2d","select2d",
                                        "toImage","sendDataToCloud"],
            "displaylogo": False,
            "doubleClick": "reset",
            "showTips": False,
            "plotGlPixelRatio": 2,
        },
    )

# ══════════════════════════════════════════════════════════════════════════════
# GUARD
# ══════════════════════════════════════════════════════════════════════════════
if not TRADIER_TOKEN:
    st.error("🔑 Add **TRADIER_TOKEN** to Streamlit secrets. Optionally add **TRADIER_ACCOUNT_ID** for trading.")
    st.markdown("""
**How to get your Tradier token:**
1. Sign up at [tradier.com](https://tradier.com) — $10/month Pro plan for real-time data
2. Go to **Account → API Access** and copy your **Bearer Token**
3. In Streamlit Cloud: **App settings → Secrets** and add:
```toml
TRADIER_TOKEN = "your_bearer_token_here"
TRADIER_ACCOUNT_ID = "VA000001"  # your account number
```
""")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
session_name,session_dot=market_session()
st.markdown(
    f'<h1 style="color:#e6edf3;margin-bottom:2px">🧠 SPY Buddy — Quant Edition</h1>'
    f'<p style="color:#8b949e;margin:0">Position-aware options manager · Powered by Tradier · '
    f'{session_dot} <span style="color:#e6edf3">{session_name}</span></p>',
    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Setup")
    symbol      =st.text_input("Underlying",value=DEFAULT_SYMBOL).upper().strip()
    option_side =st.selectbox("Direction",["Call","Put"])
    tf          =st.selectbox("Primary timeframe",["1Min","5Min","15Min","1Hour"],index=1)
    qty         =st.number_input("Contracts",min_value=1,max_value=100,value=10,step=1)
    order_style =st.selectbox("Order type",["market","limit"])

    st.divider()

    # ── Account Size + Risk Calculator ─────────────────────────────────────
    st.markdown("**💰 Account Risk Calculator**")
    account_size = st.number_input(
        "Account Size ($)",
        min_value=1000.0, max_value=10_000_000.0,
        value=float(st.session_state.get("account_size", 25000.0)),
        step=500.0, format="%.0f", key="account_size"
    )
    risk_pct = st.slider(
        "Risk per trade (%)",
        min_value=0.25, max_value=5.0,
        value=float(st.session_state.get("risk_pct", 1.0)),
        step=0.25, format="%.2f%%", key="risk_pct"
    )
    max_risk_dollars = account_size * (risk_pct / 100.0)
    max_contracts_est = max(1, int(max_risk_dollars / 100)) if max_risk_dollars >= 100 else 1

    st.markdown(
        f'<div class="risk-box">'
        f'<div class="risk-label">Max Loss This Trade</div>'
        f'<div class="risk-value">${max_risk_dollars:,.2f}</div>'
        f'<div class="risk-sub">{risk_pct:.2f}% of ${account_size:,.0f}</div>'
        f'</div>'
        f'<div class="risk-box" style="margin-top:6px">'
        f'<div class="risk-label">Est. Max Contracts</div>'
        f'<div class="risk-value">{max_contracts_est}</div>'
        f'<div class="risk-sub">at $100/contract stop loss</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.divider()
    enable_multi_tf=st.checkbox("Multi-timeframe confirmation",value=True)
    show_auto_pick =st.checkbox("Show smart contract picker",value=True)
    show_tda       =st.checkbox("Show Top-Down Analysis",value=True)
    min_dte=21; max_dte=45

# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCH
# ══════════════════════════════════════════════════════════════════════════════
# ── Real-time quote ───────────────────────────────────────────────────────────
quote = get_stock_quote(symbol)
last_trade   = quote.get("last")
daily_close  = quote.get("close")
prev_close   = quote.get("prevclose")

# ── Option expirations ────────────────────────────────────────────────────────
expirations = get_option_expirations(symbol)

colA,colB,colC=st.columns(3)
with colA:
    expiration=st.selectbox("Expiration",expirations,index=0 if expirations else None)
with colB:
    # Fetch chain for selected expiration to get strikes
    chain_for_exp = get_option_chain(symbol, expiration, option_side) if expiration else []
    strikes = sorted({float(c.get("strike",0)) for c in chain_for_exp if c.get("strike") is not None})
    default_strike_index=0
    if strikes and last_trade is not None:
        default_strike_index=min(range(len(strikes)),key=lambda i:abs(strikes[i]-float(last_trade)))
    strike=st.selectbox("Strike",strikes,index=default_strike_index if strikes else None)
with colC:
    if st.button("🔄 Refresh",use_container_width=True,key="top_refresh_btn"):
        st.cache_data.clear(); st.rerun()

# ── Find selected contract in chain ───────────────────────────────────────────
selected_contract=None
contract_symbol=None
if expiration and strike is not None:
    for c in chain_for_exp:
        if float(c.get("strike",0))==float(strike):
            selected_contract=c; break
    if selected_contract:
        contract_symbol = selected_contract.get("symbol")

# ── Bars + indicators ─────────────────────────────────────────────────────────
bars=add_indicators(get_stock_bars(symbol,tf,200))

# ── Signal ────────────────────────────────────────────────────────────────────
if enable_multi_tf:
    stock_bias,stock_score,stock_reasons,certainty_pct,tf_biases=multi_tf_signal(symbol,tf)
else:
    stock_bias,stock_score,stock_reasons,certainty_pct=stock_signal(bars)
    tf_biases={}

# ── Current RSI ───────────────────────────────────────────────────────────────
current_rsi=None
if not bars.empty and "RSI" in bars.columns and pd.notna(bars.iloc[-1].get("RSI")):
    current_rsi=float(bars.iloc[-1]["RSI"])
vix_spot=get_vix_spot()

# ── Option data from chain (Tradier returns greeks + bid/ask in one call) ─────
snapshot = {}
quote_bid = quote_ask = quote_mid = last_option_trade = None
delta = gamma = theta = vega = iv = None
option_volume = open_interest = None

if selected_contract:
    # All data is already in the chain response — no separate snapshot call needed
    quote_bid  = selected_contract.get("bid")
    quote_ask  = selected_contract.get("ask")
    last_option_trade = selected_contract.get("last")
    if quote_bid is not None and quote_ask is not None:
        quote_mid = (float(quote_bid)+float(quote_ask))/2.0
    option_volume  = selected_contract.get("volume")
    open_interest  = selected_contract.get("open_interest")
    greeks_data    = selected_contract.get("greeks") or {}
    delta = greeks_data.get("delta")
    gamma = greeks_data.get("gamma")
    theta = greeks_data.get("theta")
    vega  = greeks_data.get("vega")
    iv    = greeks_data.get("mid_iv") or greeks_data.get("smv_vol")
    snapshot = selected_contract  # for the "More details" expander

current_premium = quote_mid if quote_mid is not None else last_option_trade

quality=contract_quality(last_trade,strike,quote_bid,quote_ask,option_volume,open_interest,iv,delta)

broker_position=find_position(contract_symbol) if contract_symbol else None
has_position   =broker_position is not None or active_trade_matches(contract_symbol)

base_state=derive_options_state(stock_bias,option_side,has_position,quality["quality_ok"])
managed   =manage_active_trade(
    st.session_state.active_trade if active_trade_matches(contract_symbol) else None,
    current_premium,stock_bias)
state=managed["state"] or base_state

# All contracts for PCR (fetch both calls and puts)
all_contracts = get_option_chain(symbol, expiration) if expiration else []
pcr_val,pcr_sentiment=compute_put_call_ratio(all_contracts)

breakeven_price=None
exp_move_low=exp_move_high=None
if last_trade and iv and expiration:
    try:
        dte_days=(dt.date.fromisoformat(expiration)-dt.date.today()).days
        if dte_days>0:
            if option_side=="Call" and strike:
                ask_val=float(quote_ask) if quote_ask else 0
                breakeven_price=float(strike)+ask_val
            elif option_side=="Put" and strike:
                ask_val=float(quote_ask) if quote_ask else 0
                breakeven_price=float(strike)-ask_val
            exp_move_low,exp_move_high=expected_move(float(last_trade),float(iv),dte_days)
    except: pass

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PANELS
# ══════════════════════════════════════════════════════════════════════════════

# ── State badge + certainty ───────────────────────────────────────────────────
badge_col,cert_col,_=st.columns([1,2,1])
with badge_col:
    st.markdown(f'<div style="margin:10px 0 18px">{state_badge(state)}</div>',unsafe_allow_html=True)
with cert_col:
    st.markdown(f'<div style="padding:18px 0 0">{certainty_bar(certainty_pct)}</div>',unsafe_allow_html=True)

# ── Underlying ────────────────────────────────────────────────────────────────
section("Underlying")
u1,u2,u3,u4,u5,u6,u7=st.columns(7)
u1.markdown(f'<div style="padding:4px 0">{bias_badge(stock_bias)}</div>',unsafe_allow_html=True)
u2.metric("Score",stock_score)
u3.metric("Latest Trade",fmt_money(last_trade))
u4.metric("RSI",fmt_num(current_rsi,1))
u5.metric("VIX",fmt_num(vix_spot,2))
u6.metric("Daily / Prev",f"{fmt_money(daily_close)} / {fmt_money(prev_close)}")
u7.metric("Session",session_name)

# ── Multi-TF confirmation ─────────────────────────────────────────────────────
if tf_biases:
    tf_cols=st.columns(len(tf_biases))
    for i,(tf_name,tf_bias) in enumerate(tf_biases.items()):
        color={"BULLISH":"#4ade80","BEARISH":"#f87171","NEUTRAL":"#9ca3af","ERROR":"#6b7280"}.get(tf_bias,"#9ca3af")
        tf_cols[i].markdown(
            f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;'
            f'padding:10px;text-align:center">'
            f'<div style="color:#8b949e;font-size:.75rem">{tf_name}</div>'
            f'<div style="color:{color};font-weight:800;font-size:1.1rem">{tf_bias}</div></div>',
            unsafe_allow_html=True)

with st.expander("Why the underlying bias"):
    for r in stock_reasons:
        st.markdown(r)

st.divider()

# ── TOP-DOWN ANALYSIS PANEL ───────────────────────────────────────────────────
if show_tda:
    render_tda_panel(symbol)
    st.divider()

# ── Option Contract ────────────────────────────────────────────────────────────
section("Option Contract")
o1,o2,o3,o4,o5,o6=st.columns(6)
o1.metric("Contract",contract_symbol or "N/A")
o2.metric("Bid",fmt_money(quote_bid))
o3.metric("Ask",fmt_money(quote_ask))
o4.metric("Mid",fmt_money(quote_mid))
o5.metric("Last",fmt_money(last_option_trade))
o6.metric("Spread",fmt_money(quality["spread"]))

if any(x is not None for x in [delta,gamma,theta,vega,iv]):
    g1,g2,g3,g4,g5=st.columns(5)
    g1.metric("Delta",fmt_num(delta,3))
    g2.metric("Gamma",fmt_num(gamma,4))
    g3.metric("Theta",fmt_num(theta,4))
    g4.metric("Vega", fmt_num(vega, 4))
    g5.metric("IV",   fmt_num(iv,   3))

if breakeven_price or exp_move_low:
    em1,em2,em3,em4=st.columns(4)
    em1.metric("Breakeven",fmt_money(breakeven_price))
    em2.metric("Expected Move Low", fmt_money(exp_move_low))
    em3.metric("Expected Move High",fmt_money(exp_move_high))
    em4.metric("Put/Call Ratio",f"{pcr_val} — {pcr_sentiment}" if pcr_val else "N/A")

st.divider()

# ── Contract Quality ───────────────────────────────────────────────────────────
section("Contract Quality")
q1,q2,q3,q4=st.columns(4)
q1.markdown(f'<div style="padding:8px 0">{quality_bar(quality["score"],quality["quality_ok"])}</div>',
            unsafe_allow_html=True)
q2.metric("Volume",       fmt_num(option_volume,0))
q3.metric("Open Interest",fmt_num(open_interest,0))
q4.metric("Spread %",     f"{quality['spread_pct']*100:.1f}%" if quality["spread_pct"] else "N/A")

with st.expander("Why this contract passed / failed"):
    for r in quality["reasons"]:
        icon="✅" if any(w in r.lower() for w in ("acceptable","near","available")) else "⚠️"
        st.markdown(f"{icon} {r}")

st.divider()

# ── Smart Contract Auto-Picker ─────────────────────────────────────────────────
if show_auto_pick:
    section("🤖 Smart Contract Picker")
    with st.spinner("Scanning chain for best contract…"):
        best=auto_pick_contract(all_contracts,option_side,last_trade,min_dte,max_dte)
    if best:
        ap1,ap2,ap3,ap4,ap5,ap6=st.columns(6)
        ap1.metric("Best Contract",best.get("symbol","N/A"))
        ap2.metric("Strike",       fmt_money(best.get("strike")))
        ap3.metric("Expiry",       best.get("expiration_date","N/A"))
        ap4.metric("DTE",          best.get("_dte","N/A"))
        ap5.metric("Delta",        fmt_num(best.get("_delta"),3))
        ap6.metric("Spread %",     f"{best.get('_spread_pct',0)*100:.1f}%")
        ap_b1,ap_b2,ap_b3,ap_b4=st.columns(4)
        ap_b1.metric("Bid",  fmt_money(best.get("_bid")))
        ap_b2.metric("Ask",  fmt_money(best.get("_ask")))
        ap_b3.metric("IV",   fmt_num(safe_get(best.get("greeks",{}),"mid_iv"),3))
        ap_b4.metric("OI",   fmt_num(best.get("open_interest"),0))
        if last_trade and best.get("greeks") and best.get("expiration_date"):
            try:
                dte_ap=(dt.date.fromisoformat(best["expiration_date"])-dt.date.today()).days
                iv_ap = safe_get(best.get("greeks",{}),"mid_iv") or safe_get(best.get("greeks",{}),"smv_vol")
                if dte_ap>0 and iv_ap:
                    be_ap=(float(best["strike"])+float(best["_ask"])) if option_side=="Call" \
                          else (float(best["strike"])-float(best["_ask"]))
                    em_lo,em_hi=expected_move(float(last_trade),float(iv_ap),dte_ap)
                    ap_c1,ap_c2,ap_c3=st.columns(3)
                    ap_c1.metric("Breakeven",fmt_money(be_ap))
                    ap_c2.metric("Expected Move Low", fmt_money(em_lo))
                    ap_c3.metric("Expected Move High",fmt_money(em_hi))
            except: pass
    else:
        st.info("No contract found matching delta 0.40-0.55, DTE 21-45, OI > 100.")
    st.divider()

# ── Trade State ────────────────────────────────────────────────────────────────
section("Trade State")
s1,s2,s3=st.columns(3)
s1.metric("State",                  state)
s2.metric("Holding this contract?", "Yes" if has_position else "No")
s3.metric("Direction",              option_side)

st.divider()

# ── Trade Plan ─────────────────────────────────────────────────────────────────
section("Trade Plan")
active_trade=st.session_state.get("active_trade")
is_same_locked_trade=bool(active_trade and contract_symbol
                          and active_trade.get("contract_symbol")==contract_symbol)

if is_same_locked_trade:
    locked_entry=float(active_trade["entry_premium"]); locked_stop=float(active_trade["premium_stop"])
    locked_tp1=float(active_trade["tp1"]); locked_tp2=float(active_trade["tp2"])
else:
    locked_entry=locked_stop=locked_tp1=locked_tp2=None

default_entry=locked_entry if locked_entry is not None else (
    quote_ask if quote_ask is not None else (last_option_trade if last_option_trade is not None else 0.0))

if not is_same_locked_trade:
    st.session_state["entry_premium_input"]=float(default_entry or 0.0)

p1,p2,p3,p4,p5=st.columns(5)
with p1:
    entry_premium=st.number_input("Entry premium",min_value=0.0,value=float(default_entry or 0.0),
                                   step=0.05,key="entry_premium_input",disabled=is_same_locked_trade)
with p2: stop_pct=st.slider("Stop %", 5, 50,20,1,key="stop_pct_input", disabled=is_same_locked_trade)
with p3: tp1_pct =st.slider("TP1 %",10,100,30,1,key="tp1_pct_input",  disabled=is_same_locked_trade)
with p4: tp2_pct =st.slider("TP2 %",20,200,50,1,key="tp2_pct_input",  disabled=is_same_locked_trade)
with p5: min_rr  =st.slider("Min R/R",1.0,3.0,2.0,.1,key="min_rr_input",disabled=is_same_locked_trade)

if is_same_locked_trade:
    premium_stop=locked_stop; tp1=locked_tp1; tp2=locked_tp2
    rr1=((tp1-locked_entry)/max(.0001,locked_entry-premium_stop)) if locked_entry>premium_stop else None
    rr2=((tp2-locked_entry)/max(.0001,locked_entry-premium_stop)) if locked_entry>premium_stop else None
else:
    premium_stop=tp1=tp2=rr1=rr2=None
    if entry_premium>0:
        risk_amt=entry_premium*(stop_pct/100.0)
        premium_stop=max(.01,entry_premium-risk_amt)
        tp1=entry_premium*(1+tp1_pct/100.0); tp2=entry_premium*(1+tp2_pct/100.0)
        rr1=(tp1-entry_premium)/max(.0001,entry_premium-premium_stop)
        rr2=(tp2-entry_premium)/max(.0001,entry_premium-premium_stop)

r1,r2,r3,r4,r5=st.columns(5)
r1.metric("Premium Stop",fmt_money(premium_stop))
r2.metric("TP1",         fmt_money(tp1))
r3.metric("TP2",         fmt_money(tp2))
r4.metric("R/R to TP1",  "N/A" if rr1 is None else f"{rr1:.2f}")
r5.metric("R/R to TP2",  "N/A" if rr2 is None else f"{rr2:.2f}")

# ── Risk vs Account Size callout ──────────────────────────────────────────────
if entry_premium and entry_premium > 0 and not is_same_locked_trade:
    trade_risk_total = entry_premium * (stop_pct / 100.0) * int(qty) * 100
    pct_of_account   = (trade_risk_total / account_size * 100) if account_size > 0 else 0
    risk_color = "#4ade80" if pct_of_account <= risk_pct else ("#f59e0b" if pct_of_account <= risk_pct * 1.5 else "#ef4444")
    st.markdown(
        f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;'
        f'padding:10px 16px;margin:8px 0;display:flex;gap:24px;align-items:center">'
        f'<span style="color:#8b949e;font-size:.85rem">Trade risk ({qty} contracts):</span>'
        f'<span style="color:{risk_color};font-weight:800;font-size:1.05rem">${trade_risk_total:,.2f}</span>'
        f'<span style="color:{risk_color};font-size:.85rem">({pct_of_account:.2f}% of account)</span>'
        f'<span style="color:#8b949e;font-size:.8rem">· Max allowed: ${max_risk_dollars:,.2f} ({risk_pct:.2f}%)</span>'
        f'</div>',
        unsafe_allow_html=True
    )

if is_same_locked_trade: st.success("🔒 Locked trade active. Trade Plan is frozen until cleared or closed.")
if base_state in ["ENTER CALL","ENTER PUT"] and rr1 is not None and rr1<min_rr:
    st.warning("Direction is good, but reward/risk is below your minimum. No trade.")
elif state=="NO TRADE":   st.info("No trade right now.")
elif state in ["HOLD CALL","HOLD PUT"]: st.info(f"{state}. Manage the open position.")
elif state=="TP1 HIT":    st.success("🎯 TP1 HIT. Consider taking partial profit and tightening your stop.")
elif state in ["EXIT CALL","EXIT PUT"]: st.warning(f"⚠️ {state}. Hard exit condition triggered.")
elif state=="WEAKENING":  st.warning("⚠️ WEAKENING. Soft warning only. Momentum or bias has weakened.")
else:                     st.success(state)

if managed["notes"]:
    with st.expander("Trade manager notes"):
        for n in managed["notes"]: st.markdown(f"• {n}")

st.divider()

# ── Live P&L ───────────────────────────────────────────────────────────────────
section("Live P&L")
pos=broker_position if broker_position else None
if is_same_locked_trade and current_premium is not None:
    custom_pl=(float(current_premium)-float(st.session_state.active_trade["entry_premium"]))*int(st.session_state.active_trade["qty"])*100
    custom_pl_pct=((float(current_premium)/float(st.session_state.active_trade["entry_premium"]))-1.0)*100
else:
    custom_pl=custom_pl_pct=None

l1,l2,l3,l4,l5,l6=st.columns(6)
l1.metric("Position?",      "Yes" if has_position else "No")
l2.metric("Qty",            str(st.session_state.active_trade["qty"]) if is_same_locked_trade else (pos.get("quantity") if pos else "0"))
l3.metric("Locked Entry",   fmt_money(st.session_state.active_trade["entry_premium"]) if is_same_locked_trade else fmt_money(pos.get("cost_basis") if pos else None))
l4.metric("Current Premium",fmt_money(current_premium))
l5.metric("Custom P/L",     fmt_money(custom_pl),delta=f"{custom_pl_pct:.2f}%" if custom_pl_pct is not None else None)
l6.metric("Custom P/L %",   fmt_num(custom_pl_pct,2))

st.divider()

# ── Actions ────────────────────────────────────────────────────────────────────
section("Actions")
limit_seed=quote_ask if quote_ask is not None else entry_premium
limit_price=st.number_input("Limit price (limit orders only)",min_value=0.0,
                             value=float(limit_seed or 0.0),step=0.05,key="limit_price_input")
a1,a2,a3=st.columns(3)
can_start=(contract_symbol is not None and entry_premium>0
           and premium_stop is not None and tp1 is not None and tp2 is not None)

if a1.button("🔒 Start / Lock Trade",use_container_width=True,disabled=not can_start,key="lock_trade_btn"):
    save_active_trade({"contract_symbol":contract_symbol,"option_side":option_side.upper(),
                       "qty":int(qty),"entry_premium":float(entry_premium),
                       "premium_stop":float(premium_stop),"tp1":float(tp1),"tp2":float(tp2)})
    st.success("Trade locked. Entry, stop, and targets will now stay fixed.")

if a2.button("📈 Buy to Open",use_container_width=True,disabled=contract_symbol is None,key="buy_open_btn"):
    try:
        order=place_option_order(contract_symbol,int(qty),"buy_to_open",order_style,
                                 limit_price if order_style=="limit" else None)
        st.success(f"Submitted buy order: {order.get('id','ok')}")
    except Exception as e: st.error(f"Buy order failed: {e}")

if a3.button("📉 Sell to Close",use_container_width=True,disabled=contract_symbol is None,key="sell_close_btn"):
    try:
        order=place_option_order(contract_symbol,int(qty),"sell_to_close",order_style,
                                 limit_price if order_style=="limit" else None)
        st.success(f"Submitted sell order: {order.get('id','ok')}")
        if active_trade_matches(contract_symbol): clear_active_trade()
    except Exception as e: st.error(f"Sell order failed: {e}")

c1,c2=st.columns(2)
if c1.button("🗑️ Clear Locked Trade",use_container_width=True,key="clear_trade_btn"):
    clear_active_trade(); st.info("Locked trade cleared.")
if c2.button("🔄 Refresh data",use_container_width=True,key="bottom_refresh_btn"):
    st.cache_data.clear(); st.rerun()

st.divider()

# ── Chart ──────────────────────────────────────────────────────────────────────
section("Underlying Chart")
if not bars.empty:
    make_chart(bars, symbol, breakeven_price)
    st.caption("▲ BUY (green)  ▼ SELL (red)  ▲/▼ EXIT (orange) · Scroll = zoom · Drag = pan · Double-click = reset")
else:
    st.info("No underlying chart data returned.")

# ── News ───────────────────────────────────────────────────────────────────────
section("News")
news=get_news(symbol,limit=8)
if not news:
    st.write("No recent news returned.")
else:
    for item in news[:6]:
        headline=item.get("title","No headline")
        source  =item.get("publisher","")
        ts      =item.get("providerPublishTime","")
        if ts:
            try: ts=dt.datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")
            except: pass
        summary =item.get("summary","")
        link    =item.get("link","")
        st.markdown(
            f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;'
            f'padding:14px 18px;margin-bottom:10px">'
            f'<div style="color:#e6edf3;font-weight:700;font-size:.97rem">{headline}</div>'
            f'<div style="color:#8b949e;font-size:.78rem;margin:4px 0 8px">{source} &nbsp;•&nbsp; {ts}</div>'
            +(f'<div style="color:#c9d1d9;font-size:.88rem">{summary[:220]}{"..." if len(summary)>220 else ""}</div>' if summary else "")
            +"</div>",unsafe_allow_html=True)

# ── Trade History ──────────────────────────────────────────────────────────────
if st.session_state.trade_history:
    with st.expander(f"📜 Trade History ({len(st.session_state.trade_history)} closed)"):
        hist_df=pd.DataFrame(st.session_state.trade_history)
        st.dataframe(hist_df,use_container_width=True)

with st.expander("More contract details"):
    st.json({"contract":selected_contract or {}})

st.caption("SPY Buddy Quant Edition · Tradier Real-Time Data · Top-Down Analysis · Research / education only · Not financial advice.")
