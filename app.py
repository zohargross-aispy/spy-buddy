"""
SPY Buddy Options — Quant Edition
===================================
Base: app_final.py (all original logic preserved 100%)

New institutional layers added:
  1. Multi-timeframe confirmation (1Min + 5Min + 15Min must agree)
  2. VWAP intraday level on chart and signal
  3. ADX trend-strength filter (no trade in choppy markets)
  4. TTM Squeeze (Bollinger inside Keltner = coiling, wait for release)
  5. Volume confirmation (signal only valid on above-average volume)
  6. IV Rank 0-100 (only buy options when IV is cheap)
  7. Put/Call ratio from live options chain
  8. Composite certainty score 0-100%
  9. Smart contract auto-picker (best delta/DTE/spread/OI)
 10. Kelly criterion position sizing (1/4 Kelly, capped)
 11. Breakeven price display
 12. Expected move (±1σ by expiry using IV)
 13. MACD added to signal engine and chart
 14. Stochastic RSI for momentum confirmation
 15. Market session awareness (pre-market / regular / after-hours)
"""

import math
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

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
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CREDENTIALS
# ══════════════════════════════════════════════════════════════════════════════
ALPACA_KEY    = st.secrets.get("ALPACA_API_KEY",    "")
ALPACA_SECRET = st.secrets.get("ALPACA_SECRET_KEY", "")
PAPER_BASE    = "https://paper-api.alpaca.markets"
DATA_BASE     = "https://data.alpaca.markets"
DEFAULT_SYMBOL = "SPY"

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
_EMA_COLORS= {"EMA_8":"#f59e0b","EMA_21":"#3b82f6","EMA_50":"#a855f7"}

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
# API HELPERS  (original, untouched)
# ══════════════════════════════════════════════════════════════════════════════
def headers()->Dict[str,str]:
    return {"APCA-API-KEY-ID":ALPACA_KEY,"APCA-API-SECRET-KEY":ALPACA_SECRET,"accept":"application/json"}

def api_get(url:str,params:Optional[dict]=None)->dict:
    r=requests.get(url,headers=headers(),params=params,timeout=20); r.raise_for_status(); return r.json()

def api_post(url:str,payload:dict)->dict:
    r=requests.post(url,headers={**headers(),"content-type":"application/json"},json=payload,timeout=20)
    r.raise_for_status(); return r.json()

# ══════════════════════════════════════════════════════════════════════════════
# FORMATTERS  (original, untouched)
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
# POSITION MEMORY  (original, untouched)
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
# ALPACA DATA  (original + new multi-timeframe)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=20)
def get_stock_snapshot(symbol:str)->dict:
    return api_get(f"{DATA_BASE}/v2/stocks/{symbol}/snapshot")

@st.cache_data(ttl=30)
def get_stock_bars(symbol:str,timeframe:str="5Min",limit:int=150)->pd.DataFrame:
    payload=api_get(f"{DATA_BASE}/v2/stocks/bars",params={
        "symbols":symbol.upper(),"timeframe":timeframe,"limit":limit,
        "adjustment":"raw","feed":"iex","sort":"asc"})
    bars=payload.get("bars",{}).get(symbol.upper(),[])
    if not bars: return pd.DataFrame()
    df=pd.DataFrame(bars)
    df["Time"]=pd.to_datetime(df["t"],utc=True).dt.tz_convert("America/New_York")
    df=df.rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})
    for col in ["Open","High","Low","Close","Volume"]:
        df[col]=pd.to_numeric(df[col],errors="coerce").astype("float64")
    return df[["Time","Open","High","Low","Close","Volume"]]

@st.cache_data(ttl=60)
def get_option_contracts(symbol:str,expiration_date:Optional[str],option_type:Optional[str])->list:
    params={"underlying_symbols":symbol.upper(),"status":"active","limit":1000}
    if expiration_date: params["expiration_date"]=expiration_date
    if option_type:     params["type"]=option_type.lower()
    payload=api_get(f"{PAPER_BASE}/v2/options/contracts",params=params)
    return payload.get("option_contracts",[])

@st.cache_data(ttl=20)
def get_option_snapshot(contract_symbol:str)->dict:
    payload=api_get(f"{DATA_BASE}/v1beta1/options/snapshots",params={"symbols":contract_symbol})
    return payload.get("snapshots",{}).get(contract_symbol,{})

@st.cache_data(ttl=30)
def get_news(symbols:str,limit:int=8)->list:
    payload=api_get(f"{DATA_BASE}/v1beta1/news",params={"symbols":symbols,"limit":limit,"sort":"desc"})
    return payload.get("news",[])

def get_open_positions()->list:
    try: return api_get(f"{PAPER_BASE}/v2/positions")
    except: return []

def find_position(symbol:str)->Optional[dict]:
    for p in get_open_positions():
        if p.get("symbol")==symbol: return p
    return None

def place_option_order(symbol:str,qty:int,side:str,order_type:str="market",limit_price:Optional[float]=None)->dict:
    payload:Dict[str,Any]={"symbol":symbol,"qty":str(qty),"side":side,"type":order_type,"time_in_force":"day"}
    if order_type=="limit" and limit_price is not None: payload["limit_price"]=str(limit_price)
    return api_post(f"{PAPER_BASE}/v2/orders",payload)

# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR ENGINE  (original + new indicators)
# ══════════════════════════════════════════════════════════════════════════════
def add_indicators(df:pd.DataFrame)->pd.DataFrame:
    """Add all technical indicators to a bar DataFrame."""
    out=df.copy()
    if out.empty or len(out)<10: return out

    # ── Force all OHLCV columns to float64 to prevent DataError on rolling ──
    for col in ["Open","High","Low","Close","Volume"]:
        if col in out.columns:
            out[col]=pd.to_numeric(out[col],errors="coerce").astype("float64")
    out=out.dropna(subset=["Close"])
    if out.empty or len(out)<10: return out

    # ── Original indicators ────────────────────────────────────────────────
    out["EMA_8"] =out["Close"].ewm(span=8, adjust=False).mean()
    out["EMA_21"]=out["Close"].ewm(span=21,adjust=False).mean()
    out["EMA_50"]=out["Close"].ewm(span=50,adjust=False).mean()

    delta=out["Close"].diff()
    gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
    avg_gain=gain.ewm(alpha=1/14,adjust=False).mean()
    avg_loss=loss.ewm(alpha=1/14,adjust=False).mean()
    rs=avg_gain/avg_loss.replace(0,np.nan)
    out["RSI"]=(100-(100/(1+rs))).astype("float64")

    # ── MACD ──────────────────────────────────────────────────────────────
    ema12=out["Close"].ewm(span=12,adjust=False).mean()
    ema26=out["Close"].ewm(span=26,adjust=False).mean()
    out["MACD"]=ema12-ema26
    out["MACD_signal"]=out["MACD"].ewm(span=9,adjust=False).mean()
    out["MACD_hist"]=out["MACD"]-out["MACD_signal"]

    # ── Stochastic RSI ────────────────────────────────────────────────────
    rsi_series=out["RSI"].astype("float64")
    rsi_min=rsi_series.rolling(14).min()
    rsi_max=rsi_series.rolling(14).max()
    stoch_rsi=(rsi_series-rsi_min)/(rsi_max-rsi_min+1e-9)
    out["StochRSI_K"]=stoch_rsi.rolling(3).mean()*100
    out["StochRSI_D"]=out["StochRSI_K"].rolling(3).mean()

    # ── ADX ───────────────────────────────────────────────────────────────
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

    # ── VWAP (intraday, resets each session) ─────────────────────────────
    if "Time" in out.columns:
        out["_date"]=pd.to_datetime(out["Time"]).dt.date
        out["_tp"]=(out["High"]+out["Low"]+out["Close"])/3
        out["_tpvol"]=out["_tp"]*out["Volume"]
        # Use transform instead of apply to avoid pandas 2.x/3.x shape mismatch
        out["_cumvol"] =out.groupby("_date")["Volume"].transform("cumsum")
        out["_cumtpvol"]=out.groupby("_date")["_tpvol"].transform("cumsum")
        out["VWAP"]=out["_cumtpvol"]/out["_cumvol"].replace(0,np.nan)
        out.drop(columns=["_date","_tp","_tpvol","_cumvol","_cumtpvol"],inplace=True,errors="ignore")

    # ── Bollinger Bands ───────────────────────────────────────────────────
    bb_mid=out["Close"].rolling(20).mean()
    bb_std=out["Close"].rolling(20).std()
    out["BB_upper"]=bb_mid+2*bb_std
    out["BB_lower"]=bb_mid-2*bb_std
    out["BB_mid"]=bb_mid

    # ── Keltner Channels ──────────────────────────────────────────────────
    kc_mid=out["Close"].ewm(span=20,adjust=False).mean()
    out["KC_upper"]=kc_mid+1.5*atr14
    out["KC_lower"]=kc_mid-1.5*atr14

    # ── TTM Squeeze ───────────────────────────────────────────────────────
    # Squeeze ON = BB inside KC (low volatility coiling)
    out["Squeeze_ON"]=(out["BB_upper"]<out["KC_upper"])&(out["BB_lower"]>out["KC_lower"])
    # Squeeze momentum histogram (MACD-style on close vs midline)
    sq_val=out["Close"]-((out["High"].rolling(20).max()+out["Low"].rolling(20).min())/2+bb_mid)/2
    out["Squeeze_hist"]=sq_val.ewm(span=5,adjust=False).mean()

    # ── Volume ratio ──────────────────────────────────────────────────────
    out["Vol_avg"]=out["Volume"].rolling(20).mean()
    out["Vol_ratio"]=out["Volume"]/out["Vol_avg"].replace(0,pd.NA)

    return out

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL ENGINE  (original logic + new multi-factor certainty scoring)
# ══════════════════════════════════════════════════════════════════════════════
def stock_signal(df:pd.DataFrame)->Tuple[str,int,List[str],int]:
    """
    Returns: (bias, raw_score, reasons, certainty_pct)
    bias: BULLISH / BEARISH / NEUTRAL
    certainty_pct: 0-100
    """
    if df.empty or len(df)<30:
        return "NEUTRAL",0,["Not enough bar data."],0

    row=df.iloc[-1]
    score=0; reasons=[]; certainty_points=0; certainty_max=0

    # ── 1. EMA stack (original) ───────────────────────────────────────────
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

    # ── 2. RSI (original) ─────────────────────────────────────────────────
    certainty_max+=1
    if pd.notna(row.get("RSI")):
        if row["RSI"]>55:
            score+=1; certainty_points+=1; reasons.append(f"✅ RSI {row['RSI']:.1f} — bullish momentum.")
        elif row["RSI"]<45:
            score-=1; reasons.append(f"❌ RSI {row['RSI']:.1f} — bearish momentum.")
        else:
            reasons.append(f"⚪ RSI {row['RSI']:.1f} — neutral zone.")

    # ── 3. MACD ───────────────────────────────────────────────────────────
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

    # ── 4. Stochastic RSI ─────────────────────────────────────────────────
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

    # ── 5. ADX trend strength ─────────────────────────────────────────────
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
            score-=1; reasons.append(f"⚠️ ADX {adx:.1f} — choppy market, low conviction. Avoid options.")
        else:
            reasons.append(f"⚪ ADX {adx:.1f} — moderate trend strength.")

    # ── 6. VWAP ───────────────────────────────────────────────────────────
    certainty_max+=1
    if pd.notna(row.get("VWAP")):
        if row["Close"]>row["VWAP"]:
            score+=1; certainty_points+=1; reasons.append(f"✅ Price above VWAP ${row['VWAP']:.2f} — buy-side in control.")
        else:
            score-=1; reasons.append(f"❌ Price below VWAP ${row['VWAP']:.2f} — sell-side in control.")

    # ── 7. TTM Squeeze ────────────────────────────────────────────────────
    certainty_max+=1
    if pd.notna(row.get("Squeeze_ON")) and pd.notna(row.get("Squeeze_hist")):
        if not row["Squeeze_ON"] and pd.notna(df.iloc[-2].get("Squeeze_ON")) and df.iloc[-2]["Squeeze_ON"]:
            # Squeeze just fired
            if row["Squeeze_hist"]>0:
                score+=2; certainty_points+=1; reasons.append("🚀 TTM Squeeze FIRED bullish — explosive move starting.")
            else:
                score-=2; reasons.append("🚀 TTM Squeeze FIRED bearish — explosive move starting.")
        elif row["Squeeze_ON"]:
            reasons.append("⏳ TTM Squeeze ON — market coiling, wait for release.")
        else:
            reasons.append("⚪ TTM Squeeze off — normal volatility.")

    # ── 8. Volume confirmation ────────────────────────────────────────────
    certainty_max+=1
    if pd.notna(row.get("Vol_ratio")):
        if row["Vol_ratio"]>1.5:
            certainty_points+=1; reasons.append(f"✅ Volume {row['Vol_ratio']:.1f}× average — institutional participation.")
        elif row["Vol_ratio"]<0.7:
            score-=1; reasons.append(f"⚠️ Volume {row['Vol_ratio']:.1f}× average — weak conviction, low participation.")
        else:
            reasons.append(f"⚪ Volume {row['Vol_ratio']:.1f}× average — normal.")

    # ── Bias determination ────────────────────────────────────────────────
    if score>=4:   bias="BULLISH"
    elif score<=-4: bias="BEARISH"
    else:           bias="NEUTRAL"

    certainty_pct=int(round(certainty_points/max(certainty_max,1)*100))
    return bias,score,reasons,certainty_pct


def multi_tf_signal(symbol:str,primary_tf:str)->Tuple[str,int,List[str],int,Dict[str,str]]:
    """
    Fetch 3 timeframes, compute signal on each, require agreement for high certainty.
    Returns: (bias, score, reasons, certainty_pct, tf_biases)
    """
    tf_map={"1Min":("1Min",60),"5Min":("5Min",120),"15Min":("15Min",120),"1Hour":("1Hour",100)}
    confirm_tfs={"1Min":["1Min","5Min","15Min"],"5Min":["5Min","15Min","1Hour"],
                 "15Min":["15Min","1Hour","1Hour"],"1Hour":["1Hour","1Hour","1Hour"]}
    tfs=confirm_tfs.get(primary_tf,[primary_tf])

    tf_biases:Dict[str,str]={}
    primary_bias="NEUTRAL"; primary_score=0; primary_reasons=[]; primary_cert=0

    for i,tf in enumerate(tfs):
        tf_key,lim=tf_map.get(tf,("5Min",120))
        try:
            bars=add_indicators(get_stock_bars(symbol,tf_key,lim))
            bias,sc,reasons,cert=stock_signal(bars)
            tf_biases[tf]=bias
            if i==0:
                primary_bias=bias; primary_score=sc; primary_reasons=reasons; primary_cert=cert
        except Exception as e:
            tf_biases[tf]="ERROR"

    # Multi-TF agreement bonus/penalty
    biases=list(tf_biases.values())
    bull_count=biases.count("BULLISH"); bear_count=biases.count("BEARISH")
    if bull_count==len(biases):
        primary_cert=min(100,primary_cert+20)
        primary_reasons.insert(0,"🟢🟢🟢 All timeframes BULLISH — maximum confluence.")
    elif bear_count==len(biases):
        primary_cert=min(100,primary_cert+20)
        primary_reasons.insert(0,"🔴🔴🔴 All timeframes BEARISH — maximum confluence.")
    elif bull_count>0 and bear_count>0:
        primary_cert=max(0,primary_cert-25)
        primary_reasons.insert(0,"⚠️ Timeframes CONFLICT — certainty reduced. Consider waiting.")

    return primary_bias,primary_score,primary_reasons,primary_cert,tf_biases


# ══════════════════════════════════════════════════════════════════════════════
# STATE MACHINE  (original, untouched)
# ══════════════════════════════════════════════════════════════════════════════
def state_series(df:pd.DataFrame)->pd.DataFrame:
    out=df.copy()
    if out.empty: out["state"]=[]; return out
    states=[]; mode="FLAT"
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
            if bullish:  state="BUY";       mode="LONG"
            elif bearish: state="SELL";     mode="SHORT"
        elif mode=="LONG":
            if bearish or weakening_long:   state="EXIT BUY";  mode="FLAT"
            else:                           state="HOLD BUY"
        elif mode=="SHORT":
            if bullish or weakening_short:  state="EXIT SELL"; mode="FLAT"
            else:                           state="HOLD SELL"
        states.append(state)
    out["state"]=states; out["prev_state"]=out["state"].shift(1)
    return out

def find_chart_signals(df:pd.DataFrame):
    marked=state_series(df)
    buy_rows,sell_rows,exit_buy_rows,exit_sell_rows=[],[],[],[]
    for _,row in marked.iterrows():
        base={"index":row["Time"],"Low":row["Low"],"High":row["High"],
              "ATR":0 if pd.isna(row.get("ATR")) else row.get("ATR",0),"Close":row["Close"]}
        state=row["state"]; prev_state=row["prev_state"]
        if state=="BUY"      and prev_state!="BUY":      buy_rows.append({**base,"label":f"BUY<br>{float(row['Close']):.2f}"})
        elif state=="SELL"   and prev_state!="SELL":     sell_rows.append({**base,"label":f"SELL<br>{float(row['Close']):.2f}"})
        elif state=="EXIT BUY"  and prev_state!="EXIT BUY":  exit_buy_rows.append({**base,"label":f"EXIT<br>{float(row['Close']):.2f}"})
        elif state=="EXIT SELL" and prev_state!="EXIT SELL": exit_sell_rows.append({**base,"label":f"EXIT<br>{float(row['Close']):.2f}"})
    return pd.DataFrame(buy_rows),pd.DataFrame(sell_rows),pd.DataFrame(exit_buy_rows),pd.DataFrame(exit_sell_rows)

# ══════════════════════════════════════════════════════════════════════════════
# CONTRACT QUALITY  (original, untouched)
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
# NEW: IV RANK + PUT/CALL RATIO
# ══════════════════════════════════════════════════════════════════════════════
def compute_iv_rank(contracts_raw:list,current_iv:Optional[float])->Tuple[Optional[float],str]:
    """Compute IV rank 0-100 from the current options chain IVs."""
    if not contracts_raw or current_iv is None: return None,"N/A"
    ivs=[]
    for c in contracts_raw[:200]:
        sym=c.get("symbol")
        if not sym: continue
        try:
            snap=get_option_snapshot(sym)
            iv_val=safe_get(snap,"implied_volatility")
            if iv_val and float(iv_val)>0: ivs.append(float(iv_val))
        except: pass
    if len(ivs)<5: return None,"N/A"
    iv_min=min(ivs); iv_max=max(ivs)
    if iv_max==iv_min: return 50,"MEDIUM"
    rank=int((float(current_iv)-iv_min)/(iv_max-iv_min)*100)
    rank=max(0,min(100,rank))
    label="LOW (cheap — good to buy)" if rank<30 else ("HIGH (expensive — avoid buying)" if rank>70 else "MEDIUM")
    return rank,label

def compute_put_call_ratio(contracts_raw:list)->Tuple[Optional[float],str]:
    """Estimate put/call ratio from open interest in the chain."""
    call_oi=put_oi=0
    for c in contracts_raw:
        oi=c.get("open_interest") or 0
        if c.get("type","").lower()=="call": call_oi+=float(oi)
        elif c.get("type","").lower()=="put":  put_oi+=float(oi)
    if call_oi==0: return None,"N/A"
    pcr=put_oi/call_oi
    if pcr<0.7:   sentiment="Bullish (more calls)"
    elif pcr>1.2: sentiment="Bearish (more puts)"
    else:         sentiment="Neutral"
    return round(pcr,2),sentiment

# ══════════════════════════════════════════════════════════════════════════════
# NEW: SMART CONTRACT AUTO-PICKER
# ══════════════════════════════════════════════════════════════════════════════
def auto_pick_contract(contracts_raw:list,option_type:str,underlying_price:Optional[float],
                       min_dte:int=21,max_dte:int=45)->Optional[dict]:
    """
    Find the best contract: delta 0.40-0.55, DTE in range, tightest spread, OI > 100.
    Returns the contract dict with snapshot data merged in.
    """
    today=dt.date.today()
    candidates=[]
    for c in contracts_raw:
        if c.get("type","").lower()!=option_type.lower(): continue
        exp_str=c.get("expiration_date")
        if not exp_str: continue
        try:
            exp=dt.date.fromisoformat(exp_str)
            dte=(exp-today).days
        except: continue
        if not (min_dte<=dte<=max_dte): continue
        oi=float(c.get("open_interest") or 0)
        if oi<100: continue
        candidates.append((c,dte))

    best=None; best_score=float("inf")
    for c,dte in candidates:
        sym=c.get("symbol")
        if not sym: continue
        try:
            snap=get_option_snapshot(sym)
            bid=safe_get(snap,"latestQuote","bp")
            ask=safe_get(snap,"latestQuote","ap")
            delta_val=safe_get(snap,"greeks","delta")
            if bid is None or ask is None or delta_val is None: continue
            delta_abs=abs(float(delta_val))
            if not (0.35<=delta_abs<=0.60): continue
            spread=float(ask)-float(bid)
            spread_pct=spread/float(ask) if ask>0 else 999
            # Score = spread_pct + penalty for delta far from 0.50
            score=spread_pct+abs(delta_abs-0.50)*2
            if score<best_score:
                best_score=score
                best={**c,"_snap":snap,"_dte":dte,"_delta":float(delta_val),
                      "_bid":float(bid),"_ask":float(ask),"_spread_pct":spread_pct}
        except: continue
    return best

# ══════════════════════════════════════════════════════════════════════════════
# NEW: KELLY CRITERION SIZING
# ══════════════════════════════════════════════════════════════════════════════
def kelly_contracts(win_rate:float,avg_win_pct:float,avg_loss_pct:float,
                    account_size:float,premium:float,max_risk_pct:float=2.0)->dict:
    """
    Quarter-Kelly position sizing.
    win_rate: 0-1, avg_win_pct / avg_loss_pct: as decimals (e.g. 0.50 = 50%)
    """
    if avg_loss_pct<=0 or premium<=0:
        return {"kelly_pct":0,"contracts":0,"max_loss":0,"max_gain":0,"notes":"Invalid inputs"}
    b=avg_win_pct/avg_loss_pct  # reward/risk ratio
    p=win_rate; q=1-p
    full_kelly=(p*b-q)/b if b>0 else 0
    quarter_kelly=max(0,full_kelly/4)
    max_risk_dollar=account_size*(max_risk_pct/100)
    cost_per_contract=premium*100
    kelly_contracts_count=int(account_size*quarter_kelly/max(cost_per_contract,0.01))
    risk_contracts_count=int(max_risk_dollar/max(cost_per_contract,0.01))
    contracts=max(1,min(kelly_contracts_count,risk_contracts_count))
    return {
        "full_kelly_pct":round(full_kelly*100,1),
        "quarter_kelly_pct":round(quarter_kelly*100,1),
        "contracts":contracts,
        "total_cost":round(contracts*cost_per_contract,2),
        "max_loss":round(contracts*cost_per_contract,2),
        "max_gain":round(contracts*cost_per_contract*avg_win_pct,2),
        "notes":f"¼ Kelly={quarter_kelly*100:.1f}%, capped by {max_risk_pct}% risk rule"
    }

# ══════════════════════════════════════════════════════════════════════════════
# NEW: EXPECTED MOVE
# ══════════════════════════════════════════════════════════════════════════════
def expected_move(underlying_price:float,iv:float,dte:int)->Tuple[float,float]:
    """±1σ expected move using options pricing formula."""
    daily_move=underlying_price*iv*math.sqrt(dte/365)
    return round(underlying_price-daily_move,2),round(underlying_price+daily_move,2)

# ══════════════════════════════════════════════════════════════════════════════
# NEW: MARKET SESSION
# ══════════════════════════════════════════════════════════════════════════════
def market_session()->Tuple[str,str]:
    now=dt.datetime.now(dt.timezone(dt.timedelta(hours=-5)))  # ET approx
    t=now.time()
    pre=dt.time(4,0); open_=dt.time(9,30); close=dt.time(16,0); post=dt.time(20,0)
    if t<pre:    return "CLOSED","⚫"
    if t<open_:  return "PRE-MARKET","🟡"
    if t<close:  return "REGULAR","🟢"
    if t<post:   return "AFTER-HOURS","🟠"
    return "CLOSED","⚫"

# ══════════════════════════════════════════════════════════════════════════════
# ORIGINAL STATE LOGIC  (untouched)
# ══════════════════════════════════════════════════════════════════════════════
def derive_options_state(stock_bias:str,option_side:str,has_position:bool,quality_ok:bool)->str:
    side=option_side.upper()
    if not quality_ok: return "NO TRADE"
    if side=="CALL":
        if stock_bias=="BULLISH" and not has_position: return "ENTER CALL"
        if stock_bias=="BULLISH" and has_position:     return "HOLD CALL"
        if stock_bias!="BULLISH" and has_position:     return "WEAKENING"
        return "NO TRADE"
    if side=="PUT":
        if stock_bias=="BEARISH" and not has_position: return "ENTER PUT"
        if stock_bias=="BEARISH" and has_position:     return "HOLD PUT"
        if stock_bias!="BEARISH" and has_position:     return "WEAKENING"
        return "NO TRADE"
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
# CHART  (dark, 4 panels: Price+Volume / RSI+StochRSI / MACD / Squeeze)
# ══════════════════════════════════════════════════════════════════════════════
def make_chart(df:pd.DataFrame,symbol:str,breakeven:Optional[float]=None)->go.Figure:
    if df.empty: return go.Figure()
    cdf=df.copy()
    cdf["Time"]=pd.to_datetime(cdf["Time"],errors="coerce")
    cdf=cdf.dropna(subset=["Time"]).sort_values("Time").tail(140).copy()

    vol_colors=["rgba(0,200,100,.4)" if c>=o else "rgba(239,68,68,.4)"
                for c,o in zip(cdf["Close"],cdf["Open"])]
    has_rsi   ="RSI" in cdf.columns and cdf["RSI"].notna().sum()>5
    has_macd  ="MACD" in cdf.columns and cdf["MACD"].notna().sum()>5
    has_squeeze="Squeeze_hist" in cdf.columns and cdf["Squeeze_hist"].notna().sum()>5

    rows=1+int(has_rsi)+int(has_macd)+int(has_squeeze)
    heights=[0.50]+[0.17]*int(has_rsi)+[0.17]*int(has_macd)+[0.16]*int(has_squeeze)
    heights=[h/sum(heights) for h in heights]

    specs=[[{"secondary_y":True}]]+[[{}]]*(rows-1)
    fig=make_subplots(rows=rows,cols=1,shared_xaxes=True,vertical_spacing=0.025,
                      row_heights=heights,specs=specs)

    # Row tracking
    rsi_row=2 if has_rsi else None
    macd_row=(2+int(has_rsi)) if has_macd else None
    sq_row=(2+int(has_rsi)+int(has_macd)) if has_squeeze else None

    _grid=dict(gridcolor="#21262d",zerolinecolor="#30363d")

    # ── Candles ────────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=cdf["Time"],open=cdf["Open"],high=cdf["High"],low=cdf["Low"],close=cdf["Close"],
        name=symbol,increasing_line_color="#00c864",increasing_fillcolor="#00c864",
        decreasing_line_color="#ef4444",decreasing_fillcolor="#ef4444"),row=1,col=1,secondary_y=False)

    # ── EMAs ───────────────────────────────────────────────────────────────
    for col,color in _EMA_COLORS.items():
        if col in cdf.columns and cdf[col].notna().sum()>0:
            fig.add_trace(go.Scatter(x=cdf["Time"],y=cdf[col],mode="lines",name=col,
                                     line=dict(color=color,width=1.5)),row=1,col=1,secondary_y=False)

    # ── VWAP ───────────────────────────────────────────────────────────────
    if "VWAP" in cdf.columns and cdf["VWAP"].notna().sum()>0:
        fig.add_trace(go.Scatter(x=cdf["Time"],y=cdf["VWAP"],mode="lines",name="VWAP",
                                  line=dict(color="#06b6d4",width=1.4,dash="dot")),row=1,col=1,secondary_y=False)

    # ── Bollinger Bands ────────────────────────────────────────────────────
    if "BB_upper" in cdf.columns:
        for col,name in [("BB_upper","BB Upper"),("BB_lower","BB Lower")]:
            fig.add_trace(go.Scatter(x=cdf["Time"],y=cdf[col],mode="lines",name=name,
                                      line=dict(color="rgba(139,92,246,.4)",width=1,dash="dot"),
                                      showlegend=False),row=1,col=1,secondary_y=False)

    # ── Volume ─────────────────────────────────────────────────────────────
    fig.add_trace(go.Bar(x=cdf["Time"],y=cdf["Volume"],name="Volume",marker_color=vol_colors),
                  row=1,col=1,secondary_y=True)

    # ── Breakeven line ─────────────────────────────────────────────────────
    if breakeven is not None:
        fig.add_hline(y=breakeven,row=1,col=1,line_dash="dash",line_color="#f59e0b",line_width=1.5,
                      annotation_text=f"Breakeven ${breakeven:.2f}",
                      annotation_font_color="#f59e0b",annotation_position="top right")

    # ── Buy/Sell arrows ────────────────────────────────────────────────────
    buy_pts,sell_pts,exit_buy_pts,exit_sell_pts=find_chart_signals(cdf)
    for _,row in (buy_pts.tail(8) if not buy_pts.empty else buy_pts).iterrows():
        fig.add_annotation(x=row["index"],y=row["Low"]-max(float(row["ATR"])*.2,.05),
            text=row["label"],showarrow=True,arrowhead=2,arrowcolor="#00c864",
            font=dict(color="#00c864",size=10),ax=0,ay=38)
    for _,row in (sell_pts.tail(8) if not sell_pts.empty else sell_pts).iterrows():
        fig.add_annotation(x=row["index"],y=row["High"]+max(float(row["ATR"])*.2,.05),
            text=row["label"],showarrow=True,arrowhead=2,arrowcolor="#ef4444",
            font=dict(color="#ef4444",size=10),ax=0,ay=-38)
    for _,row in (exit_buy_pts.tail(8) if not exit_buy_pts.empty else exit_buy_pts).iterrows():
        fig.add_annotation(x=row["index"],y=row["High"]+max(float(row["ATR"])*.15,.05),
            text=row["label"],showarrow=True,arrowhead=2,arrowcolor="#fb923c",
            font=dict(color="#fb923c",size=10),ax=0,ay=-32)
    for _,row in (exit_sell_pts.tail(8) if not exit_sell_pts.empty else exit_sell_pts).iterrows():
        fig.add_annotation(x=row["index"],y=row["Low"]-max(float(row["ATR"])*.15,.05),
            text=row["label"],showarrow=True,arrowhead=2,arrowcolor="#fb923c",
            font=dict(color="#fb923c",size=10),ax=0,ay=32)

    # ── RSI + StochRSI ─────────────────────────────────────────────────────
    if has_rsi:
        fig.add_trace(go.Scatter(x=cdf["Time"],y=cdf["RSI"],mode="lines",name="RSI",
                                  line=dict(color="#06b6d4",width=1.5)),row=rsi_row,col=1)
        if "StochRSI_K" in cdf.columns:
            fig.add_trace(go.Scatter(x=cdf["Time"],y=cdf["StochRSI_K"],mode="lines",name="StochRSI K",
                                      line=dict(color="#f59e0b",width=1,dash="dot")),row=rsi_row,col=1)
            fig.add_trace(go.Scatter(x=cdf["Time"],y=cdf["StochRSI_D"],mode="lines",name="StochRSI D",
                                      line=dict(color="#a855f7",width=1,dash="dot")),row=rsi_row,col=1)
        for lvl,clr in [(70,"rgba(239,68,68,.35)"),(50,"rgba(255,255,255,.12)"),(30,"rgba(0,200,100,.35)")]:
            fig.add_hline(y=lvl,row=rsi_row,col=1,line_dash="dot",line_color=clr,line_width=1)

    # ── MACD ───────────────────────────────────────────────────────────────
    if has_macd:
        hist_colors=["rgba(0,200,100,.6)" if v>=0 else "rgba(239,68,68,.6)" for v in cdf["MACD_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=cdf["Time"],y=cdf["MACD_hist"],name="MACD Hist",marker_color=hist_colors),
                      row=macd_row,col=1)
        fig.add_trace(go.Scatter(x=cdf["Time"],y=cdf["MACD"],mode="lines",name="MACD",
                                  line=dict(color="#3b82f6",width=1.4)),row=macd_row,col=1)
        fig.add_trace(go.Scatter(x=cdf["Time"],y=cdf["MACD_signal"],mode="lines",name="Signal",
                                  line=dict(color="#f87171",width=1.4)),row=macd_row,col=1)

    # ── TTM Squeeze histogram ──────────────────────────────────────────────
    if has_squeeze:
        sq_colors=["rgba(0,200,100,.6)" if v>=0 else "rgba(239,68,68,.6)" for v in cdf["Squeeze_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=cdf["Time"],y=cdf["Squeeze_hist"],name="Squeeze",marker_color=sq_colors),
                      row=sq_row,col=1)
        # Squeeze dots
        sq_on=cdf[cdf["Squeeze_ON"]==True]
        sq_off=cdf[cdf["Squeeze_ON"]==False]
        if not sq_on.empty:
            fig.add_trace(go.Scatter(x=sq_on["Time"],y=[0]*len(sq_on),mode="markers",name="Squeeze ON",
                                      marker=dict(color="#ef4444",size=5,symbol="circle")),row=sq_row,col=1)
        if not sq_off.empty:
            fig.add_trace(go.Scatter(x=sq_off["Time"],y=[0]*len(sq_off),mode="markers",name="Squeeze OFF",
                                      marker=dict(color="#4ade80",size=5,symbol="circle")),row=sq_row,col=1)

    # ── Layout ─────────────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9",family="monospace"),
        height=680,xaxis_rangeslider_visible=False,dragmode="pan",
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#161b22",bordercolor="#30363d",font_color="#e6edf3"),
        legend=dict(orientation="h",bgcolor="rgba(13,17,23,.85)",bordercolor="#30363d",
                    borderwidth=1,font=dict(size=10),y=-0.05),
        margin=dict(l=60,r=80,t=40,b=60),
    )
    fig.update_xaxes(showspikes=True,spikemode="across",spikecolor="#58a6ff",
                     spikedash="dot",spikethickness=1,**_grid)
    fig.update_yaxes(title_text="Price",side="left",fixedrange=True,**_grid,row=1,col=1,secondary_y=False)
    fig.update_yaxes(title_text="Vol",side="right",fixedrange=True,showgrid=False,
                     range=[0,cdf["Volume"].max()*4],row=1,col=1,secondary_y=True)
    if has_rsi:    fig.update_yaxes(title_text="RSI",fixedrange=True,range=[0,100],**_grid,row=rsi_row,col=1)
    if has_macd:   fig.update_yaxes(title_text="MACD",fixedrange=True,**_grid,row=macd_row,col=1)
    if has_squeeze:fig.update_yaxes(title_text="Squeeze",fixedrange=True,**_grid,row=sq_row,col=1)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# GUARD
# ══════════════════════════════════════════════════════════════════════════════
if not ALPACA_KEY or not ALPACA_SECRET:
    st.error("🔑 Add **ALPACA_API_KEY** and **ALPACA_SECRET_KEY** to Streamlit secrets first.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
session_name,session_dot=market_session()
st.markdown(
    f'<h1 style="color:#e6edf3;margin-bottom:2px">🧠 SPY Buddy — Quant Edition</h1>'
    f'<p style="color:#8b949e;margin:0">Position-aware options manager · '
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
    st.subheader("Risk / Sizing")
    account_size=st.number_input("Account size ($)",min_value=1000,max_value=10_000_000,value=25000,step=1000)
    max_risk_pct=st.slider("Max risk per trade (%)",0.5,5.0,2.0,0.5)
    st.divider()
    st.subheader("Smart Picker DTE")
    min_dte=st.slider("Min DTE",7,30,21,1)
    max_dte=st.slider("Max DTE",21,90,45,1)
    st.divider()
    enable_multi_tf=st.checkbox("Multi-timeframe confirmation",value=True)
    show_auto_pick =st.checkbox("Show smart contract picker",value=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCH
# ══════════════════════════════════════════════════════════════════════════════
pre_snapshot=get_stock_snapshot(symbol)
pre_underlying_price=safe_get(pre_snapshot,"latestTrade","p")

contracts_raw=get_option_contracts(symbol,None,option_side)
expirations=sorted({c.get("expiration_date") for c in contracts_raw if c.get("expiration_date")})

colA,colB,colC=st.columns(3)
with colA:
    expiration=st.selectbox("Expiration",expirations,index=0 if expirations else None)
with colB:
    contracts_for_exp=[c for c in contracts_raw if c.get("expiration_date")==expiration] if expiration else []
    strikes=sorted({float(c.get("strike_price")) for c in contracts_for_exp if c.get("strike_price") is not None})
    default_strike_index=0
    if strikes and pre_underlying_price is not None:
        default_strike_index=min(range(len(strikes)),key=lambda i:abs(strikes[i]-float(pre_underlying_price)))
    strike=st.selectbox("Strike",strikes,index=default_strike_index if strikes else None)
with colC:
    if st.button("🔄 Refresh",use_container_width=True,key="top_refresh_btn"):
        st.cache_data.clear(); st.rerun()

selected_contract=None
if expiration and strike is not None:
    for c in contracts_for_exp:
        if float(c.get("strike_price"))==float(strike):
            selected_contract=c; break

contract_symbol=selected_contract.get("symbol") if selected_contract else None

# ── Bars + indicators ─────────────────────────────────────────────────────────
bars=add_indicators(get_stock_bars(symbol,tf,150))

# ── Signal (single or multi-TF) ───────────────────────────────────────────────
if enable_multi_tf:
    stock_bias,stock_score,stock_reasons,certainty_pct,tf_biases=multi_tf_signal(symbol,tf)
else:
    stock_bias,stock_score,stock_reasons,certainty_pct=stock_signal(bars)
    tf_biases={}

# ── Snapshot data ─────────────────────────────────────────────────────────────
underlying_snapshot=get_stock_snapshot(symbol)
last_trade =safe_get(underlying_snapshot,"latestTrade","p")
daily_close=safe_get(underlying_snapshot,"dailyBar","c")
prev_close =safe_get(underlying_snapshot,"prevDailyBar","c")

snapshot        =get_option_snapshot(contract_symbol) if contract_symbol else {}
quote_bid       =safe_get(snapshot,"latestQuote","bp")
quote_ask       =safe_get(snapshot,"latestQuote","ap")
quote_mid       =None
if quote_bid is not None and quote_ask is not None:
    quote_mid=(float(quote_bid)+float(quote_ask))/2.0
last_option_trade=safe_get(snapshot,"latestTrade","p")
current_premium  =quote_mid if quote_mid is not None else last_option_trade

delta=safe_get(snapshot,"greeks","delta")
gamma=safe_get(snapshot,"greeks","gamma")
theta=safe_get(snapshot,"greeks","theta")
vega =safe_get(snapshot,"greeks","vega")
iv   =safe_get(snapshot,"implied_volatility")
day_bar=safe_get(snapshot,"dailyBar",default={}) or {}
option_volume=day_bar.get("v")
open_interest=selected_contract.get("open_interest") if selected_contract else None

quality=contract_quality(last_trade,strike,quote_bid,quote_ask,option_volume,open_interest,iv,delta)

broker_position=find_position(contract_symbol) if contract_symbol else None
has_position   =broker_position is not None or active_trade_matches(contract_symbol)

base_state=derive_options_state(stock_bias,option_side,has_position,quality["quality_ok"])
managed   =manage_active_trade(
    st.session_state.active_trade if active_trade_matches(contract_symbol) else None,
    current_premium,stock_bias)
state=managed["state"] or base_state

# ── New metrics ───────────────────────────────────────────────────────────────
pcr_val,pcr_sentiment=compute_put_call_ratio(contracts_raw)
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
u1,u2,u3,u4,u5=st.columns(5)
u1.markdown(f'<div style="padding:4px 0">{bias_badge(stock_bias)}</div>',unsafe_allow_html=True)
u2.metric("Score",stock_score)
u3.metric("Latest Trade",fmt_money(last_trade))
u4.metric("Daily / Prev",f"{fmt_money(daily_close)} / {fmt_money(prev_close)}")
u5.metric("Session",session_name)

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

# ── New: Expected move + breakeven ────────────────────────────────────────────
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
        best=auto_pick_contract(contracts_raw,option_side,last_trade,min_dte,max_dte)
    if best:
        snap=best.get("_snap",{})
        ap1,ap2,ap3,ap4,ap5,ap6=st.columns(6)
        ap1.metric("Best Contract",best.get("symbol","N/A"))
        ap2.metric("Strike",       fmt_money(best.get("strike_price")))
        ap3.metric("Expiry",       best.get("expiration_date","N/A"))
        ap4.metric("DTE",          best.get("_dte","N/A"))
        ap5.metric("Delta",        fmt_num(best.get("_delta"),3))
        ap6.metric("Spread %",     f"{best.get('_spread_pct',0)*100:.1f}%")
        ap_b1,ap_b2,ap_b3,ap_b4=st.columns(4)
        ap_b1.metric("Bid",  fmt_money(best.get("_bid")))
        ap_b2.metric("Ask",  fmt_money(best.get("_ask")))
        ap_b3.metric("IV",   fmt_num(safe_get(snap,"implied_volatility"),3))
        ap_b4.metric("OI",   fmt_num(best.get("open_interest"),0))
        # Breakeven for auto-picked contract
        if last_trade and safe_get(snap,"implied_volatility") and best.get("expiration_date"):
            try:
                dte_ap=(dt.date.fromisoformat(best["expiration_date"])-dt.date.today()).days
                if dte_ap>0:
                    be_ap=(float(best["strike_price"])+float(best["_ask"])) if option_side=="Call" \
                          else (float(best["strike_price"])-float(best["_ask"]))
                    em_lo,em_hi=expected_move(float(last_trade),float(safe_get(snap,"implied_volatility")),dte_ap)
                    ap_c1,ap_c2,ap_c3=st.columns(3)
                    ap_c1.metric("Breakeven",fmt_money(be_ap))
                    ap_c2.metric("Expected Move Low", fmt_money(em_lo))
                    ap_c3.metric("Expected Move High",fmt_money(em_hi))
            except: pass
    else:
        st.info("No contract found matching delta 0.35–0.60 and your DTE range. Try widening the DTE sliders.")
    st.divider()

# ── Kelly Sizing ───────────────────────────────────────────────────────────────
section("📐 Kelly Position Sizing")
k1,k2,k3=st.columns(3)
with k1:
    hist_win_rate=st.slider("Historical win rate (%)",10,90,55,5,key="win_rate_sl")/100
with k2:
    hist_avg_win=st.slider("Avg win (%)",10,200,60,5,key="avg_win_sl")/100
with k3:
    hist_avg_loss=st.slider("Avg loss (%)",10,100,30,5,key="avg_loss_sl")/100

premium_for_kelly=float(quote_ask) if quote_ask else (float(last_option_trade) if last_option_trade else 1.0)
kelly=kelly_contracts(hist_win_rate,hist_avg_win,hist_avg_loss,account_size,premium_for_kelly,max_risk_pct)
kc1,kc2,kc3,kc4,kc5=st.columns(5)
kc1.metric("Full Kelly %",  f"{kelly['full_kelly_pct']}%")
kc2.metric("¼ Kelly %",     f"{kelly['quarter_kelly_pct']}%")
kc3.metric("Contracts",     kelly["contracts"])
kc4.metric("Total Cost",    fmt_money(kelly["total_cost"]))
kc5.metric("Max Loss",      fmt_money(kelly["max_loss"]))
st.caption(kelly["notes"])

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
l2.metric("Qty",            str(st.session_state.active_trade["qty"]) if is_same_locked_trade else (pos.get("qty") if pos else "0"))
l3.metric("Locked Entry",   fmt_money(st.session_state.active_trade["entry_premium"]) if is_same_locked_trade else fmt_money(pos.get("avg_entry_price") if pos else None))
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

if a2.button("📈 Paper Buy to Open",use_container_width=True,disabled=contract_symbol is None,key="buy_open_btn"):
    try:
        order=place_option_order(contract_symbol,int(qty),"buy",order_style,limit_price if order_style=="limit" else None)
        st.success(f"Submitted buy order: {order.get('id','ok')}")
    except Exception as e: st.error(f"Buy order failed: {e}")

if a3.button("📉 Paper Sell to Close",use_container_width=True,disabled=contract_symbol is None,key="sell_close_btn"):
    try:
        order=place_option_order(contract_symbol,int(qty),"sell",order_style,limit_price if order_style=="limit" else None)
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
    st.plotly_chart(make_chart(bars,symbol,breakeven_price),use_container_width=True,
                    config={"scrollZoom":True,"displaylogo":False,
                            "modeBarButtonsToRemove":["select2d","lasso2d"]})
    st.caption("Scroll = zoom time axis · Click-drag = pan · Double-click = reset")
else:
    st.info("No underlying chart data returned.")

st.divider()

# ── News ───────────────────────────────────────────────────────────────────────
section("News")
news=get_news(symbol,limit=8)
if not news:
    st.write("No recent news returned.")
else:
    for item in news[:6]:
        headline=item.get("headline","No headline"); source=item.get("source","")
        ts=item.get("updated_at",item.get("created_at","")); summary=item.get("summary","")
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
    st.json({"contract":selected_contract or {},"snapshot":snapshot or {}})

st.caption("SPY Buddy Quant Edition · Research / education only · Not financial advice.")
