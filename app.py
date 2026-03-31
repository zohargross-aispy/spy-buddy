"""
SPY Buddy Options — Quant Edition  (TradingView Chart + Top-Down Analysis)
==========================================================================
Changes vs app(1).py:
  • Replaced Plotly chart with TradingView Lightweight Charts
    – Native candlestick rendering, smooth zoom/pan, real crosshair
    – Sub-panels: Volume, RSI/StochRSI, MACD, TTM Squeeze
    – Markers rendered as native TV chart markers (▲ BUY / ▼ SELL / ✕ EXIT)
  • Fixed signal detection: prev_state tracked inside the loop (not via .shift)
  • Added Top-Down Analysis panel: W / D / 4H / 1H / 15min / 5min
    – Each timeframe shows bias, score, key reasons, and colour-coded badge
    – Cascade alignment indicator (all agree = high confidence)
"""

import math
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from streamlit_lightweight_charts import renderLightweightCharts, Chart

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
/* Top-Down Analysis cards */
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
# API HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def headers()->Dict[str,str]:
    return {"APCA-API-KEY-ID":ALPACA_KEY,"APCA-API-SECRET-KEY":ALPACA_SECRET,"accept":"application/json"}

def api_get(url:str,params:Optional[dict]=None)->dict:
    r=requests.get(url,headers=headers(),params=params,timeout=20); r.raise_for_status(); return r.json()

def api_post(url:str,payload:dict)->dict:
    r=requests.post(url,headers={**headers(),"content-type":"application/json"},json=payload,timeout=20)
    r.raise_for_status(); return r.json()

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
# ALPACA DATA
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=20)
def get_stock_snapshot(symbol:str)->dict:
    return api_get(f"{DATA_BASE}/v2/stocks/{symbol}/snapshot")

@st.cache_data(ttl=30)
def get_stock_bars(symbol:str,timeframe:str="5Min",limit:int=200)->pd.DataFrame:
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

@st.cache_data(ttl=120)
def get_vix_spot()->Optional[float]:
    try:
        vix = yf.Ticker("^VIX").history(period="5d", interval="1d", auto_adjust=False)
        if vix is None or vix.empty:
            return None
        return float(vix["Close"].dropna().iloc[-1])
    except Exception:
        return None

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
# INDICATOR ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def add_indicators(df:pd.DataFrame)->pd.DataFrame:
    out=df.copy()
    if out.empty or len(out)<10: return out
    for col in ["Open","High","Low","Close","Volume"]:
        if col in out.columns:
            out[col]=pd.to_numeric(out[col],errors="coerce").astype("float64")
    out=out.dropna(subset=["Close"])
    if out.empty or len(out)<10: return out

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

# ══════════════════════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME SIGNAL
# ══════════════════════════════════════════════════════════════════════════════
_TF_LIMITS = {"1Min":200,"5Min":200,"15Min":150,"1Hour":120,"1Day":100,"1Week":80}

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
# Alpaca timeframe strings for each TDA level
_TDA_TIMEFRAMES = [
    ("Weekly",  "1Week",  80,  "Macro trend — is the big picture bullish or bearish?"),
    ("Daily",   "1Day",   120, "Intermediate trend — swing direction."),
    ("4-Hour",  "4Hour",  120, "Short-term trend — intraday swing bias."),
    ("1-Hour",  "1Hour",  150, "Intraday momentum — entry zone confirmation."),
    ("15-Min",  "15Min",  150, "Execution context — is momentum aligned?"),
    ("5-Min",   "5Min",   200, "Entry trigger — fine-tune entry timing."),
]

@st.cache_data(ttl=45)
def get_tda_panel(symbol:str)->List[dict]:
    results=[]
    for label,tf,limit,purpose in _TDA_TIMEFRAMES:
        try:
            df=add_indicators(get_stock_bars(symbol,tf,limit))
            if df.empty or len(df)<20:
                results.append({"label":label,"tf":tf,"bias":"N/A","score":0,
                                 "reasons":["Not enough data."],"cert":0,"purpose":purpose})
                continue
            bias,score,reasons,cert=stock_signal(df)
            results.append({"label":label,"tf":tf,"bias":bias,"score":score,
                             "reasons":reasons[:3],"cert":cert,"purpose":purpose})
        except Exception as e:
            results.append({"label":label,"tf":tf,"bias":"ERROR","score":0,
                             "reasons":[str(e)],"cert":0,"purpose":purpose})
    return results

def render_tda_panel(symbol:str):
    section("Top-Down Analysis — W / D / 4H / 1H / 15min / 5min")
    tda=get_tda_panel(symbol)

    # Cascade alignment summary
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
        cascade_html=f'<div class="cascade-warn">⚠️ {bull_count}/{total} BULLISH — Partial alignment. Reduce size or wait for confirmation.</div>'
    elif bear_count>bull_count:
        cascade_html=f'<div class="cascade-warn">⚠️ {bear_count}/{total} BEARISH — Partial alignment. Reduce size or wait for confirmation.</div>'
    else:
        cascade_html='<div class="cascade-neut">⚪ Mixed signals across timeframes — No clear directional edge. Stay flat.</div>'

    st.markdown(cascade_html,unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)

    # 6-column grid
    cols=st.columns(6)
    for i,r in enumerate(tda):
        bias=r["bias"]
        if bias=="BULLISH":
            bias_cls="tda-bias-bull"; bias_icon="▲ BULLISH"
        elif bias=="BEARISH":
            bias_cls="tda-bias-bear"; bias_icon="▼ BEARISH"
        elif bias=="NEUTRAL":
            bias_cls="tda-bias-neut"; bias_icon="◆ NEUTRAL"
        else:
            bias_cls="tda-bias-neut"; bias_icon=f"— {bias}"

        cert_color="#4ade80" if r["cert"]>=65 else ("#f59e0b" if r["cert"]>=45 else "#ef4444")
        reasons_html="<br>".join(r["reasons"][:3])

        card=f"""
<div class="tda-card">
  <div class="tda-tf">{r['label']}</div>
  <div class="{bias_cls}">{bias_icon}</div>
  <div class="tda-score" style="color:{cert_color}">Score: {r['score']:+d} &nbsp;|&nbsp; {r['cert']}%</div>
  <div class="tda-reason">{reasons_html}</div>
</div>"""
        cols[i].markdown(card,unsafe_allow_html=True)

    with st.expander("What is Top-Down Analysis?"):
        st.markdown("""
**Top-Down Analysis** starts from the highest timeframe and works down to the execution timeframe.
The idea is simple: **trade in the direction of the dominant trend**.

| Timeframe | Role |
|-----------|------|
| **Weekly** | Macro trend — defines the big-picture direction |
| **Daily** | Intermediate trend — swing bias |
| **4-Hour** | Short-term trend — intraday swing direction |
| **1-Hour** | Intraday momentum — confirms entry zone |
| **15-Min** | Execution context — is momentum aligned? |
| **5-Min** | Entry trigger — fine-tune entry timing |

**Cascade rule:** The more timeframes that agree, the higher the conviction. A full 6/6 alignment is the highest-probability setup. Mixed signals = stay flat or reduce size.
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
        prev_state=state   # ← FIXED: track inside loop, not via .shift()
    out["state"]=states
    out["prev_state"]=prev_states
    return out

def find_chart_signals(df:pd.DataFrame):
    """Return DataFrames of BUY / SELL / EXIT BUY / EXIT SELL signal rows."""
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
def compute_iv_rank(contracts_raw:list,current_iv:Optional[float])->Tuple[Optional[float],str]:
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
# SMART CONTRACT AUTO-PICKER
# ══════════════════════════════════════════════════════════════════════════════
def auto_pick_contract(contracts_raw:list,option_type:str,underlying_price:Optional[float],
                       min_dte:int=21,max_dte:int=45)->Optional[dict]:
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
            score=spread_pct+abs(delta_abs-0.50)*2
            if score<best_score:
                best_score=score
                best={**c,"_snap":snap,"_dte":dte,"_delta":float(delta_val),
                      "_bid":float(bid),"_ask":float(ask),"_spread_pct":spread_pct}
        except: continue
    return best

# ══════════════════════════════════════════════════════════════════════════════
# KELLY CRITERION SIZING
# ══════════════════════════════════════════════════════════════════════════════
def kelly_contracts(win_rate:float,avg_win_pct:float,avg_loss_pct:float,
                    account_size:float,premium:float,max_risk_pct:float=2.0)->dict:
    if avg_loss_pct<=0 or premium<=0:
        return {"kelly_pct":0,"contracts":0,"max_loss":0,"max_gain":0,"notes":"Invalid inputs"}
    b=avg_win_pct/avg_loss_pct
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
# EXPECTED MOVE
# ══════════════════════════════════════════════════════════════════════════════
def expected_move(underlying_price:float,iv:float,dte:int)->Tuple[float,float]:
    daily_move=underlying_price*iv*math.sqrt(dte/365)
    return round(underlying_price-daily_move,2),round(underlying_price+daily_move,2)

# ══════════════════════════════════════════════════════════════════════════════
# MARKET SESSION
# ══════════════════════════════════════════════════════════════════════════════
def market_session()->Tuple[str,str]:
    now=dt.datetime.now(dt.timezone(dt.timedelta(hours=-5)))
    t=now.time()
    pre=dt.time(4,0); open_=dt.time(9,30); close=dt.time(16,0); post=dt.time(20,0)
    if t<pre:    return "CLOSED","⚫"
    if t<open_:  return "PRE-MARKET","🟡"
    if t<close:  return "REGULAR","🟢"
    if t<post:   return "AFTER-HOURS","🟠"
    return "CLOSED","⚫"

# ══════════════════════════════════════════════════════════════════════════════
# STATE LOGIC
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
# TRADINGVIEW LIGHTWEIGHT CHART
# ══════════════════════════════════════════════════════════════════════════════
def _ts(t)->int:
    """Convert pandas Timestamp to Unix epoch integer (seconds)."""
    if hasattr(t,"timestamp"):
        return int(t.timestamp())
    return int(pd.Timestamp(t).timestamp())

def make_tv_chart(df:pd.DataFrame, symbol:str, breakeven:Optional[float]=None):
    """Render a TradingView Lightweight Chart with sub-panels via renderLightweightCharts."""
    if df.empty:
        st.info("No chart data available.")
        return

    cdf=df.copy()
    cdf["Time"]=pd.to_datetime(cdf["Time"],errors="coerce")
    cdf=cdf.dropna(subset=["Time"]).sort_values("Time").tail(150).reset_index(drop=True)

    has_rsi    ="RSI"          in cdf.columns and cdf["RSI"].notna().sum()>5
    has_macd   ="MACD"         in cdf.columns and cdf["MACD"].notna().sum()>5
    has_squeeze="Squeeze_hist" in cdf.columns and cdf["Squeeze_hist"].notna().sum()>5
    has_vwap   ="VWAP"         in cdf.columns and cdf["VWAP"].notna().sum()>5
    has_bb     ="BB_upper"     in cdf.columns and cdf["BB_upper"].notna().sum()>5

    # ── Shared chart options ────────────────────────────────────────────────
    _chart_opts = {
        "layout": {
            "background": {"type":"solid","color":"#0d1117"},
            "textColor": "#c9d1d9",
            "fontFamily": "monospace",
        },
        "grid": {
            "vertLines": {"color":"#21262d"},
            "horzLines": {"color":"#21262d"},
        },
        "crosshair": {"mode": 1},
        "timeScale": {
            "borderColor": "#30363d",
            "timeVisible": True,
            "secondsVisible": False,
        },
        "rightPriceScale": {"borderColor":"#30363d"},
    }

    # ── 1. CANDLESTICK data ─────────────────────────────────────────────────
    candle_data=[
        {"time":_ts(r["Time"]),"open":round(r["Open"],4),"high":round(r["High"],4),
         "low":round(r["Low"],4),"close":round(r["Close"],4)}
        for _,r in cdf.iterrows()
        if pd.notna(r["Open"]) and pd.notna(r["Close"])
    ]

    # ── 2. Signal markers (FIXED logic) ────────────────────────────────────
    buy_df,sell_df,exit_buy_df,exit_sell_df=find_chart_signals(cdf)
    markers=[]
    for _,r in buy_df.iterrows():
        markers.append({"time":_ts(r["time"]),"position":"belowBar","color":"#00c864",
                        "shape":"arrowUp","text":f"BUY {r['close']:.2f}","size":1})
    for _,r in sell_df.iterrows():
        markers.append({"time":_ts(r["time"]),"position":"aboveBar","color":"#ef4444",
                        "shape":"arrowDown","text":f"SELL {r['close']:.2f}","size":1})
    for _,r in exit_buy_df.iterrows():
        markers.append({"time":_ts(r["time"]),"position":"aboveBar","color":"#fb923c",
                        "shape":"arrowDown","text":f"EXIT {r['close']:.2f}","size":1})
    for _,r in exit_sell_df.iterrows():
        markers.append({"time":_ts(r["time"]),"position":"belowBar","color":"#fb923c",
                        "shape":"arrowUp","text":f"EXIT {r['close']:.2f}","size":1})
    # Sort markers by time
    markers.sort(key=lambda m: m["time"])

    # ── 3. EMA series ───────────────────────────────────────────────────────
    ema_series=[]
    for col,color,width in [("EMA_8","#f59e0b",1.5),("EMA_21","#3b82f6",1.5),("EMA_50","#a855f7",1.5)]:
        if col in cdf.columns and cdf[col].notna().sum()>0:
            ema_series.append({
                "type": Chart.Line,
                "data": [{"time":_ts(r["Time"]),"value":round(float(r[col]),4)}
                         for _,r in cdf.iterrows() if pd.notna(r[col])],
                "options": {"color":color,"lineWidth":width,"priceLineVisible":False,
                            "lastValueVisible":False,"crosshairMarkerVisible":False},
            })

    # ── 4. VWAP ─────────────────────────────────────────────────────────────
    vwap_series=[]
    if has_vwap:
        vwap_series=[{
            "type": Chart.Line,
            "data": [{"time":_ts(r["Time"]),"value":round(float(r["VWAP"]),4)}
                     for _,r in cdf.iterrows() if pd.notna(r["VWAP"])],
            "options": {"color":"#06b6d4","lineWidth":1,"lineStyle":2,
                        "priceLineVisible":False,"lastValueVisible":True,
                        "crosshairMarkerVisible":False,"title":"VWAP"},
        }]

    # ── 5. Bollinger Bands ──────────────────────────────────────────────────
    bb_series=[]
    if has_bb:
        for col,title in [("BB_upper","BB+"),("BB_lower","BB-")]:
            bb_series.append({
                "type": Chart.Line,
                "data": [{"time":_ts(r["Time"]),"value":round(float(r[col]),4)}
                         for _,r in cdf.iterrows() if pd.notna(r[col])],
                "options": {"color":"rgba(139,92,246,0.45)","lineWidth":1,"lineStyle":2,
                            "priceLineVisible":False,"lastValueVisible":False,
                            "crosshairMarkerVisible":False,"title":title},
            })

    # ── 6. Breakeven line (as a price line via a dummy series point) ────────
    # We'll add it as a horizontal line annotation using a Line series with
    # a single point spanning the full range
    be_series=[]
    if breakeven is not None:
        ts_start=_ts(cdf.iloc[0]["Time"]); ts_end=_ts(cdf.iloc[-1]["Time"])
        be_series=[{
            "type": Chart.Line,
            "data": [{"time":ts_start,"value":round(breakeven,4)},
                     {"time":ts_end,"value":round(breakeven,4)}],
            "options": {"color":"#f59e0b","lineWidth":1,"lineStyle":1,
                        "priceLineVisible":True,"lastValueVisible":True,
                        "crosshairMarkerVisible":False,"title":f"BE ${breakeven:.2f}"},
        }]

    # ── 7. Volume histogram ─────────────────────────────────────────────────
    vol_data=[]
    for _,r in cdf.iterrows():
        if pd.notna(r["Volume"]):
            color="#00c86466" if r["Close"]>=r["Open"] else "#ef444466"
            vol_data.append({"time":_ts(r["Time"]),"value":float(r["Volume"]),"color":color})

    # ── 8. RSI + StochRSI ───────────────────────────────────────────────────
    rsi_data=[]; stoch_k_data=[]; stoch_d_data=[]
    if has_rsi:
        rsi_data=[{"time":_ts(r["Time"]),"value":round(float(r["RSI"]),2)}
                  for _,r in cdf.iterrows() if pd.notna(r["RSI"])]
        if "StochRSI_K" in cdf.columns:
            stoch_k_data=[{"time":_ts(r["Time"]),"value":round(float(r["StochRSI_K"]),2)}
                          for _,r in cdf.iterrows() if pd.notna(r["StochRSI_K"])]
            stoch_d_data=[{"time":_ts(r["Time"]),"value":round(float(r["StochRSI_D"]),2)}
                          for _,r in cdf.iterrows() if pd.notna(r.get("StochRSI_D"))]

    # ── 9. MACD ─────────────────────────────────────────────────────────────
    macd_data=[]; macd_sig_data=[]; macd_hist_data=[]
    if has_macd:
        macd_data=[{"time":_ts(r["Time"]),"value":round(float(r["MACD"]),6)}
                   for _,r in cdf.iterrows() if pd.notna(r["MACD"])]
        macd_sig_data=[{"time":_ts(r["Time"]),"value":round(float(r["MACD_signal"]),6)}
                       for _,r in cdf.iterrows() if pd.notna(r["MACD_signal"])]
        macd_hist_data=[{"time":_ts(r["Time"]),"value":round(float(r["MACD_hist"]),6),
                         "color":"#00c86499" if r["MACD_hist"]>=0 else "#ef444499"}
                        for _,r in cdf.iterrows() if pd.notna(r["MACD_hist"])]

    # ── 10. TTM Squeeze ─────────────────────────────────────────────────────
    sq_hist_data=[]
    if has_squeeze:
        sq_hist_data=[{"time":_ts(r["Time"]),"value":round(float(r["Squeeze_hist"]),6),
                       "color":"#00c86499" if r["Squeeze_hist"]>=0 else "#ef444499"}
                      for _,r in cdf.iterrows() if pd.notna(r["Squeeze_hist"])]

    # ══════════════════════════════════════════════════════════════════════
    # BUILD CHART LIST
    # ══════════════════════════════════════════════════════════════════════
    charts=[]

    # ── Panel 1: Price (candles + EMAs + VWAP + BB + breakeven) ────────────
    price_series=[
        {
            "type": Chart.Candlestick,
            "data": candle_data,
            "options": {
                "upColor":"#00c864","downColor":"#ef4444",
                "borderUpColor":"#00c864","borderDownColor":"#ef4444",
                "wickUpColor":"#00c864","wickDownColor":"#ef4444",
            },
            "markers": markers,
        },
        *ema_series,
        *vwap_series,
        *bb_series,
        *be_series,
    ]

    charts.append({
        "chart": {**_chart_opts, "height":380},
        "series": price_series,
    })

    # ── Panel 2: Volume ─────────────────────────────────────────────────────
    charts.append({
        "chart": {**_chart_opts, "height":80},
        "series": [{
            "type": Chart.Histogram,
            "data": vol_data,
            "options": {"priceFormat":{"type":"volume"},"priceScaleId":"vol",
                        "scaleMargins":{"top":0.1,"bottom":0}},
        }],
    })

    # ── Panel 3: RSI + StochRSI ─────────────────────────────────────────────
    if has_rsi:
        rsi_panel_series=[
            {"type":Chart.Line,"data":rsi_data,
             "options":{"color":"#06b6d4","lineWidth":1.5,"priceLineVisible":False,
                        "lastValueVisible":True,"title":"RSI"}},
        ]
        if stoch_k_data:
            rsi_panel_series.append(
                {"type":Chart.Line,"data":stoch_k_data,
                 "options":{"color":"#f59e0b","lineWidth":1,"lineStyle":2,
                            "priceLineVisible":False,"lastValueVisible":False,"title":"K"}})
        if stoch_d_data:
            rsi_panel_series.append(
                {"type":Chart.Line,"data":stoch_d_data,
                 "options":{"color":"#a855f7","lineWidth":1,"lineStyle":2,
                            "priceLineVisible":False,"lastValueVisible":False,"title":"D"}})
        charts.append({
            "chart": {**_chart_opts, "height":120},
            "series": rsi_panel_series,
        })

    # ── Panel 4: MACD ────────────────────────────────────────────────────────
    if has_macd:
        charts.append({
            "chart": {**_chart_opts, "height":120},
            "series": [
                {"type":Chart.Histogram,"data":macd_hist_data,
                 "options":{"priceLineVisible":False,"lastValueVisible":False,"title":"Hist"}},
                {"type":Chart.Line,"data":macd_data,
                 "options":{"color":"#3b82f6","lineWidth":1.4,"priceLineVisible":False,
                            "lastValueVisible":True,"title":"MACD"}},
                {"type":Chart.Line,"data":macd_sig_data,
                 "options":{"color":"#f87171","lineWidth":1.4,"priceLineVisible":False,
                            "lastValueVisible":True,"title":"Sig"}},
            ],
        })

    # ── Panel 5: TTM Squeeze ─────────────────────────────────────────────────
    if has_squeeze:
        charts.append({
            "chart": {**_chart_opts, "height":100},
            "series": [{
                "type":Chart.Histogram,"data":sq_hist_data,
                "options":{"priceLineVisible":False,"lastValueVisible":False,"title":"Squeeze"},
            }],
        })

    renderLightweightCharts(charts=charts, key=f"tv_chart_{symbol}")

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
    enable_multi_tf=st.checkbox("Multi-timeframe confirmation",value=True)
    show_auto_pick =st.checkbox("Show smart contract picker",value=True)
    show_tda       =st.checkbox("Show Top-Down Analysis",value=True)
    min_dte=21; max_dte=45

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
bars=add_indicators(get_stock_bars(symbol,tf,200))

# ── Signal ────────────────────────────────────────────────────────────────────
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
current_rsi=None
if not bars.empty and "RSI" in bars.columns and pd.notna(bars.iloc[-1].get("RSI")):
    current_rsi=float(bars.iloc[-1]["RSI"])
vix_spot=get_vix_spot()

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

# ── TradingView Chart ──────────────────────────────────────────────────────────
section("Underlying Chart  (TradingView)")
if not bars.empty:
    make_tv_chart(bars, symbol, breakeven_price)
    st.caption("▲ BUY (green)  ▼ SELL (red)  ▲/▼ EXIT (orange) · Scroll = zoom · Drag = pan · Double-click = reset")
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

st.caption("SPY Buddy Quant Edition · TradingView Charts · Top-Down Analysis · Research / education only · Not financial advice.")
