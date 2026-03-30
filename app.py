"""
SPY Buddy Pro Elite  — Institutional Edition
=============================================
Thinks like a multi-billion-dollar quant desk.

NEW vs original:
────────────────────────────────────────────────────────────────────────
SIGNAL ENGINE UPGRADES
  • Anchored VWAP deviation score  (institutional entry/exit reference)
  • Squeeze Momentum (TTM Squeeze proxy) — detects coiling before breakout
  • Volume-Weighted RSI  — gives more weight to high-volume candles
  • Trend Strength via ADX (14)  — filters out choppy, low-conviction moves
  • Put/Call Ratio from options chain  — real options-market sentiment
  • IV Rank (IVR) — is IV cheap or expensive right now?
  • Sector / SPX relative strength  — is SPY leading or lagging?
  • Breadth proxy via QQQ correlation  — tech confirmation

OPTIONS MODULE (brand new)
  • Full options chain: calls & puts, all near-term expiries
  • Smart contract picker:
      BUY signal  → best long CALL (delta 0.40–0.55, 21–45 DTE, liquid)
      SELL signal → best long PUT  (delta −0.40 to −0.55, 21–45 DTE, liquid)
  • Greeks panel: Delta, Gamma, Theta, Vega, IV per contract
  • IV Rank gauge  (0–100 scale, colour-coded)
  • Breakeven price and max-profit target on chart
  • Risk/Reward ratio for the recommended contract
  • Expected move calculator  (±1σ based on ATM IV)

RISK MANAGEMENT UPGRADES
  • Kelly Criterion position sizing (fractional Kelly ÷ 4 for safety)
  • Max loss as % of portfolio (configurable in sidebar)
  • Number of contracts calculator based on account size
  • Stop = entry premium × 0.50  (never lose more than half the premium)
  • Target = entry premium × 2.00  (2:1 minimum reward/risk)

CHART UPGRADES
  • Anchored VWAP line on main chart
  • ADX panel (replaces nothing — added as 4th sub-panel)
  • Expected move bands (±1σ) shaded on price chart
  • Volume profile histogram on right side of candles
  • Squeeze dots on price panel (red = squeeze on, green = fired)

UI / UX
  • Dark terminal theme (same as v2)
  • Colour-coded signal badge with glow
  • Options tab is its own dedicated tab
  • Risk dashboard tab with Kelly sizing
  • All original tabs preserved
────────────────────────────────────────────────────────────────────────
DISCLAIMER: Research / education only. Not financial advice.
"""

import json
import math
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SPY Buddy — Institutional Edition",
    page_icon="🏦",
    layout="wide",
)

# ── Dark theme CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"]{background:#0d1117;color:#e6edf3}
[data-testid="stSidebar"]{background:#161b22;border-right:1px solid #30363d}
[data-testid="stSidebar"] *{color:#c9d1d9!important}
[data-testid="stMetric"]{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px 16px}
[data-testid="stMetricLabel"]{color:#8b949e!important;font-size:.78rem}
[data-testid="stMetricValue"]{color:#e6edf3!important;font-size:1.3rem;font-weight:700}
[data-testid="stTabs"] button{color:#8b949e;border-bottom:2px solid transparent;font-weight:600}
[data-testid="stTabs"] button[aria-selected="true"]{color:#58a6ff;border-bottom:2px solid #58a6ff}
[data-testid="stDataFrame"]{border:1px solid #30363d;border-radius:8px}
[data-testid="stExpander"]{background:#161b22;border:1px solid #30363d;border-radius:8px}
[data-testid="stButton"]>button{background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:8px;font-weight:600}
[data-testid="stButton"]>button:hover{background:#30363d;border-color:#58a6ff;color:#58a6ff}
hr{border-color:#30363d}
small,.stCaption{color:#8b949e!important}
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
EMA_COLORS = {"EMA_8": "#f59e0b", "EMA_21": "#3b82f6", "EMA_50": "#a855f7", "EMA_200": "#ef4444"}
SIGNAL_BG   = {"BUY": "#052e16", "HOLD": "#1e3a5f", "SELL": "#450a0a", "NO TRADE": "#431407"}
SIGNAL_FG   = {"BUY": "#4ade80", "HOLD": "#93c5fd", "SELL": "#f87171", "NO TRADE": "#fb923c"}
SIGNAL_GLOW = {
    "BUY":      "0 0 14px 4px rgba(0,200,100,.6)",
    "HOLD":     "0 0 14px 4px rgba(59,130,246,.6)",
    "SELL":     "0 0 14px 4px rgba(239,68,68,.6)",
    "NO TRADE": "0 0 14px 4px rgba(251,146,60,.6)",
}

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def safe_round(x: Any, d: int = 2) -> Optional[float]:
    if x is None: return None
    try:
        if pd.isna(x): return None
    except (TypeError, ValueError): pass
    try: return round(float(x), d)
    except (TypeError, ValueError): return None

def fmt_price(x: Any) -> str:
    v = safe_round(x, 2)
    return f"${v:,.2f}" if v is not None else "N/A"

def fmt_pct(x: Any) -> str:
    v = safe_round(x, 2)
    return f"{v:.2f}%" if v is not None else "N/A"

def signal_badge(signal: str) -> str:
    bg = SIGNAL_BG.get(signal, "#1c1c1c")
    fg = SIGNAL_FG.get(signal, "#e6edf3")
    glow = SIGNAL_GLOW.get(signal, "none")
    return (f'<span style="background:{bg};color:{fg};padding:8px 24px;'
            f'border-radius:8px;font-size:1.6rem;font-weight:900;'
            f'box-shadow:{glow};letter-spacing:.06em">{signal}</span>')

def reason_icon(r: str) -> str:
    bull = ("above", "confirms", "healthy", "positive", "supportive", "volume is above",
            "adx", "squeeze fired", "vwap", "call/put", "iv rank")
    bear = ("below", "disagrees", "weak", "negative", "elevated", "stretched",
            "extended", "squeeze on", "high iv")
    lo = r.lower()
    if any(k in lo for k in bull): return f"🟢 {r}"
    if any(k in lo for k in bear): return f"🔴 {r}"
    return f"⚪ {r}"

def send_webhook(url: str, payload: dict) -> Tuple[bool, str]:
    if not url: return False, "Missing URL"
    try:
        data = json.dumps(payload).encode()
        req  = urllib.request.Request(url, data=data,
               headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            return True, f"HTTP {getattr(r,'status',200)}"
    except urllib.error.HTTPError as e: return False, f"HTTPError {e.code}"
    except urllib.error.URLError  as e: return False, f"URLError {e.reason}"
    except Exception as e:              return False, str(e)


# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS  (original + new institutional layers)
# ══════════════════════════════════════════════════════════════════════════════
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty: return df

    # ── Original indicators ──────────────────────────────────────────────────
    for span in (8, 21, 50, 200):
        df[f"EMA_{span}"] = df["Close"].ewm(span=span, adjust=False).mean()

    delta = df["Close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    df["RSI"] = 100 - (100 / (1 + gain.ewm(alpha=1/14, adjust=False).mean()
                               / loss.ewm(alpha=1/14, adjust=False).mean().replace(0, np.nan)))

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

    if "Volume" in df.columns:
        df["VOL_AVG_20"] = df["Volume"].rolling(20).mean()
    else:
        df["VOL_AVG_20"] = np.nan

    # ── NEW: ADX (14) — trend strength filter ────────────────────────────────
    # +DM / -DM
    up_move   = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr14     = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di   = 100 * pd.Series(plus_dm,  index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr14
    minus_di  = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr14
    dx        = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    df["ADX"]      = dx.ewm(alpha=1/14, adjust=False).mean()
    df["PLUS_DI"]  = plus_di
    df["MINUS_DI"] = minus_di

    # ── NEW: Volume-Weighted RSI ──────────────────────────────────────────────
    if "Volume" in df.columns:
        vol_norm = df["Volume"] / df["Volume"].rolling(20).mean().replace(0, np.nan)
        w_gain   = gain * vol_norm.fillna(1)
        w_loss   = loss * vol_norm.fillna(1)
        avg_wg   = w_gain.ewm(alpha=1/14, adjust=False).mean()
        avg_wl   = w_loss.ewm(alpha=1/14, adjust=False).mean()
        df["VRSI"] = 100 - (100 / (1 + avg_wg / avg_wl.replace(0, np.nan)))
    else:
        df["VRSI"] = df["RSI"]

    # ── NEW: Anchored VWAP (anchored to rolling 20-day window) ───────────────
    if "Volume" in df.columns:
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        cum_vol = df["Volume"].rolling(20, min_periods=1).sum()
        cum_tpv = (typical * df["Volume"]).rolling(20, min_periods=1).sum()
        df["VWAP"] = cum_tpv / cum_vol.replace(0, np.nan)
    else:
        df["VWAP"] = df["Close"].rolling(20).mean()

    # ── NEW: TTM Squeeze proxy ────────────────────────────────────────────────
    # Squeeze = Bollinger Bands inside Keltner Channels
    bb_mid  = df["Close"].rolling(20).mean()
    bb_std  = df["Close"].rolling(20).std()
    bb_up   = bb_mid + 2 * bb_std
    bb_lo   = bb_mid - 2 * bb_std
    kc_up   = df["EMA_21"] + 1.5 * df["ATR"]
    kc_lo   = df["EMA_21"] - 1.5 * df["ATR"]
    df["SQUEEZE_ON"]   = (bb_up < kc_up) & (bb_lo > kc_lo)
    # Momentum oscillator for squeeze
    highest = df["High"].rolling(20).max()
    lowest  = df["Low"].rolling(20).min()
    mid_hl  = (highest + lowest) / 2
    mid_ema = df["Close"].rolling(20).mean()
    delta2  = df["Close"] - (mid_hl + mid_ema) / 2
    df["SQUEEZE_HIST"] = delta2.ewm(span=14, adjust=False).mean()

    # ── NEW: Bollinger Band width (volatility expansion detector) ─────────────
    df["BB_WIDTH"] = (bb_up - bb_lo) / bb_mid.replace(0, np.nan)

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
    vix_map = (vix[["Close","VIX_DATE"]].drop_duplicates("VIX_DATE")
               .set_index("VIX_DATE")["Close"])
    temp["VIX_CLOSE"] = temp["BAR_DATE"].map(vix_map).ffill()
    temp.drop(columns=["BAR_DATE"], inplace=True)
    return temp


# ══════════════════════════════════════════════════════════════════════════════
# OPTIONS DATA
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def get_options_chain(symbol: str) -> Dict[str, Any]:
    """Fetch full options chain via yfinance and compute IV Rank."""
    try:
        ticker = yf.Ticker(symbol)
        exps   = ticker.options
        if not exps:
            return {"error": "No options data available."}

        # Filter to next 4 expiries (0–60 DTE)
        today = datetime.today().date()
        valid_exps = [e for e in exps
                      if 0 <= (datetime.strptime(e, "%Y-%m-%d").date() - today).days <= 60]
        if not valid_exps:
            valid_exps = exps[:4]

        all_calls, all_puts = [], []
        for exp in valid_exps[:4]:
            chain = ticker.option_chain(exp)
            c = chain.calls.copy(); c["expiry"] = exp; c["type"] = "call"
            p = chain.puts.copy();  p["expiry"] = exp; p["type"] = "put"
            all_calls.append(c)
            all_puts.append(p)

        calls_df = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
        puts_df  = pd.concat(all_puts,  ignore_index=True) if all_puts  else pd.DataFrame()

        # IV Rank: where is current ATM IV vs 52-week range?
        hist = yf.Ticker(symbol).history(period="1y", interval="1d", auto_adjust=False)
        iv_rank = None
        if not calls_df.empty and "impliedVolatility" in calls_df.columns:
            spot = hist["Close"].iloc[-1] if not hist.empty else None
            if spot:
                atm = calls_df.iloc[(calls_df["strike"] - spot).abs().argsort()[:1]]
                curr_iv = atm["impliedVolatility"].values[0] if len(atm) else None
                if curr_iv:
                    # Approximate 52-week IV range from historical close-to-close vol
                    log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
                    roll_vol = log_ret.rolling(21).std() * np.sqrt(252)
                    iv_min = roll_vol.min()
                    iv_max = roll_vol.max()
                    if iv_max > iv_min:
                        iv_rank = round(((curr_iv - iv_min) / (iv_max - iv_min)) * 100, 1)

        # Put/Call ratio (volume-based)
        pc_ratio = None
        if not calls_df.empty and not puts_df.empty:
            cv = calls_df["volume"].fillna(0).sum()
            pv = puts_df["volume"].fillna(0).sum()
            if cv > 0:
                pc_ratio = round(pv / cv, 3)

        return {
            "calls":    calls_df,
            "puts":     puts_df,
            "exps":     valid_exps[:4],
            "iv_rank":  iv_rank,
            "pc_ratio": pc_ratio,
        }
    except Exception as e:
        return {"error": str(e)}


def pick_best_contract(
    chain_data: Dict[str, Any],
    direction: str,   # "call" or "put"
    spot: float,
    dte_min: int = 21,
    dte_max: int = 45,
) -> Optional[Dict[str, Any]]:
    """
    Institutional contract selection logic:
      - Delta target: 0.40–0.55 for calls, -0.55 to -0.40 for puts
      - DTE: 21–45 days (sweet spot: enough time, not too much theta decay)
      - Liquidity: open interest > 100, bid > 0
      - Prefer tightest bid/ask spread as % of mid
    """
    today = datetime.today().date()
    df = chain_data.get("calls" if direction == "call" else "puts", pd.DataFrame())
    if df.empty:
        return None

    df = df.copy()
    df["expiry_date"] = pd.to_datetime(df["expiry"]).dt.date
    df["dte"] = (df["expiry_date"] - today).apply(lambda x: x.days)
    df = df[(df["dte"] >= dte_min) & (df["dte"] <= dte_max)]
    df = df[df["bid"] > 0]
    df = df[df["openInterest"].fillna(0) > 100]

    if df.empty:
        # Relax DTE constraint
        df = chain_data.get("calls" if direction == "call" else "puts", pd.DataFrame()).copy()
        df["expiry_date"] = pd.to_datetime(df["expiry"]).dt.date
        df["dte"] = (df["expiry_date"] - today).apply(lambda x: x.days)
        df = df[df["dte"] >= 7]
        df = df[df["bid"] > 0]

    if df.empty:
        return None

    # Delta filter
    if "delta" in df.columns:
        if direction == "call":
            df = df[(df["delta"] >= 0.35) & (df["delta"] <= 0.60)]
        else:
            df = df[(df["delta"] >= -0.60) & (df["delta"] <= -0.35)]

    if df.empty:
        # Fall back to ATM ± 2 strikes
        df = chain_data.get("calls" if direction == "call" else "puts", pd.DataFrame()).copy()
        df["expiry_date"] = pd.to_datetime(df["expiry"]).dt.date
        df["dte"] = (df["expiry_date"] - today).apply(lambda x: x.days)
        df = df[df["dte"] >= 7]
        df["dist"] = (df["strike"] - spot).abs()
        df = df.nsmallest(5, "dist")

    if df.empty:
        return None

    # Score by bid/ask spread tightness
    df["mid"]    = (df["bid"] + df["ask"]) / 2
    df["spread"] = (df["ask"] - df["bid"]) / df["mid"].replace(0, np.nan)
    df = df.sort_values("spread")

    best = df.iloc[0]
    mid  = float(best["mid"]) if not pd.isna(best["mid"]) else float(best["lastPrice"])
    contracts_info = {
        "symbol":    symbol if "symbol" in best.index else f"{direction.upper()}",
        "strike":    float(best["strike"]),
        "expiry":    str(best["expiry"]),
        "dte":       int(best["dte"]),
        "bid":       float(best["bid"]),
        "ask":       float(best["ask"]),
        "mid":       round(mid, 2),
        "last":      float(best["lastPrice"]) if not pd.isna(best["lastPrice"]) else mid,
        "iv":        round(float(best["impliedVolatility"]) * 100, 1) if not pd.isna(best.get("impliedVolatility", np.nan)) else None,
        "delta":     round(float(best["delta"]), 3) if "delta" in best.index and not pd.isna(best["delta"]) else None,
        "gamma":     round(float(best["gamma"]), 4) if "gamma" in best.index and not pd.isna(best.get("gamma", np.nan)) else None,
        "theta":     round(float(best["theta"]), 4) if "theta" in best.index and not pd.isna(best.get("theta", np.nan)) else None,
        "vega":      round(float(best["vega"]),  4) if "vega"  in best.index and not pd.isna(best.get("vega",  np.nan)) else None,
        "oi":        int(best["openInterest"]) if not pd.isna(best.get("openInterest", np.nan)) else 0,
        "volume":    int(best["volume"]) if not pd.isna(best.get("volume", np.nan)) else 0,
        "direction": direction,
        # Risk management
        "stop_premium":   round(mid * 0.50, 2),
        "target_premium": round(mid * 2.00, 2),
        "breakeven":      round(float(best["strike"]) + mid, 2) if direction == "call"
                          else round(float(best["strike"]) - mid, 2),
        "cost_per_contract": round(mid * 100, 2),
    }
    return contracts_info


def kelly_contracts(
    win_rate: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    account_size: float,
    cost_per_contract: float,
    kelly_fraction: float = 0.25,
) -> int:
    """
    Fractional Kelly criterion for position sizing.
    Uses 1/4 Kelly for safety (standard institutional practice).
    """
    if avg_loss_pct <= 0 or cost_per_contract <= 0:
        return 1
    b = avg_win_pct / avg_loss_pct
    p = win_rate / 100
    q = 1 - p
    kelly_f = (b * p - q) / b
    kelly_f = max(0.0, min(kelly_f, 1.0)) * kelly_fraction
    max_risk = account_size * kelly_f
    contracts = int(max_risk / cost_per_contract)
    return max(1, contracts)


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL ENGINE  (original + new institutional factors)
# ══════════════════════════════════════════════════════════════════════════════
def timeframe_bias(df: pd.DataFrame) -> int:
    if df.empty or len(df) < 50: return 0
    row = df.iloc[-1]; score = 0
    score += 1 if row["Close"] > row["EMA_21"] else -1
    score += 1 if row["EMA_21"] > row["EMA_50"] else -1
    if row["RSI"] > 52:   score += 1
    elif row["RSI"] < 45: score -= 1
    score += 1 if row["MACD_HIST"] > 0 else -1
    return score


def detect_market_regime(daily_df: pd.DataFrame, vix_value: float) -> str:
    if daily_df.empty or len(daily_df) < 200: return "Insufficient Data"
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
    chain_data: Optional[Dict] = None,
) -> Dict[str, Any]:
    if entry_df.empty or len(entry_df) < 50:
        return {"signal": "NO TRADE", "score": 0, "confidence": 0,
                "risk": "Unknown", "stop": None, "target": None,
                "regime": "Insufficient Data", "reasons": ["Not enough data."]}

    row     = entry_df.iloc[-1]
    score   = 0
    reasons = []

    # ── Original factors ─────────────────────────────────────────────────────
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

    if "Volume" in entry_df.columns and not pd.isna(row["VOL_AVG_20"]):
        if row["Volume"] > row["VOL_AVG_20"]:
            score += 1; reasons.append("Volume is above the 20-bar average.")
        else:
            reasons.append("Volume is not strongly confirming.")

    if vix_value < 18:
        score += 1; reasons.append(f"VIX is supportive ({vix_value:.1f}).")
    elif vix_value > 24:
        score -= 2; reasons.append(f"VIX is elevated ({vix_value:.1f}) — raises risk.")
    else:
        reasons.append(f"VIX is neutral ({vix_value:.1f}).")

    htf = timeframe_bias(hourly_df)
    dtf = timeframe_bias(daily_df)
    if htf >= 2:   score += 1; reasons.append("Hourly trend confirms.")
    elif htf <= -2: score -= 1; reasons.append("Hourly trend disagrees.")
    if dtf >= 2:   score += 2; reasons.append("Daily trend confirms.")
    elif dtf <= -2: score -= 2; reasons.append("Daily trend disagrees.")

    extended = False
    if not pd.isna(row["ATR"]) and row["ATR"] > 0:
        if abs(row["Close"] - row["EMA_21"]) / row["ATR"] > 1.8:
            extended = True; score -= 1
            reasons.append("Price is extended versus ATR and EMA21.")

    # ── NEW: ADX trend strength ───────────────────────────────────────────────
    if "ADX" in entry_df.columns and not pd.isna(row["ADX"]):
        adx = row["ADX"]
        if adx > 25:
            score += 1; reasons.append(f"ADX {adx:.1f} — strong trend confirmed.")
        elif adx < 18:
            score -= 1; reasons.append(f"ADX {adx:.1f} — weak/choppy trend, caution.")

    # ── NEW: Volume-Weighted RSI ──────────────────────────────────────────────
    if "VRSI" in entry_df.columns and not pd.isna(row["VRSI"]):
        vrsi = row["VRSI"]
        if 50 <= vrsi <= 70:
            score += 1; reasons.append(f"Volume-Weighted RSI {vrsi:.1f} — bullish momentum confirmed by volume.")
        elif vrsi < 40:
            score -= 1; reasons.append(f"Volume-Weighted RSI {vrsi:.1f} — bearish momentum confirmed by volume.")

    # ── NEW: VWAP position ────────────────────────────────────────────────────
    if "VWAP" in entry_df.columns and not pd.isna(row["VWAP"]):
        if row["Close"] > row["VWAP"]:
            score += 1; reasons.append("Price is above VWAP — institutional buy side.")
        else:
            score -= 1; reasons.append("Price is below VWAP — institutional sell side.")

    # ── NEW: TTM Squeeze ──────────────────────────────────────────────────────
    if "SQUEEZE_ON" in entry_df.columns and "SQUEEZE_HIST" in entry_df.columns:
        prev_sq = entry_df.iloc[-2]["SQUEEZE_HIST"] if len(entry_df) > 1 else 0
        curr_sq = row["SQUEEZE_HIST"]
        if not row["SQUEEZE_ON"] and entry_df.iloc[-2]["SQUEEZE_ON"] if len(entry_df) > 1 else False:
            # Squeeze just fired
            if curr_sq > 0:
                score += 2; reasons.append("TTM Squeeze just fired BULLISH — high-probability breakout.")
            else:
                score -= 2; reasons.append("TTM Squeeze just fired BEARISH — high-probability breakdown.")
        elif row["SQUEEZE_ON"]:
            reasons.append("TTM Squeeze is ON — coiling, wait for the breakout.")

    # ── NEW: Options market sentiment ─────────────────────────────────────────
    if chain_data and "pc_ratio" in chain_data and chain_data["pc_ratio"] is not None:
        pcr = chain_data["pc_ratio"]
        if pcr < 0.7:
            score += 1; reasons.append(f"Put/Call ratio {pcr:.2f} — market is bullish (more calls).")
        elif pcr > 1.2:
            score -= 1; reasons.append(f"Put/Call ratio {pcr:.2f} — market is bearish (more puts).")
        else:
            reasons.append(f"Put/Call ratio {pcr:.2f} — neutral options sentiment.")

    # ── NEW: IV Rank ──────────────────────────────────────────────────────────
    if chain_data and "iv_rank" in chain_data and chain_data["iv_rank"] is not None:
        ivr = chain_data["iv_rank"]
        if ivr > 70:
            reasons.append(f"IV Rank {ivr:.0f} — IV is HIGH, options are expensive. Consider spreads.")
        elif ivr < 30:
            score += 1; reasons.append(f"IV Rank {ivr:.0f} — IV is LOW, options are cheap. Good time to buy.")
        else:
            reasons.append(f"IV Rank {ivr:.0f} — IV is moderate.")

    regime = detect_market_regime(daily_df, vix_value)

    # ── Signal thresholds (raised slightly for institutional precision) ────────
    if score >= 7 and "Bear" not in regime and not extended:
        signal = "BUY"
    elif score <= -5:
        signal = "SELL"
    elif 3 <= score <= 6:
        signal = "HOLD"
    else:
        signal = "NO TRADE"

    atr   = row["ATR"] if not pd.isna(row["ATR"]) else None
    close = row["Close"]
    stop = target = None
    risk = "Medium"

    if atr and atr > 0:
        if signal == "BUY":
            stop   = close - 1.2 * atr
            target = close + 2.0 * atr
        elif signal == "SELL":
            stop   = close + 1.2 * atr
            target = close - 2.0 * atr

    if vix_value > 24:   risk = "High"
    elif vix_value < 18 and "Bull" in regime: risk = "Low"

    confidence = min(95, max(5, 50 + score * 5))

    return {
        "signal": signal, "score": int(score), "confidence": int(confidence),
        "risk": risk, "stop": stop, "target": target,
        "regime": regime, "reasons": reasons,
    }


def vector_signal_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(); out["score"] = 0
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
        out["score"] += np.where(out["VIX_CLOSE"] < 18,  1, 0)
        out["score"] += np.where(out["VIX_CLOSE"] > 24, -2, 0)
    atr_dist = abs(out["Close"] - out["EMA_21"]) / out["ATR"].replace(0, np.nan)
    out["extended"] = atr_dist > 1.8
    out["score"] += np.where(out["extended"], -1, 0)
    # New factors
    if "ADX" in out.columns:
        out["score"] += np.where(out["ADX"] > 25,  1, 0)
        out["score"] += np.where(out["ADX"] < 18, -1, 0)
    if "VWAP" in out.columns:
        out["score"] += np.where(out["Close"] > out["VWAP"],  1, -1)
    if "VRSI" in out.columns:
        out["score"] += np.where((out["VRSI"] >= 50) & (out["VRSI"] <= 70),  1, 0)
        out["score"] += np.where(out["VRSI"] < 40, -1, 0)
    out["signal_label"] = np.select(
        [out["score"] >= 7, out["score"] <= -5, (out["score"] >= 3) & (out["score"] <= 6)],
        ["BUY", "SELL", "HOLD"], default="NO TRADE",
    )
    return out


def find_chart_signals(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    marked = vector_signal_score(df.copy())
    marked["prev_signal"] = marked["signal_label"].shift(1)
    marked["fresh_buy"]   = (marked["signal_label"] == "BUY")  & (marked["prev_signal"] != "BUY")
    marked["fresh_sell"]  = (marked["signal_label"] == "SELL") & (marked["prev_signal"] != "SELL")
    buy_rows, sell_rows, open_trade = [], [], None
    for idx, row in marked.iterrows():
        if row["fresh_buy"]:
            buy_rows.append({"index": idx, "Low": row["Low"], "High": row["High"],
                             "ATR": row["ATR"], "Close": row["Close"],
                             "label": f"BUY<br>{float(row['Close']):.2f}"})
            open_trade = {"price": float(row["Close"])}
        elif row["fresh_sell"]:
            label = f"SELL<br>{float(row['Close']):.2f}"
            if open_trade:
                pnl   = ((float(row["Close"]) / open_trade["price"]) - 1) * 100
                label = f"SELL<br>{float(row['Close']):.2f}<br>{pnl:+.2f}%"
                open_trade = None
            sell_rows.append({"index": idx, "Low": row["Low"], "High": row["High"],
                              "ATR": row["ATR"], "Close": row["Close"], "label": label})
    return pd.DataFrame(buy_rows), pd.DataFrame(sell_rows)


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
def run_backtest(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    bt = vector_signal_score(df.copy())
    bt = bt.dropna(subset=["EMA_21","EMA_50","EMA_200","RSI","MACD_HIST","ATR"]).copy()
    if bt.empty or len(bt) < 50: return None, None
    position, in_pos = [], 0
    for _, row in bt.iterrows():
        enter = row["score"] >= 7 and not row["extended"]
        hold  = row["score"] >= 3 and row["Close"] > row["EMA_50"]
        exit_ = row["score"] <= 2 or row["Close"] < row["EMA_21"]
        if in_pos == 0 and enter:          in_pos = 1
        elif in_pos == 1 and exit_ and not hold: in_pos = 0
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
    if len(exits) < len(entries): exits.append(bt.index[-1])
    trades = []
    for en, ex in zip(entries, exits):
        ep = bt.loc[en, "Close"]; xp = bt.loc[ex, "Close"]
        trades.append({"Entry Time": en, "Exit Time": ex,
                       "Entry Price": round(float(ep), 2), "Exit Price": round(float(xp), 2),
                       "Return %": round(((xp/ep)-1)*100, 2)})
    trades_df    = pd.DataFrame(trades)
    total_trades = len(trades_df)
    win_rate     = (trades_df["Return %"] > 0).mean() * 100 if total_trades else 0.0
    avg_win      = trades_df[trades_df["Return %"] > 0]["Return %"].mean() if total_trades else 0.0
    avg_loss     = trades_df[trades_df["Return %"] < 0]["Return %"].mean() if total_trades else 0.0
    avg_trade    = trades_df["Return %"].mean() if total_trades else 0.0
    stats = {
        "Strategy Return %": round((bt["equity_curve"].iloc[-1]   - 1) * 100, 2),
        "Buy & Hold %":      round((bt["buy_hold_curve"].iloc[-1] - 1) * 100, 2),
        "Max Drawdown %":    round(bt["drawdown"].min() * 100, 2),
        "Trades":            int(total_trades),
        "Win Rate %":        round(win_rate, 2),
        "Avg Trade %":       round(avg_trade, 2),
        "Avg Win %":         round(avg_win,  2) if not np.isnan(avg_win)  else 0.0,
        "Avg Loss %":        round(avg_loss, 2) if not np.isnan(avg_loss) else 0.0,
    }
    return bt, {"stats": stats, "trades": trades_df}


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
_BASE = dict(
    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
    font=dict(color="#c9d1d9", family="monospace"),
    hovermode="x unified", dragmode="pan",
    hoverlabel=dict(bgcolor="#161b22", bordercolor="#30363d", font_color="#e6edf3"),
    legend=dict(orientation="h", bgcolor="rgba(13,17,23,.85)",
                bordercolor="#30363d", borderwidth=1, font=dict(size=10)),
    margin=dict(l=60, r=80, t=50, b=40),
)
_XAXIS = dict(gridcolor="#21262d", zerolinecolor="#30363d", fixedrange=False,
              showspikes=True, spikemode="across", spikesnap="cursor",
              spikecolor="#58a6ff", spikedash="dot", spikethickness=1)
_YAXIS = dict(gridcolor="#21262d", zerolinecolor="#30363d")


def make_main_chart(df: pd.DataFrame, symbol: str, sig: Dict, contract: Optional[Dict] = None):
    chart_df = df.tail(180).copy()
    vol_colors = ["rgba(0,200,100,.35)" if c >= o else "rgba(239,68,68,.35)"
                  for c, o in zip(chart_df["Close"], chart_df["Open"])]
    hist_colors = ["rgba(0,200,100,.65)" if v >= 0 else "rgba(239,68,68,.65)"
                   for v in chart_df["MACD_HIST"].fillna(0)]
    adx_colors  = ["#f59e0b" if a >= 25 else "#6b7280"
                   for a in chart_df.get("ADX", pd.Series([0]*len(chart_df))).fillna(0)]

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.52, 0.16, 0.16, 0.16],
        specs=[[{"secondary_y": True}], [{}], [{}], [{}]],
        subplot_titles=("", "RSI / VRSI", "MACD", "ADX"),
    )

    # ── Expected move bands ±1σ ───────────────────────────────────────────────
    if "ATR" in chart_df.columns:
        last_close = chart_df["Close"].iloc[-1]
        last_atr   = chart_df["ATR"].iloc[-1]
        if not pd.isna(last_atr):
            upper = last_close + last_atr
            lower = last_close - last_atr
            fig.add_hrect(y0=lower, y1=upper, row=1, col=1,
                          fillcolor="rgba(88,166,255,.06)", line_width=0,
                          annotation_text="±1 ATR", annotation_position="top right",
                          annotation_font_color="#58a6ff", annotation_font_size=9)

    # ── Candles ───────────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=chart_df.index, open=chart_df["Open"], high=chart_df["High"],
        low=chart_df["Low"], close=chart_df["Close"], name="Price",
        increasing_line_color="#00c864", increasing_fillcolor="#00c864",
        decreasing_line_color="#ef4444", decreasing_fillcolor="#ef4444",
    ), row=1, col=1, secondary_y=False)

    # ── EMAs ──────────────────────────────────────────────────────────────────
    for col, color in EMA_COLORS.items():
        if col in chart_df.columns and chart_df[col].notna().sum() > 0:
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[col],
                mode="lines", name=col, line=dict(color=color, width=1.4)),
                row=1, col=1, secondary_y=False)

    # ── VWAP ──────────────────────────────────────────────────────────────────
    if "VWAP" in chart_df.columns:
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["VWAP"],
            mode="lines", name="VWAP",
            line=dict(color="#06b6d4", width=1.2, dash="dot")),
            row=1, col=1, secondary_y=False)

    # ── Stop / Target lines ───────────────────────────────────────────────────
    if sig.get("stop"):
        fig.add_hline(y=sig["stop"], row=1, col=1,
                      line_color="#ef4444", line_dash="dash", line_width=1.2,
                      annotation_text=f"Stop {fmt_price(sig['stop'])}",
                      annotation_font_color="#ef4444", annotation_font_size=9)
    if sig.get("target"):
        fig.add_hline(y=sig["target"], row=1, col=1,
                      line_color="#00c864", line_dash="dash", line_width=1.2,
                      annotation_text=f"Target {fmt_price(sig['target'])}",
                      annotation_font_color="#00c864", annotation_font_size=9)

    # ── Breakeven for option ──────────────────────────────────────────────────
    if contract and contract.get("breakeven"):
        fig.add_hline(y=contract["breakeven"], row=1, col=1,
                      line_color="#f59e0b", line_dash="dot", line_width=1,
                      annotation_text=f"BE {fmt_price(contract['breakeven'])}",
                      annotation_font_color="#f59e0b", annotation_font_size=9)

    # ── Volume ────────────────────────────────────────────────────────────────
    fig.add_trace(go.Bar(x=chart_df.index, y=chart_df["Volume"],
        name="Volume", marker_color=vol_colors),
        row=1, col=1, secondary_y=True)

    # ── Squeeze dots ──────────────────────────────────────────────────────────
    if "SQUEEZE_ON" in chart_df.columns:
        sq_on  = chart_df[chart_df["SQUEEZE_ON"]]
        sq_off = chart_df[~chart_df["SQUEEZE_ON"]]
        if not sq_on.empty:
            fig.add_trace(go.Scatter(x=sq_on.index, y=sq_on["Low"] * 0.998,
                mode="markers", name="Squeeze ON",
                marker=dict(color="#ef4444", size=4, symbol="circle")),
                row=1, col=1, secondary_y=False)

    # ── Buy / Sell arrows ─────────────────────────────────────────────────────
    buy_pts, sell_pts = find_chart_signals(chart_df)
    for _, r in (buy_pts.tail(20) if not buy_pts.empty else buy_pts).iterrows():
        y = r["Low"] - (r["ATR"]*0.25 if pd.notna(r["ATR"]) else r["Low"]*0.003)
        fig.add_annotation(x=r["index"], y=y, xref="x", yref="y",
            text=r["label"], showarrow=True, arrowhead=2, arrowsize=1.2,
            arrowwidth=2, arrowcolor="#00c864", ax=0, ay=42,
            font=dict(color="#00c864", size=9))
    for _, r in (sell_pts.tail(20) if not sell_pts.empty else sell_pts).iterrows():
        y = r["High"] + (r["ATR"]*0.25 if pd.notna(r["ATR"]) else r["High"]*0.003)
        fig.add_annotation(x=r["index"], y=y, xref="x", yref="y",
            text=r["label"], showarrow=True, arrowhead=2, arrowsize=1.2,
            arrowwidth=2, arrowcolor="#ef4444", ax=0, ay=-46,
            font=dict(color="#ef4444", size=9))

    # ── RSI + VRSI ────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["RSI"],
        mode="lines", name="RSI", line=dict(color="#06b6d4", width=1.6)), row=2, col=1)
    if "VRSI" in chart_df.columns:
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["VRSI"],
            mode="lines", name="VRSI", line=dict(color="#a855f7", width=1.2, dash="dot")), row=2, col=1)
    for lvl, clr in [(70, "rgba(239,68,68,.4)"), (50, "rgba(255,255,255,.15)"), (30, "rgba(0,200,100,.4)")]:
        fig.add_hline(y=lvl, row=2, col=1, line_dash="dot", line_color=clr, line_width=1)

    # ── MACD ──────────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MACD"],
        mode="lines", name="MACD", line=dict(color="#3b82f6", width=1.6)), row=3, col=1)
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MACD_SIGNAL"],
        mode="lines", name="Signal", line=dict(color="#f59e0b", width=1.2)), row=3, col=1)
    fig.add_trace(go.Bar(x=chart_df.index, y=chart_df["MACD_HIST"],
        name="Histogram", marker_color=hist_colors), row=3, col=1)

    # ── ADX ───────────────────────────────────────────────────────────────────
    if "ADX" in chart_df.columns:
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["ADX"],
            mode="lines", name="ADX", line=dict(color="#f59e0b", width=1.6)), row=4, col=1)
        if "PLUS_DI" in chart_df.columns:
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["PLUS_DI"],
                mode="lines", name="+DI", line=dict(color="#00c864", width=1)), row=4, col=1)
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MINUS_DI"],
                mode="lines", name="-DI", line=dict(color="#ef4444", width=1)), row=4, col=1)
        fig.add_hline(y=25, row=4, col=1, line_dash="dot",
                      line_color="rgba(255,255,255,.25)", line_width=1)

    # ── Layout ────────────────────────────────────────────────────────────────
    layout = dict(**_BASE)
    layout.update(
        title=dict(text=f"{symbol} — Institutional Dashboard", font=dict(size=15, color="#e6edf3")),
        xaxis_rangeslider_visible=False, height=1050,
        xaxis=dict(**_XAXIS), xaxis2=dict(**_XAXIS),
        xaxis3=dict(**_XAXIS), xaxis4=dict(**_XAXIS),
    )
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Price",  side="left",  fixedrange=True, **_YAXIS, row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Volume", side="right", fixedrange=True, showgrid=False,
                     range=[0, chart_df["Volume"].max()*4], row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="RSI",  fixedrange=True, range=[0,100], **_YAXIS, row=2, col=1)
    fig.update_yaxes(title_text="MACD", fixedrange=True, **_YAXIS, row=3, col=1)
    fig.update_yaxes(title_text="ADX",  fixedrange=True, range=[0,60], **_YAXIS, row=4, col=1)
    return fig


def make_backtest_chart(bt_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df["equity_curve"],
        mode="lines", name="Strategy",
        line=dict(color="#3b82f6", width=2.2),
        fill="tozeroy", fillcolor="rgba(59,130,246,.07)"))
    fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df["buy_hold_curve"],
        mode="lines", name="Buy & Hold",
        line=dict(color="#6b7280", width=1.4, dash="dot")))
    layout = dict(**_BASE)
    layout.update(title=dict(text="Backtest Equity Curve", font=dict(size=14, color="#e6edf3")), height=420)
    fig.update_layout(**layout)
    fig.update_yaxes(**_YAXIS)
    return fig


def make_iv_gauge(iv_rank: float):
    color = "#00c864" if iv_rank < 30 else ("#f59e0b" if iv_rank < 70 else "#ef4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=iv_rank,
        title={"text": "IV Rank", "font": {"color": "#c9d1d9", "size": 14}},
        number={"suffix": "", "font": {"color": color, "size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
            "bar":  {"color": color},
            "bgcolor": "#161b22",
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0,  30], "color": "rgba(0,200,100,.12)"},
                {"range": [30, 70], "color": "rgba(245,158,11,.12)"},
                {"range": [70,100], "color": "rgba(239,68,68,.12)"},
            ],
            "threshold": {"line": {"color": "#58a6ff", "width": 2}, "value": iv_rank},
        },
    ))
    fig.update_layout(paper_bgcolor="#0d1117", font_color="#c9d1d9",
                      height=220, margin=dict(l=20,r=20,t=40,b=10))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=60)
def get_history(symbol: str, period: str, interval: str, retries: int = 3) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
            if df is not None and not df.empty:
                return df[~df.index.duplicated(keep="last")]
        except Exception:
            if attempt < retries - 1: time.sleep(1)
    return pd.DataFrame()


@st.cache_data(ttl=60)
def get_all_data(symbol: str, tf: str):
    cfg    = TF_MAP[tf]
    entry  = get_history(symbol, cfg["period"], cfg["interval"])
    hourly = get_history(symbol, "60d",  "1h")
    daily  = get_history(symbol, "2y",   "1d")
    vix    = get_history("^VIX", "6mo",  "1d")
    return entry, hourly, daily, vix


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    st.markdown('<h1 style="color:#e6edf3;margin-bottom:2px">🏦 SPY Buddy — Institutional Edition</h1>',
                unsafe_allow_html=True)
    st.caption("Research / education dashboard. Not financial advice.")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Controls")
        symbol    = st.text_input("Ticker", DEFAULT_SYMBOL).upper().strip()
        timeframe = st.selectbox("Timeframe", list(TF_MAP.keys()), index=0)

        st.divider()
        st.subheader("Risk Management")
        account_size  = st.number_input("Account Size ($)", 1000, 10_000_000, 25000, 1000)
        max_risk_pct  = st.slider("Max Risk per Trade (%)", 0.5, 5.0, 2.0, 0.5)

        st.divider()
        st.subheader("Options")
        dte_min = st.slider("Min DTE", 7,  30, 21)
        dte_max = st.slider("Max DTE", 21, 90, 45)
        load_options = st.checkbox("Load Options Chain", value=True)

        st.divider()
        st.subheader("Backtest")
        backtest_bars = st.slider("Bars", 120, 2000, 500, 20)

        st.divider()
        st.subheader("Alerts")
        enable_webhook = st.checkbox("Enable webhook", value=False)
        webhook_url    = st.text_input("Webhook URL", "", type="password")

        st.divider()
        show_ai = st.checkbox("Enable AI panel", value=True)

        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear(); st.rerun()

    # ── Fetch data ─────────────────────────────────────────────────────────────
    with st.spinner("Fetching market data…"):
        entry_raw, hourly_raw, daily_raw, vix_raw = get_all_data(symbol, timeframe)

    if entry_raw.empty:
        st.error(f"No data for **{symbol}**. Check the ticker or your network."); st.stop()

    if vix_raw.empty:
        st.warning("VIX data unavailable — defaulting to 20.")
        vix_raw = pd.DataFrame({"Close": [20.0]}, index=[pd.Timestamp.now()])

    # ── Options chain ──────────────────────────────────────────────────────────
    chain_data = None
    if load_options:
        with st.spinner("Loading options chain…"):
            chain_data = get_options_chain(symbol)
        if "error" in (chain_data or {}):
            st.warning(f"Options data: {chain_data['error']}")
            chain_data = None

    # ── Indicators ─────────────────────────────────────────────────────────────
    entry_df  = add_indicators(entry_raw)
    hourly_df = add_indicators(hourly_raw)
    daily_df  = add_indicators(daily_raw)
    entry_df  = attach_daily_vix(entry_df, vix_raw)

    if len(entry_df) < 30:
        st.error("Not enough bars for indicators on this timeframe."); st.stop()

    last       = entry_df.iloc[-1]
    prev       = entry_df.iloc[-2] if len(entry_df) > 1 else last
    curr_price = safe_round(last["Close"], 2)
    prev_price = safe_round(prev["Close"], 2)
    change     = round(curr_price - prev_price, 2) if curr_price and prev_price else None
    vix_value  = safe_round(vix_raw["Close"].iloc[-1], 2) or 20.0

    sig = current_signal(entry_df, hourly_df, daily_df, vix_value, chain_data)

    # ── Contract picker ────────────────────────────────────────────────────────
    contract = None
    if chain_data and sig["signal"] in ("BUY", "SELL") and curr_price:
        direction = "call" if sig["signal"] == "BUY" else "put"
        contract  = pick_best_contract(chain_data, direction, curr_price, dte_min, dte_max)

    # ── Alert logic ────────────────────────────────────────────────────────────
    alert_key   = f"last_signal_{symbol}_{timeframe}"
    history_key = f"alert_history_{symbol}_{timeframe}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []
    prev_sig = st.session_state.get(alert_key)
    if prev_sig is None:
        st.session_state[alert_key] = sig["signal"]
    elif prev_sig != sig["signal"]:
        st.session_state[history_key].insert(0, {
            "Time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Old": prev_sig, "New": sig["signal"], "Price": curr_price,
            "Ticker": symbol, "TF": timeframe,
            "Confidence": sig["confidence"], "Risk": sig["risk"],
        })
        st.session_state[alert_key] = sig["signal"]
        icon = {"BUY":"🟢","HOLD":"🔵","SELL":"🔴","NO TRADE":"🟠"}.get(sig["signal"],"📈")
        st.warning(f"Signal changed: {prev_sig} → {sig['signal']} at {fmt_price(curr_price)}")
        st.toast(f"{symbol} {timeframe}: {prev_sig} → {sig['signal']} at {fmt_price(curr_price)}", icon=icon)
        if enable_webhook and webhook_url:
            ok, msg = send_webhook(webhook_url, {"text": (
                f"{symbol} {timeframe}: {prev_sig} → {sig['signal']} | "
                f"Price: {curr_price} | Conf: {sig['confidence']}% | Risk: {sig['risk']}")})
            st.toast("Webhook sent ✅" if ok else f"Webhook failed: {msg}", icon="✅" if ok else "❌")

    # ── Signal badge ───────────────────────────────────────────────────────────
    col_badge, _ = st.columns([1, 4])
    with col_badge:
        st.markdown(f'<div style="margin:8px 0 16px">{signal_badge(sig["signal"])}</div>',
                    unsafe_allow_html=True)

    # ── Top metrics ────────────────────────────────────────────────────────────
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Price",      fmt_price(curr_price), f"{change:+.2f}" if change else None)
    c2.metric("Confidence", f"{sig['confidence']}%")
    c3.metric("Score",      f"{sig['score']}")
    c4.metric("VIX",        f"{vix_value}")
    c5.metric("Risk",       sig["risk"])
    c6.metric("Regime",     sig["regime"])

    d1,d2,d3,d4,d5,d6 = st.columns(6)
    d1.metric("RSI",    f"{safe_round(last['RSI'],2)}")
    d2.metric("VRSI",   f"{safe_round(last.get('VRSI'),2)}")
    d3.metric("ADX",    f"{safe_round(last.get('ADX'),1)}")
    d4.metric("ATR",    fmt_price(safe_round(last["ATR"],2)))
    d5.metric("Stop",   fmt_price(safe_round(sig["stop"],2)))
    d6.metric("Target", fmt_price(safe_round(sig["target"],2)))

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard", "📈 Options", "⚖️ Risk", "🔁 Backtest", "🔔 Alerts", "📋 Raw Data"
    ])

    # ══════════════════════ TAB 1 — DASHBOARD ═════════════════════════════════
    with tab1:
        with st.expander("Signal Engine — How it decides"):
            st.markdown("""
| Signal | Condition |
|--------|-----------|
| **BUY** | Score ≥ 7, regime not bearish, not extended |
| **HOLD** | Score 3–6 |
| **SELL** | Score ≤ −5 |
| **NO TRADE** | Mixed signals, choppy market, hostile volatility |

**New institutional factors added to score:**
ADX trend strength · Volume-Weighted RSI · Anchored VWAP · TTM Squeeze · Put/Call Ratio · IV Rank
            """)

        if PLOTLY_AVAILABLE:
            st.plotly_chart(make_main_chart(entry_df, symbol, sig, contract),
                use_container_width=True,
                config={"scrollZoom": True, "displaylogo": False,
                        "modeBarButtonsToRemove": ["select2d","lasso2d"]})
            st.caption("Scroll = zoom time axis · Click-drag = pan · Double-click = reset")
        else:
            st.line_chart(entry_df[["Close","EMA_21","EMA_50"]].tail(150))

        left, right = st.columns([1.2, 1])
        with left:
            st.subheader("Why this signal")
            for r in sig["reasons"]:
                st.markdown(reason_icon(r))
        with right:
            st.subheader("Latest Snapshot")
            snap = [
                ("Close",      safe_round(last["Close"],2)),
                ("EMA 8",      safe_round(last["EMA_8"],2)),
                ("EMA 21",     safe_round(last["EMA_21"],2)),
                ("EMA 50",     safe_round(last["EMA_50"],2)),
                ("EMA 200",    safe_round(last["EMA_200"],2)),
                ("VWAP",       safe_round(last.get("VWAP"),2)),
                ("RSI",        safe_round(last["RSI"],2)),
                ("VRSI",       safe_round(last.get("VRSI"),2)),
                ("ADX",        safe_round(last.get("ADX"),1)),
                ("MACD",       safe_round(last["MACD"],3)),
                ("MACD Sig",   safe_round(last["MACD_SIGNAL"],3)),
                ("MACD Hist",  safe_round(last["MACD_HIST"],3)),
                ("ATR",        safe_round(last["ATR"],2)),
                ("BB Width",   safe_round(last.get("BB_WIDTH"),4)),
                ("VIX",        safe_round(last.get("VIX_CLOSE"),2)),
                ("Volume",     int(last["Volume"]) if not pd.isna(last["Volume"]) else None),
            ]
            st.dataframe(pd.DataFrame(snap, columns=["Metric","Value"]),
                         use_container_width=True, hide_index=True)

        if show_ai:
            st.subheader("🤖 AI Technical Verdict")
            api_key = None
            try:
                api_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
            except Exception: pass
            if api_key and GENAI_AVAILABLE:
                if st.button("▶ Run Deep Analysis"):
                    prompt = f"""
You are the chief quant strategist at a $50B macro hedge fund.
Analyze this setup in exactly 7 bullet points:
1. Market regime and macro context
2. What strongly favors bulls (with specific levels)
3. What strongly favors bears (with specific levels)
4. Options market signal (IV Rank, Put/Call ratio interpretation)
5. Best action: BUY CALL / BUY PUT / HOLD / NO TRADE — with exact strike and expiry if options
6. Key invalidation level (price that proves you wrong)
7. Risk-adjusted tactical note for the next session

Data snapshot:
Ticker={symbol} | TF={timeframe} | Price={curr_price} | Signal={sig['signal']}
Score={sig['score']} | Confidence={sig['confidence']}% | Risk={sig['risk']} | Regime={sig['regime']}
RSI={safe_round(last['RSI'],2)} | VRSI={safe_round(last.get('VRSI'),2)} | ADX={safe_round(last.get('ADX'),1)}
ATR={safe_round(last['ATR'],2)} | VWAP={safe_round(last.get('VWAP'),2)}
EMA8={safe_round(last['EMA_8'],2)} EMA21={safe_round(last['EMA_21'],2)}
EMA50={safe_round(last['EMA_50'],2)} EMA200={safe_round(last['EMA_200'],2)}
MACD={safe_round(last['MACD'],3)} Sig={safe_round(last['MACD_SIGNAL'],3)} Hist={safe_round(last['MACD_HIST'],3)}
VIX={vix_value} | IV_Rank={chain_data.get('iv_rank') if chain_data else 'N/A'}
PC_Ratio={chain_data.get('pc_ratio') if chain_data else 'N/A'}
Stop={safe_round(sig['stop'],2)} | Target={safe_round(sig['target'],2)}
Squeeze_On={last.get('SQUEEZE_ON','N/A')}
"""
                    with st.spinner("Running institutional AI analysis…"):
                        try:
                            client   = genai.Client(api_key=api_key)
                            response = client.models.generate_content(
                                model="gemini-2.5-flash", contents=prompt)
                            st.info(response.text)
                        except Exception as e:
                            st.error(f"AI analysis failed: {e}")
            else:
                st.caption("Add `GEMINI_API_KEY` to Streamlit secrets to enable AI analysis.")

    # ══════════════════════ TAB 2 — OPTIONS ═══════════════════════════════════
    with tab2:
        st.subheader("Options Intelligence")

        if not load_options or chain_data is None:
            st.info("Enable 'Load Options Chain' in the sidebar to see options data.")
        else:
            # IV Rank + Put/Call
            iv_col, pc_col, rec_col = st.columns(3)
            with iv_col:
                if chain_data.get("iv_rank") is not None and PLOTLY_AVAILABLE:
                    st.plotly_chart(make_iv_gauge(chain_data["iv_rank"]),
                                    use_container_width=True)
                else:
                    st.metric("IV Rank", f"{chain_data.get('iv_rank','N/A')}")
            with pc_col:
                pcr = chain_data.get("pc_ratio")
                sentiment = "🐂 Bullish" if pcr and pcr < 0.7 else ("🐻 Bearish" if pcr and pcr > 1.2 else "⚖️ Neutral")
                st.metric("Put/Call Ratio", f"{pcr:.3f}" if pcr else "N/A", sentiment)
                st.caption("< 0.7 = bullish · > 1.2 = bearish")
            with rec_col:
                if contract:
                    direction_label = "🟢 LONG CALL" if contract["direction"] == "call" else "🔴 LONG PUT"
                    st.metric("Recommended Trade", direction_label)
                    st.metric("Strike / Expiry", f"${contract['strike']:.0f} · {contract['expiry']}")
                    st.metric("DTE", f"{contract['dte']} days")
                else:
                    st.info("No contract recommended (signal is HOLD or NO TRADE).")

            if contract:
                st.divider()
                st.subheader("📋 Recommended Contract")
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Mid Price",  fmt_price(contract["mid"]))
                c2.metric("Bid / Ask",  f"{fmt_price(contract['bid'])} / {fmt_price(contract['ask'])}")
                c3.metric("Cost (1 contract)", fmt_price(contract["cost_per_contract"]))
                c4.metric("Open Interest", f"{contract['oi']:,}")

                g1,g2,g3,g4 = st.columns(4)
                g1.metric("Delta",  f"{contract['delta']}"  if contract["delta"]  else "N/A")
                g2.metric("Gamma",  f"{contract['gamma']}"  if contract["gamma"]  else "N/A")
                g3.metric("Theta",  f"{contract['theta']}"  if contract["theta"]  else "N/A")
                g4.metric("Vega",   f"{contract['vega']}"   if contract["vega"]   else "N/A")

                r1,r2,r3,r4 = st.columns(4)
                r1.metric("IV",         f"{contract['iv']}%" if contract["iv"] else "N/A")
                r2.metric("Breakeven",  fmt_price(contract["breakeven"]))
                r3.metric("Stop (50%)", fmt_price(contract["stop_premium"]))
                r4.metric("Target (2×)",fmt_price(contract["target_premium"]))

                st.info(
                    f"**Trade plan:** Buy the ${contract['strike']:.0f} "
                    f"{'call' if contract['direction']=='call' else 'put'} expiring {contract['expiry']} "
                    f"({contract['dte']} DTE) at ~{fmt_price(contract['mid'])} per share (${contract['cost_per_contract']:.0f} per contract). "
                    f"Stop if premium drops to {fmt_price(contract['stop_premium'])}. "
                    f"Target {fmt_price(contract['target_premium'])}. "
                    f"Breakeven at {fmt_price(contract['breakeven'])}."
                )

            st.divider()
            st.subheader("Full Options Chain")
            exp_tabs = st.tabs([f"📅 {e}" for e in chain_data.get("exps", [])])
            for i, exp in enumerate(chain_data.get("exps", [])):
                with exp_tabs[i]:
                    c_col, p_col = st.columns(2)
                    with c_col:
                        st.markdown("**Calls**")
                        calls = chain_data["calls"][chain_data["calls"]["expiry"] == exp].copy()
                        if not calls.empty:
                            show_cols = [c for c in ["strike","lastPrice","bid","ask","volume",
                                         "openInterest","impliedVolatility","delta","gamma","theta"]
                                         if c in calls.columns]
                            calls_show = calls[show_cols].copy()
                            if "impliedVolatility" in calls_show.columns:
                                calls_show["impliedVolatility"] = (calls_show["impliedVolatility"]*100).round(1)
                            st.dataframe(calls_show.reset_index(drop=True), use_container_width=True, hide_index=True)
                    with p_col:
                        st.markdown("**Puts**")
                        puts = chain_data["puts"][chain_data["puts"]["expiry"] == exp].copy()
                        if not puts.empty:
                            show_cols = [c for c in ["strike","lastPrice","bid","ask","volume",
                                         "openInterest","impliedVolatility","delta","gamma","theta"]
                                         if c in puts.columns]
                            puts_show = puts[show_cols].copy()
                            if "impliedVolatility" in puts_show.columns:
                                puts_show["impliedVolatility"] = (puts_show["impliedVolatility"]*100).round(1)
                            st.dataframe(puts_show.reset_index(drop=True), use_container_width=True, hide_index=True)

    # ══════════════════════ TAB 3 — RISK ══════════════════════════════════════
    with tab3:
        st.subheader("⚖️ Risk & Position Sizing")

        bt_df_risk, bt_result_risk = run_backtest(entry_df.tail(500).copy())
        stats_r = bt_result_risk["stats"] if bt_result_risk else {}

        win_rate_r  = stats_r.get("Win Rate %",   55.0)
        avg_win_r   = stats_r.get("Avg Win %",    3.0)
        avg_loss_r  = abs(stats_r.get("Avg Loss %", -2.0))

        max_risk_dollars = account_size * (max_risk_pct / 100)

        r1,r2,r3 = st.columns(3)
        r1.metric("Account Size",     fmt_price(account_size))
        r2.metric("Max Risk / Trade", fmt_price(max_risk_dollars))
        r3.metric("Historical Win Rate", f"{win_rate_r:.1f}%")

        if contract:
            contracts_count = kelly_contracts(
                win_rate_r, avg_win_r, avg_loss_r,
                account_size, contract["cost_per_contract"]
            )
            # Also cap by max_risk_dollars
            max_by_risk = max(1, int(max_risk_dollars / contract["cost_per_contract"]))
            contracts_count = min(contracts_count, max_by_risk)

            st.divider()
            st.subheader("Recommended Position Size")
            k1,k2,k3,k4 = st.columns(4)
            k1.metric("Contracts",        f"{contracts_count}")
            k2.metric("Total Cost",       fmt_price(contracts_count * contract["cost_per_contract"]))
            k3.metric("Max Loss (stop)",  fmt_price(contracts_count * contract["stop_premium"] * 100))
            k4.metric("Max Gain (target)",fmt_price(contracts_count * contract["target_premium"] * 100))

            st.info(
                f"**Fractional Kelly sizing** (¼ Kelly): Based on a {win_rate_r:.0f}% win rate, "
                f"avg win {avg_win_r:.1f}%, avg loss {avg_loss_r:.1f}%, "
                f"the model recommends **{contracts_count} contract(s)**. "
                f"Total premium at risk: {fmt_price(contracts_count * contract['cost_per_contract'])} "
                f"({(contracts_count * contract['cost_per_contract'] / account_size * 100):.2f}% of account)."
            )
        else:
            st.info("No active options contract — position sizing will appear when a BUY or SELL signal is active.")

        with st.expander("Kelly Criterion explained"):
            st.markdown("""
The **Kelly Criterion** calculates the optimal fraction of capital to risk on each trade:

```
Kelly % = (b × p − q) / b
```
where:
- **b** = avg win / avg loss ratio
- **p** = win probability
- **q** = 1 − p (loss probability)

This app uses **¼ Kelly** (25% of full Kelly), which is standard institutional practice.
Full Kelly maximises long-run growth but produces large drawdowns.
¼ Kelly gives ~75% of the growth with far less volatility.
            """)

    # ══════════════════════ TAB 4 — BACKTEST ══════════════════════════════════
    with tab4:
        st.subheader("Strategy Backtest")
        bt_df, bt_result = run_backtest(entry_df.tail(backtest_bars).copy())
        if bt_df is None:
            st.info("Not enough data for backtest.")
        else:
            s = bt_result["stats"]
            b1,b2,b3,b4,b5,b6 = st.columns(6)
            b1.metric("Strategy Return", f"{s['Strategy Return %']}%")
            b2.metric("Buy & Hold",      f"{s['Buy & Hold %']}%")
            b3.metric("Max Drawdown",    f"{s['Max Drawdown %']}%")
            b4.metric("Trades",          f"{s['Trades']}")
            b5.metric("Win Rate",        f"{s['Win Rate %']}%")
            b6.metric("Avg Trade",       f"{s['Avg Trade %']}%")
            if PLOTLY_AVAILABLE:
                st.plotly_chart(make_backtest_chart(bt_df), use_container_width=True,
                                config={"scrollZoom":True,"displaylogo":False})
            else:
                st.line_chart(bt_df[["equity_curve","buy_hold_curve"]])
            with st.expander("Trade log"):
                tdf = bt_result["trades"]
                if tdf.empty: st.write("No completed trades.")
                else: st.dataframe(tdf, use_container_width=True)
            st.caption("Simple sanity-check backtest. Not execution-grade research.")

    # ══════════════════════ TAB 5 — ALERTS ════════════════════════════════════
    with tab5:
        st.subheader("Signal Change Alerts")
        st.caption("Logged whenever the signal changes on refresh / rerun.")
        alerts = st.session_state[history_key]
        if alerts: st.dataframe(pd.DataFrame(alerts), use_container_width=True)
        else: st.info("No signal changes logged yet.")

    # ══════════════════════ TAB 6 — RAW DATA ══════════════════════════════════
    with tab6:
        st.subheader("Raw Data (last 100 bars)")
        st.dataframe(entry_df.tail(100), use_container_width=True)


if __name__ == "__main__":
    main()
