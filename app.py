import math
from typing import Any, Dict, Optional

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="SPY Buddy Options V1", page_icon="🎯", layout="wide")
st.title("🎯 SPY Buddy Options V1")
st.caption("Live P&L, better options states, contract quality filter, and news panel.")

ALPACA_KEY = st.secrets.get("ALPACA_API_KEY", "")
ALPACA_SECRET = st.secrets.get("ALPACA_SECRET_KEY", "")

PAPER_BASE = "https://paper-api.alpaca.markets"
DATA_BASE = "https://data.alpaca.markets"
DEFAULT_SYMBOL = "SPY"


def headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
        "accept": "application/json",
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


@st.cache_data(ttl=20)
def get_stock_snapshot(symbol: str) -> dict:
    return api_get(f"{DATA_BASE}/v2/stocks/{symbol}/snapshot")


@st.cache_data(ttl=30)
def get_stock_bars(symbol: str, timeframe: str = "5Min", limit: int = 120) -> pd.DataFrame:
    payload = api_get(
        f"{DATA_BASE}/v2/stocks/bars",
        params={
            "symbols": symbol.upper(),
            "timeframe": timeframe,
            "limit": limit,
            "adjustment": "raw",
            "feed": "iex",
            "sort": "asc",
        },
    )
    bars = payload.get("bars", {}).get(symbol.upper(), [])
    if not bars:
        return pd.DataFrame()

    df = pd.DataFrame(bars)
    df["Time"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert("America/New_York")
    df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    return df[["Time", "Open", "High", "Low", "Close", "Volume"]]


def add_underlying_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    out["EMA_8"] = out["Close"].ewm(span=8, adjust=False).mean()
    out["EMA_21"] = out["Close"].ewm(span=21, adjust=False).mean()
    out["EMA_50"] = out["Close"].ewm(span=50, adjust=False).mean()

    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    out["RSI"] = 100 - (100 / (1 + rs))
    return out


def stock_signal(df: pd.DataFrame) -> tuple[str, int, list]:
    if df.empty or len(df) < 30:
        return "NO TRADE", 0, ["Not enough bar data."]

    row = df.iloc[-1]
    score = 0
    reasons = []

    if row["Close"] > row["EMA_8"]:
        score += 1
        reasons.append("Price above EMA 8.")
    else:
        score -= 1
        reasons.append("Price below EMA 8.")

    if row["EMA_8"] > row["EMA_21"]:
        score += 1
        reasons.append("EMA 8 above EMA 21.")
    else:
        score -= 1
        reasons.append("EMA 8 below EMA 21.")

    if row["EMA_21"] > row["EMA_50"]:
        score += 1
        reasons.append("EMA 21 above EMA 50.")
    else:
        score -= 1
        reasons.append("EMA 21 below EMA 50.")

    if pd.notna(row["RSI"]):
        if row["RSI"] > 55:
            score += 1
            reasons.append("RSI supportive.")
        elif row["RSI"] < 45:
            score -= 1
            reasons.append("RSI weak.")

    if score >= 3:
        return "BULLISH", score, reasons
    if score <= -3:
        return "BEARISH", score, reasons
    return "NEUTRAL", score, reasons


@st.cache_data(ttl=60)
def get_option_contracts(symbol: str, expiration_date: Optional[str], option_type: Optional[str]) -> list:
    params = {
        "underlying_symbols": symbol.upper(),
        "status": "active",
        "limit": 1000,
    }
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


def place_option_order(symbol: str, qty: int, side: str, order_type: str = "market", limit_price: Optional[float] = None) -> dict:
    payload: Dict[str, Any] = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": order_type,
        "time_in_force": "day",
    }
    if order_type == "limit" and limit_price is not None:
        payload["limit_price"] = str(limit_price)
    return api_post(f"{PAPER_BASE}/v2/orders", payload)


def contract_quality(
    underlying_price: Optional[float],
    strike: Optional[float],
    bid: Optional[float],
    ask: Optional[float],
    volume: Optional[float],
    open_interest: Optional[float],
    iv: Optional[float],
    delta: Optional[float],
) -> dict:
    score = 100
    reasons = []

    spread = None
    spread_pct = None

    if bid is not None and ask is not None:
        spread = float(ask) - float(bid)
        if ask and ask > 0:
            spread_pct = spread / float(ask)

    if spread_pct is None:
        score -= 25
        reasons.append("No usable bid/ask spread.")
    elif spread_pct > 0.20:
        score -= 30
        reasons.append("Spread too wide.")
    elif spread_pct > 0.10:
        score -= 15
        reasons.append("Spread somewhat wide.")
    else:
        reasons.append("Spread acceptable.")

    if ask is None:
        score -= 15
        reasons.append("No ask price.")
    elif ask < 0.10:
        score -= 20
        reasons.append("Premium too tiny / noisy.")
    elif ask > 10:
        score -= 10
        reasons.append("Premium is expensive.")
    else:
        reasons.append("Premium in a workable range.")

    if underlying_price is not None and strike is not None and underlying_price > 0:
        moneyness = abs(float(strike) - float(underlying_price)) / float(underlying_price)
        if moneyness > 0.05:
            score -= 20
            reasons.append("Strike far from underlying.")
        elif moneyness > 0.02:
            score -= 8
            reasons.append("Strike slightly far from underlying.")
        else:
            reasons.append("Strike near the underlying.")

    if volume is not None:
        if float(volume) < 10:
            score -= 15
            reasons.append("Low contract volume.")
        else:
            reasons.append("Volume acceptable.")

    if open_interest is not None:
        if float(open_interest) < 50:
            score -= 15
            reasons.append("Low open interest.")
        else:
            reasons.append("Open interest acceptable.")

    if iv is not None and float(iv) > 2.0:
        score -= 10
        reasons.append("Implied volatility very high.")

    if delta is None:
        reasons.append("Greeks unavailable.")
    else:
        reasons.append("Delta available.")

    score = max(0, min(100, score))
    quality_ok = score >= 55

    return {
        "score": score,
        "quality_ok": quality_ok,
        "spread": spread,
        "spread_pct": spread_pct,
        "reasons": reasons,
    }


def derive_options_state(stock_bias: str, option_side: str, has_position: bool, quality_ok: bool) -> str:
    side = option_side.upper()

    if not quality_ok:
        return "NO TRADE"

    if side == "CALL":
        if stock_bias == "BULLISH" and not has_position:
            return "ENTER CALL"
        if stock_bias == "BULLISH" and has_position:
            return "HOLD CALL"
        if stock_bias != "BULLISH" and has_position:
            return "EXIT CALL"
        return "NO TRADE"

    if side == "PUT":
        if stock_bias == "BEARISH" and not has_position:
            return "ENTER PUT"
        if stock_bias == "BEARISH" and has_position:
            return "HOLD PUT"
        if stock_bias != "BEARISH" and has_position:
            return "EXIT PUT"
        return "NO TRADE"

    return "NO TRADE"


if not ALPACA_KEY or not ALPACA_SECRET:
    st.error("Add ALPACA_API_KEY and ALPACA_SECRET_KEY to Streamlit secrets first.")
    st.stop()

with st.sidebar:
    st.header("Setup")
    symbol = st.text_input("Underlying", value=DEFAULT_SYMBOL).upper().strip()
    option_side = st.selectbox("Direction", ["Call", "Put"])
    tf = st.selectbox("Underlying timeframe", ["1Min", "5Min", "15Min", "1Hour"], index=1)
    qty = st.number_input("Contracts", min_value=1, max_value=100, value=1, step=1)
    order_style = st.selectbox("Order type", ["market", "limit"])

contracts_raw = get_option_contracts(symbol, None, option_side)
expirations = sorted({c.get("expiration_date") for c in contracts_raw if c.get("expiration_date")})

colA, colB, colC = st.columns(3)
with colA:
    expiration = st.selectbox("Expiration", expirations, index=0 if expirations else None)
with colB:
    contracts_for_exp = [c for c in contracts_raw if c.get("expiration_date") == expiration] if expiration else []
    strikes = sorted({float(c.get("strike_price")) for c in contracts_for_exp if c.get("strike_price") is not None})
    strike = st.selectbox("Strike", strikes, index=0 if strikes else None)
with colC:
    refresh_clicked = st.button("Refresh data", use_container_width=True)

if refresh_clicked:
    st.cache_data.clear()
    st.rerun()

selected_contract = None
if expiration and strike is not None:
    for c in contracts_for_exp:
        if float(c.get("strike_price")) == float(strike):
            selected_contract = c
            break

contract_symbol = selected_contract.get("symbol") if selected_contract else None

underlying_snapshot = get_stock_snapshot(symbol)
bars = add_underlying_indicators(get_stock_bars(symbol, tf, 120))
stock_bias, stock_score, stock_reasons = stock_signal(bars)

last_trade = safe_get(underlying_snapshot, "latestTrade", "p")
daily_close = safe_get(underlying_snapshot, "dailyBar", "c")
prev_close = safe_get(underlying_snapshot, "prevDailyBar", "c")

snapshot = get_option_snapshot(contract_symbol) if contract_symbol else {}
quote_bid = safe_get(snapshot, "latestQuote", "bp")
quote_ask = safe_get(snapshot, "latestQuote", "ap")
quote_mid = None
if quote_bid is not None and quote_ask is not None:
    quote_mid = (float(quote_bid) + float(quote_ask)) / 2.0
last_option_trade = safe_get(snapshot, "latestTrade", "p")
delta = safe_get(snapshot, "greeks", "delta")
gamma = safe_get(snapshot, "greeks", "gamma")
theta = safe_get(snapshot, "greeks", "theta")
vega = safe_get(snapshot, "greeks", "vega")
iv = safe_get(snapshot, "implied_volatility")
day_bar = safe_get(snapshot, "dailyBar", default={}) or {}
option_volume = day_bar.get("v")
open_interest = selected_contract.get("open_interest") if selected_contract else None

quality = contract_quality(
    underlying_price=last_trade,
    strike=strike,
    bid=quote_bid,
    ask=quote_ask,
    volume=option_volume,
    open_interest=open_interest,
    iv=iv,
    delta=delta,
)

position = find_position(contract_symbol) if contract_symbol else None
has_position = position is not None
state = derive_options_state(stock_bias, option_side, has_position, quality["quality_ok"])

st.subheader("Underlying")
u1, u2, u3, u4 = st.columns(4)
u1.metric("Bias", stock_bias)
u2.metric("Score", stock_score)
u3.metric("Latest Trade", fmt_money(last_trade))
u4.metric("Daily / Prev", f"{fmt_money(daily_close)} / {fmt_money(prev_close)}")

with st.expander("Why the underlying bias"):
    for r in stock_reasons:
        st.write(f"- {r}")

st.subheader("Option Contract")
o1, o2, o3, o4, o5, o6 = st.columns(6)
o1.metric("Contract", contract_symbol or "N/A")
o2.metric("Bid", fmt_money(quote_bid))
o3.metric("Ask", fmt_money(quote_ask))
o4.metric("Mid", fmt_money(quote_mid))
o5.metric("Last", fmt_money(last_option_trade))
o6.metric("Spread", fmt_money(quality["spread"]))

g1, g2, g3, g4, g5 = st.columns(5)
g1.metric("Delta", fmt_num(delta, 3))
g2.metric("Gamma", fmt_num(gamma, 4))
g3.metric("Theta", fmt_num(theta, 4))
g4.metric("Vega", fmt_num(vega, 4))
g5.metric("IV", fmt_num(iv, 3))

st.subheader("Contract Quality Filter")
q1, q2, q3, q4 = st.columns(4)
q1.metric("Quality Score", quality["score"])
q2.metric("Quality Verdict", "PASS" if quality["quality_ok"] else "FAIL")
q3.metric("Volume", fmt_num(option_volume, 0))
q4.metric("Open Interest", fmt_num(open_interest, 0))

with st.expander("Why this contract passed / failed"):
    for r in quality["reasons"]:
        st.write(f"- {r}")

st.subheader("Options State Engine")
s1, s2, s3 = st.columns(3)
s1.metric("State", state)
s2.metric("Holding this contract?", "Yes" if has_position else "No")
s3.metric("Direction", option_side)

default_entry = quote_ask if quote_ask is not None else (last_option_trade if last_option_trade is not None else 0.0)

st.subheader("Trade Plan")
p1, p2, p3, p4, p5 = st.columns(5)
with p1:
    entry_premium = st.number_input("Entry premium", min_value=0.0, value=float(default_entry or 0.0), step=0.05)
with p2:
    stop_pct = st.slider("Stop %", 5, 50, 20, 1)
with p3:
    tp1_pct = st.slider("TP1 %", 10, 100, 30, 1)
with p4:
    tp2_pct = st.slider("TP2 %", 20, 200, 50, 1)
with p5:
    min_rr = st.slider("Min R/R", 1.0, 3.0, 1.5, 0.1)

premium_stop = None
tp1 = None
tp2 = None
rr1 = None
rr2 = None
if entry_premium > 0:
    risk_amt = entry_premium * (stop_pct / 100.0)
    premium_stop = max(0.01, entry_premium - risk_amt)
    tp1 = entry_premium * (1 + tp1_pct / 100.0)
    tp2 = entry_premium * (1 + tp2_pct / 100.0)
    rr1 = (tp1 - entry_premium) / max(0.0001, entry_premium - premium_stop)
    rr2 = (tp2 - entry_premium) / max(0.0001, entry_premium - premium_stop)

r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("Premium Stop", fmt_money(premium_stop))
r2.metric("TP1", fmt_money(tp1))
r3.metric("TP2", fmt_money(tp2))
r4.metric("R/R to TP1", "N/A" if rr1 is None else f"{rr1:.2f}")
r5.metric("R/R to TP2", "N/A" if rr2 is None else f"{rr2:.2f}")

if state in ["ENTER CALL", "ENTER PUT"] and rr1 is not None and rr1 < min_rr:
    st.warning("Direction is good, but reward/risk is below your minimum. No trade.")
elif state == "NO TRADE":
    st.info("No trade right now.")
elif state in ["HOLD CALL", "HOLD PUT"]:
    st.info(f"{state}. Manage the open position.")
elif state in ["EXIT CALL", "EXIT PUT"]:
    st.warning(f"{state}. Consider closing the open position.")
else:
    st.success(state)

st.subheader("Live Position P&L")
l1, l2, l3, l4, l5, l6 = st.columns(6)
l1.metric("Position?", "Yes" if position else "No")
l2.metric("Qty", position.get("qty") if position else "0")
l3.metric("Avg Entry", fmt_money(position.get("avg_entry_price") if position else None))
l4.metric("Market Value", fmt_money(position.get("market_value") if position else None))
l5.metric("Unrealized P/L", fmt_money(position.get("unrealized_pl") if position else None))
l6.metric("Unrealized P/L %", fmt_num(position.get("unrealized_plpc") * 100 if position and position.get("unrealized_plpc") is not None else None, 2))

st.subheader("Recent News")
news = get_news(symbol, limit=8)
if not news:
    st.write("No recent news returned.")
else:
    for item in news[:6]:
        headline = item.get("headline", "No headline")
        source = item.get("source", "")
        ts = item.get("updated_at", item.get("created_at", ""))
        summary = item.get("summary", "")
        st.markdown(f"**{headline}**")
        st.caption(f"{source} • {ts}")
        if summary:
            st.write(summary[:240] + ("..." if len(summary) > 240 else ""))
        st.markdown("---")

st.subheader("Paper Actions")
limit_seed = quote_ask if quote_ask is not None else entry_premium
limit_price = st.number_input("Limit price (used only for limit orders)", min_value=0.0, value=float(limit_seed or 0.0), step=0.05)

a1, a2 = st.columns(2)
if a1.button("Paper Buy to Open", use_container_width=True, disabled=contract_symbol is None):
    try:
        order = place_option_order(contract_symbol, int(qty), "buy", order_style, limit_price if order_style == "limit" else None)
        st.success(f"Submitted buy order: {order.get('id', 'ok')}")
    except Exception as e:
        st.error(f"Buy order failed: {e}")

if a2.button("Paper Sell to Close", use_container_width=True, disabled=contract_symbol is None):
    try:
        order = place_option_order(contract_symbol, int(qty), "sell", order_style, limit_price if order_style == "limit" else None)
        st.success(f"Submitted sell order: {order.get('id', 'ok')}")
    except Exception as e:
        st.error(f"Sell order failed: {e}")

with st.expander("Contract details"):
    st.json(selected_contract or {})

with st.expander("Snapshot payload"):
    st.json(snapshot or {})

st.caption("This version adds: live P&L, better trade states, contract quality filter, and news panel.")
