import json
import urllib.request
import urllib.error
from typing import Tuple, Optional

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Optional Plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# Optional Gemini
try:
    from google import genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False


# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="SPY Buddy Pro Elite X",
    page_icon="📈",
    layout="wide"
)

st.title("📈 SPY Buddy Pro Elite X")
st.caption("Research / education dashboard. Not financial advice.")


# ----------------------------
# CONFIG
# ----------------------------
TF_MAP = {
    "1 Day": {"period": "2y", "interval": "1d", "bars_year": 252, "chart_bars": 220},
    "1 Hour": {"period": "730d", "interval": "1h", "bars_year": 252 * 7, "chart_bars": 220},
    "15 Min": {"period": "60d", "interval": "15m", "bars_year": 252 * 26, "chart_bars": 260},
    "5 Min": {"period": "60d", "interval": "5m", "bars_year": 252 * 78, "chart_bars": 300},
    "1 Min": {"period": "7d", "interval": "1m", "bars_year": 252 * 390, "chart_bars": 390},
}

OPENING_RANGE_BARS = {
    "1 Hour": 1,
    "15 Min": 2,
    "5 Min": 6,
    "1 Min": 30,
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


def send_webhook_alert(webhook_url: str, payload: dict) -> Tuple[bool, str]:
    if not webhook_url:
        return False, "Missing webhook URL"

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = getattr(resp, "status", 200)
        return True, f"HTTP {status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTPError {e.code}"
    except urllib.error.URLError as e:
        return False, f"URLError {e.reason}"
    except Exception as e:
        return False, str(e)


def _normalized_dates(index_like) -> pd.Series:
    idx = pd.to_datetime(index_like)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    return pd.Series(idx).dt.normalize()


def is_intraday_df(df: pd.DataFrame) -> bool:
    if df.empty or len(df.index) < 2:
        return False
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    diffs = pd.Series(idx).diff().dropna()
    if diffs.empty:
        return False
    return diffs.median() < pd.Timedelta(days=1)


def add_indicators(df: pd.DataFrame, timeframe_label: str) -> pd.DataFrame:
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
    df["VOL_AVG_20"] = df["Volume"].rolling(20).mean() if "Volume" in df.columns else np.nan

    # Session VWAP for intraday charts
    if is_intraday_df(df) and "Volume" in df.columns:
        session_dates = _normalized_dates(df.index).values
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
        tpv = typical_price * df["Volume"]
        df["VWAP"] = pd.Series(tpv).groupby(session_dates).cumsum().values / pd.Series(df["Volume"]).groupby(session_dates).cumsum().replace(0, np.nan).values
    else:
        df["VWAP"] = np.nan

    # Structure placeholders
    df["PREV_DAY_HIGH"] = np.nan
    df["PREV_DAY_LOW"] = np.nan
    df["OPENING_RANGE_HIGH"] = np.nan
    df["OPENING_RANGE_LOW"] = np.nan

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


def attach_reference_levels(entry_df: pd.DataFrame, daily_df: pd.DataFrame, timeframe_label: str) -> pd.DataFrame:
    df = entry_df.copy()
    if df.empty:
        return df

    if timeframe_label == "1 Day":
        df["PREV_DAY_HIGH"] = df["High"].shift(1)
        df["PREV_DAY_LOW"] = df["Low"].shift(1)
        return df

    if daily_df.empty:
        return df

    daily = daily_df.copy()
    daily.index = pd.to_datetime(daily.index)
    if getattr(daily.index, "tz", None) is not None:
        daily.index = daily.index.tz_convert(None)
    daily["SESSION_DATE"] = daily.index.normalize()
    daily["PREV_DAY_HIGH"] = daily["High"].shift(1)
    daily["PREV_DAY_LOW"] = daily["Low"].shift(1)

    date_map = daily.set_index("SESSION_DATE")[["PREV_DAY_HIGH", "PREV_DAY_LOW"]]

    temp = df.copy()
    temp.index = pd.to_datetime(temp.index)
    if getattr(temp.index, "tz", None) is not None:
        temp.index = temp.index.tz_convert(None)
    temp["SESSION_DATE"] = temp.index.normalize()
    temp["PREV_DAY_HIGH"] = temp["SESSION_DATE"].map(date_map["PREV_DAY_HIGH"])
    temp["PREV_DAY_LOW"] = temp["SESSION_DATE"].map(date_map["PREV_DAY_LOW"])
    temp.drop(columns=["SESSION_DATE"], inplace=True)
    return temp


def attach_opening_range(df: pd.DataFrame, timeframe_label: str) -> pd.DataFrame:
    out = df.copy()
    if out.empty or timeframe_label not in OPENING_RANGE_BARS or not is_intraday_df(out):
        return out

    bars_n = OPENING_RANGE_BARS[timeframe_label]
    temp = out.copy()
    temp.index = pd.to_datetime(temp.index)
    if getattr(temp.index, "tz", None) is not None:
        temp.index = temp.index.tz_convert(None)
    temp["SESSION_DATE"] = temp.index.normalize()

    or_high = temp.groupby("SESSION_DATE").apply(lambda g: g["High"].iloc[: min(bars_n, len(g))].max())
    or_low = temp.groupby("SESSION_DATE").apply(lambda g: g["Low"].iloc[: min(bars_n, len(g))].min())

    temp["OPENING_RANGE_HIGH"] = temp["SESSION_DATE"].map(or_high)
    temp["OPENING_RANGE_LOW"] = temp["SESSION_DATE"].map(or_low)
    temp.drop(columns=["SESSION_DATE"], inplace=True)
    return temp


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

    if pd.notna(row.get("VWAP", np.nan)):
        score += 1 if row["Close"] > row["VWAP"] else -1

    return score


def detect_market_regime(daily_df: pd.DataFrame, vix_value: float) -> str:
    if daily_df.empty or len(daily_df) < 200:
        return "Insufficient Data"

    row = daily_df.iloc[-1]
    atr_pct = row["ATR"] / row["Close"] if pd.notna(row["ATR"]) and row["Close"] else np.nan

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
    if pd.notna(atr_pct) and atr_pct > 0.025:
        return "Volatility Expansion"
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

    # VWAP
    if pd.notna(row.get("VWAP", np.nan)):
        if row["Close"] > row["VWAP"]:
            score += 1
            reasons.append("Price is above VWAP.")
        else:
            score -= 1
            reasons.append("Price is below VWAP.")

    # Previous day structure
    if pd.notna(row.get("PREV_DAY_HIGH", np.nan)) and row["Close"] > row["PREV_DAY_HIGH"]:
        score += 1
        reasons.append("Price is above the previous day high.")
    elif pd.notna(row.get("PREV_DAY_LOW", np.nan)) and row["Close"] < row["PREV_DAY_LOW"]:
        score -= 1
        reasons.append("Price is below the previous day low.")

    # Opening range structure
    if pd.notna(row.get("OPENING_RANGE_HIGH", np.nan)) and row["Close"] > row["OPENING_RANGE_HIGH"]:
        score += 1
        reasons.append("Price is above the opening range high.")
    elif pd.notna(row.get("OPENING_RANGE_LOW", np.nan)) and row["Close"] < row["OPENING_RANGE_LOW"]:
        score -= 1
        reasons.append("Price is below the opening range low.")

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

    if score >= 8 and "Bear" not in regime and not extended:
        signal = "BUY"
    elif score <= -5:
        signal = "SELL"
    elif 3 <= score <= 7:
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
            target = close + 2.2 * atr
        elif signal == "SELL":
            stop = close + 1.2 * atr
            target = close - 2.2 * atr

    if vix_value > 24:
        risk = "High"
    elif vix_value < 18 and "Bull" in regime:
        risk = "Low"

    confidence = min(95, max(5, 48 + score * 5))

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

    out["score"] += np.where(out["Close"] > out["EMA_21"], 1, -1)
    out["score"] += np.where(out["EMA_21"] > out["EMA_50"], 1, -1)
    out["score"] += np.where(out["EMA_50"] > out["EMA_200"], 1, -1)

    out["score"] += np.where((out["RSI"] >= 52) & (out["RSI"] <= 68), 1, 0)
    out["score"] += np.where(out["RSI"] < 45, -1, 0)
    out["score"] += np.where(out["RSI"] > 72, -1, 0)

    out["score"] += np.where(out["MACD_HIST"] > 0, 1, -1)
    out["score"] += np.where(out["Volume"] > out["VOL_AVG_20"], 1, 0)

    if "VIX_CLOSE" in out.columns:
        out["score"] += np.where(out["VIX_CLOSE"] < 18, 1, 0)
        out["score"] += np.where(out["VIX_CLOSE"] > 24, -2, 0)

    if "VWAP" in out.columns:
        out["score"] += np.where(pd.notna(out["VWAP"]) & (out["Close"] > out["VWAP"]), 1, 0)
        out["score"] += np.where(pd.notna(out["VWAP"]) & (out["Close"] < out["VWAP"]), -1, 0)

    if "PREV_DAY_HIGH" in out.columns:
        out["score"] += np.where(pd.notna(out["PREV_DAY_HIGH"]) & (out["Close"] > out["PREV_DAY_HIGH"]), 1, 0)
    if "PREV_DAY_LOW" in out.columns:
        out["score"] += np.where(pd.notna(out["PREV_DAY_LOW"]) & (out["Close"] < out["PREV_DAY_LOW"]), -1, 0)

    if "OPENING_RANGE_HIGH" in out.columns:
        out["score"] += np.where(pd.notna(out["OPENING_RANGE_HIGH"]) & (out["Close"] > out["OPENING_RANGE_HIGH"]), 1, 0)
    if "OPENING_RANGE_LOW" in out.columns:
        out["score"] += np.where(pd.notna(out["OPENING_RANGE_LOW"]) & (out["Close"] < out["OPENING_RANGE_LOW"]), -1, 0)

    atr_dist = abs(out["Close"] - out["EMA_21"]) / out["ATR"].replace(0, np.nan)
    out["extended"] = atr_dist > 1.8
    out["score"] += np.where(out["extended"], -1, 0)

    out["signal_label"] = np.select(
        [
            out["score"] >= 8,
            out["score"] <= -5,
            (out["score"] >= 3) & (out["score"] <= 7),
        ],
        [
            "BUY",
            "SELL",
            "HOLD",
        ],
        default="NO TRADE"
    )

    return out


def find_chart_signals(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    marked = vector_signal_score(df.copy())
    marked["prev_signal"] = marked["signal_label"].shift(1)

    marked["fresh_buy"] = (marked["signal_label"] == "BUY") & (marked["prev_signal"] != "BUY")
    marked["fresh_sell"] = (marked["signal_label"] == "SELL") & (marked["prev_signal"] != "SELL")

    buy_rows = []
    sell_rows = []
    open_trade = None

    for idx, row in marked.iterrows():
        if row["fresh_buy"]:
            buy_label = f"BUY<br>{float(row['Close']):.2f}"
            buy_rows.append({
                "index": idx,
                "Low": row["Low"],
                "High": row["High"],
                "ATR": row["ATR"],
                "Close": row["Close"],
                "label": buy_label,
            })
            open_trade = {
                "index": idx,
                "price": float(row["Close"]),
            }

        elif row["fresh_sell"]:
            sell_label = f"SELL<br>{float(row['Close']):.2f}"

            if open_trade is not None:
                pnl_pct = ((float(row["Close"]) / open_trade["price"]) - 1.0) * 100.0
                sell_label = f"SELL<br>{float(row['Close']):.2f}<br>{pnl_pct:+.2f}%"
                open_trade = None

            sell_rows.append({
                "index": idx,
                "Low": row["Low"],
                "High": row["High"],
                "ATR": row["ATR"],
                "Close": row["Close"],
                "label": sell_label,
            })

    buy_points = pd.DataFrame(buy_rows)
    sell_points = pd.DataFrame(sell_rows)

    return buy_points, sell_points


def run_backtest(df: pd.DataFrame, timeframe_label: str):
    bt = df.copy()
    bt = vector_signal_score(bt)
    bt = bt.dropna(subset=["EMA_21", "EMA_50", "EMA_200", "RSI", "MACD_HIST", "ATR"]).copy()

    if bt.empty or len(bt) < 80:
        return None, None

    position = []
    in_pos = 0

    for _, row in bt.iterrows():
        enter_long = row["score"] >= 8 and not row["extended"]
        hold_long = row["score"] >= 3 and row["Close"] > row["EMA_50"]
        exit_long = row["score"] <= 1 or row["Close"] < row["EMA_21"]

        if in_pos == 0 and enter_long:
            in_pos = 1
        elif in_pos == 1 and exit_long and not hold_long:
            in_pos = 0

        position.append(in_pos)

    bt["position"] = position
    bt["ret"] = bt["Close"].pct_change().fillna(0)
    bt["strategy_ret"] = bt["ret"] * bt["position"].shift(1).fillna(0)
    bt["equity_curve"] = (1 + bt["strategy_ret"]).cumprod()
    bt["buy_hold_curve"] = (1 + bt["ret"]).cumprod()

    bt["equity_peak"] = bt["equity_curve"].cummax()
    bt["drawdown"] = bt["equity_curve"] / bt["equity_peak"] - 1

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
        gross_profit = trades_df.loc[trades_df["Return %"] > 0, "Return %"].sum()
        gross_loss = abs(trades_df.loc[trades_df["Return %"] < 0, "Return %"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan
    else:
        win_rate = 0.0
        avg_trade = 0.0
        profit_factor = np.nan

    exposure = bt["position"].mean() * 100

    bars_year = TF_MAP[timeframe_label]["bars_year"]
    sharpe = np.nan
    if bt["strategy_ret"].std() not in [0, np.nan] and pd.notna(bt["strategy_ret"].std()):
        std = bt["strategy_ret"].std()
        if std > 0:
            sharpe = (bt["strategy_ret"].mean() / std) * np.sqrt(bars_year)

    stats = {
        "Strategy Return %": round(total_return, 2),
        "Buy & Hold %": round(buy_hold_return, 2),
        "Max Drawdown %": round(max_drawdown, 2),
        "Trades": int(total_trades),
        "Win Rate %": round(win_rate, 2),
        "Avg Trade %": round(avg_trade, 2),
        "Profit Factor": None if pd.isna(profit_factor) else round(float(profit_factor), 2),
        "Exposure %": round(float(exposure), 2),
        "Sharpe": None if pd.isna(sharpe) else round(float(sharpe), 2),
    }

    return bt, {"stats": stats, "trades": trades_df}


def make_candlestick_chart(df: pd.DataFrame, symbol: str, timeframe_label: str):
    chart_bars = TF_MAP[timeframe_label]["chart_bars"]
    chart_df = df.tail(chart_bars).copy()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.62, 0.18, 0.20],
        specs=[[{"secondary_y": True}], [{}], [{}]]
    )

    # Candles -> RIGHT axis
    fig.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df["Open"],
            high=chart_df["High"],
            low=chart_df["Low"],
            close=chart_df["Close"],
            name="Candles"
        ),
        row=1, col=1, secondary_y=True
    )

    # EMAs + VWAP -> RIGHT axis
    price_lines = [
        ("EMA_8", "rgba(99, 102, 241, 0.90)", "solid"),
        ("EMA_21", "rgba(245, 158, 11, 0.95)", "solid"),
        ("EMA_50", "rgba(14, 165, 233, 0.95)", "solid"),
        ("EMA_200", "rgba(244, 63, 94, 0.90)", "solid"),
        ("VWAP", "rgba(168, 85, 247, 0.95)", "solid"),
        ("PREV_DAY_HIGH", "rgba(34, 197, 94, 0.60)", "dash"),
        ("PREV_DAY_LOW", "rgba(239, 68, 68, 0.60)", "dash"),
        ("OPENING_RANGE_HIGH", "rgba(250, 204, 21, 0.65)", "dot"),
        ("OPENING_RANGE_LOW", "rgba(250, 204, 21, 0.65)", "dot"),
    ]

    for col, color, dash in price_lines:
        if col in chart_df.columns and chart_df[col].notna().sum() > 0:
            fig.add_trace(
                go.Scatter(
                    x=chart_df.index,
                    y=chart_df[col],
                    mode="lines",
                    name=col,
                    line=dict(color=color, dash=dash, width=1.6)
                ),
                row=1, col=1, secondary_y=True
            )

    # Volume -> LEFT axis
    volume_colors = np.where(chart_df["Close"] >= chart_df["Open"], "rgba(34, 197, 94, 0.30)", "rgba(239, 68, 68, 0.30)")
    fig.add_trace(
        go.Bar(
            x=chart_df.index,
            y=chart_df["Volume"],
            name="Volume",
            marker_color=volume_colors,
            opacity=0.45
        ),
        row=1, col=1, secondary_y=False
    )

    # RSI
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["RSI"],
            mode="lines",
            name="RSI",
            line=dict(color="rgba(59, 130, 246, 0.95)", width=1.8)
        ),
        row=2, col=1
    )
    fig.add_hline(y=70, row=2, col=1, line_dash="dot", line_color="rgba(239, 68, 68, 0.55)")
    fig.add_hline(y=30, row=2, col=1, line_dash="dot", line_color="rgba(34, 197, 94, 0.55)")

    # MACD
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["MACD"],
            mode="lines",
            name="MACD",
            line=dict(color="rgba(59, 130, 246, 0.95)", width=1.8)
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["MACD_SIGNAL"],
            mode="lines",
            name="MACD Signal",
            line=dict(color="rgba(245, 158, 11, 0.95)", width=1.8)
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Bar(
            x=chart_df.index,
            y=chart_df["MACD_HIST"],
            name="MACD Hist",
            marker_color=np.where(
                chart_df["MACD_HIST"] >= 0,
                "rgba(34, 197, 94, 0.65)",
                "rgba(239, 68, 68, 0.65)"
            ),
            opacity=0.45
        ),
        row=3, col=1
    )

    # Buy / Sell arrows
    buy_points, sell_points = find_chart_signals(chart_df)

    if not buy_points.empty:
        buy_points = buy_points.tail(20)
    if not sell_points.empty:
        sell_points = sell_points.tail(20)

    for _, row in buy_points.iterrows():
        x_val = row["index"]
        y_val = row["Low"] - (row["ATR"] * 0.25 if pd.notna(row["ATR"]) else row["Low"] * 0.003)

        fig.add_annotation(
            x=x_val,
            y=y_val,
            xref="x",
            yref="y2",
            text=row["label"],
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=2,
            arrowcolor="green",
            ax=0,
            ay=38,
            font=dict(color="green", size=10),
            align="center"
        )

    for _, row in sell_points.iterrows():
        x_val = row["index"]
        y_val = row["High"] + (row["ATR"] * 0.25 if pd.notna(row["ATR"]) else row["High"] * 0.003)

        fig.add_annotation(
            x=x_val,
            y=y_val,
            xref="x",
            yref="y2",
            text=row["label"],
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=2,
            arrowcolor="red",
            ax=0,
            ay=-42,
            font=dict(color="red", size=10),
            align="center"
        )

    fig.update_layout(
        title=f"{symbol} Candlestick + Structure / RSI / MACD",
        xaxis_rangeslider_visible=False,
        height=940,
        legend_orientation="h",
        dragmode="pan",
        hovermode="x unified"
    )

    # Swapped axes: volume left, price right
    fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)

    return fig


def make_backtest_chart(bt_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bt_df.index,
            y=bt_df["equity_curve"],
            mode="lines",
            name="Strategy",
            line=dict(width=2)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bt_df.index,
            y=bt_df["buy_hold_curve"],
            mode="lines",
            name="Buy & Hold",
            line=dict(width=2, dash="dot")
        )
    )
    fig.update_layout(
        title="Backtest Equity Curve",
        height=460,
        dragmode="pan"
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
    st.subheader("Alerts")
    enable_webhook = st.checkbox("Enable webhook alerts", value=False)
    webhook_url = st.text_input(
        "Webhook URL",
        value="",
        type="password",
        help="Paste a Discord, Slack-compatible, or automation webhook URL."
    )

    st.divider()
    show_ai = st.checkbox("Enable AI panel", value=True)

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ----------------------------
# MAIN
# ----------------------------
try:
    entry_raw, hourly_raw, daily_raw, vix_raw = get_all_data(symbol, timeframe)

    if entry_raw.empty:
        st.error(f"No data returned for {symbol}.")
        st.stop()

    if vix_raw.empty:
        st.error("No VIX data returned.")
        st.stop()

    entry_df = add_indicators(entry_raw, timeframe)
    hourly_df = add_indicators(hourly_raw, "1 Hour")
    daily_df = add_indicators(daily_raw, "1 Day")

    entry_df = attach_daily_vix(entry_df, vix_raw)
    hourly_df = attach_daily_vix(hourly_df, vix_raw)
    daily_df = attach_daily_vix(daily_df, vix_raw)

    entry_df = attach_reference_levels(entry_df, daily_df, timeframe)
    entry_df = attach_opening_range(entry_df, timeframe)

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

    # Alerts
    alert_key = f"last_signal_{symbol}_{timeframe}"
    history_key = f"alert_history_{symbol}_{timeframe}"

    if history_key not in st.session_state:
        st.session_state[history_key] = []

    prev_signal = st.session_state.get(alert_key)

    if prev_signal is None:
        st.session_state[alert_key] = sig["signal"]

    elif prev_signal != sig["signal"]:
        event_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        alert_row = {
            "Time": event_time,
            "Old Signal": prev_signal,
            "New Signal": sig["signal"],
            "Price": curr_price,
            "Ticker": symbol,
            "Timeframe": timeframe,
            "Confidence": sig["confidence"],
            "Risk": sig["risk"],
        }

        st.session_state[history_key].insert(0, alert_row)
        st.session_state[alert_key] = sig["signal"]

        st.warning(f"Signal changed: {prev_signal} → {sig['signal']} at {fmt_price(curr_price)}")

        toast_icon = {
            "BUY": "🟢",
            "HOLD": "🔵",
            "SELL": "🔴",
            "NO TRADE": "🟠"
        }.get(sig["signal"], "📈")

        st.toast(
            f"{symbol} {timeframe}: {prev_signal} → {sig['signal']} at {fmt_price(curr_price)}",
            icon=toast_icon
        )

        if enable_webhook and webhook_url:
            payload = {
                "text": (
                    f"{symbol} {timeframe} signal changed: "
                    f"{prev_signal} -> {sig['signal']} | "
                    f"Price: {curr_price} | "
                    f"Confidence: {sig['confidence']}% | "
                    f"Risk: {sig['risk']} | "
                    f"Regime: {sig['regime']}"
                )
            }

            ok, msg = send_webhook_alert(webhook_url, payload)

            if ok:
                st.toast("Webhook alert sent", icon="✅")
            else:
                st.toast(f"Webhook failed: {msg}", icon="❌")

    # Header
    st.subheader(f"Signal: :{signal_color(sig['signal'])}[{sig['signal']}]")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Price", fmt_price(curr_price), None if change is None else f"{change:+.2f}")
    c2.metric("Confidence", f"{sig['confidence']}%")
    c3.metric("Score", f"{sig['score']}")
    c4.metric("VIX", f"{vix_value}")
    c5.metric("Risk", sig["risk"])
    c6.metric("Regime", sig["regime"])

    d1, d2, d3, d4, d5, d6 = st.columns(6)
    d1.metric("RSI", f"{safe_round(last['RSI'], 2)}")
    d2.metric("ATR", fmt_price(safe_round(last["ATR"], 2)))
    d3.metric("VWAP", fmt_price(safe_round(last.get("VWAP", np.nan), 2)))
    d4.metric("Prev Day High", fmt_price(safe_round(last.get("PREV_DAY_HIGH", np.nan), 2)))
    d5.metric("Prev Day Low", fmt_price(safe_round(last.get("PREV_DAY_LOW", np.nan), 2)))
    d6.metric("OR High", fmt_price(safe_round(last.get("OPENING_RANGE_HIGH", np.nan), 2)))

    e1, e2 = st.columns(2)
    e1.metric("Stop", fmt_price(safe_round(sig["stop"], 2)))
    e2.metric("Target", fmt_price(safe_round(sig["target"], 2)))

    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Backtest", "Alerts", "Raw Data"])

    with tab1:
        with st.expander("How the engine decides"):
            st.markdown(
                """
**BUY**
- Price above EMA21, EMA50, and usually VWAP
- Daily trend agrees
- RSI healthy, MACD positive
- Price behaves well vs previous-day and opening-range structure
- VIX not too hot
- Price not overly extended

**HOLD**
- Trend is okay, but setup is not strong enough for a fresh buy

**SELL**
- Trend and momentum break down
- Higher timeframe disagrees
- Price loses VWAP / structure
- Risk rises

**NO TRADE**
- Signals are mixed
- Price is too stretched
- Volatility is hostile
- Structure is messy
                """
            )

        st.subheader(f"{symbol} Chart")

        if PLOTLY_AVAILABLE:
            fig = make_candlestick_chart(entry_df, symbol, timeframe)
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "scrollZoom": True,
                    "displaylogo": False
                }
            )
            st.caption("Mouse wheel zooms. Mouse drag pans left and right. Double-click resets.")
        else:
            st.warning("Plotly is not installed, so showing a simpler chart without arrows or structure overlays.")
            fallback_cols = ["Close", "EMA_8", "EMA_21", "EMA_50"]
            if "EMA_200" in entry_df.columns and entry_df["EMA_200"].notna().sum() > 0:
                fallback_cols.append("EMA_200")
            if "VWAP" in entry_df.columns and entry_df["VWAP"].notna().sum() > 0:
                fallback_cols.append("VWAP")
            st.line_chart(entry_df[fallback_cols].tail(TF_MAP[timeframe]["chart_bars"]))

        left, right = st.columns([1.2, 1])

        with left:
            st.subheader("Why this signal")
            for reason in sig["reasons"]:
                st.write(f"- {reason}")

        with right:
            st.subheader("Latest Snapshot")
            snap = pd.DataFrame({
                "Close": [safe_round(last["Close"], 2)],
                "VWAP": [safe_round(last.get("VWAP", np.nan), 2)],
                "EMA_8": [safe_round(last["EMA_8"], 2)],
                "EMA_21": [safe_round(last["EMA_21"], 2)],
                "EMA_50": [safe_round(last["EMA_50"], 2)],
                "EMA_200": [safe_round(last["EMA_200"], 2)],
                "RSI": [safe_round(last["RSI"], 2)],
                "MACD": [safe_round(last["MACD"], 3)],
                "MACD_SIGNAL": [safe_round(last["MACD_SIGNAL"], 3)],
                "MACD_HIST": [safe_round(last["MACD_HIST"], 3)],
                "ATR": [safe_round(last["ATR"], 2)],
                "PREV_DAY_HIGH": [safe_round(last.get("PREV_DAY_HIGH", np.nan), 2)],
                "PREV_DAY_LOW": [safe_round(last.get("PREV_DAY_LOW", np.nan), 2)],
                "OPENING_RANGE_HIGH": [safe_round(last.get("OPENING_RANGE_HIGH", np.nan), 2)],
                "OPENING_RANGE_LOW": [safe_round(last.get("OPENING_RANGE_LOW", np.nan), 2)],
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

Analyze this setup in 7 short bullet points:
1. Market regime
2. What favors bulls
3. What favors bears
4. What structure matters now
5. Best action now: BUY / HOLD / SELL / NO TRADE
6. Key invalidation
7. Tactical note

Data:
Ticker: {symbol}
Timeframe: {timeframe}
Price: {curr_price}
Signal: {sig['signal']}
Confidence: {sig['confidence']}%
Score: {sig['score']}
Risk: {sig['risk']}
Regime: {sig['regime']}
VWAP: {safe_round(last.get('VWAP', np.nan), 2)}
Prev Day High: {safe_round(last.get('PREV_DAY_HIGH', np.nan), 2)}
Prev Day Low: {safe_round(last.get('PREV_DAY_LOW', np.nan), 2)}
Opening Range High: {safe_round(last.get('OPENING_RANGE_HIGH', np.nan), 2)}
Opening Range Low: {safe_round(last.get('OPENING_RANGE_LOW', np.nan), 2)}
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
        st.subheader("Advanced Backtest")

        bt_source = entry_df.tail(backtest_bars).copy()
        bt_df, bt_result = run_backtest(bt_source, timeframe)

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

            c7, c8, c9 = st.columns(3)
            c7.metric("Profit Factor", "N/A" if s["Profit Factor"] is None else f"{s['Profit Factor']}")
            c8.metric("Exposure", f"{s['Exposure %']}%")
            c9.metric("Sharpe", "N/A" if s["Sharpe"] is None else f"{s['Sharpe']}")

            if PLOTLY_AVAILABLE:
                st.plotly_chart(
                    make_backtest_chart(bt_df),
                    use_container_width=True,
                    config={
                        "scrollZoom": True,
                        "displaylogo": False
                    }
                )
            else:
                st.line_chart(bt_df[["equity_curve", "buy_hold_curve"]])

            with st.expander("Backtest trade log"):
                trades_df = bt_result["trades"]
                if trades_df.empty:
                    st.write("No completed trades in this window.")
                else:
                    st.dataframe(trades_df, use_container_width=True)

            st.caption("This backtest is still simplified, but it is stronger than the earlier version: more structure-aware, more stats, still not execution-grade research.")

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
        st.dataframe(entry_df.tail(120), use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
