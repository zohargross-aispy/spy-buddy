
import json
import urllib.request
import urllib.error
from typing import Tuple

import streamlit as st
refresh_ms = 30000 if timeframe != "1 Day" else 60000
if AUTOREFRESH_AVAILABLE:
    st_autorefresh(interval=refresh_ms, key=f"autorefresh_{timeframe}")

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except Exception:
    AUTOREFRESH_AVAILABLE = False

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


st.set_page_config(
    page_title="SPY Buddy Pro Elite X",
    page_icon="📈",
    layout="wide"
)

st.title("📈 SPY Buddy Pro Elite X")
st.caption("Lean chart view. Research / education only, not financial advice.")


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


def _normalize_index(idx_like):
    idx = pd.to_datetime(idx_like)
    if getattr(idx, "tz", None) is not None:
        try:
            idx = idx.tz_convert("America/New_York")
        except Exception:
            idx = idx.tz_convert(None)
    return idx


def _normalized_dates(index_like) -> pd.Series:
    idx = _normalize_index(index_like)
    return pd.Series(idx).dt.normalize()


def is_intraday_df(df: pd.DataFrame) -> bool:
    if df.empty or len(df.index) < 2:
        return False
    idx = _normalize_index(df.index)
    diffs = pd.Series(idx).diff().dropna()
    if diffs.empty:
        return False
    return diffs.median() < pd.Timedelta(days=1)


def add_live_daily_bar(daily_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Append an in-progress daily bar built from recent intraday data.
    This lets the 1 Day chart show today's session instead of only the last completed day.
    """
    if daily_df.empty:
        return daily_df

    intraday = yf.Ticker(symbol).history(period="5d", interval="1m", auto_adjust=False)
    if intraday is None or intraday.empty:
        return daily_df

    intraday = intraday.copy()
    intraday.index = _normalize_index(intraday.index)
    intraday = intraday[intraday.index.notna()]
    if intraday.empty:
        return daily_df

    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    today_rows = intraday[intraday.index.normalize() == today]
    if today_rows.empty:
        return daily_df

    live_bar = pd.DataFrame({
        "Open": [float(today_rows["Open"].iloc[0])],
        "High": [float(today_rows["High"].max())],
        "Low": [float(today_rows["Low"].min())],
        "Close": [float(today_rows["Close"].iloc[-1])],
        "Volume": [float(today_rows["Volume"].sum())],
    }, index=[today])

    out = daily_df.copy()
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert(None)

    if today in out.index:
        out.loc[today, ["Open", "High", "Low", "Close", "Volume"]] = live_bar.iloc[0][["Open", "High", "Low", "Close", "Volume"]]
    else:
        out = pd.concat([out, live_bar])

    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def add_indicators(df: pd.DataFrame, timeframe_label: str) -> pd.DataFrame:
    df = df.copy()

    if df.empty:
        return df

    df["EMA_8"] = df["Close"].ewm(span=8, adjust=False).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    df["VOL_AVG_20"] = df["Volume"].rolling(20).mean() if "Volume" in df.columns else np.nan

    if is_intraday_df(df) and "Volume" in df.columns:
        session_dates = _normalized_dates(df.index).values
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
        tpv = typical_price * df["Volume"]
        df["VWAP"] = (
            pd.Series(tpv).groupby(session_dates).cumsum().values
            / pd.Series(df["Volume"]).groupby(session_dates).cumsum().replace(0, np.nan).values
        )
    else:
        df["VWAP"] = np.nan

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

    if row["Close"] > row["EMA_21"]:
        score += 1
        reasons.append("Price above EMA21.")
    else:
        score -= 1
        reasons.append("Price below EMA21.")

    if row["EMA_21"] > row["EMA_50"]:
        score += 1
        reasons.append("EMA21 above EMA50.")
    else:
        score -= 1
        reasons.append("EMA21 below EMA50.")

    if not pd.isna(row["EMA_200"]):
        if row["EMA_50"] > row["EMA_200"]:
            score += 1
            reasons.append("EMA50 above EMA200.")
        else:
            score -= 1
            reasons.append("EMA50 below EMA200.")

    if 52 <= row["RSI"] <= 68:
        score += 1
        reasons.append("RSI healthy.")
    elif row["RSI"] < 45:
        score -= 1
        reasons.append("RSI weak.")
    elif row["RSI"] > 72:
        score -= 1
        reasons.append("RSI stretched.")

    if row["MACD_HIST"] > 0:
        score += 1
        reasons.append("MACD histogram positive.")
    else:
        score -= 1
        reasons.append("MACD histogram negative.")

    if "Volume" in entry_df.columns and not pd.isna(row["VOL_AVG_20"]):
        if row["Volume"] > row["VOL_AVG_20"]:
            score += 1
            reasons.append("Volume confirms.")

    if pd.notna(row.get("VWAP", np.nan)):
        if row["Close"] > row["VWAP"]:
            score += 1
            reasons.append("Price above VWAP.")
        else:
            score -= 1
            reasons.append("Price below VWAP.")

    if pd.notna(row.get("PREV_DAY_HIGH", np.nan)) and row["Close"] > row["PREV_DAY_HIGH"]:
        score += 1
        reasons.append("Above previous day high.")
    elif pd.notna(row.get("PREV_DAY_LOW", np.nan)) and row["Close"] < row["PREV_DAY_LOW"]:
        score -= 1
        reasons.append("Below previous day low.")

    if pd.notna(row.get("OPENING_RANGE_HIGH", np.nan)) and row["Close"] > row["OPENING_RANGE_HIGH"]:
        score += 1
        reasons.append("Above opening range high.")
    elif pd.notna(row.get("OPENING_RANGE_LOW", np.nan)) and row["Close"] < row["OPENING_RANGE_LOW"]:
        score -= 1
        reasons.append("Below opening range low.")

    if vix_value < 18:
        score += 1
        reasons.append("VIX supportive.")
    elif vix_value > 24:
        score -= 2
        reasons.append("VIX elevated.")

    htf = timeframe_bias(hourly_df)
    dtf = timeframe_bias(daily_df)

    if htf >= 2:
        score += 1
    elif htf <= -2:
        score -= 1

    if dtf >= 2:
        score += 2
    elif dtf <= -2:
        score -= 2

    extended = False
    if not pd.isna(row["ATR"]) and row["ATR"] > 0:
        dist = abs(row["Close"] - row["EMA_21"]) / row["ATR"]
        if dist > 1.8:
            extended = True
            score -= 1
            reasons.append("Extended versus ATR.")

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
        buffer = 0.18 * atr

        if signal == "BUY":
            support_candidates = sorted(
                [float(v) for v in [
                    row.get("VWAP", np.nan),
                    row.get("EMA_8", np.nan),
                    row.get("EMA_21", np.nan),
                    row.get("OPENING_RANGE_LOW", np.nan),
                    row.get("PREV_DAY_HIGH", np.nan),
                    row.get("PREV_DAY_LOW", np.nan),
                ] if pd.notna(v) and float(v) < float(close)],
                reverse=True,
            )
            structure_stop = support_candidates[0] if support_candidates else float(close - 0.9 * atr)
            stop = structure_stop - buffer
            risk_dist = max(float(close - stop), float(0.55 * atr))
            target_candidates = sorted(
                [float(v) for v in [
                    row.get("OPENING_RANGE_HIGH", np.nan),
                    row.get("PREV_DAY_HIGH", np.nan),
                    close + 1.15 * risk_dist,
                    close + 1.0 * atr,
                ] if pd.notna(v) and float(v) > float(close)]
            )
            target = target_candidates[0] if target_candidates else float(close + 1.15 * risk_dist)

        elif signal == "SELL":
            resistance_candidates = sorted(
                [float(v) for v in [
                    row.get("VWAP", np.nan),
                    row.get("EMA_8", np.nan),
                    row.get("EMA_21", np.nan),
                    row.get("OPENING_RANGE_HIGH", np.nan),
                    row.get("PREV_DAY_LOW", np.nan),
                    row.get("PREV_DAY_HIGH", np.nan),
                ] if pd.notna(v) and float(v) > float(close)]
            )
            structure_stop = resistance_candidates[0] if resistance_candidates else float(close + 0.9 * atr)
            stop = structure_stop + buffer
            risk_dist = max(float(stop - close), float(0.55 * atr))
            target_candidates = sorted(
                [float(v) for v in [
                    row.get("OPENING_RANGE_LOW", np.nan),
                    row.get("PREV_DAY_LOW", np.nan),
                    close - 1.15 * risk_dist,
                    close - 1.0 * atr,
                ] if pd.notna(v) and float(v) < float(close)],
                reverse=True,
            )
            target = target_candidates[0] if target_candidates else float(close - 1.15 * risk_dist)

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
        [out["score"] >= 8, out["score"] <= -5, (out["score"] >= 3) & (out["score"] <= 7)],
        ["BUY", "SELL", "HOLD"],
        default="NO TRADE"
    )
    return out


def find_chart_signals(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    marked = vector_signal_score(df.copy())
    marked["prev_signal"] = marked["signal_label"].shift(1)
    marked["fresh_buy"] = (marked["signal_label"] == "BUY") & (marked["prev_signal"] != "BUY")
    marked["fresh_sell"] = (marked["signal_label"] == "SELL") & (marked["prev_signal"] != "SELL")

    buy_rows, sell_rows = [], []
    open_trade = None

    for idx, row in marked.iterrows():
        if row["fresh_buy"]:
            buy_rows.append({
                "index": idx, "Low": row["Low"], "High": row["High"], "ATR": row["ATR"],
                "Close": row["Close"], "label": f"BUY<br>{float(row['Close']):.2f}",
            })
            open_trade = {"index": idx, "price": float(row["Close"])}

        elif row["fresh_sell"]:
            sell_label = f"SELL<br>{float(row['Close']):.2f}"
            if open_trade is not None:
                pnl_pct = ((float(row["Close"]) / open_trade["price"]) - 1.0) * 100.0
                sell_label = f"SELL<br>{float(row['Close']):.2f}<br>{pnl_pct:+.2f}%"
                open_trade = None

            sell_rows.append({
                "index": idx, "Low": row["Low"], "High": row["High"], "ATR": row["ATR"],
                "Close": row["Close"], "label": sell_label,
            })

    return pd.DataFrame(buy_rows), pd.DataFrame(sell_rows)


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
    std = bt["strategy_ret"].std()
    if pd.notna(std) and std > 0:
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


def build_options_plan(signal: str, premium_entry: float, stop_pct: float, tp1_pct: float, tp2_pct: float, min_rr: float) -> dict:
    if premium_entry <= 0:
        return {
            "enabled": False, "status": "No option premium entered.",
            "premium_stop": None, "premium_tp1": None, "premium_tp2": None,
            "rr1": None, "rr2": None, "trade_ok": None,
        }

    if signal not in ["BUY", "SELL"]:
        return {
            "enabled": True, "status": "Options plan only activates on BUY or SELL.",
            "premium_stop": None, "premium_tp1": None, "premium_tp2": None,
            "rr1": None, "rr2": None, "trade_ok": False,
        }

    risk_amt = premium_entry * (stop_pct / 100.0)
    premium_stop = max(0.01, premium_entry - risk_amt)
    premium_tp1 = premium_entry * (1 + tp1_pct / 100.0)
    premium_tp2 = premium_entry * (1 + tp2_pct / 100.0)
    rr1 = (premium_tp1 - premium_entry) / max(0.0001, premium_entry - premium_stop)
    rr2 = (premium_tp2 - premium_entry) / max(0.0001, premium_entry - premium_stop)
    trade_ok = rr1 >= min_rr

    return {
        "enabled": True,
        "status": "Options setup passes minimum reward/risk." if trade_ok else "NO TRADE for options. Reward/risk is below your minimum.",
        "premium_stop": premium_stop,
        "premium_tp1": premium_tp1,
        "premium_tp2": premium_tp2,
        "rr1": rr1,
        "rr2": rr2,
        "trade_ok": trade_ok,
    }


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

    volume_colors = np.where(
        chart_df["Close"] >= chart_df["Open"],
        "rgba(34, 197, 94, 0.30)",
        "rgba(239, 68, 68, 0.30)"
    )
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

    buy_points, sell_points = find_chart_signals(chart_df)
    if not buy_points.empty:
        buy_points = buy_points.tail(20)
    if not sell_points.empty:
        sell_points = sell_points.tail(20)

    for _, row in buy_points.iterrows():
        y_val = row["Low"] - (row["ATR"] * 0.25 if pd.notna(row["ATR"]) else row["Low"] * 0.003)
        fig.add_annotation(
            x=row["index"], y=y_val, xref="x", yref="y2", text=row["label"],
            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2,
            arrowcolor="green", ax=0, ay=38, font=dict(color="green", size=10), align="center"
        )

    for _, row in sell_points.iterrows():
        y_val = row["High"] + (row["ATR"] * 0.25 if pd.notna(row["ATR"]) else row["High"] * 0.003)
        fig.add_annotation(
            x=row["index"], y=y_val, xref="x", yref="y2", text=row["label"],
            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2,
            arrowcolor="red", ax=0, ay=-42, font=dict(color="red", size=10), align="center"
        )

    fig.update_layout(
        title=f"{symbol} Candlestick + Structure / RSI / MACD",
        xaxis_rangeslider_visible=False,
        height=940,
        legend_orientation="h",
        dragmode="pan",
        hovermode="x unified"
    )
    fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    return fig


def make_backtest_chart(bt_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df["equity_curve"], mode="lines", name="Strategy", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df["buy_hold_curve"], mode="lines", name="Buy & Hold", line=dict(width=2, dash="dot")))
    fig.update_layout(title="Backtest Equity Curve", height=460, dragmode="pan")
    return fig


with st.sidebar:
    st.header("Controls")
    symbol = st.text_input("Ticker", value=DEFAULT_SYMBOL).upper().strip()
    timeframe = st.selectbox("Chart Timeframe", list(TF_MAP.keys()), index=0)

    st.divider()
    st.subheader("Options Mode")
    options_mode = st.checkbox("Use daily options mode", value=True)
    premium_entry = st.number_input("Option entry premium ($)", min_value=0.0, value=0.0, step=0.05)
    stop_loss_pct = st.slider("Premium stop loss %", 10, 40, 20, 1)
    take_profit_1_pct = st.slider("Premium take profit 1 %", 15, 80, 35, 1)
    take_profit_2_pct = st.slider("Premium take profit 2 %", 25, 150, 60, 1)
    min_rr = st.slider("Minimum reward/risk", 1.0, 3.0, 1.5, 0.1)

    st.divider()
    st.subheader("Backtest")
    backtest_bars = st.slider("Bars to backtest", 120, 2000, 500, 20)

    st.divider()
    show_ai = st.checkbox("Enable AI panel", value=False)

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

refresh_ms = 30000 if timeframe != "1 Day" else 60000
if AUTOREFRESH_AVAILABLE:
    st_autorefresh(interval=refresh_ms, key=f"autorefresh_{timeframe}")

try:
    cfg = TF_MAP[timeframe]
    entry_raw = yf.Ticker(symbol).history(period=cfg["period"], interval=cfg["interval"], auto_adjust=False)
    hourly_raw = yf.Ticker(symbol).history(period="60d", interval="1h", auto_adjust=False)
    daily_raw = yf.Ticker(symbol).history(period="2y", interval="1d", auto_adjust=False)
    vix_raw = yf.Ticker("^VIX").history(period="6mo", interval="1d", auto_adjust=False)

    if timeframe == "1 Day":
        entry_raw = add_live_daily_bar(entry_raw, symbol)
        daily_raw = add_live_daily_bar(daily_raw, symbol)

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

    options_plan = build_options_plan(
        signal=sig["signal"],
        premium_entry=float(premium_entry),
        stop_pct=float(stop_loss_pct),
        tp1_pct=float(take_profit_1_pct),
        tp2_pct=float(take_profit_2_pct),
        min_rr=float(min_rr),
    ) if options_mode else {
        "enabled": False, "status": "Options mode is off.",
        "premium_stop": None, "premium_tp1": None, "premium_tp2": None,
        "rr1": None, "rr2": None, "trade_ok": None,
    }

    display_signal = sig["signal"]
    if options_mode and options_plan["enabled"] and options_plan["trade_ok"] is False:
        display_signal = "NO TRADE"

    st.subheader(f"Signal: :{signal_color(display_signal)}[{display_signal}]")

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
    d3.metric("Underlying Stop", fmt_price(safe_round(sig["stop"], 2)))
    d4.metric("Underlying Target", fmt_price(safe_round(sig["target"], 2)))

    if options_mode:
        st.subheader("🎯 Daily Options Plan")
        o1, o2, o3, o4, o5 = st.columns(5)
        o1.metric("Entry Premium", fmt_price(premium_entry) if premium_entry > 0 else "N/A")
        o2.metric("Premium Stop", fmt_price(options_plan["premium_stop"]))
        o3.metric("TP1", fmt_price(options_plan["premium_tp1"]))
        o4.metric("TP2", fmt_price(options_plan["premium_tp2"]))
        o5.metric("R/R TP1", "N/A" if options_plan["rr1"] is None else f"{options_plan['rr1']:.2f}")

        if options_plan["enabled"]:
            if options_plan["trade_ok"] is False:
                st.warning(options_plan["status"])
            elif options_plan["trade_ok"] is True:
                st.success(options_plan["status"])
            else:
                st.info(options_plan["status"])

    tab1, tab2, tab3 = st.tabs(["Chart", "Backtest", "Alerts"])

    with tab1:
        if PLOTLY_AVAILABLE:
            fig = make_candlestick_chart(entry_df, symbol, timeframe)
            st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
            st.caption("Mouse wheel zooms. Mouse drag pans left and right. Double-click resets.")
        else:
            fallback_cols = ["Close", "EMA_8", "EMA_21", "EMA_50"]
            if "EMA_200" in entry_df.columns and entry_df["EMA_200"].notna().sum() > 0:
                fallback_cols.append("EMA_200")
            if "VWAP" in entry_df.columns and entry_df["VWAP"].notna().sum() > 0:
                fallback_cols.append("VWAP")
            st.line_chart(entry_df[fallback_cols].tail(TF_MAP[timeframe]["chart_bars"]))

        if show_ai:
            st.subheader("🤖 AI Technical Verdict")
            api_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
            if api_key and GENAI_AVAILABLE:
                if st.button("Run Deep Analysis"):
                    prompt = f"""
You are a disciplined institutional market strategist.

Analyze this setup in 5 short bullet points:
1. Market regime
2. Bull case
3. Bear case
4. Best action now
5. Key invalidation

Ticker: {symbol}
Timeframe: {timeframe}
Price: {curr_price}
Signal: {display_signal}
Confidence: {sig['confidence']}%
Score: {sig['score']}
Risk: {sig['risk']}
Regime: {sig['regime']}
VWAP: {safe_round(last.get('VWAP', np.nan), 2)}
RSI: {safe_round(last['RSI'], 2)}
ATR: {safe_round(last['ATR'], 2)}
EMA21: {safe_round(last['EMA_21'], 2)}
EMA50: {safe_round(last['EMA_50'], 2)}
VIX: {vix_value}
Underlying stop: {safe_round(sig['stop'], 2)}
Underlying target: {safe_round(sig['target'], 2)}
"""
                    with st.spinner("Running AI analysis..."):
                        client = genai.Client(api_key=api_key)
                        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                        st.info(response.text)
            else:
                st.caption("To enable AI, install `google-genai` and add `GEMINI_API_KEY` or `GOOGLE_API_KEY` to Streamlit secrets.")

    with tab2:
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

            if PLOTLY_AVAILABLE:
                st.plotly_chart(make_backtest_chart(bt_df), use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
            else:
                st.line_chart(bt_df[["equity_curve", "buy_hold_curve"]])

    with tab3:
        st.write("Refresh or change timeframe to update the latest signal view.")

except Exception as e:
    st.error(f"Error: {e}")
