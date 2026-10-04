import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
from scipy.signal import find_peaks

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="Crypto Swing Assistant",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Crypto Swing Trading Signal Dashboard")
st.caption("Noise-filtered swing entry, ATR support/resistance, and dynamic stop-loss suggestions.")

# --- SIDEBAR: USER INPUTS ---
st.sidebar.header("⚙️ Settings")

default_pairs = ["GRT/USDT", "OP/USDT", "REZ/USDT", "PAXG/USDT", "BTC/USDT", "ETH/USDT"]
custom_pair = st.sidebar.text_input("Add Custom Pair (e.g. SOL/USDT)", "").upper().strip()

if custom_pair and custom_pair not in default_pairs:
    default_pairs.append(custom_pair)

selected_symbol = st.sidebar.selectbox("Select Trading Pair", default_pairs, index=0)
timeframe = st.sidebar.selectbox("Swing Timeframe", ["4h", "1d", "1h"], index=0)
atr_multiplier = st.sidebar.slider("ATR Stop Buffer Multiplier", 1.0, 3.0, 1.5, 0.1)

# --- DATA FETCHING (CACHE FOR EFFICIENCY) ---
@st.cache_data(ttl=180)  # Refresh market data every 3 minutes
def fetch_binance_data(symbol: str, timeframe: str, limit: int = 250):
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Error fetching data from Binance: {e}")
        return None

# --- TECHNICAL ANALYSIS ENGINE ---
def analyze_market(df: pd.DataFrame, atr_mult: float):
    df['ema_20'] = ta.ema(df['close'], length=20)
    df['ema_50'] = ta.ema(df['close'], length=50)
    df['ema_200'] = ta.ema(df['close'], length=200)
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    curr_close = df['close'].iloc[-1]
    curr_rsi = df['rsi'].iloc[-1]
    curr_atr = df['atr'].iloc[-1]
    ema20 = df['ema_20'].iloc[-1]
    ema50 = df['ema_50'].iloc[-1]
    
    if curr_close > ema20 > ema50 and curr_rsi > 50:
        regime = "Bullish Trend 🚀"
        regime_desc = "Price is above key EMAs with positive momentum. Favorable for swing longs."
        status_color = "green"
    elif curr_close < ema20 < ema50 and curr_rsi < 50:
        regime = "Bearish Trend 🔻"
        regime_desc = "Price is below key EMAs with downward pressure. High risk for long entries."
        status_color = "red"
    else:
        regime = "Consolidation / Stagnant ⚖️"
        regime_desc = "Price is moving sideways in a range. Look for support bounces."
        status_color = "orange"
        
    highs = df['high'].values
    lows = -df['low'].values
    res_idx, _ = find_peaks(highs, distance=10)
    sup_idx, _ = find_peaks(lows, distance=10)
    
    supports = sorted([df['low'].iloc[i] for i in sup_idx if df['low'].iloc[i] < curr_close])
    resistances = sorted([df['high'].iloc[i] for i in res_idx if df['high'].iloc[i] > curr_close])
    
    key_support = supports[-1] if supports else df['low'].tail(20).min()
    key_resistance = resistances[0] if resistances else df['high'].tail(20).max()
    
    suggested_sl = key_support - (curr_atr * atr_mult)
    risk_per_unit = curr_close - suggested_sl
    
    tp1 = curr_close + (risk_per_unit * 1.5)
    tp2 = min(key_resistance, curr_close + (risk_per_unit * 2.5))
    
    return {
        "price": curr_close,
        "rsi": curr_rsi,
        "atr": curr_atr,
        "regime": regime,
        "regime_desc": regime_desc,
        "status_color": status_color,
        "support": key_support,
        "resistance": key_resistance,
        "suggested_sl": suggested_sl,
        "tp1": tp1,
        "tp2": tp2,
        "risk_pct": ((curr_close - suggested_sl) / curr_close) * 100
    }

# --- MAIN DISPLAY LOGIC ---
df = fetch_binance_data(selected_symbol, timeframe)

if df is not None:
    data = analyze_market(df, atr_multiplier)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${data['price']:.4f}")
    col2.metric("RSI (14)", f"{data['rsi']:.1f}")
    col3.metric("Key Support", f"${data['support']:.4f}")
    col4.metric("Key Resistance", f"${data['resistance']:.4f}")
    
    st.markdown("---")
    
    st.subheader("📊 Market Regime & Condition")
    if data['status_color'] == "green":
        st.success(f"**{data['regime']}**: {data['regime_desc']}")
    elif data['status_color'] == "red":
        st.error(f"**{data['regime']}**: {data['regime_desc']}")
    else:
        st.warning(f"**{data['regime']}**: {data['regime_desc']}")
        
    st.markdown("---")
    
    st.subheader("🎯 Suggested Trade Structure (Manual Execution)")
    
    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
    t_col1.metric("Suggested Entry Zone", f"${data['support']:.4f} - ${data['price']:.4f}")
    t_col2.metric("Take Profit 1 (Scale Out 50%)", f"${data['tp1']:.4f}", "+1.5 R:R")
    t_col3.metric("Take Profit 2 (Final Target)", f"${data['tp2']:.4f}", "+2.5 R:R")
    t_col4.metric("Hard Stop Loss", f"${data['suggested_sl']:.4f}", f"-{data['risk_pct']:.2f}% Risk")
    
    st.info(
        "⚠️ **Wick Protection Rule:** Do not exit on an intraday spike touch alone. "
        f"Only execute a stop-loss exit if a **{timeframe.upper()} candle closes below ${data['suggested_sl']:.4f}**."
    )
    
    st.subheader("📉 Price & Indicator Trend")
    st.line_chart(df.set_index('timestamp')[['close', 'ema_20', 'ema_50']])