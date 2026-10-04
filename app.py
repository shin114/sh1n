import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import requests
import time

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="SH1N",
    page_icon="⚡",
    layout="wide"
)

# --- 1. Exchange & API Initialization ---
@st.cache_resource
def get_exchange():
    return ccxt.bybit({'enableRateLimit': True})

exchange = get_exchange()
exchange = ccxt.bybit({
    'hostname': 'bytick.com',  # Redirects requests from api.bybit.com -> api.bytick.com
    'enableRateLimit': True,
})

# Test market fetching
markets = exchange.load_markets()
print(f"Successfully loaded {len(markets)} markets.")

# --- 2. Data Fetching Utilities ---
@st.cache_data(ttl=3600)
def fetch_all_usdt_markets():
    """Fetch all active Bybit USDT Spot and Perpetual markets."""
    try:
        markets = exchange.load_markets()
        market_list = []
        for symbol, market in markets.items():
            if market.get('active') and market.get('quote') == 'USDT':
                m_type = 'Spot' if market.get('spot') else ('Perp' if market.get('swap') else None)
                if m_type:
                    market_list.append({
                        'symbol': symbol,
                        'base': market['base'],
                        'type': m_type,
                        'label': f"{symbol} ({m_type})"
                    })
        return market_list
    except Exception as e:
        st.error(f"Error loading Bybit markets: {e}")
        return []

@st.cache_data(ttl=300)
def fetch_ohlcv_data(symbol, timeframe='1d', limit=150):
    """Fetch candlestick OHLCV data."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Error fetching OHLCV data for {symbol}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_derivatives_data(symbol):
    """Fetch Funding Rate & Open Interest if the market is a Perpetual/Swap."""
    if ":" not in symbol:  # CCXT swap symbols contain ':'
        return {"funding_rate_pct": 0.0, "open_interest": 0.0, "is_perp": False}
    try:
        funding_info = exchange.fetch_funding_rate(symbol)
        funding_rate = funding_info.get('fundingRate', 0.0) * 100
        oi_info = exchange.fetch_open_interest(symbol)
        open_interest = oi_info.get('openInterestAmount', 0.0)
        return {
            "funding_rate_pct": funding_rate,
            "open_interest": open_interest,
            "is_perp": True
        }
    except Exception:
        return {"funding_rate_pct": 0.0, "open_interest": 0.0, "is_perp": False}

@st.cache_data(ttl=3600)
def fetch_fear_and_greed():
    """Fetch global Crypto Fear & Greed Index from Alternative.me."""
    try:
        res = requests.get("https://api.alternative.me/fng/", timeout=5).json()
        data = res['data'][0]
        return {
            "value": int(data['value']),
            "classification": data['value_classification']
        }
    except Exception:
        return {"value": 50, "classification": "Neutral"}

# --- 3. Indicator & Technical Analysis Engine ---
def calculate_technical_indicators(df):
    """Calculate EMAs, RSI, MACD, ATR, OBV, and Volume SMA."""
    df = df.copy()
    
    # EMAs
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

    # RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # ATR (14)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()

    # Volume Indicators
    df['vol_sma_20'] = df['volume'].rolling(window=20).mean()
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    return df

def calculate_fibonacci_levels(df, lookback=50):
    """Calculate swing highs, lows, and Fibonacci retracements over lookback period."""
    recent_df = df.tail(lookback)
    high_val = recent_df['high'].max()
    low_val = recent_df['low'].min()
    diff = high_val - low_val if high_val != low_val else 1.0

    fibs = {
        '1.000 (Swing Low)': low_val,
        '0.786': high_val - (diff * 0.786),
        '0.618 (Golden Pocket)': high_val - (diff * 0.618),
        '0.500 (Mid Point)': high_val - (diff * 0.500),
        '0.382': high_val - (diff * 0.382),
        '0.236': high_val - (diff * 0.236),
        '0.000 (Swing High)': high_val,
    }
    return fibs, high_val, low_val

# --- 4. Weighted Confluence Scoring Engine ---
def evaluate_confluence(df, fibs, derivatives, fng):
    """
    Evaluates 6 market layers and returns a normalized score (-100 to +100).
    Positive = Bullish, Negative = Bearish.
    """
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price = latest['close']

    breakdown = {}

    # 1. Market Structure & Trend Alignment (Max 30 pts)
    trend_score = 0
    if price > latest['ema_20']: trend_score += 10
    if latest['ema_20'] > latest['ema_50']: trend_score += 10
    if latest['ema_50'] > latest['ema_200']: trend_score += 10
    breakdown['Trend & Structure'] = {
        'score': trend_score, 'max': 30,
        'detail': f"Price vs EMAs (20/50/200). Current: {'Bullish Stack' if trend_score == 30 else 'Mixed/Bearish'}"
    }

    # 2. Momentum & Oscillators (Max 20 pts)
    mom_score = 0
    rsi = latest['rsi']
    if rsi < 35: mom_score += 10       # Oversold (Bullish setup)
    elif 45 <= rsi <= 60: mom_score += 5 # Healthy trend momentum
    elif rsi > 70: mom_score -= 10     # Overbought warning

    if latest['macd_hist'] > 0 and latest['macd_hist'] > prev['macd_hist']:
        mom_score += 10  # Bullish MACD expansion
    elif latest['macd_hist'] < 0 and latest['macd_hist'] < prev['macd_hist']:
        mom_score -= 10  # Bearish MACD expansion
    breakdown['Momentum (RSI & MACD)'] = {
        'score': max(-20, min(20, mom_score)), 'max': 20,
        'detail': f"RSI: {rsi:.1f} | MACD Hist: {latest['macd_hist']:.4f}"
    }

    # 3. Volume & Order Flow (Max 15 pts)
    vol_score = 0
    if latest['volume'] > latest['vol_sma_20']: vol_score += 8
    if latest['obv'] > prev['obv']: vol_score += 7
    else: vol_score -= 7
    breakdown['Volume & Order Flow'] = {
        'score': max(-15, min(15, vol_score)), 'max': 15,
        'detail': f"Vol > 20-SMA: {latest['volume'] > latest['vol_sma_20']} | OBV Trending Up: {latest['obv'] > prev['obv']}"
    }

    # 4. Fibonacci & Key Levels (Max 15 pts)
    fib_score = 0
    golden_pocket = fibs['0.618 (Golden Pocket)']
    mid_point = fibs['0.500 (Mid Point)']
    dist_0618 = abs(price - golden_pocket) / price
    dist_0500 = abs(price - mid_point) / price

    if dist_0618 < 0.015 or dist_0500 < 0.015:
        fib_score = 15  # Sitting directly in Golden Pocket / Retracement zone
    elif dist_0618 < 0.03:
        fib_score = 10
    breakdown['Fibonacci & Key Levels'] = {
        'score': fib_score, 'max': 15,
        'detail': f"0.618 Level: {golden_pocket:.4f} (Distance: {dist_0618*100:.2f}%)"
    }

    # 5. Derivatives Positioning (Max 10 pts)
    deriv_score = 0
    if derivatives['is_perp']:
        fr = derivatives['funding_rate_pct']
        if fr < -0.01:
            deriv_score += 10  # Heavy shorting = Short squeeze potential
        elif -0.01 <= fr <= 0.015:
            deriv_score += 5   # Healthy / Balanced funding
        elif fr > 0.03:
            deriv_score -= 10  # Over-leveraged long = Risk of long liquidation
    else:
        deriv_score = 5  # Spot default
    breakdown['Derivatives Setup'] = {
        'score': max(-10, min(10, deriv_score)), 'max': 10,
        'detail': f"Funding Rate: {derivatives['funding_rate_pct']:.4f}%" if derivatives['is_perp'] else "Spot Pair (N/A)"
    }

    # 6. Market Sentiment (Max 10 pts)
    sent_score = 0
    fng_val = fng['value']
    if fng_val < 25:
        sent_score += 10  # Extreme Fear = Contrarian buying opportunity
    elif 25 <= fng_val <= 55:
        sent_score += 5   # Neutral
    elif fng_val > 75:
        sent_score -= 10  # Extreme Greed = High risk of market pullback
    breakdown['Market Sentiment'] = {
        'score': max(-10, min(10, sent_score)), 'max': 10,
        'detail': f"Fear & Greed Index: {fng_val} ({fng['classification']})"
    }

    # Total Net Score Calculation (-100 to +100)
    total_score = sum(item['score'] for item in breakdown.values())

    return total_score, breakdown

# --- 5. Streamlit User Interface ---
st.title("⚡ SH1N Crypto Swing Analysis Engine")
st.caption("Integrated Price Action, Volume, Technicals, Bybit Derivatives, and Macro Sentiment")

# Sidebar Controls
all_markets = fetch_all_usdt_markets()

search_query = st.sidebar.text_input("Search Ticker / Pair:", value="BTC").strip().upper()
timeframe = st.sidebar.selectbox("Select Timeframe:", options=['15m', '1h', '4h', '1d'], index=3)
lookback_period = st.sidebar.slider("Fibonacci Lookback Bars:", min_value=20, max_value=200, value=50)

# Filter matching pairs
matching_markets = [
    m for m in all_markets 
    if search_query in m['base'] or search_query in m['symbol']
]

if matching_markets:
    selected_label = st.sidebar.selectbox("Select Available Market:", [m['label'] for m in matching_markets])
    selected_market = next(m for m in matching_markets if m['label'] == selected_label)
    symbol = selected_market['symbol']
    
    # Run Complete Pipeline
    with st.spinner(f"Analyzing {symbol}..."):
        df = fetch_ohlcv_data(symbol, timeframe=timeframe)
        derivatives = fetch_derivatives_data(symbol)
        fng = fetch_fear_and_greed()

    if not df.empty and len(df) >= 30:
        df = calculate_technical_indicators(df)
        fibs, swing_high, swing_low = calculate_fibonacci_levels(df, lookback=lookback_period)
        score, score_breakdown = evaluate_confluence(df, fibs, derivatives, fng)

        latest_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        price_change_pct = ((latest_price - prev_price) / prev_price) * 100

        # High Level Metric Header
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Current Price", f"${latest_price:,.4f}", f"{price_change_pct:+.2f}%")
        
        # Recommendation Gauge
        rec_label = "STRONG BUY" if score >= 60 else ("BUY" if score >= 25 else ("NEUTRAL" if score > -25 else ("SELL" if score >= -60 else "STRONG SELL")))
        rec_color = "normal" if score == 0 else ("inverse" if score < 0 else "off")
        col2.metric("Confluence Score", f"{score} / 100", delta=rec_label)
        
        col3.metric("Fear & Greed", f"{fng['value']}/100", fng['classification'])
        col4.metric("Funding Rate", f"{derivatives['funding_rate_pct']:.4f}%" if derivatives['is_perp'] else "N/A (Spot)")
        col5.metric("24h ATR (Volatility)", f"${df['atr'].iloc[-1]:,.4f}")

        st.divider()

        # Detailed Breakdown Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Confluence Breakdown", "📈 Technicals & Fibonacci", "⚡ Derivatives & Sentiment", "📋 Raw Data"])

        with tab1:
            st.subheader("Confluence Layer Breakdown")
            st.progress(max(0, min(100, int((score + 100) / 2))))  # Scale -100..100 to 0..100 for progress bar
            
            for category, data in score_breakdown.items():
                c1, c2, c3 = st.columns([2, 1, 4])
                c1.write(f"**{category}**")
                c2.write(f"`{data['score']} / {data['max']} pts`")
                c3.caption(data['detail'])

        with tab2:
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("Auto-Fibonacci Retracement Levels")
                fib_df = pd.DataFrame(list(fibs.items()), columns=['Fib Level', 'Price Level'])
                fib_df['Distance to Price'] = fib_df['Price Level'].apply(lambda x: f"{((latest_price - x)/x)*100:+.2f}%")
                fib_df['Price Level'] = fib_df['Price Level'].apply(lambda x: f"${x:,.4f}")
                st.table(fib_df)

            with col_b:
                st.subheader("Technical Indicator Values")
                latest = df.iloc[-1]
                tech_data = {
                    "Metric": ["EMA 20", "EMA 50", "EMA 200", "RSI (14)", "MACD Histogram", "20-Period Vol SMA"],
                    "Value": [
                        f"${latest['ema_20']:,.4f}",
                        f"${latest['ema_50']:,.4f}",
                        f"${latest['ema_200']:,.4f}",
                        f"{latest['rsi']:.2f}",
                        f"{latest['macd_hist']:.4f}",
                        f"{latest['vol_sma_20']:,.2f}"
                    ]
                }
                st.table(pd.DataFrame(tech_data))

        with tab3:
            st.subheader("Derivatives & Macro Metrics")
            d1, d2 = st.columns(2)
            with d1:
                st.markdown("##### Bybit Perpetual Data")
                st.write(f"**Market Type:** {selected_market['type']}")
                st.write(f"**Funding Rate:** `{derivatives['funding_rate_pct']:.4f}%`")
                st.write(f"**Open Interest:** `{derivatives['open_interest']:,.2f}`")
                st.caption("High positive funding (>0.03%) indicates crowded long positions (liquidation risk). Negative funding indicates short squeeze potential.")
            
            with d2:
                st.markdown("##### Sentiment Index")
                st.write(f"**Crypto Fear & Greed Value:** `{fng['value']}`")
                st.write(f"**Market State:** `{fng['classification']}`")
                st.caption("Extreme Fear (<25) often marks local bottoms, while Extreme Greed (>75) signals heightened risk of top reversals.")

        with tab4:
            st.subheader("Recent OHLCV Data")
            st.dataframe(df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'rsi', 'ema_20', 'ema_50']].tail(20), use_container_width=True)

    else:
        st.warning("Insufficient OHLCV data returned for this asset/timeframe combination.")
else:
    st.warning(f"No active USDT Spot or Perpetual pairs found matching '{search_query}'.")
