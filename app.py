import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import requests
import time

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="SH1N Spot Dip Catcher",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Developer Mode & Branding Visibility ---
# Append ?dev=true to your app URL to reveal Streamlit header & dev tools
is_dev_mode = st.query_params.get("dev", ["false"])[0].lower() == "true" if isinstance(st.query_params.get("dev"), list) else st.query_params.get("dev", "false").lower() == "true"

if not is_dev_mode:
    st.markdown("""
        <style>
        /* 1. Keep top header transparent without collapsing its flexbox container */
        header[data-testid="stHeader"], [data-testid="stHeader"] {
            background: transparent !important;
        }

        /* 2. Hide ONLY top-right menu actions, status widget, decoration, and footer */
        [data-testid="stHeaderActionElements"],
        [data-testid="stToolbarActionElements"],
        [data-testid="stStatusWidget"],
        [data-testid="stDecoration"],
        #MainMenu,
        footer {
            visibility: hidden !important;
            display: none !important;
        }

        /* 3. Preserve sidebar toggle arrow (expand/collapse controls) */
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="stSidebarCollapseButton"] {
            visibility: visible !important;
            display: flex !important;
            opacity: 1 !important;
            z-index: 999999 !important;
        }

        /* Ensure the collapsed icon button retains pointer interactions */
        [data-testid="stSidebarCollapsedControl"] button {
            visibility: visible !important;
            display: inline-flex !important;
        }
        </style>
    """, unsafe_allow_html=True)

# --- Best Practice Constants ---
DEFAULT_FIB_LOOKBACK = 100

# --- Helper Functions: Dynamic Precision Formatting ---
def format_price(val):
    """
    Dynamically format prices based on order of magnitude.
    Uses escaped dollar signs (\$) to prevent Streamlit from treating
    price pairs as LaTeX math formulas.
    """
    if val is None or np.isnan(val):
        return r"\$0.00"
    abs_val = abs(val)
    if abs_val == 0:
        return r"\$0.00"
    elif abs_val >= 100:
        return f"\\${val:,.2f}"
    elif abs_val >= 1:
        return f"\\${val:,.4f}"
    elif abs_val >= 0.001:
        return f"\\${val:,.6f}"
    else:
        return f"\\${val:,.8f}"

def format_decimal(val, default_dp=4):
    """Dynamically format technical indicators (like MACD) without zero-clipping."""
    if val is None or np.isnan(val):
        return "0.00"
    abs_val = abs(val)
    if abs_val == 0:
        return "0.00"
    elif abs_val >= 1:
        return f"{val:,.{default_dp}f}"
    elif abs_val >= 0.001:
        return f"{val:,.6f}"
    else:
        return f"{val:,.8f}"


# --- Responsive CSS & Typography Styling ---
st.markdown("""
    <style>
    /* 1. Constrain container width for optimal portrait readability */
    .main .block-container {
        max-width: 680px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    /* 2. Fluid Headers */
    h1 { font-size: clamp(1.4rem, 5vw, 2.1rem) !important; font-weight: 700 !important; }
    h2 { font-size: clamp(1.2rem, 4vw, 1.6rem) !important; font-weight: 600 !important; }
    h3 { font-size: clamp(1.05rem, 3.2vw, 1.3rem) !important; font-weight: 600 !important; }

    /* 3. Metric Value Adjustments - Full visibility without clipping */
    [data-testid="stMetricValue"] {
        font-size: clamp(0.95rem, 3.6vw, 1.30rem) !important;
        font-weight: 700 !important;
        white-space: normal !important;
        word-break: break-word !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: clamp(0.75rem, 2.6vw, 0.88rem) !important;
        font-weight: 600 !important;
        color: #64748b !important;
    }

    /* 4. Column Padding & Responsive Spacing */
    @media (max-width: 640px) {
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
        }
        [data-testid="stColumn"] {
            width: 48% !important;
            min-width: 48% !important;
            flex: 1 1 48% !important;
            margin-bottom: 0.5rem !important;
        }
    }

    /* 5. Safe Text Wrapping */
    p, li, div, span {
        overflow-wrap: break-word !important;
        word-break: break-word !important;
    }

    /* 6. Mobile-friendly Tab Scroll */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        overflow-x: auto;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: clamp(0.75rem, 2.5vw, 0.88rem) !important;
        padding: 6px 10px !important;
    }

    /* 7. Selected Market Callout Banner Styling */
    .selected-market-banner {
        background-color: #1e293b;
        border-left: 4px solid #10b981;
        padding: 8px 12px;
        border-radius: 6px;
        margin-top: 8px;
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)


# --- 1. Exchange Initialization ---
exchange = ccxt.bitget({
    'enableRateLimit': True,
})

# --- 2. Data Fetching Utilities ---
@st.cache_data(ttl=3600)
def fetch_all_usdt_markets():
    """Fetch active Bitget USDT Spot markets."""
    try:
        markets = exchange.load_markets()
        market_list = []
        for symbol, market in markets.items():
            if market.get('active', True) and market.get('quote') == 'USDT' and market.get('spot'):
                market_list.append({
                    'symbol': symbol,
                    'base': market['base'],
                    'type': 'Spot',
                    'label': f"{symbol} (Spot)"
                })
        return market_list
    except Exception as e:
        st.error(f"Error loading Bitget spot markets: {e}")
        return []

@st.cache_data(ttl=300)
def fetch_ohlcv_data(symbol, timeframe='4h', limit=150):
    """Fetch candlestick OHLCV data."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Error fetching OHLCV data for {symbol}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_fear_and_greed():
    """Fetch global Crypto Fear & Greed Index."""
    try:
        res = requests.get("https://api.alternative.me/fng/", timeout=5).json()
        data = res['data'][0]
        return {
            "value": int(data['value']),
            "classification": data['value_classification']
        }
    except Exception:
        return {"value": 50, "classification": "Neutral"}

# --- 3. Technical Indicator Engine ---
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

def calculate_fibonacci_levels(df, lookback=DEFAULT_FIB_LOOKBACK):
    """Calculate swing highs, lows, and Fibonacci retracements."""
    recent_df = df.tail(lookback)
    high_val = recent_df['high'].max()
    low_val = recent_df['low'].min()
    diff = high_val - low_val if high_val != low_val else (high_val * 0.01 if high_val != 0 else 1e-8)

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

def fetch_recent_pivots(df):
    """
    Identifies recent major local swing highs (Resistance) and swing lows (Support)
    using standard 5-candle Williams Fractal patterns.
    """
    if len(df) < 5:
        return [], []
        
    highs = df['high']
    lows = df['low']
    
    is_pivot_high = (
        (highs > highs.shift(1)) & (highs > highs.shift(2)) &
        (highs > highs.shift(-1)) & (highs > highs.shift(-2))
    )
    
    is_pivot_low = (
        (lows < lows.shift(1)) & (lows < lows.shift(2)) &
        (lows < lows.shift(-1)) & (lows < lows.shift(-2))
    )
    
    recent_res = df[is_pivot_high]['high'].dropna().tail(3).tolist()
    recent_sup = df[is_pivot_low]['low'].dropna().tail(3).tolist()
    
    return recent_res, recent_sup

# --- 4. Spot Dip Scoring Engine (0 to 100) ---
def evaluate_dip_confluence(df, fibs, fng):
    """
    Evaluates Spot dip-buying quality on a 0-100 scale:
    1. Trend Health (EMAs) - avoids buying into broken downtrends
    2. Pullback / Dip Momentum (RSI & MACD) - rewards cooled RSI & turning MACD
    3. Volume & Accumulation (Volume SMA & OBV) - checks real demand/absorption
    4. Support & Resistance Confluence (Fibs + Williams Fractals)
    5. Market Sentiment (Fear & Greed) - favors fear/pullbacks over greed spikes
    """
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price = latest['close']

    breakdown = {}

    # 1. Trend & Structure Health (Max 25 pts)
    trend_score = 0
    trend_notes = []
    if len(df) >= 200 and not np.isnan(latest['ema_200']):
        if price >= latest['ema_200']:
            trend_score += 10
            trend_notes.append("Above 200 EMA")
        else:
            trend_notes.append("Below 200 EMA")
    if latest['ema_20'] >= latest['ema_50']:
        trend_score += 8
        trend_notes.append("20>50 EMA")
    if price >= latest['ema_50'] * 0.97:
        trend_score += 7
        trend_notes.append("Holding near/above 50 EMA")

    breakdown['Trend Health'] = {
        'score': trend_score, 'max': 25,
        'detail': " | ".join(trend_notes) if trend_notes else "Evaluating trend"
    }

    # 2. Dip Cool-Off & Reversal Momentum (Max 25 pts)
    mom_score = 0
    rsi = latest['rsi']
    if 28 <= rsi <= 45:
        mom_score += 15
        rsi_state = f"Dip Zone (RSI {rsi:.1f})"
    elif rsi < 28:
        mom_score += 12
        rsi_state = f"Deep Oversold (RSI {rsi:.1f})"
    elif 45 < rsi <= 55:
        mom_score += 8
        rsi_state = f"Mild Pullback (RSI {rsi:.1f})"
    elif rsi > 70:
        mom_score -= 5
        rsi_state = f"Overbought/Extended (RSI {rsi:.1f})"
    else:
        rsi_state = f"Mid RSI ({rsi:.1f})"

    if latest['macd_hist'] > prev['macd_hist']:
        mom_score += 10
        macd_state = "MACD Hist Rising"
    elif latest['macd_hist'] > 0:
        mom_score += 5
        macd_state = "MACD Hist Positive"
    else:
        macd_state = "MACD Hist Falling"

    breakdown['Dip Momentum'] = {
        'score': max(0, min(25, mom_score)), 'max': 25,
        'detail': f"{rsi_state} | {macd_state}"
    }

    # 3. Volume & Order Flow Absorption (Max 15 pts)
    vol_score = 0
    vol_spike = latest['volume'] > latest['vol_sma_20']
    obv_rising = latest['obv'] > prev['obv']
    if vol_spike and obv_rising:
        vol_score += 10
    elif vol_spike or obv_rising:
        vol_score += 6
    if latest['obv'] >= df['obv'].tail(5).min():
        vol_score += 5

    breakdown['Volume & OBV'] = {
        'score': min(15, vol_score), 'max': 15,
        'detail': f"Vol>20SMA: {vol_spike} | OBV Up: {obv_rising}"
    }

    # 4. Support Proximity & Fib/Pivot Confluence (Max 20 pts)
    sr_score = 0
    sr_notes = []
    recent_res, recent_sup = fetch_recent_pivots(df)
    golden_pocket = fibs['0.618 (Golden Pocket)']
    mid_point = fibs['0.500 (Mid Point)']
    fib_0786 = fibs['0.786']

    support_candidates = [s for s in (recent_sup + [golden_pocket, mid_point, fib_0786]) if s <= price * 1.005]
    closest_sup = max(support_candidates) if support_candidates else None
    dist_to_sup = ((price - closest_sup) / price) if (closest_sup and closest_sup > 0) else 1.0

    if dist_to_sup <= 0.015:
        sr_score += 15
        sr_notes.append(f"At Support ({dist_to_sup*100:.2f}%)")
    elif dist_to_sup <= 0.035:
        sr_score += 10
        sr_notes.append(f"Near Support ({dist_to_sup*100:.2f}%)")
    else:
        sr_notes.append(f"Support {dist_to_sup*100:.2f}% below")

    # Overlap bonus between Fibonacci pullback levels and Williams Fractal pivots
    has_overlap = False
    for fib_lvl in [golden_pocket, mid_point, fib_0786]:
        for piv in (recent_sup + recent_res):
            if piv > 0 and abs(fib_lvl - piv) / fib_lvl <= 0.015:
                has_overlap = True
                break
    if has_overlap:
        sr_score += 5
        sr_notes.append("⭐ Fib+Pivot Confluence")

    breakdown['Support Confluence'] = {
        'score': min(20, sr_score), 'max': 20,
        'detail': " | ".join(sr_notes)
    }

    # 5. Market Sentiment / Fear (Max 15 pts)
    sent_score = 0
    fng_val = fng['value']
    if fng_val < 30:
        sent_score += 15
    elif fng_val <= 50:
        sent_score += 10
    elif fng_val <= 65:
        sent_score += 5
    else:
        sent_score += 2

    breakdown['Market Fear'] = {
        'score': sent_score, 'max': 15,
        'detail': f"Fear & Greed: {fng_val} ({fng['classification']})"
    }

    total_score = sum(item['score'] for item in breakdown.values())
    return max(0, min(100, int(total_score))), breakdown

# --- 5. Spot Dip & Resistance Sell Target Generator ---
def generate_spot_dip_plan(df, fibs, score):
    """
    Builds a Spot dip-buying zone and overhead resistance sell targets (TP1, TP2)
    plus a structural invalidation level (SL).
    """
    latest = df.iloc[-1]
    price = latest['close']
    atr = latest['atr'] if not np.isnan(latest['atr']) and latest['atr'] > 0 else price * 0.02

    recent_res, recent_sup = fetch_recent_pivots(df)
    golden_pocket = fibs['0.618 (Golden Pocket)']
    mid_point = fibs['0.500 (Mid Point)']
    swing_high = fibs['0.000 (Swing High)']
    fib_0382 = fibs['0.382']
    fib_0236 = fibs['0.236']

    # Support candidates below or slightly above current price for dip entry
    supports_below = sorted([s for s in (recent_sup + [fibs['0.786'], golden_pocket, mid_point]) if s <= price * 1.003])
    key_support = supports_below[-1] if supports_below else max(price - (1.5 * atr), price * 0.95)

    # Overhead resistance candidates above current price for selling into strength
    resistances_above = sorted([r for r in (recent_res + [fib_0382, fib_0236, swing_high]) if r > price * 1.005])
    tp1 = resistances_above[0] if resistances_above else price + (2.0 * atr)
    if len(resistances_above) > 1:
        tp2 = resistances_above[1]
    else:
        tp2 = max(swing_high, tp1 * 1.04) if swing_high > tp1 * 1.01 else tp1 * 1.05

    # Ensure TP2 is strictly above TP1
    if tp2 <= tp1:
        tp2 = tp1 * 1.04

    # Structural invalidation below nearest support with ATR cushion
    sl = min(key_support - (1.2 * atr), price * 0.90) if key_support < price else price - (2.0 * atr)
    if sl >= price:
        sl = price * 0.95

    if score >= 60:
        action = "BUY DIP 🟢"
        confidence = "High Dip Setup (~72%-82%)"
        entry_min = min(key_support, price * 0.992)
        entry_max = price
        rationale = [
            f"**Dip Quality:** Score is **{score}/100**, showing solid dip alignment across trend, pullback momentum, and support.",
            f"**Dip Entry Zone:** Accumulate spot bids between **{format_price(entry_min)}** and **{format_price(entry_max)}**.",
            f"**Resistance Sell Plan:** Trim/sell into overhead resistance candidates at **{format_price(tp1)}** (TP1) and **{format_price(tp2)}** (TP2); invalidation below **{format_price(sl)}**."
        ]
    elif score >= 40:
        action = "SCALE / DCA DIP 🟡"
        confidence = "Moderate Dip Setup (~58%-70%)"
        entry_min = min(key_support, price * 0.985)
        entry_max = price * 0.998
        rationale = [
            f"**Partial Alignment:** Score is **{score}/100**—decent pullback structure, best suited for smaller staggered spot buys.",
            f"**Staggered Bids:** Look for pullbacks toward **{format_price(entry_min)} - {format_price(entry_max)}** rather than chasing green candles.",
            f"**Sell Targets:** Overhead resistance candidates sit near **{format_price(tp1)}** and **{format_price(tp2)}** (invalidation: **{format_price(sl)}**)."
        ]
    else:
        action = "WAIT FOR DIP ⏸️"
        confidence = "Low Edge / Wait (~50%)"
        entry_min = min(key_support, golden_pocket) if min(key_support, golden_pocket) < price else price * 0.97
        entry_max = key_support if key_support < price else price * 0.985
        rationale = [
            f"**Not Ready Yet:** Score is **{score}/100**—either price is too extended/overbought or lacks clean support confluence.",
            f"**Patient Bid Zone:** Wait for price to dip closer to **{format_price(entry_min)} - {format_price(entry_max)}** instead of buying right now.",
            f"**Overhead Targets (Once Filled):** Resistance candidates are around **{format_price(tp1)}** and **{format_price(tp2)}**."
        ]

    dip_depth_pct = abs(((entry_min - price) / price) * 100) if price > 0 else 0
    tp1_pct = abs(((tp1 - price) / price) * 100) if price > 0 else 0
    tp2_pct = abs(((tp2 - price) / price) * 100) if price > 0 else 0
    sl_pct = abs(((price - sl) / price) * 100) if price > 0 else 0

    return {
        "action": action,
        "confidence": confidence,
        "entry_range": f"{format_price(entry_min)} - {format_price(entry_max)}",
        "entry_min": entry_min,
        "entry_max": entry_max,
        "dip_depth_pct": dip_depth_pct,
        "sl": sl,
        "sl_pct": sl_pct,
        "tp1": tp1,
        "tp1_pct": tp1_pct,
        "tp2": tp2,
        "tp2_pct": tp2_pct,
        "rationale": rationale
    }

# --- 6. User Interface & State Initialization ---

# Initialize Session State
if 'recent_markets' not in st.session_state:
    st.session_state['recent_markets'] = []

if 'search_query_val' not in st.session_state:
    st.session_state['search_query_val'] = "GRT"

# Main Title & Description
st.title("🎯 SH1N Spot Dip Catcher")
st.caption("Spot Accumulation Engine: Catch Dips into Support & Sell into Resistance")

# Fetch Markets
all_markets = fetch_all_usdt_markets()

# Sidebar Market Selection Inputs
st.sidebar.header("Spot Market Selection")
search_query = st.sidebar.text_input(
    "Search Ticker / Pair:", 
    value=st.session_state.get('search_query_val', 'GRT')
).strip().upper()

st.session_state['search_query_val'] = search_query

timeframe = st.sidebar.selectbox(
    "Select Timeframe:", 
    options=['15m', '1h', '4h', '1d'], 
    index=2
)

# Filter matching markets (prefix match)
matching_markets = [
    m for m in all_markets 
    if m['base'].startswith(search_query) or m['symbol'].startswith(search_query)
]

if matching_markets:
    labels = [m['label'] for m in matching_markets]
    
    # Selected label from dropdown
    selected_label = st.sidebar.selectbox("Select Available Market:", labels)
    pending_market = next(m for m in matching_markets if m['label'] == selected_label)

    # Manual Trigger Button to run analysis
    analyze_click = st.sidebar.button("🚀 Analyze Dip Setup", type="primary", use_container_width=True)

    # Initial boot default
    if 'active_market' not in st.session_state:
        st.session_state['active_market'] = pending_market
        st.session_state['recent_markets'] = [pending_market]

    # Trigger analysis ONLY on explicit button click
    if analyze_click:
        st.session_state['active_market'] = pending_market
        
        # Add to recent markets (deduplicated, max 5)
        recent = [m for m in st.session_state['recent_markets'] if m['symbol'] != pending_market['symbol']]
        recent.insert(0, pending_market)
        st.session_state['recent_markets'] = recent[:5]

    active_market = st.session_state['active_market']

    # --- Selected Market Banner ---
    st.markdown(
        f"""
        <div class="selected-market-banner">
            <span style="color: #94a3b8; font-size: 0.82rem; font-weight: 600; text-transform: uppercase;">Currently Analyzing (Spot)</span><br/>
            <span style="font-size: 1.15rem; font-weight: 700; color: #f8fafc;">📍 {active_market['label']}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Recent Selections Quick-Switch Pills ---
    recent_list = st.session_state['recent_markets']
    if len(recent_list) > 1:
        st.markdown("**🕒 Recent Selections:**")
        rec_cols = st.columns(len(recent_list))
        for idx, r_m in enumerate(recent_list):
            is_active = (r_m['symbol'] == active_market['symbol'])
            btn_label = f"● {r_m['base']}" if is_active else r_m['base']
            
            # Clicking a recent pill triggers immediate switch
            if rec_cols[idx].button(
                btn_label, 
                key=f"rec_btn_{r_m['symbol']}_{idx}", 
                disabled=is_active, 
                use_container_width=True
            ):
                st.session_state['active_market'] = r_m
                st.session_state['search_query_val'] = r_m['base']
                st.rerun()

    st.write("")  # Spacing divider

    # --- Fetch Data & Process Analysis ---
    symbol = active_market['symbol']
    with st.spinner(f"Scanning spot dip setup for {symbol}..."):
        df = fetch_ohlcv_data(symbol, timeframe=timeframe)
        fng = fetch_fear_and_greed()

    if not df.empty and len(df) >= 30:
        df = calculate_technical_indicators(df)
        fibs, swing_high, swing_low = calculate_fibonacci_levels(df, lookback=DEFAULT_FIB_LOOKBACK)
        score, score_breakdown = evaluate_dip_confluence(df, fibs, fng)
        dip_plan = generate_spot_dip_plan(df, fibs, score)

        latest_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        price_change_pct = ((latest_price - prev_price) / prev_price) * 100

        # --- HIGH-LEVEL OVERVIEW ---
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.metric("Current Price", format_price(latest_price), f"{price_change_pct:+.2f}%")
        with row1_col2:
            st.metric("Spot Action", dip_plan['action'])

        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            st.metric("Setup Read", dip_plan['confidence'])
        with row2_col2:
            st.metric(
                "Dip Setup Score", 
                f"{score} / 100", 
                help="Higher scores (60+) mean stronger dip alignment near support with cooling momentum. Below 40 means wait for a better dip."
            )

        st.divider()

        # --- SPOT DIP & RESISTANCE EXECUTION SUMMARY CARD ---
        st.subheader("🎯 Spot Dip & Sell Plan")
        
        if "BUY DIP" in dip_plan['action']:
            st.success("🟢 **PRIME SPOT DIP SETUP**")
        elif "SCALE" in dip_plan['action']:
            st.warning("🟡 **MODERATE PULLBACK / SCALE-IN ZONE**")
        else:
            st.info("⏸️ **WAIT FOR DEEPER DIP**")
        
        with st.container():
            p_col1, p_col2 = st.columns(2)
            p_col1.metric("1. Dip Buy Zone", dip_plan['entry_range'], f"~{dip_plan['dip_depth_pct']:.2f}% lower bound")
            p_col2.metric("2. Invalidation Level", format_price(dip_plan['sl']), f"-{dip_plan['sl_pct']:.2f}%")
            
            p_col3, p_col4 = st.columns(2)
            p_col3.metric("3. Resistance Sell TP1", format_price(dip_plan['tp1']), f"+{dip_plan['tp1_pct']:.2f}%")
            p_col4.metric("4. Resistance Sell TP2", format_price(dip_plan['tp2']), f"+{dip_plan['tp2_pct']:.2f}%")

        st.divider()

        # --- DETAILED TABBED VIEWS ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Dip Strategy", 
            "📊 Score Breakdown", 
            "📈 Support & Resistance", 
            "📋 Raw Data"
        ])

        with tab1:
            st.subheader("💡 Why this Spot Plan?")
            for item in dip_plan['rationale']:
                st.markdown(f"- {item}")
                
            st.info(
                "🛡 **Spot Patience Rule:** "
                "Spot trading works best when you buy pullbacks into support and sell into resistance—not green spikes. "
                f"Only consider invalidation if a **{timeframe.upper()} candle closes below {format_price(dip_plan['sl'])}**."
            )

            st.divider()
            st.subheader("🧮 Spot Position & Target Calculator")
            st.caption("Estimate token quantity and net spot gain at TP1 / TP2 after round-trip trading fees.")

            calc_c1, calc_c2 = st.columns(2)
            with calc_c1:
                alloc_usdt = st.number_input(
                    "Allocation (USDT)",
                    min_value=1.0,
                    value=100.0,
                    step=10.0,
                    key="spot_alloc_usdt"
                )
            with calc_c2:
                fee_pct = st.number_input(
                    "Round-Trip Fee (%)",
                    min_value=0.0,
                    max_value=5.0,
                    value=0.20,
                    step=0.02,
                    key="spot_round_trip_fee",
                    help="Combined buy + sell spot fee percentage (e.g., 0.20% = 0.10% maker/taker each side)."
                )

            ref_entry = dip_plan['entry_max'] if dip_plan['entry_max'] > 0 else latest_price
            est_qty = alloc_usdt / ref_entry if ref_entry > 0 else 0.0
            fee_drag_usdt = alloc_usdt * (fee_pct / 100.0)

            tp1_gross_pct = ((dip_plan['tp1'] - ref_entry) / ref_entry) * 100.0 if ref_entry > 0 else 0.0
            tp2_gross_pct = ((dip_plan['tp2'] - ref_entry) / ref_entry) * 100.0 if ref_entry > 0 else 0.0
            sl_gross_pct = ((dip_plan['sl'] - ref_entry) / ref_entry) * 100.0 if ref_entry > 0 else 0.0

            tp1_net_usdt = (est_qty * (dip_plan['tp1'] - ref_entry)) - fee_drag_usdt
            tp2_net_usdt = (est_qty * (dip_plan['tp2'] - ref_entry)) - fee_drag_usdt
            sl_net_usdt = (est_qty * (dip_plan['sl'] - ref_entry)) - fee_drag_usdt

            tp1_net_pct = (tp1_net_usdt / alloc_usdt) * 100.0 if alloc_usdt > 0 else 0.0
            tp2_net_pct = (tp2_net_usdt / alloc_usdt) * 100.0 if alloc_usdt > 0 else 0.0
            sl_net_pct = (sl_net_usdt / alloc_usdt) * 100.0 if alloc_usdt > 0 else 0.0

            m_c1, m_c2, m_c3 = st.columns(3)
            m_c1.metric(f"Est. {active_market['base']} Qty", f"{est_qty:,.4f}", f"@ {format_price(ref_entry)}")
            m_c2.metric("Net Profit @ TP1", f"{tp1_net_usdt:+,.2f} USDT", f"{tp1_net_pct:+.2f}% net ({tp1_gross_pct:+.2f}% gross)")
            m_c3.metric("Net Profit @ TP2", f"{tp2_net_usdt:+,.2f} USDT", f"{tp2_net_pct:+.2f}% net ({tp2_gross_pct:+.2f}% gross)")

            st.caption(
                f"Est. Invalidation impact @ {format_price(dip_plan['sl'])}: "
                f"**{sl_net_usdt:+,.2f} USDT** ({sl_net_pct:+.2f}% net / {sl_gross_pct:+.2f}% gross) · "
                f"Est. round-trip fee drag: **{fee_drag_usdt:.2f} USDT**."
            )

        with tab2:
            st.subheader("5-Layer Spot Dip Score Breakdown")
            st.caption(
                "Scores **0 to 100** based on Trend Health, Dip Momentum (RSI/MACD), Volume/OBV absorption, "
                "Fibonacci + Williams Fractal support confluence, and Market Fear."
            )
            st.progress(max(0, min(100, score)))
            
            for category, data in score_breakdown.items():
                c1, c2, c3 = st.columns([2, 1, 3])
                c1.write(f"**{category}**")
                c2.write(f"`{data['score']}/{data['max']} pts`")
                c3.caption(data['detail'])

        with tab3:
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("Fibonacci Levels")
                fib_df = pd.DataFrame(list(fibs.items()), columns=['Fib Level', 'Price Level'])
                fib_df['Distance'] = fib_df['Price Level'].apply(lambda x: f"{((latest_price - x)/x)*100:+.2f}%" if x else "0.00%")
                fib_df['Price Level'] = fib_df['Price Level'].apply(format_price)
                st.table(fib_df)

            with col_b:
                st.subheader("Fractal S/R Candidates")
                recent_res, recent_sup = fetch_recent_pivots(df)
                
                pivot_data = []
                if recent_res:
                    for res in reversed(recent_res):
                        pct_move = ((res - latest_price) / latest_price) * 100
                        pivot_data.append({
                            "Candidate": f"Resistance ({pct_move:+.1f}%)",
                            "Level": format_price(res)
                        })
                if recent_sup:
                    for sup in reversed(recent_sup):
                        pct_move = ((sup - latest_price) / latest_price) * 100
                        pivot_data.append({
                            "Candidate": f"Support ({pct_move:+.1f}%)",
                            "Level": format_price(sup)
                        })
                
                if pivot_data:
                    st.table(pd.DataFrame(pivot_data))
                else:
                    st.info("No fractal pivots detected in recent candles.")

            st.subheader("Key Indicators")
            latest = df.iloc[-1]
            tech_data = {
                "Metric": ["EMA 20", "EMA 50", "EMA 200", "RSI (14)", "MACD Hist", "20-SMA Vol", "Fear & Greed"],
                "Value": [
                    format_price(latest['ema_20']),
                    format_price(latest['ema_50']),
                    format_price(latest['ema_200']) if len(df) >= 200 else "N/A (<200 bars)",
                    f"{latest['rsi']:.2f}",
                    format_decimal(latest['macd_hist']),
                    f"{latest['vol_sma_20']:,.2f}",
                    f"{fng['value']} ({fng['classification']})"
                ]
            }
            st.table(pd.DataFrame(tech_data))

        with tab4:
            st.subheader("Recent Market Data")
            st.dataframe(
                df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'rsi', 'ema_20', 'ema_50']].tail(20), 
                use_container_width=True
            )

    else:
        st.warning("Insufficient historical data returned for this spot asset/timeframe combination.")
else:
    st.warning(f"No active Spot USDT pairs found starting with '{search_query}'.")
