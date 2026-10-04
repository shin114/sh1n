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
    layout="centered"  # Enforces portrait proportions on desktop
)

# --- Developer Mode & Branding Visibility ---
# Append ?dev=true to your app URL to reveal Streamlit header & dev tools
is_dev_mode = st.query_params.get("dev", ["false"])[0].lower() == "true" if isinstance(st.query_params.get("dev"), list) else st.query_params.get("dev", "false").lower() == "true"

if not is_dev_mode:
    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {background: transparent !important;}
        div[data-testid="stToolbar"] {visibility: hidden !important;}
        div[data-testid="stDecoration"] {visibility: hidden !important;}
        div[data-testid="stStatusWidget"] {visibility: hidden !important;}
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
        border-left: 4px solid #3b82f6;
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
    """Fetch all active Bitget USDT Spot and Perpetual markets."""
    try:
        markets = exchange.load_markets()
        market_list = []
        for symbol, market in markets.items():
            if market.get('active', True) and market.get('quote') == 'USDT':
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
        st.error(f"Error loading Bitget markets: {e}")
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

@st.cache_data(ttl=300)
def fetch_derivatives_data(symbol):
    """Fetch Funding Rate & Open Interest if market is Perpetual/Swap."""
    if ":" not in symbol:  # CCXT swap symbols contain ':'
        return {"funding_rate_pct": 0.0, "open_interest": 0.0, "is_perp": False}
    try:
        funding_info = exchange.fetch_funding_rate(symbol)
        funding_rate = funding_info.get('fundingRate', 0.0) * 100
        oi_info = exchange.fetch_open_interest(symbol)
        open_interest = oi_info.get('openInterestAmount') or oi_info.get('openInterestValue') or oi_info.get('openInterest') or 0.0
        return {
            "funding_rate_pct": funding_rate,
            "open_interest": open_interest,
            "is_perp": True
        }
    except Exception:
        return {"funding_rate_pct": 0.0, "open_interest": 0.0, "is_perp": False}

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

# --- 4. Setup Scoring Engine ---
def evaluate_confluence(df, fibs, derivatives, fng):
    """Evaluates 6 key market indicators and compiles a single 'Setup Strength Score' out of 100."""
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price = latest['close']

    breakdown = {}

    # 1. Market Structure & Trend Alignment
    trend_score = 0
    if price > latest['ema_20']: trend_score += 10
    if latest['ema_20'] > latest['ema_50']: trend_score += 10
    if latest['ema_50'] > latest['ema_200']: trend_score += 10
    breakdown['Trend Alignment'] = {
        'score': trend_score, 'max': 30,
        'detail': f"Price vs EMAs (20/50/200): {'Bullish Stack' if trend_score == 30 else 'Mixed/Bearish'}"
    }

    # 2. Momentum & Oscillators
    mom_score = 0
    rsi = latest['rsi']
    if rsi < 35: mom_score += 10       
    elif 45 <= rsi <= 60: mom_score += 5 
    elif rsi > 70: mom_score -= 10     

    if latest['macd_hist'] > 0 and latest['macd_hist'] > prev['macd_hist']:
        mom_score += 10  
    elif latest['macd_hist'] < 0 and latest['macd_hist'] < prev['macd_hist']:
        mom_score -= 10  
    breakdown['Momentum (RSI & MACD)'] = {
        'score': max(-20, min(20, mom_score)), 'max': 20,
        'detail': f"RSI: {rsi:.1f} | MACD Hist: {format_decimal(latest['macd_hist'])}"
    }

    # 3. Volume & Order Flow
    vol_score = 0
    if latest['volume'] > latest['vol_sma_20']: vol_score += 8
    if latest['obv'] > prev['obv']: vol_score += 7
    else: vol_score -= 7
    breakdown['Volume Strength'] = {
        'score': max(-15, min(15, vol_score)), 'max': 15,
        'detail': f"Above 20-SMA Vol: {latest['volume'] > latest['vol_sma_20']} | OBV Up: {latest['obv'] > prev['obv']}"
    }

    # 4. Directional S/R & Confluence (Fibonacci + Williams Fractal Pivots)
    sr_score = 0
    detail_notes = []

    recent_res, recent_sup = fetch_recent_pivots(df)
    golden_pocket = fibs['0.618 (Golden Pocket)']
    mid_point = fibs['0.500 (Mid Point)']

    # Identify closest support below or at price
    valid_supports = [s for s in (recent_sup + [golden_pocket, mid_point]) if s <= price * 1.005]
    closest_sup = max(valid_supports) if valid_supports else None

    # Identify closest resistance above or at price
    valid_resistances = [r for r in (recent_res + [fibs['0.000 (Swing High)']]) if r >= price * 0.995]
    closest_res = min(valid_resistances) if valid_resistances else None

    # Calculate proximity percentages
    dist_to_sup = ((price - closest_sup) / price) if closest_sup else 1.0
    dist_to_res = ((closest_res - price) / price) if closest_res else 1.0

    # Bullish Support Reaction Points
    if dist_to_sup <= 0.015:
        sr_score += 10
        detail_notes.append("Holding Key Support (<=1.5%)")
    elif dist_to_sup <= 0.03:
        sr_score += 5
        detail_notes.append("Near Support Zone (<=3%)")

    # Bearish Resistance Penalty (buying into overhead resistance)
    if dist_to_res <= 0.015:
        sr_score -= 10
        detail_notes.append("Testing Heavy Resistance (<=1.5%)")
    elif dist_to_res <= 0.03:
        sr_score -= 5
        detail_notes.append("Approaching Resistance (<=3%)")

    # Confluence Bonus: Check if Fibonacci Golden/Mid level overlaps with a Williams Fractal Pivot
    has_confluence = False
    for fib_lvl in [golden_pocket, mid_point]:
        for piv in (recent_sup + recent_res):
            if abs(fib_lvl - piv) / fib_lvl <= 0.015:  # Within 1.5% overlap
                has_confluence = True
                break

    if has_confluence:
        sr_score += 5
        detail_notes.append("⭐ Golden Fib + Pivot Overlap")

    final_sr_score = max(-15, min(15, sr_score))
    sr_detail_text = " | ".join(detail_notes) if detail_notes else "Mid-range trading (No immediate S/R bounce)"

    breakdown['Key S/R & Confluence'] = {
        'score': final_sr_score, 'max': 15,
        'detail': sr_detail_text
    }

    # 5. Derivatives Setup
    deriv_score = 0
    if derivatives['is_perp']:
        fr = derivatives['funding_rate_pct']
        if fr < -0.01:
            deriv_score += 10  
        elif -0.01 <= fr <= 0.015:
            deriv_score += 5   
        elif fr > 0.03:
            deriv_score -= 10  
    else:
        deriv_score = 5  
    breakdown['Derivatives Data'] = {
        'score': max(-10, min(10, deriv_score)), 'max': 10,
        'detail': f"Funding Rate: {derivatives['funding_rate_pct']:.4f}%" if derivatives['is_perp'] else "Spot Market"
    }

    # 6. Overall Market Sentiment
    sent_score = 0
    fng_val = fng['value']
    if fng_val < 25:
        sent_score += 10  
    elif 25 <= fng_val <= 55:
        sent_score += 5   
    elif fng_val > 75:
        sent_score -= 10  
    breakdown['Market Sentiment'] = {
        'score': max(-10, min(10, sent_score)), 'max': 10,
        'detail': f"Fear & Greed Index: {fng_val} ({fng['classification']})"
    }

    total_score = sum(item['score'] for item in breakdown.values())
    return total_score, breakdown

# --- 5. Simplified Trade Plan Deductor ---
def generate_plain_english_trade_plan(df, fibs, score):
    """
    Translates technical indicators into a simplified recommendation
    with estimated win probability, entry range, SL, TP1, and TP2.
    """
    latest = df.iloc[-1]
    price = latest['close']
    atr = latest['atr'] if not np.isnan(latest['atr']) else price * 0.02

    golden_pocket = fibs['0.618 (Golden Pocket)']
    swing_high = fibs['0.000 (Swing High)']
    swing_low = fibs['1.000 (Swing Low)']

    # Case 1: BULLISH BUY SIGNAL (Score >= 25)
    if score >= 25:
        action = "BUY / LONG 🟢"
        
        if score >= 65:
            probability = "High (~75%-82%)"
        elif score >= 45:
            probability = "Mod-High (~65%-74%)"
        else:
            probability = "Moderate (~55%-64%)"

        support_levels = [v for k, v in fibs.items() if v <= price]
        key_support = max(support_levels) if support_levels else price - (atr * 1.5)
        
        entry_min = min(key_support, price * 0.992)
        entry_max = price
        
        sl = key_support - (1.5 * atr)
        risk_per_unit = price - sl
        if risk_per_unit <= 0:
            risk_per_unit = atr * 1.5
            sl = price - risk_per_unit

        tp1 = price + (risk_per_unit * 1.5)
        tp2 = max(swing_high, price + (risk_per_unit * 2.5))

        rationale = [
            f"**Trend Control:** High setup score (+{score}/100) indicates buyers dominate this market.",
            f"**Entry Strategy:** Buy between **{format_price(entry_min)}** and **{format_price(entry_max)}** on minor dips.",
            f"**Risk Management:** Cut losses if a candle closes below **{format_price(sl)}** (Risk: -{((price-sl)/price)*100:.2f}%)."
        ]

    # Case 2: BEARISH SELL SIGNAL (Score <= -25)
    elif score <= -25:
        action = "SELL / SHORT 🔴"
        
        if score <= -65:
            probability = "High (~75%-82%)"
        elif score <= -45:
            probability = "Mod-High (~65%-74%)"
        else:
            probability = "Moderate (~55%-64%)"

        resistance_levels = [v for k, v in fibs.items() if v >= price]
        key_resistance = min(resistance_levels) if resistance_levels else price + (atr * 1.5)
        
        entry_min = price
        entry_max = max(key_resistance, price * 1.008)
        
        sl = key_resistance + (1.5 * atr)
        risk_per_unit = sl - price
        if risk_per_unit <= 0:
            risk_per_unit = atr * 1.5
            sl = price + risk_per_unit

        tp1 = price - (risk_per_unit * 1.5)
        tp2 = min(swing_low, price - (risk_per_unit * 2.5))

        rationale = [
            f"**Trend Control:** Low setup score ({score}/100) indicates sellers are pushing price lower.",
            f"**Entry Strategy:** Sell or short relief bounces between **{format_price(entry_min)}** and **{format_price(entry_max)}**.",
            f"**Risk Management:** Cut losses if a candle closes above **{format_price(sl)}** (Risk: -{((sl-price)/price)*100:.2f}%)."
        ]

    # Case 3: NEUTRAL / RANGEBOUND (Score between -25 and +25)
    else:
        action = "STAND BY ⏸️"
        probability = "Low Edge (~50%)"
        entry_min, entry_max = price, price
        sl, tp1, tp2 = price, price, price

        rationale = [
            f"**Market Indecision:** Setup score is neutral ({score}/100). Technical indicators conflict.",
            "**Strategy:** No high-probability entry right now. Cash is a valid position.",
            f"**Action Plan:** Wait for price to pull back to key Fib levels ({format_price(golden_pocket)}) or break out."
        ]

    return {
        "action": action,
        "probability": probability,
        "entry_range": f"{format_price(entry_min)} - {format_price(entry_max)}",
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "risk_pct": abs(((price - sl) / price) * 100) if sl != price and price > 0 else 0,
        "tp1_pct": abs(((tp1 - price) / price) * 100) if tp1 != price and price > 0 else 0,
        "tp2_pct": abs(((tp2 - price) / price) * 100) if tp2 != price and price > 0 else 0,
        "rationale": rationale
    }

# --- 6. User Interface & State Initialization ---

# Initialize Session State
if 'recent_markets' not in st.session_state:
    st.session_state['recent_markets'] = []

if 'search_query_val' not in st.session_state:
    st.session_state['search_query_val'] = "GRT"

# Main Title & Description
st.title("⚡ SH1N Swing Analysis Engine")
st.caption("Simplified Trading Signals Driven by Multi-Layer Market Analysis")

# Fetch Markets
all_markets = fetch_all_usdt_markets()

# Sidebar Market Selection Inputs
st.sidebar.header("Market Selection")
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

# Filter matching markets
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
    analyze_click = st.sidebar.button("🚀 Analyze Market", type="primary", use_container_width=True)

    # Initial boot default
    if 'active_market' not in st.session_state:
        st.session_state['active_market'] = pending_market
        # Store in recent on initial boot
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
            <span style="color: #94a3b8; font-size: 0.82rem; font-weight: 600; text-transform: uppercase;">Currently Analyzing</span><br/>
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
    with st.spinner(f"Evaluating market probability for {symbol}..."):
        df = fetch_ohlcv_data(symbol, timeframe=timeframe)
        derivatives = fetch_derivatives_data(symbol)
        fng = fetch_fear_and_greed()

    if not df.empty and len(df) >= 30:
        df = calculate_technical_indicators(df)
        fibs, swing_high, swing_low = calculate_fibonacci_levels(df, lookback=DEFAULT_FIB_LOOKBACK)
        score, score_breakdown = evaluate_confluence(df, fibs, derivatives, fng)
        trade_plan = generate_plain_english_trade_plan(df, fibs, score)

        latest_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        price_change_pct = ((latest_price - prev_price) / prev_price) * 100

        # --- HIGH-LEVEL OVERVIEW ---
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.metric("Current Price", format_price(latest_price), f"{price_change_pct:+.2f}%")
        with row1_col2:
            st.metric("Market Action", trade_plan['action'])

        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            st.metric("Est. Win Rate", trade_plan['probability'])
        with row2_col2:
            st.metric(
                "Setup Strength Score", 
                f"{score} / 100", 
                help="Scores above +25 suggest strong buying setups. Scores below -25 suggest selling setups."
            )

        st.divider()

        # --- SIMPLE EXECUTION SUMMARY CARD ---
        st.subheader("🎯 Execution Summary")
        
        if "BUY" in trade_plan['action']:
            st.success("🟢 **BULLISH SETUP DETECTED**")
        elif "SELL" in trade_plan['action']:
            st.error("🔴 **BEARISH SETUP DETECTED**")
        else:
            st.warning("⏸️ **NEUTRAL MARKET / STAND BY**")
        
        # Bordered Execution Box
        with st.container():
            p_col1, p_col2 = st.columns(2)
            p_col1.metric("1. Entry Zone", trade_plan['entry_range'])
            p_col2.metric("2. Hard Stop Loss", format_price(trade_plan['sl']), f"-{trade_plan['risk_pct']:.2f}%")
            
            p_col3, p_col4 = st.columns(2)
            p_col3.metric("3. Take Profit 1", format_price(trade_plan['tp1']), f"+{trade_plan['tp1_pct']:.2f}%")
            p_col4.metric("4. Take Profit 2", format_price(trade_plan['tp2']), f"+{trade_plan['tp2_pct']:.2f}%")

        st.divider()

        # --- DETAILED TABBED VIEWS ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Strategy", 
            "📊 Setup Analysis", 
            "📈 Technicals", 
            "⚡ Derivatives", 
            "📋 Raw Data"
        ])

        with tab1:
            st.subheader("💡 Why is this recommended?")
            for item in trade_plan['rationale']:
                st.markdown(f"- {item}")
                
            st.info(
                "🛡 **Wick Protection Rule:** "
                "Avoid panicking on temporary price spikes. "
                f"Only close your trade if a **{timeframe.upper()} candle closes beyond {format_price(trade_plan['sl'])}**."
            )

        with tab2:
            st.subheader("6-Layer Setup Strength Breakdown")
            st.caption(
                "**What is Setup Strength?** Instead of relying on a single indicator, "
                "we analyze 6 separate market layers (Trend, Momentum, Volume, Directional S/R Confluence, "
                "Derivatives, and Sentiment). The higher the score out of 100, the more indicators agree on the move."
            )
            st.progress(max(0, min(100, int((score + 100) / 2))))
            
            for category, data in score_breakdown.items():
                c1, c2, c3 = st.columns([2, 1, 3])
                c1.write(f"**{category}**")
                c2.write(f"`{data['score']}/{data['max']} pts`")
                c3.caption(data['detail'])

        with tab3:
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("Fibonacci Retracements")
                fib_df = pd.DataFrame(list(fibs.items()), columns=['Fib Level', 'Price Level'])
                fib_df['Distance'] = fib_df['Price Level'].apply(lambda x: f"{((latest_price - x)/x)*100:+.2f}%")
                fib_df['Price Level'] = fib_df['Price Level'].apply(format_price)
                st.table(fib_df)

            with col_b:
                st.subheader("Historical Pivot S/R")
                recent_res, recent_sup = fetch_recent_pivots(df)
                
                pivot_data = []
                if recent_res:
                    for res in reversed(recent_res):
                        pivot_data.append({"Type": "Resistance 🔴", "Level": format_price(res)})
                if recent_sup:
                    for sup in reversed(recent_sup):
                        pivot_data.append({"Type": "Support 🟢", "Level": format_price(sup)})
                
                if pivot_data:
                    st.table(pd.DataFrame(pivot_data))
                else:
                    st.info("No fractal pivots detected in recent candles.")

        with tab4:
            st.subheader("Derivatives & Market Sentiment")
            d1, d2 = st.columns(2)
            with d1:
                st.markdown("##### Bitget Futures Data")
                st.write(f"**Market Type:** {active_market['type']}")
                st.write(f"**Funding Rate:** `{derivatives['funding_rate_pct']:.4f}%`")
                st.write(f"**Open Interest:** `{derivatives['open_interest']:,.2f}`")
            
            with d2:
                st.markdown("##### Global Sentiment")
                st.write(f"**Fear & Greed Index:** `{fng['value']}`")
                st.write(f"**Market State:** `{fng['classification']}`")

        with tab5:
            st.subheader("Recent Market Data")
            st.dataframe(
                df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'rsi', 'ema_20', 'ema_50']].tail(20), 
                use_container_width=True
            )

    else:
        st.warning("Insufficient historical data returned for this asset/timeframe combination.")
else:
    st.warning(f"No active USDT Spot or Perpetual pairs found starting with '{search_query}'.")
