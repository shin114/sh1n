import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import requests
import time
import json

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="SH1N Spot Dip Catcher v5",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Developer Mode & Branding Visibility ---
# Append ?dev=true to your app URL to reveal Streamlit header & dev tools
is_dev_mode = (
    st.query_params.get("dev", ["false"])[0].lower() == "true"
    if isinstance(st.query_params.get("dev"), list)
    else st.query_params.get("dev", "false").lower() == "true"
)

if not is_dev_mode:
    st.markdown("""
        <style>
        header[data-testid="stHeader"], [data-testid="stHeader"] {
            background: transparent !important;
        }
        [data-testid="stHeaderActionElements"],
        [data-testid="stToolbarActionElements"],
        [data-testid="stStatusWidget"],
        [data-testid="stDecoration"],
        [data-testid="stViewerBadge"],
        a[href*="github.com"],
        #MainMenu,
        footer {
            visibility: hidden !important;
            display: none !important;
        }
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="stSidebarCollapseButton"] {
            visibility: visible !important;
            display: flex !important;
            opacity: 1 !important;
            z-index: 999999 !important;
        }
        [data-testid="stSidebarCollapsedControl"] button {
            visibility: visible !important;
            display: inline-flex !important;
        }
        </style>
    """, unsafe_allow_html=True)

# --- Constants ---
DEFAULT_FIB_LOOKBACK = 100
MIN_BARS = 220
TIMEFRAME_MS = {
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}

# --- Helper Functions: Dynamic Precision Formatting ---
def format_price(val):
    """
    Dynamically format prices based on order of magnitude.
    Uses escaped dollar signs to prevent Streamlit from treating
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
    .main .block-container {
        max-width: 680px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    h1 { font-size: clamp(1.4rem, 5vw, 2.1rem) !important; font-weight: 700 !important; }
    h2 { font-size: clamp(1.2rem, 4vw, 1.6rem) !important; font-weight: 600 !important; }
    h3 { font-size: clamp(1.05rem, 3.2vw, 1.3rem) !important; font-weight: 600 !important; }

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

    p, li, div, span {
        overflow-wrap: break-word !important;
        word-break: break-word !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        overflow-x: auto;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: clamp(0.75rem, 2.5vw, 0.88rem) !important;
        padding: 6px 10px !important;
    }

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
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_usdt_markets():
    """Fetch active Bitget USDT Spot markets."""
    try:
        markets = exchange.load_markets()
        market_list = []
        for symbol, market in markets.items():
            if (
                market.get('active', True)
                and market.get('quote') == 'USDT'
                and market.get('spot')
                and market.get('base') != 'USDT'
            ):
                market_list.append({
                    'symbol': symbol,
                    'base': market['base'],
                    'type': 'Spot',
                    'label': f"{symbol} (Spot)"
                })
        return sorted(market_list, key=lambda x: x['symbol'])
    except Exception as e:
        st.error(f"Error loading Bitget spot markets: {e}")
        return []

@st.cache_data(ttl=60, show_spinner=False)
def fetch_spot_tickers_snapshot():
    """Fetch spot tickers for liquidity, spread, and volatility pre-filtering."""
    try:
        return exchange.fetch_tickers()
    except Exception:
        return {}

@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlcv_data(symbol, timeframe='15m', limit=260):
    """Fetch closed candlestick OHLCV data (excludes live unclosed candle)."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not ohlcv:
            return pd.DataFrame()
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

        now_ms = exchange.milliseconds()
        duration = TIMEFRAME_MS[timeframe]
        if len(df) and int(df['timestamp'].iloc[-1].timestamp() * 1000) + duration > now_ms:
            df = df.iloc[:-1].copy()

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna(subset=['open', 'high', 'low', 'close', 'volume']).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=900, show_spinner=False)
def fetch_fear_and_greed():
    """Fetch global Crypto Fear & Greed Index (context only)."""
    try:
        res = requests.get("https://api.alternative.me/fng/", timeout=5).json()
        data = res['data'][0]
        return {
            "value": int(data['value']),
            "classification": data['value_classification']
        }
    except Exception:
        return {"value": None, "classification": "Unavailable"}


# --- 3. Technical Indicator Engine (Wilder Smoothing) ---
def calculate_technical_indicators(df):
    """Calculate EMAs, Wilder RSI, MACD, Wilder ATR, OBV, and Relative Volume."""
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    for period in (20, 50, 200):
        df[f'ema_{period}'] = close.ewm(span=period, adjust=False, min_periods=period).mean()

    # Wilder RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    df.loc[(avg_loss == 0) & (avg_gain > 0), 'rsi'] = 100
    df.loc[(avg_gain == 0) & (avg_loss > 0), 'rsi'] = 0

    # MACD (12, 26, 9)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Wilder ATR (14)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    df['atr'] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    df['atr_pct'] = (df['atr'] / close) * 100

    # Volume & OBV
    df['vol_sma_20'] = volume.rolling(window=20, min_periods=20).mean()
    df['relative_volume'] = volume / df['vol_sma_20'].replace(0, np.nan)
    df['obv'] = (np.sign(close.diff()).fillna(0) * volume).cumsum()
    df['obv_ema_20'] = df['obv'].ewm(span=20, adjust=False).mean()

    # Candle Anatomy & Previous Values
    candle_range = (high - low).replace(0, np.nan)
    df['close_location'] = (close - low) / candle_range
    df['green'] = close > df['open']
    df['prior_high'] = high.shift(1)
    df['rsi_prev'] = df['rsi'].shift(1)
    df['macd_hist_prev'] = df['macd_hist'].shift(1)

    return df

def fetch_confirmed_pivots(df, right=2):
    """
    Identifies confirmed 5-candle Williams Fractal swing highs & lows.
    Enforces a `right`-bar confirmation delay so unconfirmed pivots are excluded.
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

    if len(df) > right:
        is_pivot_high.iloc[-right:] = False
        is_pivot_low.iloc[-right:] = False

    recent_res = df.loc[is_pivot_high, 'high'].dropna().tail(6).tolist()
    recent_sup = df.loc[is_pivot_low, 'low'].dropna().tail(6).tolist()
    return recent_res, recent_sup

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

def get_trend_state(df):
    """Determine trend state using EMA 50/200 alignment and slope."""
    if df is None or len(df) < 10:
        return "UNAVAILABLE", False
    last = df.iloc[-1]
    if pd.isna(last['ema_200']) or pd.isna(last['ema_50']):
        return "INSUFFICIENT", False

    bullish = (
        last['close'] > last['ema_200']
        and last['ema_50'] > last['ema_200']
        and last['ema_50'] >= df['ema_50'].iloc[-4]
    )
    bearish = (
        last['close'] < last['ema_200']
        and last['ema_50'] < last['ema_200']
        and last['ema_50'] < df['ema_50'].iloc[-4]
    )
    if bullish:
        return "BULLISH", True
    if bearish:
        return "BEARISH", False
    return "MIXED", False


# --- 4. Three-Layer Intraday Dip Engine (15m Primary + 4H/1H + Optional 5m) ---
def evaluate_intraday_dip_setup(
    m15_df, h1_df, h4_df, btc_df, fibs, fee_per_side_pct, m5_df=None
):
    """
    Three-Layer Scalp-to-Swing Dip Catcher:
    1. Mandatory Filters: Avoid broken 4H+1H downtrends or sharp BTC 4H selloffs.
    2. Entry Confirmation: Pullback near support/EMA + bullish price action & momentum recovery.
    3. Supporting Evidence: 15m RSI (35-50 recovery zone), MACD histogram, relative volume, OBV.
    """
    e = m15_df.iloc[-1]
    price = float(e['close'])
    atr = float(e['atr']) if pd.notna(e['atr']) and e['atr'] > 0 else price * 0.015
    atr = max(atr, price * 0.0015)

    trend4h, trend4h_ok = get_trend_state(h4_df)
    trend1h, trend1h_ok = get_trend_state(h1_df)
    btc_trend, btc_ok = get_trend_state(btc_df)

    recent_res, recent_sup = fetch_confirmed_pivots(m15_df)
    golden_pocket = fibs['0.618 (Golden Pocket)']
    mid_point = fibs['0.500 (Mid Point)']
    fib_0786 = fibs['0.786']

    # Classify levels strictly by their position relative to the current price.
    # A level above price is resistance, never support (and vice versa).
    support_candidates = [float(x) for x in recent_sup if pd.notna(x) and 0 < float(x) < price]
    for lvl in (golden_pocket, mid_point, fib_0786, e['ema_20'], e['ema_50']):
        if pd.notna(lvl) and 0 < float(lvl) < price:
            support_candidates.append(float(lvl))
    supports = sorted(set(support_candidates), reverse=True)

    resistance_candidates = [float(x) for x in recent_res if pd.notna(x) and float(x) > price]
    for lvl in (fibs['0.382'], fibs['0.236'], fibs['0.000 (Swing High)']):
        if pd.notna(lvl) and float(lvl) > price:
            resistance_candidates.append(float(lvl))
    resistances = sorted(set(resistance_candidates))

    nearest_support = supports[0] if supports else price - (1.5 * atr)
    nearest_resistance = resistances[0] if resistances else price + (1.8 * atr)
    support_dist_pct = max(0.0, ((price - nearest_support) / price) * 100)

    # Pullback context on 15m (not chasing green breakouts)
    near_ema20 = pd.notna(e['ema_20']) and abs(price - float(e['ema_20'])) / price <= 0.016
    near_ema50 = pd.notna(e['ema_50']) and abs(price - float(e['ema_50'])) / price <= 0.022
    near_support = support_dist_pct <= 2.0
    pullback_ok = near_ema20 or near_ema50 or near_support

    # Momentum & price action confirmation
    rsi_val = float(e['rsi']) if pd.notna(e['rsi']) else 50.0
    rsi_prev = float(e['rsi_prev']) if pd.notna(e['rsi_prev']) else rsi_val
    rsi_recovery_sweet = (35.0 <= rsi_val <= 52.0) and (rsi_val > rsi_prev)
    rsi_bounce_broader = (28.0 <= rsi_val <= 58.0) and (rsi_val > rsi_prev)

    bullish_reclaim = bool(e['green'] and price > float(e['prior_high']))
    strong_green_close = bool(e['green'] and pd.notna(e['close_location']) and float(e['close_location']) >= 0.60)
    macd_improving = (
        pd.notna(e['macd_hist'])
        and pd.notna(e['macd_hist_prev'])
        and float(e['macd_hist']) > float(e['macd_hist_prev'])
    )
    rel_vol = float(e['relative_volume']) if pd.notna(e['relative_volume']) else 1.0
    vol_ok = rel_vol >= 0.95
    obv_ok = bool(e['obv'] > e['obv_ema_20'])

    # Optional 5m fine-tuning check
    m5_note = "Off"
    m5_bonus = 0
    if m5_df is not None and len(m5_df) >= 20:
        m5_last = m5_df.iloc[-1]
        m5_green_bounce = bool(m5_last['green'] and pd.notna(m5_last['rsi']) and float(m5_last['rsi']) >= float(m5_df['rsi'].iloc[-2]))
        if m5_green_bounce:
            m5_note = "5m Bullish Bounce"
            m5_bonus = 4
        else:
            m5_note = "5m Cooling/Mixed"

    has_overlap = any(
        piv > 0 and abs(fib_lvl - piv) / fib_lvl <= 0.012
        for fib_lvl in (golden_pocket, mid_point, fib_0786)
        for piv in (recent_sup + recent_res)
    )

    breakdown = {
        "4H & 1H Trend Filter": {
            "score": (12 if trend4h_ok else (5 if trend4h == "MIXED" else 0)) + (10 if trend1h_ok else (4 if trend1h == "MIXED" else 0)),
            "max": 22,
            "detail": f"4H: {trend4h} | 1H: {trend1h}"
        },
        "15m Pullback & S/R": {
            "score": min(18, (14 if pullback_ok else 0) + (4 if has_overlap else 0)),
            "max": 18,
            "detail": f"Sup {support_dist_pct:.2f}% below" + (" | Fib+Pivot" if has_overlap else "")
        },
        "15m RSI Recovery (35-58)": {
            "score": 16 if rsi_recovery_sweet else (10 if rsi_bounce_broader else 0),
            "max": 16,
            "detail": f"RSI {rsi_val:.1f} (prev {rsi_prev:.1f})"
        },
        "15m Price & MACD Reversal": {
            "score": (11 if bullish_reclaim else (7 if strong_green_close else 0)) + (8 if macd_improving else 0),
            "max": 19,
            "detail": f"{'Reclaimed prior high' if bullish_reclaim else ('Green bounce' if strong_green_close else 'No reclaim')} | MACD {'up' if macd_improving else 'flat/down'}"
        },
        "Rel. Volume & OBV": {
            "score": (6 if vol_ok else 0) + (6 if obv_ok else 0),
            "max": 12,
            "detail": f"RelVol {rel_vol:.2f}x | OBV {'>' if obv_ok else '<='} EMA20"
        },
        "BTC 4H & 5m Timing": {
            "score": (5 if btc_ok else (2 if btc_trend == "MIXED" else 0)) + m5_bonus,
            "max": 9,
            "detail": f"BTC 4H: {btc_trend} | {m5_note}"
        },
    }

    raw_score = sum(item["score"] for item in breakdown.values())
    raw_score_max = sum(item["max"] for item in breakdown.values())
    score = int(round((raw_score / raw_score_max) * 100)) if raw_score_max else 0

    # Dynamic ATR + intraday swing stop & targets
    stop = min(nearest_support - (0.55 * atr), price - (1.1 * atr))
    if stop <= 0 or stop >= price:
        stop = price - (1.3 * atr)

    # Use the nearest overhead resistance for TP1; use the next level or ATR extension for TP2.
    tp1 = nearest_resistance
    tp2 = resistances[1] if len(resistances) > 1 else max(tp1 + (0.8 * atr), price + (2.9 * atr))
    if tp2 <= tp1:
        tp2 = tp1 + (0.8 * atr)

    fee = fee_per_side_pct / 100.0
    entry = price
    entry_cost_per_unit = entry * (1.0 + fee)
    stop_proceeds_per_unit = stop * (1.0 - fee)
    risk_per_unit_net = max(0.0, entry_cost_per_unit - stop_proceeds_per_unit)
    tp1_profit_per_unit_net = (tp1 * (1.0 - fee)) - entry_cost_per_unit
    tp2_profit_per_unit_net = (tp2 * (1.0 - fee)) - entry_cost_per_unit
    risk_pct = (risk_per_unit_net / entry_cost_per_unit) * 100.0 if entry_cost_per_unit > 0 else 0.0
    tp1_net_pct = (tp1_profit_per_unit_net / entry_cost_per_unit) * 100.0 if entry_cost_per_unit > 0 else 0.0
    tp2_net_pct = (tp2_profit_per_unit_net / entry_cost_per_unit) * 100.0 if entry_cost_per_unit > 0 else 0.0
    net_r_multiple = (tp1_profit_per_unit_net / risk_per_unit_net) if risk_per_unit_net > 0 else 0.0

    # Layer 1 (Mandatory) + Layer 2 (Confirmation) + Layer 3 (Supporting)
    broken_macro = (trend4h == "BEARISH" and trend1h == "BEARISH")
    btc_risk_off = (btc_trend == "BEARISH") and (score < 58)
    # A reclaim is useful, but still require at least one independent momentum recovery signal.
    momentum_recovering = rsi_bounce_broader or macd_improving
    entry_confirmed = pullback_ok and momentum_recovering and (
        bullish_reclaim or strong_green_close
    )

    if btc_risk_off:
        state = "Waiting"
        action = "WAIT — BTC 4H WEAK 🛑"
        reason = "BTC 4H is bearish and setup score is under 58; wait for stronger dip alignment."
    elif broken_macro:
        state = "Waiting"
        action = "WAIT — BEARISH HTF ⏸️"
        reason = "Both 4H and 1H are bearish; avoid catching an unconfirmed falling knife."
    elif entry_confirmed and score >= 60 and net_r_multiple >= 1.25:
        state = "Confirmed"
        action = "BUY DIP 🟢"
        reason = f"Confirmed 15m dip setup ({score}/100) with {net_r_multiple:.2f}R net upside to TP1."
    elif pullback_ok and score >= 43:
        state = "Watch"
        action = "WATCH / SCALE DIP 🟡"
        reason = "In pullback zone—watch for a bullish 15m close/reclaim or scale cautiously."
    else:
        state = "Waiting"
        action = "WAIT FOR DIP ⏸️"
        reason = "Awaiting cleaner pullback proximity or 15m momentum recovery."

    entry_min = min(nearest_support, price * 0.995) if nearest_support < price else price * 0.993
    entry_max = price

    return {
        "state": state,
        "score": score,
        "breakdown": breakdown,
        "action": action,
        "reason": reason,
        "price": price,
        "atr_pct": float(e['atr_pct']) if pd.notna(e['atr_pct']) else np.nan,
        "rsi": rsi_val,
        "trend4h": trend4h,
        "trend1h": trend1h,
        "btc_trend": btc_trend,
        "m5_note": m5_note,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "support_dist_pct": support_dist_pct,
        "entry_min": entry_min,
        "entry_max": entry_max,
        "entry_range": f"{format_price(entry_min)} - {format_price(entry_max)}",
        "stop": stop,
        "risk_pct": risk_pct,
        "tp1": tp1,
        "tp1_net_pct": tp1_net_pct,
        "tp2": tp2,
        "tp2_net_pct": tp2_net_pct,
        "net_r_multiple": net_r_multiple,
        "supports": supports[:6],
        "resistances": resistances[:6],
    }

@st.cache_data(ttl=120, show_spinner=False)
def run_market_dip_scanner(max_coins=25, min_quote_vol=500_000.0, max_spread_pct=0.35, fee_side_pct=0.10):
    """
    Scans top liquid Bitget USDT spot pairs for 15m dip setups.
    Filters out low liquidity, wide spreads, and extreme single-bar volatility spikes.
    """
    markets = fetch_all_usdt_markets()
    tickers = fetch_spot_tickers_snapshot()
    if not markets:
        return []

    candidates = []
    for m in markets:
        sym = m['symbol']
        t = tickers.get(sym, {}) if tickers else {}
        quote_vol = float(t.get('quoteVolume') or 0.0)
        bid = t.get('bid')
        ask = t.get('ask')
        last = t.get('last') or t.get('close')

        spread_pct = 0.0
        if bid and ask and bid > 0 and ask >= bid:
            spread_pct = ((ask - bid) / bid) * 100.0

        # Reject missing market-quality data instead of treating it as zero spread / valid liquidity.
        if not tickers or quote_vol <= 0 or quote_vol < min_quote_vol:
            continue
        if bid is None or ask is None or float(bid or 0) <= 0 or float(ask or 0) < float(bid or 0):
            continue
        if spread_pct <= 0 or spread_pct > max_spread_pct:
            continue

        candidates.append({
            "symbol": sym,
            "base": m['base'],
            "label": m['label'],
            "quote_vol": quote_vol,
            "spread_pct": spread_pct,
            "last": last,
        })

    # Prioritize highest 24h USDT volume pairs (always include BTC/USDT if present)
    candidates.sort(key=lambda x: x["quote_vol"], reverse=True)
    selected_batch = candidates[:max_coins]
    if not any(c["symbol"] == "BTC/USDT" for c in selected_batch):
        btc_match = next((c for c in candidates if c["symbol"] == "BTC/USDT"), None)
        if btc_match:
            selected_batch = [btc_match] + selected_batch[:max(0, max_coins - 1)]

    raw_btc = fetch_ohlcv_data("BTC/USDT", timeframe="4h", limit=240)
    btc_df = calculate_technical_indicators(raw_btc) if len(raw_btc) >= MIN_BARS else None

    scan_results = []
    for item in selected_batch:
        sym = item["symbol"]
        m15_raw = fetch_ohlcv_data(sym, timeframe="15m", limit=240)
        if len(m15_raw) < MIN_BARS:
            continue
        h1_raw = fetch_ohlcv_data(sym, timeframe="1h", limit=240)
        h4_raw = fetch_ohlcv_data(sym, timeframe="4h", limit=240)
        if len(h1_raw) < MIN_BARS or len(h4_raw) < MIN_BARS:
            continue

        m15_df = calculate_technical_indicators(m15_raw)
        # Exclude extreme volatility spikes (> 8% 15m ATR)
        if pd.notna(m15_df['atr_pct'].iloc[-1]) and float(m15_df['atr_pct'].iloc[-1]) > 8.0:
            continue

        h1_df = calculate_technical_indicators(h1_raw)
        h4_df = calculate_technical_indicators(h4_raw)
        fibs, _, _ = calculate_fibonacci_levels(m15_df, lookback=DEFAULT_FIB_LOOKBACK)
        eval_res = evaluate_intraday_dip_setup(m15_df, h1_df, h4_df, btc_df, fibs, fee_side_pct)

        scan_results.append({
            "symbol": sym,
            "base": item["base"],
            "label": item["label"],
            "candle_ts": m15_df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M UTC'),
            "score": eval_res["score"],
            "state": eval_res["state"],
            "action": eval_res["action"],
            "price": eval_res["price"],
            "tp1": eval_res["tp1"],
            "stop": eval_res["stop"],
            "rr": eval_res["net_r_multiple"],
            "rsi": eval_res["rsi"],
            "quote_vol": item["quote_vol"],
            "spread_pct": item["spread_pct"],
        })

    rank_order = {"Confirmed": 0, "Watch": 1, "Waiting": 2}
    scan_results.sort(key=lambda x: (rank_order.get(x["state"], 9), -x["score"]))
    return scan_results


# --- 5. User Interface & State Initialization ---
if 'recent_markets' not in st.session_state:
    st.session_state['recent_markets'] = []
if 'search_query_val' not in st.session_state:
    st.session_state['search_query_val'] = "SOL"

st.title("🎯 SH1N Spot Dip Catcher v5")
st.caption("15m Spot Setups · 1H/4H Trend Context · Scalp-to-Swing · Bitget USDT Spot")
st.caption("Scores rank setup confluence; they are not win probabilities. Always verify spread, fees, and order-book conditions before entry.")

all_markets = fetch_all_usdt_markets()

# Sidebar Controls
st.sidebar.header("Spot Market & Scanner")
search_query = st.sidebar.text_input(
    "Search Ticker / Pair:",
    value=st.session_state.get('search_query_val', 'SOL')
).strip().upper()
st.session_state['search_query_val'] = search_query

use_5m_refine = st.sidebar.checkbox(
    "Enable optional 5m timing check",
    value=True,
    help="Checks the latest closed 5m candle for finer dip entry confirmation."
)

max_risk_pct = st.sidebar.slider(
    "Max Risk per Trade (%)",
    min_value=0.25,
    max_value=2.0,
    value=1.0,
    step=0.25,
    help="Target 0.5%-1% risk per spot trade across 2-3 max concurrent positions."
)

alloc_usdt = st.sidebar.number_input(
    "Planned Allocation (USDT)",
    min_value=1.0,
    value=100.0,
    step=10.0,
    key="spot_alloc_usdt"
)

account_equity_usdt = st.sidebar.number_input(
    "Total Spot Account Equity (USDT)",
    min_value=1.0,
    value=500.0,
    step=25.0,
    help="Used only to calculate the risk budget. Enter your actual available spot account value."
)

fee_side_pct = st.sidebar.number_input(
    "Fee Per Side (%)",
    min_value=0.0,
    max_value=2.0,
    value=0.10,
    step=0.01,
    key="spot_side_fee"
)

matching_markets = [
    m for m in all_markets
    if m['base'].startswith(search_query) or m['symbol'].startswith(search_query)
]

if matching_markets:
    labels = [m['label'] for m in matching_markets]
    selected_label = st.sidebar.selectbox("Select Available Spot Market:", labels)
    pending_market = next(m for m in matching_markets if m['label'] == selected_label)

    analyze_click = st.sidebar.button("🚀 Analyze Dip Setup", type="primary", use_container_width=True)
    scan_click = st.sidebar.button("🔍 Scan Market (15m)", use_container_width=True)

    if 'active_market' not in st.session_state:
        st.session_state['active_market'] = pending_market
        st.session_state['recent_markets'] = [pending_market]

    if analyze_click:
        st.session_state['active_market'] = pending_market
        recent = [m for m in st.session_state['recent_markets'] if m['symbol'] != pending_market['symbol']]
        recent.insert(0, pending_market)
        st.session_state['recent_markets'] = recent[:5]

    active_market = st.session_state['active_market']

    # Selected Market Banner
    st.markdown(
        f"""
        <div class="selected-market-banner">
            <span style="color: #94a3b8; font-size: 0.82rem; font-weight: 600; text-transform: uppercase;">Currently Analyzing (15m Primary)</span><br/>
            <span style="font-size: 1.15rem; font-weight: 700; color: #f8fafc;">📍 {active_market['label']}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Recent Selections Quick-Switch Pills
    recent_list = st.session_state['recent_markets']
    if len(recent_list) > 1:
        st.markdown("**🕒 Recent Selections:**")
        rec_cols = st.columns(len(recent_list))
        for idx, r_m in enumerate(recent_list):
            is_active = (r_m['symbol'] == active_market['symbol'])
            btn_label = f"● {r_m['base']}" if is_active else r_m['base']
            if rec_cols[idx].button(
                btn_label,
                key=f"rec_btn_{r_m['symbol']}_{idx}",
                disabled=is_active,
                use_container_width=True
            ):
                st.session_state['active_market'] = r_m
                st.session_state['search_query_val'] = r_m['base']
                st.rerun()

    st.write("")

    # Fetch Closed Candles: 15m Primary + 1H + 4H + BTC 4H (+ optional 5m)
    symbol = active_market['symbol']
    with st.spinner(f"Evaluating 15m dip setup for {symbol}..."):
        m15_raw = fetch_ohlcv_data(symbol, timeframe="15m", limit=260)
        h1_raw = fetch_ohlcv_data(symbol, timeframe="1h", limit=260)
        h4_raw = fetch_ohlcv_data(symbol, timeframe="4h", limit=260)
        btc_raw = fetch_ohlcv_data("BTC/USDT", timeframe="4h", limit=260)
        m5_raw = fetch_ohlcv_data(symbol, timeframe="5m", limit=60) if use_5m_refine else pd.DataFrame()
        fng = fetch_fear_and_greed()

    if len(m15_raw) >= MIN_BARS and len(h1_raw) >= MIN_BARS and len(h4_raw) >= MIN_BARS:
        m15_df = calculate_technical_indicators(m15_raw)
        h1_df = calculate_technical_indicators(h1_raw)
        h4_df = calculate_technical_indicators(h4_raw)
        btc_df = calculate_technical_indicators(btc_raw) if len(btc_raw) >= MIN_BARS else None
        m5_df = calculate_technical_indicators(m5_raw) if (use_5m_refine and len(m5_raw) >= 20) else None

        fibs, _, _ = calculate_fibonacci_levels(m15_df, lookback=DEFAULT_FIB_LOOKBACK)
        res = evaluate_intraday_dip_setup(m15_df, h1_df, h4_df, btc_df, fibs, fee_side_pct, m5_df=m5_df)

        latest_price = res['price']
        prev_price = float(m15_df['close'].iloc[-2])
        price_change_pct = ((latest_price - prev_price) / prev_price) * 100

        st.caption(
            f"Closed 15m candle: **{m15_df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M UTC')}** · "
            f"Net R:R: **{res['net_r_multiple']:.2f}R** · {res['m5_note']}"
        )

        # --- HIGH-LEVEL OVERVIEW ---
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.metric("15m Close Price", format_price(latest_price), f"{price_change_pct:+.2f}%")
        with row1_col2:
            st.metric("Intraday Action", res['action'])

        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            st.metric("4H / 1H Trend", f"{res['trend4h']} / {res['trend1h']}", f"BTC 4H: {res['btc_trend']}")
        with row2_col2:
            st.metric("Dip Score", f"{res['score']} / 100", f"State: {res['state']}")

        st.divider()

        # --- INTRADAY PLAN CARD ---
        st.subheader("🎯 15m Intraday Dip & Resistance Plan")
        if res["state"] == "Confirmed":
            st.success(f"🟢 **CONFIRMED INTRADAY DIP** — {res['reason']}")
        elif res["state"] == "Watch":
            st.warning(f"🟡 **WATCH / PULLBACK ZONE** — {res['reason']}")
        else:
            st.info(f"⏸️ **WAITING** — {res['reason']}")

        with st.container():
            p_col1, p_col2 = st.columns(2)
            p_col1.metric("1. Dip Buy Zone", res['entry_range'], f"Sup {res['support_dist_pct']:.2f}% below")
            p_col2.metric("2. ATR/Swing Stop", format_price(res['stop']), f"-{res['risk_pct']:.2f}%")

            p_col3, p_col4 = st.columns(2)
            p_col3.metric("3. Resistance TP1", format_price(res['tp1']), f"{res['tp1_net_pct']:+.2f}% net")
            p_col4.metric("4. Resistance TP2", format_price(res['tp2']), f"{res['tp2_net_pct']:+.2f}% net")

        st.divider()

        # --- TABS ---
        tab1, tab2, tab3 = st.tabs([
            "🔍 Market Scanner",
            "🧮 Position & Sizing",
            "📊 Layers & S/R"
        ])

        with tab1:
            st.subheader("Bitget USDT Spot Scanner (15m)")
            st.caption("Designed to surface a handful of liquid candidates on active days; signal frequency is market-dependent.")

            sc1, sc2 = st.columns(2)
            with sc1:
                min_vol_k = st.selectbox("Min 24h Volume", options=[250, 500, 1000, 2500], index=1, format_func=lambda v: f"${v:,}K")
            with sc2:
                max_coins = st.selectbox("Pairs to Scan", options=[15, 25, 40], index=1)

            if scan_click or st.button("🔄 Run Spot Scanner Now", use_container_width=True):
                st.session_state['last_scan'] = run_market_dip_scanner(
                    max_coins=max_coins,
                    min_quote_vol=float(min_vol_k * 1000),
                    max_spread_pct=0.35,
                    fee_side_pct=fee_side_pct
                )

            scan_rows = st.session_state.get('last_scan', [])
            if scan_rows:
                disp = pd.DataFrame([
                    {
                        "Pair": r["symbol"],
                        "State": r["state"],
                        "Score": f"{r['score']}/100",
                        "Price": format_price(r["price"]),
                        "TP1": format_price(r["tp1"]),
                        "Net R:R": f"{r['rr']:.2f}R",
                        "RSI": f"{r['rsi']:.1f}",
                    }
                    for r in scan_rows
                ])
                st.dataframe(disp, use_container_width=True, hide_index=True)

                pick_symbol = st.selectbox("Load scanned pair into analyzer:", [r["symbol"] for r in scan_rows])
                if st.button("Load Selected Pair", key="load_scanned_pair_btn"):
                    found_m = next((m for m in all_markets if m["symbol"] == pick_symbol), None)
                    if found_m:
                        st.session_state['active_market'] = found_m
                        st.session_state['search_query_val'] = found_m['base']
                        st.rerun()
            else:
                st.info("Click **Run Spot Scanner Now** or **Scan Market (15m)** in the sidebar to rank liquid USDT spot dip candidates.")

        with tab2:
            st.subheader("Fee-Aware Position & Risk Sizing")
            fee = fee_side_pct / 100.0
            entry_cost_per_unit = latest_price * (1.0 + fee)
            stop_proceeds_per_unit = res['stop'] * (1.0 - fee)
            net_risk_per_unit = max(entry_cost_per_unit - stop_proceeds_per_unit, 0.0)
            risk_budget_usdt = account_equity_usdt * (max_risk_pct / 100.0)

            # Risk-based quantity, capped by the planned notional allocation.
            qty_by_risk = risk_budget_usdt / net_risk_per_unit if net_risk_per_unit > 0 else 0.0
            qty_by_allocation = alloc_usdt / entry_cost_per_unit if entry_cost_per_unit > 0 else 0.0
            est_qty = min(qty_by_risk, qty_by_allocation)
            buy_cost = est_qty * entry_cost_per_unit

            tp1_pnl = (est_qty * res['tp1'] * (1.0 - fee)) - buy_cost
            tp2_pnl = (est_qty * res['tp2'] * (1.0 - fee)) - buy_cost
            sl_pnl = (est_qty * res['stop'] * (1.0 - fee)) - buy_cost
            actual_stop_risk = abs(sl_pnl)
            implied_risk_pct = (actual_stop_risk / account_equity_usdt) * 100.0 if account_equity_usdt > 0 else 0.0

            m1, m2, m3 = st.columns(3)
            m1.metric(f"Est. {active_market['base']} Qty", f"{est_qty:,.4f}", f"@ {format_price(latest_price)}")
            m2.metric("Net P/L @ TP1", f"{tp1_pnl:+.2f} USDT", f"{(tp1_pnl / buy_cost * 100):+.2f}% net")
            m3.metric("Net P/L @ TP2", f"{tp2_pnl:+.2f} USDT", f"{(tp2_pnl / buy_cost * 100):+.2f}% net")

            st.caption(
                f"Stop impact @ {format_price(res['stop'])}: **{sl_pnl:+.2f} USDT** ({(sl_pnl / buy_cost * 100):+.2f}% net) · "
                f"Account risk at stop: **{implied_risk_pct:.2f}%** (Target: **{max_risk_pct:.2f}% / {risk_budget_usdt:.2f} USDT**) · "
                f"Fear & Greed (context): **{fng['value'] if fng['value'] is not None else 'N/A'} ({fng['classification']})**."
            )

            st.divider()
            st.subheader("💰 Spot Profit Calculator")
            st.caption("Estimate your net result for a buy-and-sell trade. Trading fees are applied separately on both sides.")

            calc_c1, calc_c2 = st.columns(2)
            with calc_c1:
                calc_buy_price = st.number_input(
                    "Buy Price (USDT)",
                    min_value=0.0,
                    value=0.0,
                    step=0.0001,
                    format="%.8f",
                    key="profit_calc_buy_price",
                )
            with calc_c2:
                calc_sell_price = st.number_input(
                    "Sell Price (USDT)",
                    min_value=0.0,
                    value=0.0,
                    step=0.0001,
                    format="%.8f",
                    key="profit_calc_sell_price",
                )

            calc_c3, calc_c4 = st.columns(2)
            with calc_c3:
                calc_buy_amount = st.number_input(
                    "Buying Amount (USDT)",
                    min_value=0.0,
                    value=100.0,
                    step=10.0,
                    format="%.2f",
                    key="profit_calc_buy_amount",
                )
            with calc_c4:
                calc_include_fee = st.checkbox(
                    "Include trading fee",
                    value=True,
                    key="profit_calc_include_fee",
                )
                calc_fee_pct = st.number_input(
                    "Fee per side (%)",
                    min_value=0.0,
                    max_value=5.0,
                    value=0.1,
                    step=0.01,
                    format="%.2f",
                    disabled=not calc_include_fee,
                    key="profit_calc_fee_pct",
                )

            calc_fee = (calc_fee_pct / 100.0) if calc_include_fee else 0.0
            if calc_buy_price > 0 and calc_sell_price > 0 and calc_buy_amount > 0:
                calc_qty = calc_buy_amount / calc_buy_price
                calc_buy_fee = calc_buy_amount * calc_fee
                calc_total_cost = calc_buy_amount + calc_buy_fee
                calc_gross_sell = calc_qty * calc_sell_price
                calc_sell_fee = calc_gross_sell * calc_fee
                calc_net_sell = calc_gross_sell - calc_sell_fee
                calc_net_pnl = calc_net_sell - calc_total_cost
                calc_roi = (calc_net_pnl / calc_total_cost * 100.0) if calc_total_cost > 0 else 0.0
                calc_break_even = calc_buy_price * (1.0 + calc_fee) / (1.0 - calc_fee) if calc_fee < 1.0 else 0.0

                r1, r2, r3 = st.columns(3)
                r1.metric("Estimated Quantity", f"{calc_qty:,.6f}")
                r2.metric("Net P/L", f"{calc_net_pnl:+,.4f} USDT", f"{calc_roi:+.2f}% net ROI")
                r3.metric("Break-even Sell Price", format_price(calc_break_even))

                st.caption(
                    f"Total buy cost: **{calc_total_cost:,.4f} USDT** "
                    f"(buy fee: {calc_buy_fee:,.4f}) · "
                    f"Net sell proceeds: **{calc_net_sell:,.4f} USDT** "
                    f"(sell fee: {calc_sell_fee:,.4f})."
                )
                if calc_net_pnl > 0:
                    st.success("This price scenario results in an estimated net profit after fees.")
                elif calc_net_pnl < 0:
                    st.warning("This price scenario results in an estimated net loss after fees.")
                else:
                    st.info("This price scenario is approximately break-even after fees.")
            else:
                st.info("Enter a buy price, sell price, and buying amount above zero to calculate the result.")

        with tab3:
            st.subheader("Three-Layer Score Breakdown")
            st.progress(res['score'] / 100)
            for cat, info in res['breakdown'].items():
                c1, c2, c3 = st.columns([2, 1, 3])
                c1.write(f"**{cat}**")
                c2.write(f"`{info['score']}/{info['max']} pts`")
                c3.caption(info['detail'])

            st.divider()
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("15m Supports")
                if res['supports']:
                    st.table(pd.DataFrame([{"Support": format_price(s)} for s in res['supports']]))
                else:
                    st.info("No confirmed support below price.")
            with col_b:
                st.subheader("15m Resistances")
                if res['resistances']:
                    st.table(pd.DataFrame([{"Resistance": format_price(r)} for r in res['resistances']]))
                else:
                    st.info("No confirmed resistance above price.")

    else:
        st.warning(f"Insufficient closed candles across 15m/1H/4H (need at least {MIN_BARS} bars for 200 EMA).")
else:
    st.warning(f"No active Spot USDT pairs found starting with '{search_query}'.")
