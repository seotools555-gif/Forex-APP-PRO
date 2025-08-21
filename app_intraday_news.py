
# app_intraday_news.py — Forex Pro (Intraday + Enhanced News per Pair, No Uploads)
# - No file uploads; focused on live analysis
# - Real-time intraday chart (1m..60m) with EMA overlays
# - 5-minute ASI predictor + short-term suggestion
# - Per-pair cards: signals + detailed "News Analysis Points" (counts, top keywords, sources)
# - NewsAPI integration (optional) with RSS fallback
# - Safe secrets/env handling; Streamlit Cloud ready

import os, time, math, re
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import requests, feedparser
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ---------- Safe config ----------
DEFAULT_NEWSAPI_KEY = "1502cba32d134f4095aa03d4bd5bfe3c"
def get_cfg(name: str, default: str = "") -> str:
    val = os.getenv(name)
    if val not in (None, "", "None"): return val
    try:
        _ = st.secrets
        if name in st.secrets:
            sv = st.secrets.get(name)
            if sv not in (None, "", "None"): return str(sv)
    except Exception:
        pass
    return default

NEWSAPI_KEY = get_cfg("NEWSAPI_KEY", DEFAULT_NEWSAPI_KEY)
ACCOUNT_BALANCE_DEFAULT = float(get_cfg("ACCOUNT_BALANCE", "100000"))

# Ensure VADER
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# ---------- UI ----------
st.set_page_config(page_title="Forex Pro — Intraday + News", layout="wide")
st.title("🟢 Forex Pro — Intraday Signals + Real-Time News Analysis")
st.caption("Educational only — not financial advice.")

# Pairs
MAJOR = ["EURUSD=X","GBPUSD=X","USDJPY=X","USDCHF=X","AUDUSD=X","NZDUSD=X","USDCAD=X"]
CROSS = ["EURGBP=X","EURJPY=X","GBPJPY=X","AUDJPY=X","CHFJPY=X","NZDJPY=X","EURAUD=X","GBPAUD=X"]
EXOTIC = ["USDINR=X","USDSGD=X","USDHKD=X","USDTRY=X"]
ALL_PAIRS = MAJOR + CROSS + EXOTIC

# Sidebar
st.sidebar.header("Settings")
pairs = st.sidebar.multiselect("Pairs to analyze", ALL_PAIRS, MAJOR)
rt_pair = st.sidebar.selectbox("Real-Time Pair", ALL_PAIRS, index=0)
account_balance = st.sidebar.number_input("Account balance (USD)", value=ACCOUNT_BALANCE_DEFAULT, step=1000.0)
refresh = st.sidebar.slider("Auto-refresh (seconds)", 30, 1800, 300)
st.sidebar.write("NewsAPI key:", "✅ found" if NEWSAPI_KEY else "⚠️ not set (using RSS fallback)")
if st.sidebar.button("Force refresh now"): st.cache_data.clear()

# ---------- Data helpers ----------
@st.cache_data(ttl=60, show_spinner=False)
def download_history(symbol: str, period="60d", interval="1h"):
    try:
        t = yf.Ticker(symbol)
        df = t.history(period=period, interval=interval, actions=False, auto_adjust=False)
        if df is None or df.empty: return None
        for c in ("Open","High","Low","Close"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Open","High","Low","Close"])
        return df if not df.empty else None
    except Exception:
        return None

@st.cache_data(ttl=15, show_spinner=False)
def download_intraday(symbol: str, period="2d", interval="5m"):
    try:
        t = yf.Ticker(symbol)
        df = t.history(period=period, interval=interval, actions=False, auto_adjust=False)
        if df is None or df.empty: return None
        for c in ("Open","High","Low","Close"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Open","High","Low","Close"])
        return df if not df.empty else None
    except Exception:
        return None

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # EMAs
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    # RSI
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.ewm(alpha=1/14, adjust=False).mean() / (down.ewm(alpha=1/14, adjust=False).mean().replace(0,1e-10))
    df["RSI"] = 100 - (100/(1+rs))
    # ATR
    tr1 = (df["High"]-df["Low"]).abs()
    tr2 = (df["High"]-df["Close"].shift()).abs()
    tr3 = (df["Low"]-df["Close"].shift()).abs()
    df["ATR"] = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1).rolling(14, min_periods=1).mean()
    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]
    return df

# Candlestick pattern
def detect_candle(df: pd.DataFrame) -> str:
    if df is None or len(df) < 3: return "No Data"
    last = df.iloc[-1]; prev = df.iloc[-2]
    body = float(last["Close"] - last["Open"])
    prev_body = float(prev["Close"] - prev["Open"])
    rng = float(last["High"] - last["Low"]); 
    if rng <= 0: return "No Data"
    upper = float(last["High"] - max(last["Close"], last["Open"]))
    lower = float(min(last["Close"], last["Open"]) - last["Low"])
    if body > 0 and prev_body < 0 and last["Close"] > prev["Open"] and last["Open"] < prev["Close"]:
        return "Bullish Engulfing"
    if body < 0 and prev_body > 0 and last["Open"] > prev["Close"] and last["Close"] < prev["Open"]:
        return "Bearish Engulfing"
    if abs(body) <= 0.1 * rng: return "Doji"
    if abs(body) < (rng * 0.3) and lower > (2 * abs(body)): return "Hammer"
    if abs(body) < (rng * 0.3) and upper > (2 * abs(body)): return "Inverted Hammer"
    if body > 0 and upper > 2 * abs(body) and lower < abs(body): return "Shooting Star"
    return "No Clear Pattern"

# ASI
def compute_asi(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) < 3:
        return pd.Series(index=df.index if df is not None else pd.RangeIndex(0), dtype=float)
    df = df.copy()
    si_vals = [np.nan]
    for i in range(1, len(df)):
        H, L, C, O = df["High"].iloc[i], df["Low"].iloc[i], df["Close"].iloc[i], df["Open"].iloc[i]
        Hp, Lp, Cp, Op = df["High"].iloc[i-1], df["Low"].iloc[i-1], df["Close"].iloc[i-1], df["Open"].iloc[i-1]
        A = abs(H - Cp); B = abs(L - Cp); Cw = abs(H - L)
        if L <= Cp <= H: R = Cw
        elif Cp < L: R = abs(H - Cp)
        else: R = abs(Cp - L)
        if R == 0: si = 0.0
        else:
            K = max(A, B)
            si = 50.0 * ((C - Cp) + 0.5*(C - O) + 0.25*(Cp - Op)) / R
            if (A + B) != 0:
                si *= (K / (A + B))
        si_vals.append(si)
    return pd.Series(si_vals, index=df.index).fillna(0).cumsum()

def asi_trend_signal(df: pd.DataFrame):
    if df is None or len(df) < 20:
        return "Neutral", 0.0
    df = compute_indicators(df)
    asi = compute_asi(df)
    N = min(20, len(asi)-1)
    if N <= 1: return "Neutral", 0.0
    slope = (asi.iloc[-1] - asi.iloc[-N]) / N
    atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns and not np.isnan(df["ATR"].iloc[-1]) else 0.0
    conf = min(1.0, max(0.0, abs(slope) / (atr if atr>0 else 1.0)))
    if slope > 0: return "Uptrend", conf
    if slope < 0: return "Downtrend", conf
    return "Neutral", 0.0

# News & sentiment
RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/marketsNews",
    "https://www.fxstreet.com/rss",
    "https://www.investing.com/rss/news_25.rss",
]
CUR_KEYS = {
    "USD":["usd","fed","powell","cpi","pce","fomc","treasury"],
    "EUR":["eur","ecb","lagarde","eurozone","germany","euro"],
    "GBP":["gbp","boe","uk","britain"],
    "JPY":["jpy","boj","yen","japan"],
    "AUD":["aud","rba","australia"],
    "CAD":["cad","boc","canada","oil"],
    "CHF":["chf","snb","swiss","switzerland"],
    "INR":["inr","rbi","india","rupee"]
}

@st.cache_data(ttl=120, show_spinner=False)
def fetch_newsapi(q="forex OR currency", page_size=25):
    if not NEWSAPI_KEY: return []
    try:
        r = requests.get("https://newsapi.org/v2/everything",
                         params={"q":q,"language":"en","pageSize":page_size,"sortBy":"publishedAt","apiKey":NEWSAPI_KEY},
                         timeout=10)
        items=[]
        for a in r.json().get("articles", []):
            items.append({
                "title": a.get("title") or "",
                "desc": a.get("description") or "",
                "source": (a.get("source") or {}).get("name","")
            })
        return items
    except Exception:
        return []

@st.cache_data(ttl=120, show_spinner=False)
def fetch_rss(limit=25):
    items=[]
    for url in RSS_FEEDS:
        try:
            f=feedparser.parse(url)
            for e in f.entries[: max(3, limit//len(RSS_FEEDS))]:
                items.append({"title": e.get("title") or "", "desc": e.get("summary") or "", "source": url})
        except Exception:
            continue
    # dedupe
    seen=set(); out=[]
    for it in items:
        t=it.get("title","").strip()
        if t and t not in seen: seen.add(t); out.append(it)
    return out[:limit]

def news_for_pair_points(pair: str):
    """Return sentiment counts, bias, top keywords, and sources for the pair."""
    items = fetch_newsapi(f"{pair.split('=')[0]} OR forex OR currency", 30) or fetch_rss(30)
    pos=neg=neu=0
    per = {k:0.0 for k in CUR_KEYS}
    src_counts = {}
    kw_counts = {k:0 for k in set(sum(CUR_KEYS.values(), []))}
    details=[]
    for it in items:
        text = (it["title"] + " " + it["desc"]).strip()
        if not text: continue
        score = float(SentimentIntensityAnalyzer().polarity_scores(text)["compound"])
        if score >= 0.05: pos += 1
        elif score <= -0.05: neg += 1
        else: neu += 1
        low = text.lower()
        for cur, kws in CUR_KEYS.items():
            if any(kw in low for kw in kws):
                per[cur] += (1 if score>=0 else -1) * abs(score)
        # keyword counts
        for kw in kw_counts.keys():
            if re.search(r'\b' + re.escape(kw) + r'\b', low):
                kw_counts[kw] += 1
        src = (it.get("source") or "").strip()
        if src: src_counts[src] = src_counts.get(src,0) + 1
        details.append({"title": it["title"], "score": score, "source": src})
    p = pair.replace("=X","").upper()
    base, quote = p[:3], p[3:6]
    bias = float(per.get(base,0.0) - per.get(quote,0.0))
    # top keywords & sources
    top_kws = sorted(kw_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_src = sorted(src_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    return {
        "pos": pos, "neg": neg, "neu": neu, "bias": bias,
        "top_keywords": [k for k,c in top_kws if c>0],
        "top_sources": [s for s,c in top_src if c>0],
        "details": details
    }

# ---------- Composite & suggestions ----------
def per_pair_signal(df: pd.DataFrame, balance: float, pair: str):
    if df is None or len(df) < 3:
        return {"error":"no price data"}, {}
    df = compute_indicators(df)
    last, prev = df.iloc[-1], df.iloc[-2]
    last_close, prev_close = float(last["Close"]), float(prev["Close"])
    action = "HOLD"
    if last_close > prev_close and last["Close"] >= last["EMA20"]:
        action = "BUY"
    elif last_close < prev_close and last["Close"] <= last["EMA20"]:
        action = "SELL"
    # ATR fallback
    atr = float(last.get("ATR", np.nan))
    if not np.isfinite(atr) or atr <= 0:
        atr = max(df["Close"].pct_change().std() * last_close, last_close*0.0001)
    # SL/TP
    sl=tp1=tp2=None
    if action=="BUY":
        sl = last_close - 1.2*atr; tp1 = last_close + 1.5*(last_close - sl); tp2 = last_close + 2.5*(last_close - sl)
    elif action=="SELL":
        sl = last_close + 1.2*atr; tp1 = last_close - 1.5*(sl - last_close); tp2 = last_close - 2.5*(sl - last_close)
    # candle + flow
    pattern = detect_candle(df)
    flow = "Neutral"
    if pattern in ["Bullish Engulfing","Hammer"] and last["Close"] > last["EMA20"]:
        flow = "📈 Uptrend Likely"
    elif pattern in ["Bearish Engulfing","Shooting Star"] and last["Close"] < last["EMA20"]:
        flow = "📉 Downtrend Likely"
    elif pattern == "Doji":
        flow = "⏸ Sideways / Wait"
    # news points
    npnts = news_for_pair_points(pair)
    # composite
    tech = 1.0 if action=="BUY" else -1.0 if action=="SELL" else 0.0
    vol = 0.5 if atr>0 else 0.0
    composite = 0.30*(npnts["bias"]/3.0) + 0.25*tech + 0.15*vol
    composite = float(max(-1.0, min(1.0, composite)))
    final = "HOLD"
    if composite >= 0.4: final="STRONG BUY"
    elif composite >= 0.1: final="BUY"
    elif composite <= -0.4: final="STRONG SELL"
    elif composite <= -0.1: final="SELL"
    conf = "High" if abs(composite)>=0.6 else "Medium" if abs(composite)>=0.3 else "Low"
    return {
        "final": final, "confidence": conf, "price": round(last_close,6),
        "entry": round(last_close,6), "sl": round(sl,6) if sl else None,
        "tp1": round(tp1,6) if tp1 else None, "tp2": round(tp2,6) if tp2 else None,
        "composite": round(composite,3), "atr": round(atr,6),
        "pattern": pattern, "flow": flow, "rsi": round(float(last["RSI"]),2),
        "news_bias": round(npnts["bias"],3),
        "news_pos": npnts["pos"], "news_neg": npnts["neg"], "news_neu": npnts["neu"],
        "news_keywords": npnts["top_keywords"], "news_sources": npnts["top_sources"]
    }, npnts

# ---------- Real-Time Chart (top) ----------
st.markdown("## ⚡ Real-Time Intraday Chart")
col1, col2, col3 = st.columns([1,1,1])
with col1:
    rt_interval = st.selectbox("Interval", ["1m","2m","5m","15m","30m","60m"], index=2)
with col2:
    rt_period = st.selectbox("Lookback", ["1d","5d","7d"], index=0)
with col3:
    show_ema = st.checkbox("Show EMA20/EMA50", value=True)

rt_df = download_intraday(rt_pair, period=rt_period, interval=rt_interval)
if rt_df is None:
    st.warning(f"No intraday data for {rt_pair} ({rt_interval}, {rt_period}). Try a different interval/period.")
else:
    dfc = compute_indicators(rt_df)
    fig_rt = go.Figure()
    fig_rt.add_trace(go.Candlestick(x=dfc.index, open=dfc["Open"], high=dfc["High"], low=dfc["Low"], close=dfc["Close"], name="Candles"))
    if show_ema:
        fig_rt.add_trace(go.Scatter(x=dfc.index, y=dfc["EMA20"], mode="lines", name="EMA20"))
        fig_rt.add_trace(go.Scatter(x=dfc.index, y=dfc["EMA50"], mode="lines", name="EMA50"))
    fig_rt.update_layout(height=360, margin=dict(l=0,r=0,t=10,b=0), template="plotly_dark")
    st.plotly_chart(fig_rt, use_container_width=True)
st.markdown("---")

# ---------- 5m ASI Predictor for the Real-Time Pair ----------
st.subheader("🧭 5‑Minute Trend (ASI Ensemble)")
df5 = download_intraday(rt_pair, period="2d", interval="5m")
if df5 is None or len(df5) < 30:
    st.warning(f"Insufficient 5m data for {rt_pair}.")
else:
    df5i = compute_indicators(df5)
    tlabel, tconf = asi_trend_signal(df5i)
    # micro action
    last = df5i.iloc[-1]
    atr = float(last["ATR"]) if not math.isnan(float(last["ATR"])) else (df5i["Close"].pct_change().std() * float(last["Close"]))
    action = "HOLD"; sl=tp=None
    if tlabel=="Uptrend" and last["Close"] > last["EMA20"]:
        action = "BUY"; sl = float(last["Close"]) - 1.0*atr; tp = float(last["Close"]) + 1.5*(float(last["Close"]) - sl)
    elif tlabel=="Downtrend" and last["Close"] < last["EMA20"]:
        action = "SELL"; sl = float(last["Close"]) + 1.0*atr; tp = float(last["Close"]) - 1.5*(sl - float(last["Close"]))
    st.write(f"**Pair:** {rt_pair} | **ASI Trend:** {tlabel} | **Confidence:** {tconf:.2f}")
    st.write(f"**Short-term Suggestion (next 5–15m):** {action}  |  **SL:** {None if sl is None else round(sl,6)}  |  **TP:** {None if tp is None else round(tp,6)}")

st.markdown("---")

# ---------- Per-Pair Cards with News Analysis Points ----------
st.subheader("📈 Per‑Pair Intraday Signals + News Analysis")
for pair in pairs:
    df = download_history(pair, period="30d", interval="1h")
    if df is None:
        st.warning(f"{pair}: no price data"); 
        continue
    res, npnts = per_pair_signal(df, account_balance, pair)
    if "error" in res:
        st.warning(f"{pair}: {res['error']}"); 
        continue
    st.markdown(f"<div class='card'><h3 style='margin:0'>{pair} — <span style='color:#a7f3d0'>{res['final']}</span></h3>", unsafe_allow_html=True)
    st.markdown(f"<div class='small'>Price: {res['price']} | RSI: {res['rsi']} | ATR: {res['atr']} | Confidence: {res['confidence']}</div>", unsafe_allow_html=True)
    st.markdown(f"<p>Entry: <b>{res['entry']}</b>  &nbsp; SL: <b>{res['sl']}</b>  &nbsp; TP1: <b>{res['tp1']}</b>  &nbsp; TP2: <b>{res['tp2']}</b></p>", unsafe_allow_html=True)
    st.markdown(f"<div class='small'>Composite: {res['composite']} | News bias: {res['news_bias']} | News Sentiment — +{res['news_pos']} / −{res['news_neg']} / ◼ {res['news_neu']}</div>", unsafe_allow_html=True)
    # News analysis bullets
    bullets = []
    if res["news_keywords"]: bullets.append("Top keywords: " + ", ".join(res["news_keywords"][:5]))
    if res["news_sources"]: bullets.append("Top sources: " + ", ".join(res["news_sources"][:5]))
    bullets.append(f"Candlestick: {res['pattern']}  |  Market Flow: {res['flow']}")
    st.write("• " + "\n• ".join(bullets))

    # small intraday chart for each pair (optional)
    dfc = download_intraday(pair, period="1d", interval="15m")
    if dfc is not None:
        dfc = compute_indicators(dfc)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=dfc.index, open=dfc['Open'], high=dfc['High'], low=dfc['Low'], close=dfc['Close']))
        fig.add_trace(go.Scatter(x=dfc.index, y=dfc['EMA20'], mode='lines', name='EMA20'))
        fig.add_trace(go.Scatter(x=dfc.index, y=dfc['EMA50'], mode='lines', name='EMA50'))
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0), template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # expandable raw headlines
    if npnts and npnts.get("details"):
        with st.expander("Latest headlines & sentiment"):
            for h in npnts["details"][:10]:
                st.write(f"- ({h.get('score'):+.3f}) {h.get('title')} — {h.get('source','')}")

    st.markdown("---")
    
# ---- Auto-refresh (version-safe) ----
import time as _t
# Sleep for the chosen refresh seconds, then rerun.
_t.sleep(refresh)
try:
    # Streamlit >= 1.27
    st.rerun()
except Exception:
    # Older Streamlit
    try:
        st.experimental_rerun()
    except Exception:
        pass  # As a last resort, do nothing

