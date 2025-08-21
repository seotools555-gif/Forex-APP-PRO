# 📊 Forex Pro — Real-Time Dashboard (with ASI + File Uploads)

A professional-grade **Streamlit dashboard** for **Forex trading analysis**.  
It combines **technical indicators, news sentiment, ASI trend prediction, and candlestick charts** — plus lets you **upload your own data (CSV/TXT)** for deeper insights.

---

## 🚀 Features
- ✅ **Real-Time Forex Data** via [Yahoo Finance](https://pypi.org/project/yfinance/)  
- ✅ **Indicators**: EMA, RSI, MACD, ATR  
- ✅ **ASI (Accumulation Swing Index)** trend analysis (5-minute + hourly)  
- ✅ **Candlestick Charts** (Plotly)  
- ✅ **News Sentiment** using [NewsAPI](https://newsapi.org/) (or RSS fallback)  
- ✅ **Buy/Sell Suggestions** with Stop Loss & Take Profit levels  
- ✅ **Confidence Scoring** with position sizing  
- ✅ **File Uploads**:  
  - Custom pairs list (`.txt` or `.csv`)  
  - OHLC CSV (with Date, Open, High, Low, Close)  
  - Signals CSV (to view past trades/logs)  

---

## 📂 File Upload Examples

### Custom Pairs List
- **TXT file:**  
  ```
  EURUSD=X
  GBPUSD=X
  USDJPY=X
  ```

- **CSV file:**  
  ```csv
  symbol
  EURUSD=X
  GBPUSD=X
  USDJPY=X
  ```

### OHLC CSV
Columns required: `Date, Open, High, Low, Close` (case-insensitive). Example:
```csv
Date,Open,High,Low,Close
2025-08-20 09:00,1.0850,1.0862,1.0842,1.0858
2025-08-20 10:00,1.0858,1.0870,1.0850,1.0865
```

---

## ⚙️ Installation (Local)

### 1. Clone repo or copy files
```bash
git clone https://github.com/your-username/forex-pro-dashboard.git
cd forex-pro-dashboard
```

### 2. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app_realtime_asi_fileio.py
```

Visit **http://localhost:8501** in your browser.

---

## 🌐 Deploy to Streamlit Cloud

1. Push this project to **GitHub**  
2. Go to [Streamlit Cloud](https://share.streamlit.io/) → **New App**  
3. Connect your repo & branch  
4. Select `app_realtime_asi_fileio.py` as the main file  
5. Add **Secrets** (optional):  
   - `NEWSAPI_KEY = "your_api_key_here"`  
   - `ACCOUNT_BALANCE = 100000`  

Streamlit Cloud will auto-install from `requirements.txt`.

---

## 🔑 API Keys
- **NewsAPI** (optional): [Get a free key here](https://newsapi.org/register)  
- If not set, the app uses **RSS feeds** as fallback.

---

## ⚠️ Disclaimer
This dashboard is for **educational purposes only**.  
It is **not financial advice**. Use at your own risk.
