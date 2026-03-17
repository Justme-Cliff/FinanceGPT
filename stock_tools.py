"""
FinanceGPT — Live Stock Tools
==============================
Uses yfinance (free, no API key) to fetch live stock/crypto/index data.
Provides stock detection from natural language queries.
"""

import re

# Lazy import so startup isn't slowed when audio/other modes don't need it
def _yf():
    import yfinance as yf
    return yf


# ── Company name → ticker map ───────────────────────────────────────────

NAME_MAP = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN", "tesla": "TSLA", "nvidia": "NVDA", "meta": "META",
    "facebook": "META", "netflix": "NFLX", "berkshire": "BRK-B",
    "jpmorgan": "JPM", "jp morgan": "JPM", "goldman sachs": "GS", "goldman": "GS",
    "morgan stanley": "MS", "bank of america": "BAC", "wells fargo": "WFC",
    "visa": "V", "mastercard": "MA", "paypal": "PYPL",
    "disney": "DIS", "walmart": "WMT", "target": "TGT", "costco": "COST",
    "coca cola": "KO", "pepsi": "PEP", "mcdonalds": "MCD", "starbucks": "SBUX",
    "intel": "INTC", "amd": "AMD", "qualcomm": "QCOM", "broadcom": "AVGO",
    "salesforce": "CRM", "adobe": "ADBE", "oracle": "ORCL", "ibm": "IBM",
    "exxon": "XOM", "chevron": "CVX", "shell": "SHEL",
    "pfizer": "PFE", "johnson": "JNJ", "unitedhealth": "UNH",
    "bitcoin": "BTC-USD", "btc": "BTC-USD",
    "ethereum": "ETH-USD", "eth": "ETH-USD",
    "solana": "SOL-USD", "sol": "SOL-USD",
    "dogecoin": "DOGE-USD", "doge": "DOGE-USD",
    "s&p 500": "^GSPC", "s&p": "^GSPC", "sp500": "^GSPC",
    "dow jones": "^DJI", "dow": "^DJI",
    "nasdaq": "^IXIC",
    "vix": "^VIX",
}

STOCK_KEYWORDS = {
    "stock", "price", "share", "trading", "worth", "value",
    "doing", "up", "down", "gain", "loss", "ticker",
}


# ── Core fetch ──────────────────────────────────────────────────────────

def get_stock_info(ticker: str) -> dict:
    """Fetch live data for a ticker. Returns dict with price, change, etc."""
    try:
        yf = _yf()
        t    = yf.Ticker(ticker.upper())
        hist = t.history(period="5d")
        if hist.empty:
            return {"error": f"No data found for '{ticker.upper()}'"}

        price = float(hist["Close"].iloc[-1])
        prev  = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price
        change = price - prev
        pct    = (change / prev * 100) if prev else 0.0

        info = {}
        try:
            info = t.info or {}
        except Exception:
            pass

        return {
            "ticker":     ticker.upper(),
            "name":       info.get("longName") or info.get("shortName") or ticker.upper(),
            "price":      round(price, 4),
            "change":     round(change, 4),
            "pct_change": round(pct, 2),
            "volume":     info.get("volume", 0),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio":   info.get("trailingPE"),
            "52w_high":   info.get("fiftyTwoWeekHigh"),
            "52w_low":    info.get("fiftyTwoWeekLow"),
            "sector":     info.get("sector", ""),
            "currency":   info.get("currency", "USD"),
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker.upper()}


def format_stock_summary(data: dict) -> str:
    """Turn a stock info dict into a human-readable string."""
    if "error" in data:
        return f"Could not fetch data for {data.get('ticker', 'that ticker')}: {data['error']}"

    sign  = "+" if data["change"] >= 0 else ""
    arrow = "▲" if data["change"] >= 0 else "▼"
    cap   = f"${data['market_cap']/1e9:.2f}B" if data.get("market_cap") else "N/A"
    pe    = f"{data['pe_ratio']:.1f}"          if data.get("pe_ratio")   else "N/A"
    hi    = f"${data['52w_high']:.2f}"         if data.get("52w_high")   else "N/A"
    lo    = f"${data['52w_low']:.2f}"          if data.get("52w_low")    else "N/A"

    lines = [
        f"{data['name']} ({data['ticker']})",
        f"  Price    : ${data['price']}  {arrow} {sign}{data['change']} ({sign}{data['pct_change']}%)",
        f"  Mkt Cap  : {cap}   P/E: {pe}",
        f"  52w range: {lo} – {hi}",
    ]
    if data.get("sector"):
        lines.append(f"  Sector   : {data['sector']}")
    return "\n".join(lines)


# ── Query detection ─────────────────────────────────────────────────────

def detect_ticker(text: str) -> str | None:
    """Return a ticker symbol if the query looks like a stock lookup, else None."""
    text_up = text.upper()

    # $AAPL or explicit ticker patterns
    for pat in [
        r'\$([A-Z]{1,5})\b',
        r'\b([A-Z]{2,5})\s+stock\b',
        r'\bstock\s+(?:price\s+of\s+)?([A-Z]{2,5})\b',
        r'\bprice\s+of\s+([A-Z]{2,5})\b',
        r'\bhow\s+is\s+([A-Z]{2,5})\s+(?:stock\s+)?doing\b',
        r'\b([A-Z]{2,5})\s+(?:stock\s+)?price\b',
        r'\bcheck\s+([A-Z]{2,5})\b',
        r'\blook\s+up\s+([A-Z]{2,5})\b',
    ]:
        m = re.search(pat, text_up)
        if m:
            candidate = m.group(1)
            # Skip common English words that look like tickers
            if candidate not in {"IS", "IT", "AT", "BE", "DO", "GO", "MY", "NO", "OR", "SO", "TO", "UP"}:
                return candidate

    # Company name match
    text_lo = text.lower()
    for name, ticker in NAME_MAP.items():
        if name in text_lo:
            if any(kw in text_lo for kw in STOCK_KEYWORDS):
                return ticker

    return None
