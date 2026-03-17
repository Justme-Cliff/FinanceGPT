"""
FinanceGPT — Live Stock Tools
==============================
Uses yfinance (free, no API key) to fetch live stock/crypto/index data.
Provides stock detection from natural language queries.
"""

import re

def _yf():
    import yfinance as yf
    return yf


# ── Company name → ticker map ───────────────────────────────────────────

NAME_MAP = {
    # ── US Big Tech ──────────────────────────────────────────────────────
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN", "tesla": "TSLA", "nvidia": "NVDA", "meta": "META",
    "facebook": "META", "netflix": "NFLX", "berkshire": "BRK-B",
    "berkshire hathaway": "BRK-B", "warren buffett": "BRK-B",

    # ── Social / Consumer Tech ───────────────────────────────────────────
    "uber": "UBER", "lyft": "LYFT", "airbnb": "ABNB", "spotify": "SPOT",
    "twitter": "X", "x corp": "X", "snapchat": "SNAP", "snap": "SNAP",
    "pinterest": "PINS", "reddit": "RDDT", "roblox": "RBLX",
    "tiktok": "BDNCE", "duolingo": "DUOL", "bumble": "BMBL",
    "match group": "MTCH", "tinder": "MTCH",

    # ── Enterprise Tech / Cloud ──────────────────────────────────────────
    "palantir": "PLTR", "snowflake": "SNOW", "cloudflare": "NET",
    "shopify": "SHOP", "square": "SQ", "block": "SQ",
    "zoom": "ZM", "dropbox": "DBX", "docusign": "DOCU",
    "twilio": "TWLO", "sendgrid": "TWLO", "okta": "OKTA",
    "datadog": "DDOG", "mongodb": "MDB", "elastic": "ESTC",
    "confluent": "CFLT", "hashicorp": "HCP", "gitlab": "GTLB",
    "github": "MSFT", "linkedin": "MSFT", "nuance": "MSFT",
    "salesforce": "CRM", "slack": "CRM", "tableau": "CRM",
    "adobe": "ADBE", "oracle": "ORCL", "ibm": "IBM",
    "sap": "SAP", "servicenow": "NOW", "workday": "WDAY",
    "atlassian": "TEAM", "zendesk": "ZEN", "hubspot": "HUBS",
    "veeva": "VEEV", "splunk": "SPLK",

    # ── Semiconductors / Hardware ────────────────────────────────────────
    "intel": "INTC", "amd": "AMD", "qualcomm": "QCOM", "broadcom": "AVGO",
    "samsung": "005930.KS", "tsmc": "TSM", "taiwan semiconductor": "TSM",
    "asml": "ASML", "arm": "ARM", "arm holdings": "ARM",
    "micron": "MU", "western digital": "WDC", "seagate": "STX",
    "marvell": "MRVL", "lattice semiconductor": "LSCC",
    "on semiconductor": "ON", "texas instruments": "TXN",
    "analog devices": "ADI", "microchip": "MCHP",
    "applied materials": "AMAT", "lam research": "LRCX",
    "kla": "KLAC", "entegris": "ENTG",

    # ── Finance / Banking ────────────────────────────────────────────────
    "jpmorgan": "JPM", "jp morgan": "JPM", "chase": "JPM",
    "goldman sachs": "GS", "goldman": "GS",
    "morgan stanley": "MS", "bank of america": "BAC", "bofa": "BAC",
    "wells fargo": "WFC", "citigroup": "C", "citi": "C",
    "us bancorp": "USB", "pnc": "PNC", "truist": "TFC",
    "capital one": "COF", "discover": "DFS", "synchrony": "SYF",
    "blackrock": "BLK", "blackstone": "BX", "apollo": "APO",
    "kkr": "KKR", "carlyle": "CG", "ares": "ARES",
    "charles schwab": "SCHW", "schwab": "SCHW",
    "fidelity": "FNF", "vanguard": "V", "tdameritrade": "SCHW",
    "interactive brokers": "IBKR", "etrade": "MS",
    "visa": "V", "mastercard": "MA", "paypal": "PYPL",
    "american express": "AXP", "amex": "AXP",
    "stripe": "STRP", "affirm": "AFRM", "klarna": "KLAR",
    "sofi": "SOFI", "chime": "CHMEF", "wise": "WISE.L",

    # ── Insurance ────────────────────────────────────────────────────────
    "berkshire reinsurance": "BRK-B", "allstate": "ALL",
    "progressive": "PGR", "travelers": "TRV", "chubb": "CB",
    "aig": "AIG", "metlife": "MET", "prudential": "PRU",
    "unum": "UNM", "aflac": "AFL", "lincoln national": "LNC",

    # ── Consumer / Retail ────────────────────────────────────────────────
    "disney": "DIS", "walmart": "WMT", "target": "TGT", "costco": "COST",
    "coca cola": "KO", "coke": "KO", "pepsi": "PEP", "pepsico": "PEP",
    "mcdonalds": "MCD", "starbucks": "SBUX", "chipotle": "CMG",
    "yum brands": "YUM", "dominos": "DPZ", "shake shack": "SHAK",
    "nike": "NKE", "adidas": "ADDYY", "lululemon": "LULU",
    "under armour": "UAA", "vf corporation": "VFC",
    "home depot": "HD", "lowes": "LOW", "best buy": "BBY",
    "macys": "M", "nordstrom": "JWN", "gap": "GPS", "old navy": "GPS",
    "ross": "ROST", "tjx": "TJX", "tj maxx": "TJX",
    "dollar general": "DG", "dollar tree": "DLTR", "five below": "FIVE",
    "kroger": "KR", "albertsons": "ACI", "whole foods": "AMZN",
    "procter gamble": "PG", "pg": "PG", "unilever": "UL",
    "colgate": "CL", "kimberly clark": "KMB", "church dwight": "CHD",
    "estee lauder": "EL", "ulta": "ULTA", "revlon": "REV",
    "mondelez": "MDLZ", "kraft heinz": "KHC", "general mills": "GIS",
    "kellogg": "K", "campbell soup": "CPB", "hershey": "HSY",
    "constellation brands": "STZ", "anheuser busch": "BUD",
    "molson coors": "TAP", "boston beer": "SAM",

    # ── Media / Entertainment ────────────────────────────────────────────
    "warner bros": "WBD", "wbd": "WBD", "hbo": "WBD",
    "comcast": "CMCSA", "nbcuniversal": "CMCSA",
    "paramount": "PARA", "cbs": "PARA", "mtv": "PARA",
    "fox": "FOX", "fox news": "FOX",
    "new york times": "NYT", "nyt": "NYT",
    "live nation": "LYV", "ticketmaster": "LYV",
    "imax": "IMAX", "amc": "AMC", "amc networks": "AMCX",
    "electronic arts": "EA", "ea": "EA",
    "activision": "MSFT", "blizzard": "MSFT",
    "take two": "TTWO", "rockstar": "TTWO",
    "nintendo": "NTDOY", "sony": "SONY", "playstation": "SONY",

    # ── Automotive ───────────────────────────────────────────────────────
    "ford": "F", "gm": "GM", "general motors": "GM",
    "toyota": "TM", "honda": "HMC", "nissan": "NSANY",
    "volkswagen": "VWAGY", "bmw": "BMWYY", "mercedes": "MBGYY",
    "porsche": "POAHY", "ferrari": "RACE", "lamborghini": "VWAGY",
    "stellantis": "STLA", "jeep": "STLA", "chrysler": "STLA", "dodge": "STLA",
    "rivian": "RIVN", "lucid": "LCID", "fisker": "FSR",
    "nio": "NIO", "xpeng": "XPEV", "li auto": "LI",
    "harley davidson": "HOG",

    # ── Airlines / Travel ────────────────────────────────────────────────
    "delta": "DAL", "united airlines": "UAL", "american airlines": "AAL",
    "southwest": "LUV", "spirit": "SAVE", "frontier": "ULCC",
    "jetblue": "JBLU", "alaska airlines": "ALK",
    "lufthansa": "DLAKY", "air france": "AFLYY", "british airways": "IAG.L",
    "ryanair": "RYAAY", "easyjet": "EZJ.L",
    "booking": "BKNG", "booking.com": "BKNG", "priceline": "BKNG",
    "expedia": "EXPE", "tripadvisor": "TRIP", "airbnb": "ABNB",
    "carnival": "CCL", "royal caribbean": "RCL", "norwegian": "NCLH",
    "hilton": "HLT", "marriott": "MAR", "hyatt": "H", "wyndham": "WH",

    # ── Energy ───────────────────────────────────────────────────────────
    "exxon": "XOM", "exxonmobil": "XOM", "chevron": "CVX",
    "shell": "SHEL", "bp": "BP", "totalenergies": "TTE",
    "conocophillips": "COP", "pioneer natural": "PXD",
    "devon energy": "DVN", "marathon oil": "MRO",
    "schlumberger": "SLB", "slb": "SLB", "halliburton": "HAL",
    "baker hughes": "BKR",
    "nextera energy": "NEE", "duke energy": "DUK",
    "southern company": "SO", "dominion": "D",
    "enphase": "ENPH", "first solar": "FSLR", "sunrun": "RUN",
    "plug power": "PLUG", "ballard power": "BLDP",

    # ── Healthcare / Pharma ──────────────────────────────────────────────
    "pfizer": "PFE", "johnson": "JNJ", "j&j": "JNJ",
    "unitedhealth": "UNH", "unitedhealthcare": "UNH",
    "moderna": "MRNA", "abbvie": "ABBV", "merck": "MRK",
    "eli lilly": "LLY", "lilly": "LLY", "novo nordisk": "NVO",
    "astrazeneca": "AZN", "novartis": "NVS", "roche": "RHHBY",
    "bristol myers": "BMY", "gilead": "GILD", "biogen": "BIIB",
    "regeneron": "REGN", "vertex": "VRTX", "illumina": "ILMN",
    "intuitive surgical": "ISRG", "edwards lifesciences": "EW",
    "stryker": "SYK", "medtronic": "MDT", "boston scientific": "BSX",
    "abbott": "ABT", "baxter": "BAX", "becton dickinson": "BDX",
    "cvs": "CVS", "walgreens": "WBA", "cigna": "CI",
    "humana": "HUM", "centene": "CNC", "molina": "MOH",
    "hca healthcare": "HCA", "davita": "DVA",

    # ── Real Estate ──────────────────────────────────────────────────────
    "zillow": "Z", "redfin": "RDFN", "opendoor": "OPEN",
    "prologis": "PLD", "american tower": "AMT", "crown castle": "CCI",
    "equinix": "EQIX", "simon property": "SPG",
    "realty income": "O", "welltower": "WELL", "ventas": "VTR",

    # ── Telecom ──────────────────────────────────────────────────────────
    "at&t": "T", "att": "T", "verizon": "VZ", "t-mobile": "TMUS",
    "tmobile": "TMUS", "sprint": "TMUS", "comcast": "CMCSA",
    "charter": "CHTR", "cox": "N/A", "dish": "DISH",
    "vodafone": "VOD", "deutsche telekom": "DTEGY",
    "softbank": "SFTBY",

    # ── Industrial / Defense ─────────────────────────────────────────────
    "boeing": "BA", "airbus": "EADSY", "lockheed martin": "LMT",
    "raytheon": "RTX", "northrop grumman": "NOC",
    "general dynamics": "GD", "l3harris": "LHX",
    "caterpillar": "CAT", "deere": "DE", "john deere": "DE",
    "3m": "MMM", "honeywell": "HON", "emerson": "EMR",
    "ge": "GE", "general electric": "GE", "ge aerospace": "GE",
    "parker hannifin": "PH", "illinois tool": "ITW",
    "united parcel": "UPS", "ups": "UPS", "fedex": "FDX",
    "dhl": "DPW.DE", "xpo": "XPO", "old dominion": "ODFL",
    "csx": "CSX", "union pacific": "UNP", "norfolk southern": "NSC",
    "waste management": "WM", "republic services": "RSG",

    # ── Luxury / International ───────────────────────────────────────────
    "lvmh": "LVMHF", "louis vuitton": "LVMHF", "hermes": "HESAY",
    "kering": "PPRUY", "gucci": "PPRUY", "richemont": "CFRUY",
    "cartier": "CFRUY", "rolex": "N/A",
    "burberry": "BURBY", "prada": "PRDSF", "moncler": "MONRF",

    # ── Food / Agriculture ───────────────────────────────────────────────
    "cargill": "N/A", "archer daniels": "ADM", "adm": "ADM",
    "bunge": "BG", "tyson": "TSN", "hormel": "HRL",
    "sysco": "SYY", "us foods": "USFD",
    "nestle": "NSRGY", "danone": "DANOY",

    # ── ETFs ─────────────────────────────────────────────────────────────
    "spy": "SPY", "qqq": "QQQ", "vti": "VTI", "voo": "VOO",
    "iwm": "IWM", "dia": "DIA", "arkk": "ARKK", "arkg": "ARKG",
    "xlk": "XLK", "xlf": "XLF", "xle": "XLE", "xly": "XLY",
    "gld": "GLD", "slv": "SLV", "uso": "USO", "tlt": "TLT",
    "hyg": "HYG", "lqd": "LQD", "bnd": "BND", "agg": "AGG",

    # ── Crypto ───────────────────────────────────────────────────────────
    "bitcoin": "BTC-USD", "btc": "BTC-USD",
    "ethereum": "ETH-USD", "eth": "ETH-USD",
    "solana": "SOL-USD", "sol": "SOL-USD",
    "dogecoin": "DOGE-USD", "doge": "DOGE-USD",
    "cardano": "ADA-USD", "ada": "ADA-USD",
    "ripple": "XRP-USD", "xrp": "XRP-USD",
    "binance coin": "BNB-USD", "bnb": "BNB-USD",
    "avalanche": "AVAX-USD", "avax": "AVAX-USD",
    "chainlink": "LINK-USD", "link": "LINK-USD",
    "polkadot": "DOT-USD", "dot": "DOT-USD",
    "shiba inu": "SHIB-USD", "shib": "SHIB-USD",
    "uniswap": "UNI-USD", "uni": "UNI-USD",
    "polygon": "MATIC-USD", "matic": "MATIC-USD",
    "litecoin": "LTC-USD", "ltc": "LTC-USD",
    "stellar": "XLM-USD", "xlm": "XLM-USD",
    "tron": "TRX-USD", "trx": "TRX-USD",
    "near protocol": "NEAR-USD", "near": "NEAR-USD",
    "cosmos": "ATOM-USD", "atom": "ATOM-USD",

    # ── Indices ──────────────────────────────────────────────────────────
    "s&p 500": "^GSPC", "s&p": "^GSPC", "sp500": "^GSPC",
    "dow jones": "^DJI", "dow": "^DJI",
    "nasdaq": "^IXIC", "nasdaq composite": "^IXIC",
    "nasdaq 100": "^NDX", "ndx": "^NDX",
    "russell 2000": "^RUT", "russell": "^RUT",
    "vix": "^VIX", "fear index": "^VIX",
    "nikkei": "^N225", "nikkei 225": "^N225",
    "ftse": "^FTSE", "ftse 100": "^FTSE",
    "dax": "^GDAXI", "cac 40": "^FCHI",
    "hang seng": "^HSI", "shanghai": "000001.SS",
    "sensex": "^BSESN", "nifty": "^NSEI",

    # ── Commodities ──────────────────────────────────────────────────────
    "gold": "GC=F", "silver": "SI=F", "platinum": "PL=F",
    "crude oil": "CL=F", "wti": "CL=F", "oil": "CL=F",
    "brent crude": "BZ=F", "brent": "BZ=F",
    "natural gas": "NG=F", "gasoline": "RB=F",
    "copper": "HG=F", "aluminum": "ALI=F",
    "corn": "ZC=F", "wheat": "ZW=F", "soybeans": "ZS=F",
    "coffee": "KC=F", "sugar": "SB=F", "cotton": "CT=F",
}

STOCK_KEYWORDS = {
    "stock", "price", "share", "trading", "worth", "value",
    "doing", "up", "down", "gain", "loss", "ticker", "chart",
    "buy", "sell", "invest", "market cap", "dividend",
}

_COMPANY_INTENT_PATTERNS = [
    r'\bwhat\s+is\s+',
    r'\bwhat\'?s\s+',
    r'\btell\s+me\s+about\s+',
    r'\bwho\s+is\s+',
    r'\babout\s+',
    r'\bexplain\s+',
]

# Extract what the user is asking about from a company-intent query
_INTENT_STRIP = re.compile(
    r'^(what\s+is|what\'?s|tell\s+me\s+about|who\s+is|about|explain)\s+',
    re.IGNORECASE,
)


# ── Core fetch ──────────────────────────────────────────────────────────

def get_company_overview(ticker: str) -> str:
    """
    Return a human-readable company overview: what it does + live stock snapshot.
    Used when someone asks 'what is Amazon' rather than 'AMZN stock price'.
    """
    if ticker == "N/A":
        return "That company isn't publicly traded, so I don't have live market data for it."

    try:
        yf = _yf()
        t    = yf.Ticker(ticker.upper())
        info = t.info or {}
        hist = t.history(period="5d")

        name     = info.get("longName") or info.get("shortName") or ticker.upper()
        summary  = info.get("longBusinessSummary", "")
        sector   = info.get("sector", "")
        industry = info.get("industry", "")
        country  = info.get("country", "")
        cap      = info.get("marketCap", 0)
        cap_s    = f"${cap/1e9:.1f}B" if cap else "N/A"
        employees = info.get("fullTimeEmployees", 0)

        # Trim summary to ~2 sentences
        if summary:
            sentences = summary.split(". ")
            summary = ". ".join(sentences[:2]).strip()
            if not summary.endswith("."):
                summary += "."

        lines = [f"{name} ({ticker.upper()})"]
        if summary:
            lines.append(f"\n{summary}")

        details = []
        if sector:
            details.append(f"Sector: {sector}")
        if industry:
            details.append(f"Industry: {industry}")
        if country:
            details.append(f"Country: {country}")
        if cap_s != "N/A":
            details.append(f"Market Cap: {cap_s}")
        if employees:
            details.append(f"Employees: {employees:,}")
        if details:
            lines.append("\n" + "  |  ".join(details))

        if not hist.empty:
            price  = round(float(hist["Close"].iloc[-1]), 2)
            prev   = round(float(hist["Close"].iloc[-2]), 2) if len(hist) > 1 else price
            change = price - prev
            pct    = (change / prev * 100) if prev else 0.0
            sign   = "+" if change >= 0 else ""
            arrow  = "▲" if change >= 0 else "▼"
            lines.append(f"\nLive price: ${price}  {arrow} {sign}{pct:.2f}% today")

        return "\n".join(lines)
    except Exception as e:
        return f"I couldn't fetch live data for {ticker.upper()} ({e}). Try /stock {ticker.upper()} or check the ticker symbol."


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
            if candidate not in {"IS", "IT", "AT", "BE", "DO", "GO", "MY", "NO", "OR", "SO", "TO", "UP"}:
                return candidate

    text_lo = text.lower()
    has_stock_kw   = any(kw in text_lo for kw in STOCK_KEYWORDS)
    has_company_kw = any(re.search(p, text_lo) for p in _COMPANY_INTENT_PATTERNS)

    for name, ticker in NAME_MAP.items():
        if name in text_lo and (has_stock_kw or has_company_kw):
            return ticker

    return None


def detect_company_query(text: str) -> tuple[str | None, str]:
    """
    Returns (ticker, query_type):
      'overview'        — "what is Amazon" → company description + price
      'price'           — "AMZN stock price" → price only
      'unknown_company' — looks like a company query but not in NAME_MAP
      'none'            — not a company query at all
    """
    text_lo = text.lower()
    ticker  = detect_ticker(text)

    if ticker:
        has_stock_kw = any(kw in text_lo for kw in STOCK_KEYWORDS)
        return ticker, "price" if has_stock_kw else "overview"

    # Check if it looks like a company intent with an unknown entity
    has_company_kw = any(re.search(p, text_lo) for p in _COMPANY_INTENT_PATTERNS)
    if has_company_kw:
        # Extract the entity name after the intent phrase
        stripped = _INTENT_STRIP.sub("", text.strip())
        stripped = stripped.rstrip("?").strip()
        # If it looks like a proper noun (capitalized) and is 1-4 words, likely unknown company
        words = stripped.split()
        if 1 <= len(words) <= 4 and stripped[0].isupper():
            return stripped, "unknown_company"

    return None, "none"
