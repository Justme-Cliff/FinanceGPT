"""
FinanceGPT — Live Data Fetcher
================================
Pulls live market data via yfinance and generates Q&A CSV files,
then automatically fine-tunes the model on the new data in one shot.

Each topic always writes to a fixed filename (no date suffix) so
running /fetch multiple times never creates duplicate CSV files.

Usage:
  python main.py /fetch stocks        fetch + train on top 20 S&P stocks
  python main.py /fetch crypto        fetch + train on top 10 crypto
  python main.py /fetch market        fetch + train on major indices
  python main.py /fetch sectors       fetch + train on sector ETFs
  python main.py /fetch all           fetch + train on everything above
  python main.py /fetch stocks --no-train   fetch only, skip training

Output files (always overwritten, never duplicated):
  data/fetched_stocks.csv
  data/fetched_crypto.csv
  data/fetched_market.csv
  data/fetched_sectors.csv
"""

import csv
import os
from datetime import datetime

from config import DATA_DIR


# ── Top symbols ─────────────────────────────────────────────────────────

TOP_STOCKS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B",
    "JPM", "V", "JNJ", "WMT", "XOM", "MA", "UNH", "PG", "HD", "BAC",
    "AVGO", "COST",
]

TOP_CRYPTO = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
    "USDC-USD", "ADA-USD", "DOGE-USD", "TON-USD", "AVAX-USD",
]

INDICES = {
    "S&P 500":   "^GSPC",
    "Dow Jones": "^DJI",
    "NASDAQ":    "^IXIC",
    "VIX":       "^VIX",
    "Russell 2000": "^RUT",
}

SECTOR_ETFS = {
    "Technology":        "XLK",
    "Healthcare":        "XLV",
    "Financials":        "XLF",
    "Energy":            "XLE",
    "Consumer Discretionary": "XLY",
    "Consumer Staples":  "XLP",
    "Industrials":       "XLI",
    "Materials":         "XLB",
    "Real Estate":       "XLRE",
    "Utilities":         "XLU",
    "Communication":     "XLC",
}


# ── Helpers ──────────────────────────────────────────────────────────────

def _pct(new, old):
    return ((new - old) / old * 100) if old else 0.0


def _fetch_history(ticker, period="5d"):
    import yfinance as yf
    t = yf.Ticker(ticker)
    return t.history(period=period), t


def _write_csv(rows: list[tuple[str, str]], filename: str):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "answer"])
        writer.writerows(rows)
    print(f"  ✓ Saved {len(rows)} Q&A pairs → {path}")
    return path


def _today():
    return datetime.now().strftime("%B %d, %Y")


# ── Fetch functions ──────────────────────────────────────────────────────

def fetch_stocks() -> str:
    from tqdm import tqdm
    import yfinance as yf

    rows = []
    for ticker in tqdm(TOP_STOCKS, desc="  Stocks", ncols=70, unit="ticker"):
        try:
            t    = yf.Ticker(ticker)
            hist = t.history(period="5d")
            if hist.empty:
                continue
            info  = t.info or {}
            price = round(float(hist["Close"].iloc[-1]), 2)
            prev  = round(float(hist["Close"].iloc[-2]), 2) if len(hist) > 1 else price
            pct   = round(_pct(price, prev), 2)
            name  = info.get("longName") or ticker
            cap   = info.get("marketCap", 0)
            pe    = info.get("trailingPE")
            hi52  = info.get("fiftyTwoWeekHigh")
            lo52  = info.get("fiftyTwoWeekLow")
            sector = info.get("sector", "")

            sign  = "+" if pct >= 0 else ""
            cap_s = f"${cap/1e9:.1f}B" if cap else "N/A"
            pe_s  = f"{pe:.1f}" if pe else "N/A"

            rows.append((
                f"What is the current stock price of {name} ({ticker})?",
                f"As of {_today()}, {name} ({ticker}) is trading at ${price} per share, "
                f"{sign}{pct}% from the previous close of ${prev}. "
                f"Market cap: {cap_s}. P/E ratio: {pe_s}."
                + (f" 52-week range: ${lo52:.2f}–${hi52:.2f}." if hi52 and lo52 else "")
                + (f" Sector: {sector}." if sector else ""),
            ))
            direction = "up" if pct >= 0 else "down"
            rows.append((
                f"How is {ticker} performing today?",
                f"Step 1: Check current price — {name} ({ticker}) is at ${price}. "
                f"Step 2: Compare to previous close of ${prev}. "
                f"Step 3: The stock is {direction} {abs(pct):.2f}% today. "
                f"Therefore, {ticker} is {direction} {sign}{pct}% as of {_today()}.",
            ))
        except Exception as e:
            tqdm.write(f"    {ticker}: skipped ({e})")

    return _write_csv(rows, "fetched_stocks.csv")


def fetch_crypto() -> str:
    from tqdm import tqdm
    import yfinance as yf

    rows = []
    for symbol in tqdm(TOP_CRYPTO, desc="  Crypto ", ncols=70, unit="coin"):
        try:
            t    = yf.Ticker(symbol)
            hist = t.history(period="5d")
            if hist.empty:
                continue
            info  = t.info or {}
            price = round(float(hist["Close"].iloc[-1]), 4)
            prev  = round(float(hist["Close"].iloc[-2]), 4) if len(hist) > 1 else price
            pct   = round(_pct(price, prev), 2)
            name  = info.get("longName") or symbol.replace("-USD", "")
            cap   = info.get("marketCap", 0)
            cap_s = f"${cap/1e9:.1f}B" if cap else "N/A"
            sign  = "+" if pct >= 0 else ""
            direction = "up" if pct >= 0 else "down"

            rows.append((
                f"What is the current price of {name}?",
                f"As of {_today()}, {name} ({symbol}) is trading at ${price} USD, "
                f"{direction} {sign}{pct}% from the prior session. Market cap: {cap_s}.",
            ))
            rows.append((
                f"How is {name} doing today?",
                f"Step 1: Current price — ${price}. "
                f"Step 2: Previous close — ${prev}. "
                f"Step 3: Change — {sign}{pct}%. "
                f"Therefore, {name} is {direction} {abs(pct):.2f}% today as of {_today()}.",
            ))
        except Exception as e:
            tqdm.write(f"    {symbol}: skipped ({e})")

    return _write_csv(rows, "fetched_crypto.csv")


def fetch_market() -> str:
    print("  Fetching market indices…")
    import yfinance as yf

    from tqdm import tqdm

    rows = []
    summary_lines = []

    for name, symbol in tqdm(INDICES.items(), desc="  Market ", ncols=70, unit="index"):
        try:
            hist  = yf.Ticker(symbol).history(period="5d")
            if hist.empty:
                continue
            price = round(float(hist["Close"].iloc[-1]), 2)
            prev  = round(float(hist["Close"].iloc[-2]), 2) if len(hist) > 1 else price
            pct   = round(_pct(price, prev), 2)
            sign  = "+" if pct >= 0 else ""
            direction = "up" if pct >= 0 else "down"

            rows.append((
                f"What is the {name} at today?",
                f"As of {_today()}, the {name} ({symbol}) is at {price:.2f} points, "
                f"{direction} {sign}{pct}% from the previous session.",
            ))
            rows.append((
                f"How are the markets performing today?",
                f"As of {_today()}: {name} at {price:.2f} ({sign}{pct}%). "
                f"Markets are {'rising' if pct >= 0 else 'declining'} today.",
            ))
            summary_lines.append(f"{name}: {price:.2f} ({sign}{pct}%)")
        except Exception as e:
            tqdm.write(f"    {name}: skipped ({e})")

    if summary_lines:
        rows.append((
            "Give me a market overview for today.",
            f"Market summary as of {_today()}:\n" + "\n".join(f"  • {l}" for l in summary_lines),
        ))

    return _write_csv(rows, "fetched_market.csv")


def fetch_sectors() -> str:
    from tqdm import tqdm
    import yfinance as yf

    rows = []
    perf = []

    for sector, etf in tqdm(SECTOR_ETFS.items(), desc="  Sectors", ncols=70, unit="sector"):
        try:
            hist  = yf.Ticker(etf).history(period="5d")
            if hist.empty:
                continue
            price = round(float(hist["Close"].iloc[-1]), 2)
            prev  = round(float(hist["Close"].iloc[-2]), 2) if len(hist) > 1 else price
            pct   = round(_pct(price, prev), 2)
            sign  = "+" if pct >= 0 else ""
            direction = "up" if pct >= 0 else "down"

            rows.append((
                f"How is the {sector} sector performing today?",
                f"The {sector} sector ETF ({etf}) is {direction} {sign}{pct}% today "
                f"at ${price}, as of {_today()}.",
            ))
            perf.append((sector, pct, sign))
        except Exception as e:
            tqdm.write(f"    {sector}: skipped ({e})")

    if perf:
        perf.sort(key=lambda x: x[1], reverse=True)
        best  = perf[0]
        worst = perf[-1]
        rows.append((
            "Which stock market sector is performing best today?",
            f"As of {_today()}, the top-performing sector is {best[0]} "
            f"({best[2]}{best[1]:.2f}%), while the weakest is {worst[0]} "
            f"({worst[2]}{worst[1]:.2f}%).",
        ))

    return _write_csv(rows, "fetched_sectors.csv")


# ── CLI entry ────────────────────────────────────────────────────────────

TOPICS = {
    "stocks":  fetch_stocks,
    "crypto":  fetch_crypto,
    "market":  fetch_market,
    "sectors": fetch_sectors,
}


def fetch_cli(args: list[str]):
    topic    = args[0].lower() if args else "all"
    no_train = "--no-train" in args   # skip auto-train if flag passed

    print(f"\n  FinanceGPT — Live Data Fetcher")
    print(f"  Fetching: {topic}  ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n")

    fetched = []
    if topic == "all":
        for fn in TOPICS.values():
            fetched.append(fn())
    elif topic in TOPICS:
        fetched.append(TOPICS[topic]())
    else:
        print(f"  Unknown topic '{topic}'. Available: {', '.join(TOPICS)}, 'all'")
        print(f"  Add --no-train to skip automatic training after fetch.")
        return

    print(f"\n  ✓ {len(fetched)} CSV(s) written (always overwrites — no duplicates).")

    if no_train:
        print("  Skipping training (--no-train flag set).")
        print("  To train manually:")
        for path in fetched:
            print(f"    python main.py /train {path}")
        print()
        return

    # ── Auto-train on each fetched CSV ───────────────────────────────────
    from trainer import train as _train
    for path in fetched:
        print(f"\n  ── Auto-training on {path} ──────────────────────────────")
        _train(csv_file=path)

    print(f"\n  ✓ Fetch + train complete. Your model now knows today's market data.\n")
