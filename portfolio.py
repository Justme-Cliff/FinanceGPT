"""
FinanceGPT — Portfolio Tracker
================================
Stores holdings locally in checkpoints/portfolio.json.
Fetches live prices via yfinance to calculate P&L and % gains.

CLI usage (via main.py):
  python main.py /portfolio add AAPL 10 150.00
  python main.py /portfolio remove AAPL
  python main.py /portfolio show
  python main.py /portfolio clear
"""

import json
import os
from datetime import datetime

from config import PORTFOLIO_FILE
from stock_tools import get_stock_info


# ── Load / Save ─────────────────────────────────────────────────────────

def _load() -> dict:
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    return {"holdings": {}}


def _save(data: dict):
    os.makedirs(os.path.dirname(PORTFOLIO_FILE), exist_ok=True)
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── CRUD ────────────────────────────────────────────────────────────────

def add_holding(ticker: str, shares: float, buy_price: float):
    data = _load()
    ticker = ticker.upper()
    data["holdings"][ticker] = {
        "shares":    shares,
        "buy_price": buy_price,
        "added":     datetime.now().strftime("%Y-%m-%d"),
    }
    _save(data)
    print(f"  ✓ Added {shares} shares of {ticker} @ ${buy_price:.2f}")


def remove_holding(ticker: str):
    data = _load()
    ticker = ticker.upper()
    if ticker in data["holdings"]:
        del data["holdings"][ticker]
        _save(data)
        print(f"  ✓ Removed {ticker} from portfolio")
    else:
        print(f"  {ticker} not found in portfolio")


def clear_portfolio():
    _save({"holdings": {}})
    print("  Portfolio cleared.")


# ── Display ─────────────────────────────────────────────────────────────

def show_portfolio(use_color: bool = True) -> str:
    data = _load()
    holdings = data.get("holdings", {})

    if not holdings:
        return "  Portfolio is empty. Add a holding with:\n  python main.py /portfolio add TICKER SHARES BUY_PRICE"

    try:
        from colorama import Fore, Style
        GREEN = Fore.GREEN   if use_color else ""
        RED   = Fore.RED     if use_color else ""
        CYAN  = Fore.CYAN    if use_color else ""
        DIM   = Style.DIM    if use_color else ""
        RST   = Style.RESET_ALL if use_color else ""
    except ImportError:
        GREEN = RED = CYAN = DIM = RST = ""

    lines = [f"\n{CYAN}  ── Portfolio ──────────────────────────────────────────{RST}"]
    lines.append(f"  {'Ticker':<8} {'Shares':>8} {'Buy':>8} {'Now':>8} {'Change':>10} {'P&L':>12}")
    lines.append("  " + "─" * 60)

    total_cost  = 0.0
    total_value = 0.0
    errors      = []

    for ticker, h in holdings.items():
        shares    = h["shares"]
        buy_price = h["buy_price"]
        info      = get_stock_info(ticker)

        if "error" in info:
            errors.append(f"  {ticker}: {info['error']}")
            continue

        now_price  = info["price"]
        cost       = shares * buy_price
        value      = shares * now_price
        pnl        = value - cost
        pct        = (pnl / cost * 100) if cost else 0.0
        sign       = "+" if pnl >= 0 else ""
        color      = GREEN if pnl >= 0 else RED

        total_cost  += cost
        total_value += value

        lines.append(
            f"  {ticker:<8} {shares:>8.2f} ${buy_price:>7.2f} ${now_price:>7.2f} "
            f"{color}{sign}{pct:>8.2f}%{RST} {color}{sign}${pnl:>9.2f}{RST}"
        )

    lines.append("  " + "─" * 60)
    total_pnl = total_value - total_cost
    total_pct = (total_pnl / total_cost * 100) if total_cost else 0.0
    sign  = "+" if total_pnl >= 0 else ""
    color = GREEN if total_pnl >= 0 else RED
    lines.append(
        f"  {'TOTAL':<8} {'':>8} {'':>8} ${total_value:>7.2f} "
        f"{color}{sign}{total_pct:>8.2f}%{RST} {color}{sign}${total_pnl:>9.2f}{RST}"
    )
    lines.append(f"  {DIM}Cost basis: ${total_cost:.2f}{RST}")

    if errors:
        lines.append(f"\n{RED}  Errors:{RST}")
        lines.extend(errors)

    return "\n".join(lines)


def get_portfolio_summary() -> str:
    """Short summary string for injection into chat context."""
    data = _load()
    holdings = data.get("holdings", {})
    if not holdings:
        return ""

    lines = ["Live Portfolio:"]
    for ticker, h in holdings.items():
        info = get_stock_info(ticker)
        if "error" in info:
            continue
        cost  = h["shares"] * h["buy_price"]
        value = h["shares"] * info["price"]
        pnl   = value - cost
        pct   = (pnl / cost * 100) if cost else 0.0
        sign  = "+" if pnl >= 0 else ""
        lines.append(f"  {ticker}: {h['shares']} shares @ ${info['price']} ({sign}{pct:.1f}%)")
    return "\n".join(lines)


# ── CLI entry ────────────────────────────────────────────────────────────

def portfolio_cli(args: list[str]):
    if not args:
        print(show_portfolio())
        return

    sub = args[0].lower()

    if sub == "show":
        print(show_portfolio())

    elif sub == "add":
        if len(args) < 4:
            print("  Usage: python main.py /portfolio add TICKER SHARES BUY_PRICE")
            return
        try:
            ticker    = args[1].upper()
            shares    = float(args[2])
            buy_price = float(args[3])
            add_holding(ticker, shares, buy_price)
        except ValueError:
            print("  SHARES and BUY_PRICE must be numbers.")

    elif sub == "remove":
        if len(args) < 2:
            print("  Usage: python main.py /portfolio remove TICKER")
            return
        remove_holding(args[1].upper())

    elif sub == "clear":
        clear_portfolio()

    else:
        print(f"  Unknown sub-command '{sub}'. Options: show, add, remove, clear")
        print(show_portfolio())
