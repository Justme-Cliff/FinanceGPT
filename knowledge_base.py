"""
Knowledge Base — TF-IDF retrieval over all CSV Q&A pairs.
Built entirely from scratch (no external libraries beyond stdlib + csv).

Optimisations vs original:
  1. Inverted index  — search only touches docs that share a term with the query
                       instead of iterating all N docs.  ~10-50× faster at scale.
  2. Bigram tokens   — "credit score" is indexed as one feature, improving phrase
                       matching for compound finance terms.
  3. Disk cache      — TF-IDF index pickled to checkpoints/kb_cache.pkl.
                       If no CSV has changed since last run, load takes ~0.1 s
                       instead of several seconds of rebuilding.
  4. Query expansion — search also runs on a normalised form of the query so
                       "what's a loan" and "What is a loan?" both hit the same doc.
"""

import csv
import hashlib
import math
import os
import pickle
import re
from collections import defaultdict


# ── Stop words ─────────────────────────────────────────────────────────
_STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'in', 'on', 'at',
    'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into',
    'through', 'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
    'neither', 'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just',
    'that', 'this', 'it', 'its', 'if', 'as', 'also', 'how', 'what', 'when',
    'where', 'which', 'who', 'whom', 'why', 'then', 'there', 'here', 'some',
    'any', 'all', 'no', 'more', 'most', 'other', 'such', 'between', 'each',
    'over', 'under', 'while', 'because', 'since', 'before', 'after', 'does',
    'use', 'used', 'using', 'make', 'made', 'making', 'get', 'set', 'put',
}

# Column-name aliases recognised in CSV files
_Q_COLS = {'question', 'q', 'query', 'input', 'prompt', 'ask'}
_A_COLS = {'answer', 'a', 'response', 'output', 'reply', 'text'}

# Conversational normalisation rules (same as reasoning_engine)
_REPHRASE_RULES = [
    (r"\bwhat'?s\b", "what is"),
    (r"\bhow'?s\b", "how does"),
    (r"\bi don'?t understand\b", "explain"),
    (r"\bhelp me understand\b", "explain"),
    (r"\bcan you explain\b", "explain"),
    (r"\btell me about\b", "what is"),
    (r"\bplease\b", ""),
    (r"\bkind of\b", ""),
    (r"\bsort of\b", ""),
]

_CACHE_VERSION = 5   # bump to invalidate old caches after format changes


# ── Tokeniser ───────────────────────────────────────────────────────────

# Finance synonym map — built from actual training CSV vocabulary
# Each key expands into related terms so different phrasings hit the same docs
_SYNONYMS: dict[str, list[str]] = {
    # ── Stocks / Equities ──────────────────────────────────────────────
    "stocks":           ["equities", "shares", "equity", "stock"],
    "equities":         ["stocks", "shares", "equity"],
    "shares":           ["stocks", "equities", "stock"],
    "equity":           ["stock", "shares", "equities"],
    "stock":            ["equities", "shares"],
    "bluechip":         ["large", "cap", "dividend", "stable"],
    "penny":            ["small", "cap", "speculative"],
    "growth":           ["appreciation", "momentum", "expansion"],
    "value":            ["undervalued", "cheap", "discount", "fundamental"],

    # ── Market indices ─────────────────────────────────────────────────
    "sp500":            ["index", "benchmark", "large", "cap", "market"],
    "nasdaq":           ["tech", "composite", "technology", "index"],
    "dow":              ["djia", "jones", "industrial", "index"],
    "djia":             ["dow", "jones", "industrial"],
    "index":            ["benchmark", "etf", "fund", "market"],
    "indices":          ["index", "benchmark", "market"],
    "russell":          ["small", "cap", "index"],
    "ftse":             ["uk", "british", "index", "market"],
    "nikkei":           ["japan", "japanese", "index", "market"],

    # ── Bonds / Fixed income ───────────────────────────────────────────
    "bonds":            ["fixed", "income", "debt", "treasuries", "securities"],
    "bond":             ["treasury", "note", "fixed", "income", "coupon"],
    "treasuries":       ["bonds", "government", "debt", "tbills"],
    "treasury":         ["government", "bonds", "debt", "risk", "free"],
    "tbills":           ["treasury", "short", "term", "government"],
    "munis":            ["municipal", "bonds", "tax", "exempt"],
    "municipal":        ["munis", "bonds", "tax", "exempt"],
    "junk":             ["high", "yield", "bonds", "speculative", "credit"],
    "highyield":        ["junk", "bonds", "credit", "risk"],
    "coupon":           ["interest", "payment", "bond", "yield"],
    "ytm":              ["yield", "maturity", "bond", "return"],
    "duration":         ["interest", "rate", "risk", "bond", "sensitivity"],
    "convexity":        ["bond", "duration", "interest", "rate"],
    "yield":            ["return", "dividend", "interest", "coupon", "rate"],
    "inverted":         ["yield", "curve", "recession", "warning"],
    "laddering":        ["bonds", "maturity", "diversification", "fixed", "income"],
    "tips":             ["inflation", "protected", "treasury", "bonds"],
    "mbs":              ["mortgage", "backed", "securities", "bonds"],

    # ── Returns / Performance ──────────────────────────────────────────
    "return":           ["roi", "yield", "profit", "gain", "performance"],
    "roi":              ["return", "profit", "gain", "investment"],
    "roic":             ["return", "invested", "capital", "profitability"],
    "roe":              ["return", "equity", "profitability"],
    "roa":              ["return", "assets", "profitability"],
    "alpha":            ["outperformance", "excess", "return", "skill"],
    "beta":             ["market", "risk", "volatility", "correlation", "systematic"],
    "sharpe":           ["risk", "adjusted", "return", "ratio"],
    "sortino":          ["downside", "risk", "return", "ratio"],
    "calmar":           ["drawdown", "return", "ratio", "risk"],
    "cagr":             ["compound", "annual", "growth", "rate", "return"],
    "performance":      ["return", "gain", "profit", "results"],
    "drawdown":         ["loss", "decline", "peak", "trough", "risk"],

    # ── Valuation ratios ───────────────────────────────────────────────
    "pe":               ["price", "earnings", "valuation", "multiple", "ratio"],
    "peg":              ["pe", "growth", "valuation", "ratio"],
    "pb":               ["price", "book", "valuation", "ratio"],
    "ps":               ["price", "sales", "valuation", "ratio"],
    "ebitda":           ["earnings", "operating", "income", "profit", "margin"],
    "ev":               ["enterprise", "value", "valuation"],
    "wacc":             ["weighted", "average", "cost", "capital", "discount"],
    "dcf":              ["discounted", "cash", "flow", "valuation", "intrinsic"],
    "npv":              ["net", "present", "value", "discounted", "cash"],
    "irr":              ["internal", "rate", "return", "investment"],
    "eps":              ["earnings", "per", "share", "profit"],
    "fcf":              ["free", "cash", "flow", "earnings"],
    "capex":            ["capital", "expenditure", "investment", "spending"],
    "intrinsic":        ["value", "dcf", "fundamental", "worth"],
    "valuation":        ["price", "worth", "multiple", "pe", "dcf"],
    "multiple":         ["valuation", "pe", "ebitda", "ratio"],

    # ── Financial ratios ───────────────────────────────────────────────
    "current":          ["ratio", "liquidity", "assets", "liabilities"],
    "quick":            ["ratio", "liquidity", "acid", "test"],
    "debt":             ["leverage", "liability", "borrowing", "credit"],
    "leverage":         ["debt", "margin", "borrowing", "risk"],
    "liquidity":        ["cash", "quick", "current", "ratio", "assets"],
    "solvency":         ["debt", "leverage", "long", "term", "viability"],
    "profitability":    ["margin", "roe", "roa", "return", "profit"],
    "margin":           ["profit", "gross", "operating", "net", "profitability"],
    "turnover":         ["efficiency", "asset", "inventory", "ratio"],
    "efficiency":       ["turnover", "ratio", "operational", "productivity"],
    "altman":           ["zscore", "bankruptcy", "distress", "risk"],

    # ── Crypto / Blockchain ────────────────────────────────────────────
    "bitcoin":          ["btc", "crypto", "cryptocurrency", "digital"],
    "btc":              ["bitcoin", "crypto", "digital", "currency"],
    "ethereum":         ["eth", "crypto", "smart", "contracts", "blockchain"],
    "eth":              ["ethereum", "crypto", "blockchain"],
    "crypto":           ["cryptocurrency", "bitcoin", "blockchain", "digital", "assets"],
    "cryptocurrency":   ["crypto", "bitcoin", "ethereum", "digital", "assets"],
    "blockchain":       ["distributed", "ledger", "crypto", "decentralized"],
    "defi":             ["decentralized", "finance", "crypto", "protocol"],
    "nft":              ["non", "fungible", "token", "digital", "art"],
    "stablecoin":       ["usdc", "usdt", "peg", "dollar", "crypto"],
    "staking":          ["yield", "proof", "stake", "crypto", "rewards"],
    "altcoin":          ["crypto", "alternative", "coin", "token"],
    "halving":          ["bitcoin", "supply", "mining", "reward"],
    "pow":              ["proof", "work", "mining", "bitcoin"],
    "pos":              ["proof", "stake", "ethereum", "validator"],
    "wallet":           ["crypto", "keys", "storage", "cold", "hot"],
    "dex":              ["decentralized", "exchange", "defi", "swap"],
    "dao":              ["decentralized", "autonomous", "organization", "governance"],

    # ── Options ────────────────────────────────────────────────────────
    "options":          ["call", "put", "derivative", "contract", "premium"],
    "call":             ["option", "buy", "right", "bullish", "upside"],
    "put":              ["option", "sell", "right", "bearish", "downside"],
    "strike":           ["price", "exercise", "option", "contract"],
    "premium":          ["option", "cost", "price", "insurance"],
    "delta":            ["options", "greek", "sensitivity", "hedge"],
    "gamma":            ["options", "greek", "delta", "change"],
    "theta":            ["time", "decay", "option", "greek", "erosion"],
    "vega":             ["volatility", "option", "greek", "sensitivity"],
    "iv":               ["implied", "volatility", "options", "vega"],
    "itm":              ["in", "money", "option", "intrinsic"],
    "otm":              ["out", "money", "option", "speculative"],
    "covered":          ["call", "option", "income", "strategy"],
    "straddle":         ["options", "volatility", "strategy", "earnings"],
    "condor":           ["iron", "options", "strategy", "range"],
    "leaps":            ["long", "term", "options", "equity", "anticipation"],
    "blackscholes":     ["options", "pricing", "model", "formula"],

    # ── Technical analysis ─────────────────────────────────────────────
    "rsi":              ["relative", "strength", "index", "momentum", "overbought"],
    "macd":             ["moving", "average", "convergence", "divergence", "momentum"],
    "bollinger":        ["bands", "volatility", "standard", "deviation"],
    "vwap":             ["volume", "weighted", "average", "price", "institutional"],
    "ema":              ["exponential", "moving", "average", "trend"],
    "sma":              ["simple", "moving", "average", "trend"],
    "golden":           ["cross", "moving", "average", "bullish"],
    "death":            ["cross", "moving", "average", "bearish"],
    "support":          ["level", "price", "floor", "technical"],
    "resistance":       ["level", "price", "ceiling", "technical"],
    "breakout":         ["support", "resistance", "price", "momentum"],
    "candlestick":      ["chart", "technical", "pattern", "ohlc"],
    "fibonacci":        ["retracement", "technical", "levels", "support"],
    "momentum":         ["rsi", "macd", "trend", "velocity"],
    "overbought":       ["rsi", "high", "reversal", "technical"],
    "oversold":         ["rsi", "low", "reversal", "technical", "cheap"],

    # ── Risk ───────────────────────────────────────────────────────────
    "risk":             ["volatility", "uncertainty", "loss", "exposure"],
    "volatility":       ["risk", "vix", "standard", "deviation", "fluctuation"],
    "vix":              ["volatility", "fear", "index", "options"],
    "var":              ["value", "risk", "loss", "quantile"],
    "cvar":             ["conditional", "var", "expected", "shortfall", "tail"],
    "hedge":            ["protection", "risk", "management", "offset"],
    "hedging":          ["hedge", "risk", "management", "protection"],
    "systematic":       ["market", "risk", "beta", "undiversifiable"],
    "unsystematic":     ["specific", "risk", "diversifiable", "company"],
    "correlation":      ["diversification", "relationship", "covariance", "beta"],
    "diversification":  ["portfolio", "correlation", "risk", "spread", "allocation"],
    "rebalancing":      ["portfolio", "allocation", "rebalance", "adjustment"],
    "allocation":       ["portfolio", "asset", "diversification", "weights"],

    # ── Economy / Fed / Macro ──────────────────────────────────────────
    "fed":              ["federal", "reserve", "fomc", "central", "bank", "monetary"],
    "federal":          ["fed", "reserve", "fomc", "central", "bank"],
    "fomc":             ["fed", "federal", "reserve", "rate", "decision"],
    "reserve":          ["fed", "federal", "central", "bank"],
    "cpi":              ["inflation", "consumer", "price", "index", "cost"],
    "pce":              ["inflation", "personal", "consumption", "fed", "preferred"],
    "inflation":        ["cpi", "pce", "prices", "purchasing", "power", "rate"],
    "deflation":        ["prices", "falling", "economy", "contraction"],
    "stagflation":      ["inflation", "recession", "stagnation", "unemployment"],
    "gdp":              ["growth", "output", "economy", "economic", "production"],
    "gnp":              ["gross", "national", "product", "gdp", "economy"],
    "recession":        ["downturn", "contraction", "gdp", "negative", "growth"],
    "depression":       ["severe", "recession", "economic", "crisis", "1929"],
    "unemployment":     ["jobs", "labor", "market", "rate", "employment"],
    "pmi":              ["purchasing", "managers", "index", "manufacturing", "economic"],
    "qe":               ["quantitative", "easing", "fed", "money", "supply", "bonds"],
    "qt":               ["quantitative", "tightening", "fed", "contraction", "balance"],
    "sofr":             ["libor", "overnight", "rate", "benchmark", "loans"],
    "libor":            ["sofr", "benchmark", "rate", "interbank", "lending"],
    "monetary":         ["policy", "fed", "interest", "rate", "money", "supply"],
    "fiscal":           ["policy", "government", "spending", "tax", "deficit"],
    "stimulus":         ["fiscal", "government", "spending", "economy", "recovery"],
    "austerity":        ["fiscal", "cuts", "government", "spending", "debt"],
    "yield":            ["return", "interest", "curve", "bond", "rate"],
    "interest":         ["rate", "yield", "fed", "borrowing", "cost"],
    "rate":             ["interest", "yield", "fed", "borrowing", "inflation"],
    "rates":            ["interest", "yield", "borrowing", "fed", "policy"],
    "hawkish":          ["interest", "rate", "hike", "fed", "inflation"],
    "dovish":           ["interest", "rate", "cut", "fed", "easing"],
    "pivot":            ["fed", "rate", "change", "policy", "shift"],
    "tariff":           ["trade", "tax", "import", "protectionism"],
    "trade":            ["deficit", "surplus", "tariff", "export", "import"],
    "globalization":    ["trade", "international", "markets", "exports"],

    # ── Personal finance / Savings ─────────────────────────────────────
    "budget":           ["spending", "income", "expenses", "financial", "plan"],
    "saving":           ["savings", "emergency", "fund", "investment"],
    "savings":          ["saving", "account", "emergency", "fund", "interest"],
    "debt":             ["loan", "credit", "liability", "borrowing", "payoff"],
    "loan":             ["debt", "credit", "borrowing", "interest", "mortgage"],
    "credit":           ["score", "debt", "loan", "rating", "borrowing"],
    "mortgage":         ["home", "loan", "real", "estate", "interest", "rate"],
    "compound":         ["interest", "growth", "reinvest", "exponential"],
    "simple":           ["interest", "calculation", "linear"],
    "networth":         ["wealth", "assets", "liabilities", "balance"],
    "emergency":        ["fund", "savings", "cash", "reserve", "liquid"],
    "insurance":        ["protection", "risk", "coverage", "premium", "policy"],
    "annuity":          ["retirement", "income", "fixed", "payment", "insurance"],
    "social":           ["security", "retirement", "benefit", "government"],

    # ── Retirement ────────────────────────────────────────────────────
    "401k":             ["retirement", "pension", "ira", "employer", "savings"],
    "ira":              ["retirement", "account", "tax", "traditional", "roth"],
    "roth":             ["ira", "tax", "free", "retirement", "after", "tax"],
    "traditional":      ["ira", "401k", "pretax", "deductible", "retirement"],
    "rmd":              ["required", "minimum", "distribution", "retirement", "ira"],
    "fire":             ["financial", "independence", "retire", "early"],
    "pension":          ["defined", "benefit", "retirement", "income"],
    "sepira":           ["self", "employed", "retirement", "ira"],
    "403b":             ["retirement", "nonprofit", "403", "plan"],
    "backdoor":         ["roth", "conversion", "retirement", "high", "income"],

    # ── Tax ───────────────────────────────────────────────────────────
    "tax":              ["taxes", "capital", "gains", "income", "deduction"],
    "taxes":            ["tax", "filing", "irs", "deduction", "liability"],
    "capitalgains":     ["tax", "profit", "investment", "short", "long", "term"],
    "taxloss":          ["harvesting", "offset", "gains", "tax", "strategy"],
    "deduction":        ["tax", "itemized", "standard", "expense", "write"],
    "depreciation":     ["real", "estate", "tax", "deduction", "asset"],
    "1031":             ["exchange", "real", "estate", "tax", "defer"],
    "estate":           ["planning", "inheritance", "trust", "tax", "wealth"],
    "trust":            ["estate", "planning", "legal", "asset", "protection"],

    # ── Real estate ───────────────────────────────────────────────────
    "reit":             ["real", "estate", "investment", "trust", "dividend"],
    "caprate":          ["capitalization", "rate", "real", "estate", "yield"],
    "noi":              ["net", "operating", "income", "real", "estate"],
    "cashoncash":       ["return", "real", "estate", "rental", "income"],
    "brrrr":            ["buy", "rehab", "rent", "refinance", "repeat", "real", "estate"],
    "househacking":     ["real", "estate", "rental", "income", "primary"],
    "fixflip":          ["real", "estate", "renovation", "profit", "strategy"],
    "ltv":              ["loan", "to", "value", "mortgage", "real", "estate"],
    "dscr":             ["debt", "service", "coverage", "ratio", "loan"],

    # ── ETFs / Funds ──────────────────────────────────────────────────
    "etf":              ["fund", "index", "exchange", "traded", "basket"],
    "mutual":           ["fund", "etf", "managed", "portfolio", "investor"],
    "index":            ["fund", "passive", "benchmark", "etf", "sp500"],
    "passive":          ["investing", "index", "fund", "etf", "low", "cost"],
    "active":           ["management", "fund", "alpha", "stock", "picking"],

    # ── Private equity / VC ───────────────────────────────────────────
    "pe":               ["private", "equity", "buyout", "leverage", "lbo"],
    "lbo":              ["leveraged", "buyout", "private", "equity", "debt"],
    "vc":               ["venture", "capital", "startup", "seed", "funding"],
    "moic":             ["multiple", "invested", "capital", "return", "private", "equity"],
    "tvpi":             ["total", "value", "paid", "private", "equity", "return"],
    "jcurve":           ["private", "equity", "returns", "early", "loss"],
    "carry":            ["carried", "interest", "profit", "private", "equity"],
    "drypowder":        ["uninvested", "capital", "private", "equity", "cash"],
    "gp":               ["general", "partner", "private", "equity", "fund"],
    "lp":               ["limited", "partner", "private", "equity", "investor"],

    # ── Hedge funds ───────────────────────────────────────────────────
    "hedgefund":        ["fund", "alternative", "investment", "strategy"],
    "longshort":        ["equity", "hedge", "fund", "strategy"],
    "globalmacro":      ["macro", "fund", "strategy", "interest", "rates"],
    "arbitrage":        ["risk", "free", "profit", "spread", "mispricing"],
    "statarb":          ["statistical", "arbitrage", "pairs", "trading"],

    # ── Behavioral finance ────────────────────────────────────────────
    "bias":             ["behavioral", "cognitive", "psychology", "decision"],
    "fomo":             ["fear", "missing", "out", "behavioral", "greed"],
    "lossaversion":     ["loss", "behavioral", "bias", "pain", "gain"],
    "anchoring":        ["bias", "behavioral", "reference", "point"],
    "herding":          ["behavioral", "group", "crowd", "bias", "market"],
    "overconfidence":   ["bias", "behavioral", "trading", "skill"],
    "bubble":           ["asset", "price", "inflation", "speculation", "crash"],
    "sentiment":        ["investor", "market", "fear", "greed", "behavioral"],

    # ── Commodities ───────────────────────────────────────────────────
    "commodities":      ["commodity", "gold", "oil", "futures", "raw"],
    "commodity":        ["futures", "spot", "gold", "oil", "silver"],
    "gold":             ["commodity", "safe", "haven", "inflation", "hedge"],
    "oil":              ["crude", "energy", "commodity", "wti", "brent"],
    "silver":           ["gold", "commodity", "precious", "metals"],
    "futures":          ["contract", "derivative", "commodity", "forward"],
    "contango":         ["futures", "spot", "commodity", "curve"],
    "backwardation":    ["futures", "spot", "commodity", "curve"],

    # ── Forex ─────────────────────────────────────────────────────────
    "forex":            ["foreign", "exchange", "currency", "fx", "trading"],
    "fx":               ["forex", "currency", "exchange", "foreign"],
    "currency":         ["forex", "exchange", "rate", "pair", "appreciation"],
    "pip":              ["forex", "currency", "price", "movement", "point"],
    "carry":            ["trade", "forex", "interest", "rate", "strategy"],
    "ppp":              ["purchasing", "power", "parity", "exchange", "rate"],
    "eurusd":           ["euro", "dollar", "forex", "major", "pair"],

    # ── Fintech / Blockchain ──────────────────────────────────────────
    "fintech":          ["technology", "financial", "digital", "banking", "innovation"],
    "roboadviser":      ["automated", "investing", "portfolio", "algorithm"],
    "bnpl":             ["buy", "now", "pay", "later", "credit", "debt"],
    "openbanking":      ["api", "banking", "data", "sharing", "fintech"],
    "kyc":              ["know", "your", "customer", "compliance", "identity"],
    "aml":              ["anti", "money", "laundering", "compliance", "regulation"],
    "cbdc":             ["central", "bank", "digital", "currency", "fed"],

    # ── IPO / Corporate actions ───────────────────────────────────────
    "ipo":              ["initial", "public", "offering", "listing", "shares"],
    "spac":             ["blank", "check", "company", "merger", "ipo"],
    "spinoff":          ["corporate", "action", "separation", "subsidiary"],
    "merger":           ["acquisition", "takeover", "consolidation", "deal"],
    "acquisition":      ["merger", "buyout", "takeover", "purchase"],
    "dividend":         ["yield", "income", "payout", "distribution", "reinvest"],
    "buyback":          ["share", "repurchase", "return", "capital", "shareholders"],

    # ── Quantitative finance ──────────────────────────────────────────
    "quant":            ["quantitative", "algorithmic", "systematic", "model"],
    "backtesting":      ["strategy", "historical", "performance", "simulation"],
    "montecarlo":       ["simulation", "risk", "probability", "model"],
    "regression":       ["analysis", "statistical", "factor", "model"],
    "timeseries":       ["data", "analysis", "forecasting", "statistical"],
    "ml":               ["machine", "learning", "algorithm", "prediction", "model"],
    "hft":              ["high", "frequency", "trading", "algorithm", "speed"],
    "factor":           ["investing", "value", "momentum", "quality", "model"],
    "capm":             ["capital", "asset", "pricing", "model", "beta", "risk"],
    "mpt":              ["modern", "portfolio", "theory", "efficient", "frontier"],
}


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, remove stop words, add bigrams."""
    text = text.lower()
    words = re.findall(r"[a-z0-9]+(?:['\-][a-z]+)*", text)
    unigrams = [t for t in words if t not in _STOPWORDS and len(t) > 1]
    # Bigrams improve matching of compound terms like "credit score", "stock market"
    bigrams = [f"{unigrams[i]}_{unigrams[i+1]}" for i in range(len(unigrams) - 1)]
    return unigrams + bigrams


def _normalize_query(query: str) -> str:
    """Map informal phrasings to canonical forms that match training data."""
    q = query.strip()
    for pattern, replacement in _REPHRASE_RULES:
        q = re.sub(pattern, replacement, q, flags=re.IGNORECASE)
    return re.sub(r" {2,}", " ", q).strip()


# ── Cache helpers ───────────────────────────────────────────────────────

def _dir_fingerprint(data_dir: str) -> str:
    """MD5 of all CSV filenames + mtimes + sizes.  Changes whenever any file does."""
    h = hashlib.md5()
    h.update(str(_CACHE_VERSION).encode())
    if not os.path.isdir(data_dir):
        return h.hexdigest()
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".csv"):
            path = os.path.join(data_dir, fname)
            stat = os.stat(path)
            h.update(f"{fname}:{stat.st_mtime:.1f}:{stat.st_size}".encode())
    return h.hexdigest()


# ── Main class ──────────────────────────────────────────────────────────

class KnowledgeBase:
    """
    TF-IDF index over all CSV Q&A pairs with cosine-similarity search.

    The inverted index means search time is proportional to the number of
    documents that share *at least one token* with the query, not the total
    number of documents — a 10-50× speedup for typical queries at scale.
    """

    def __init__(self, data_dir: str = "data", cache_path: str | None = None):
        self.qa_pairs:  list[tuple[str, str, str]] = []
        self.idf:       dict[str, float] = {}
        self.doc_tfidf: list[dict[str, float]] = []
        self._inv_idx:  dict[str, list[tuple[int, float]]] = {}

        cache_path = cache_path or os.path.join("checkpoints", "kb_cache.pkl")

        if self._load_cache(data_dir, cache_path):
            return  # fast path — everything loaded from disk

        self._load_all_csvs(data_dir)
        self._build_index()
        self._save_cache(data_dir, cache_path)

    # ── Cache I/O ─────────────────────────────────────────────────────

    def _load_cache(self, data_dir: str, cache_path: str) -> bool:
        try:
            if not os.path.exists(cache_path):
                return False
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            if cache.get("fingerprint") != _dir_fingerprint(data_dir):
                return False
            self.qa_pairs  = cache["qa_pairs"]
            self.idf       = cache["idf"]
            self.doc_tfidf = cache["doc_tfidf"]
            self._inv_idx  = cache["inv_idx"]
            return True
        except Exception:
            return False

    def _save_cache(self, data_dir: str, cache_path: str) -> None:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "fingerprint": _dir_fingerprint(data_dir),
                        "qa_pairs":    self.qa_pairs,
                        "idf":         self.idf,
                        "doc_tfidf":   self.doc_tfidf,
                        "inv_idx":     self._inv_idx,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        except Exception:
            pass  # caching is best-effort

    # ── CSV loading ───────────────────────────────────────────────────

    def _load_all_csvs(self, data_dir: str) -> None:
        if not os.path.isdir(data_dir):
            return
        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(".csv"):
                continue
            source = fname[:-4]
            fpath  = os.path.join(data_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        q_key = next(
                            (k for k in row if k.strip().lower() in _Q_COLS), None
                        )
                        a_key = next(
                            (k for k in row if k.strip().lower() in _A_COLS), None
                        )
                        if q_key and a_key:
                            q = row[q_key].strip()
                            a = row[a_key].strip()
                            if q and a:
                                self.qa_pairs.append((q, a, source))
            except Exception:
                pass

    # ── Index building ────────────────────────────────────────────────

    def _build_index(self) -> None:
        N = len(self.qa_pairs)
        if N == 0:
            return

        df: dict[str, int] = defaultdict(int)
        all_tf: list[dict[str, float]] = []

        for q, a, _ in self.qa_pairs:
            tokens  = _tokenize(q + " " + a)
            raw_tf: dict[str, int] = defaultdict(int)
            for t in tokens:
                raw_tf[t] += 1
            tf = {t: math.log(1.0 + c) for t, c in raw_tf.items()}
            all_tf.append(tf)
            for t in raw_tf:
                df[t] += 1

        # Smoothed IDF
        self.idf = {
            t: math.log((N + 1) / (count + 1)) + 1.0
            for t, count in df.items()
        }

        # Normalised TF-IDF vectors
        self.doc_tfidf = []
        for tf in all_tf:
            vec  = {t: tf[t] * self.idf.get(t, 0.0) for t in tf}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self.doc_tfidf.append({t: v / norm for t, v in vec.items()})

        # Inverted index: term → [(doc_id, tfidf_value), ...]
        inv: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for doc_id, vec in enumerate(self.doc_tfidf):
            for term, val in vec.items():
                inv[term].append((doc_id, val))
        self._inv_idx = dict(inv)

    # ── Search ────────────────────────────────────────────────────────

    def _query_vec(self, query: str) -> dict[str, float]:
        """Build a normalised TF-IDF vector for an arbitrary query string.
        Expands tokens with finance synonyms so different phrasings hit the same docs.
        Synonym tokens get half weight so they assist but don't dominate.
        """
        tokens = _tokenize(query)
        if not tokens:
            return {}
        raw: dict[str, float] = defaultdict(float)
        for t in tokens:
            raw[t] += 1.0
            # Expand with synonyms at half weight
            for syn in _SYNONYMS.get(t, []):
                for syn_tok in _tokenize(syn):
                    raw[syn_tok] += 0.5
        vec = {}
        for t, c in raw.items():
            idf_val = self.idf.get(t, 0.0)
            if idf_val > 0:
                vec[t] = math.log(1.0 + c) * idf_val
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Return top_k most relevant Q&A pairs for *query*.

        Uses the inverted index to skip documents that share no terms with the
        query, then optionally boosts exact / substring question matches.
        Both the original query and a normalised variant are searched and merged.
        """
        if not self.qa_pairs:
            return []

        # Build score maps for original query and its normalised form
        q_vec      = self._query_vec(query)
        q_norm_str = _normalize_query(query)
        q_vec_norm = self._query_vec(q_norm_str) if q_norm_str != query else q_vec

        candidate_scores: dict[int, float] = {}

        for vec in (q_vec, q_vec_norm):
            for t, q_val in vec.items():
                for doc_id, doc_val in self._inv_idx.get(t, []):
                    candidate_scores[doc_id] = (
                        candidate_scores.get(doc_id, 0.0) + q_val * doc_val
                    )

        if not candidate_scores:
            return []

        # Exact / substring question-match boost
        query_canon = query.lower().strip().rstrip("?").strip()
        norm_canon  = q_norm_str.lower().strip().rstrip("?").strip()

        for doc_id in list(candidate_scores):
            stored_q = self.qa_pairs[doc_id][0].lower().strip().rstrip("?").strip()
            if stored_q in (query_canon, norm_canon):
                candidate_scores[doc_id] += 0.5
            elif (
                query_canon in stored_q
                or stored_q in query_canon
                or norm_canon in stored_q
                or stored_q in norm_canon
            ):
                candidate_scores[doc_id] += 0.15

        sorted_docs = sorted(candidate_scores.items(), key=lambda x: -x[1])

        results = []
        for doc_id, score in sorted_docs[:top_k]:
            if score <= 0.0:
                break
            q, a, source = self.qa_pairs[doc_id]
            results.append(
                {
                    "question": q,
                    "answer":   a,
                    "source":   source,
                    "score":    round(score, 4),
                }
            )
        return results

    def __len__(self) -> int:
        return len(self.qa_pairs)
