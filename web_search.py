"""
FinanceGPT — Web Search
========================
Falls back to DuckDuckGo (free, no API key) when the KB has no good match.
Always searches with "finance" context appended so results stay relevant.
Filters out results that have no finance relevance before returning them.
"""

# Confidence threshold — trigger web search when best KB score is below this
WEB_SEARCH_THRESHOLD = 0.18

# Finance-related keywords — used to check if query/results are on-topic
_FINANCE_KEYWORDS = {
    "stock", "stocks", "share", "shares", "market", "invest", "investing",
    "investment", "portfolio", "fund", "funds", "etf", "bond", "bonds",
    "crypto", "bitcoin", "ethereum", "trading", "trade", "price", "equity",
    "dividend", "yield", "return", "profit", "loss", "revenue", "earnings",
    "valuation", "ipo", "forex", "currency", "interest", "rate", "inflation",
    "recession", "gdp", "economy", "economic", "financial", "finance",
    "bank", "banking", "loan", "debt", "credit", "mortgage", "tax", "taxes",
    "budget", "saving", "savings", "retirement", "pension", "hedge",
    "asset", "assets", "liability", "balance", "income", "expense",
    "nasdaq", "nyse", "s&p", "dow", "index", "indices", "commodity",
    "oil", "gold", "silver", "real estate", "reit", "options", "futures",
    "derivative", "capital", "venture", "startup", "vc", "private equity",
    "quarterly", "annual", "fiscal", "ebitda", "pe ratio", "sharpe",
    "volatility", "risk", "hedge fund", "mutual fund", "roi", "net worth",
    "wealth", "cash flow", "liquidity", "solvency", "exchange",
}

# Clearly off-topic patterns — no point searching with finance context
_OFF_TOPIC_PATTERNS = [
    r"\b(movie|film|actor|actress|character|superhero|marvel|dc|anime|game|gaming)\b",
    r"\b(recipe|cook|food|restaurant|travel|tourism|sport|athlete|team|league)\b",
    r"\b(weather|temperature|rain|sun|forecast)\b",
    r"\b(celebrity|singer|band|album|song|music|concert)\b",
    r"\b(ironman|iron man|superman|batman|spiderman|avengers)\b",
]


def _is_finance_query(query: str) -> bool:
    """Return True if the query has any finance-related intent."""
    q = query.lower()
    return any(kw in q for kw in _FINANCE_KEYWORDS)


def _is_off_topic(query: str) -> bool:
    """Return True if the query is clearly outside finance scope."""
    import re
    q = query.lower()
    return any(re.search(p, q) for p in _OFF_TOPIC_PATTERNS)


def _results_are_finance_relevant(results: list[dict]) -> bool:
    """Return True if at least one result snippet contains finance keywords."""
    for r in results:
        text = (r.get("title", "") + " " + r.get("snippet", "")).lower()
        if any(kw in text for kw in _FINANCE_KEYWORDS):
            return True
    return False


def web_search(query: str, max_results: int = 4) -> list[dict]:
    """
    Search DuckDuckGo with finance context appended.
    Returns empty list if results aren't finance-relevant.
    """
    try:
        from ddgs import DDGS

        # Always search with finance context to keep results on-topic
        finance_query = f"{query} finance investing"

        results = []
        with DDGS() as ddg:
            for r in ddg.text(finance_query, max_results=max_results):
                results.append({
                    "title":   r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url":     r.get("href", ""),
                })

        # Only return results if they're actually finance-related
        if results and _results_are_finance_relevant(results):
            return results
        return []

    except ImportError:
        return []
    except Exception:
        return []


def format_web_results(results: list[dict]) -> str:
    """Format web results into a context block for the model prompt."""
    if not results:
        return ""
    lines = ["[Web Search Results]"]
    for i, r in enumerate(results, 1):
        title   = r.get("title", "").strip()
        snippet = r.get("snippet", "").strip()
        if snippet:
            lines.append(f"{i}. {title}\n   {snippet}")
    return "\n".join(lines)


def format_web_answer(results: list[dict]) -> str:
    """
    Format web results as a clean readable answer — not raw bullet dumps.
    Picks the best non-empty snippets and writes them as flowing text.
    """
    if not results:
        return ""
    snippets = []
    sources  = []
    for r in results[:3]:
        snippet = r.get("snippet", "").strip()
        title   = r.get("title", "").strip()
        url     = r.get("url", "").strip()
        if snippet and len(snippet) > 30:
            snippets.append(snippet)
            if title:
                sources.append(title)
    if not snippets:
        return ""
    # Join snippets into a coherent block
    body = " ".join(snippets)
    # Trim to ~600 chars at a sentence boundary
    if len(body) > 600:
        cut = body.rfind(".", 0, 600)
        body = body[: cut + 1] if cut > 100 else body[:600] + "..."
    source_line = f"\n[Source: Web — {sources[0]}]" if sources else "\n[Source: Web Search]"
    return body + source_line


# Time-sensitive keywords — user wants live/current info, always search
_TIME_SENSITIVE = [
    r"\b(latest|current|right now|today|this week|this month|this year|recently|recent|now)\b",
    r"\b(right now|as of|at the moment|currently|what('s| is) happening)\b",
    r"\b(2024|2025|2026)\b",
]


def _is_time_sensitive(query: str) -> bool:
    """Return True if the query asks for current/live information."""
    import re
    q = query.lower()
    return any(re.search(p, q) for p in _TIME_SENSITIVE)


def needs_web_search(kb_results: list[dict], query: str = "") -> bool:
    """
    Return True if web search should fire.
    Rules (in order):
      1. Off-topic query → never search
      2. KB has no results → always search
      3. Time-sensitive query AND KB score < 0.28 → search (KB answer is probably stale/generic)
      4. KB score below threshold → search
      If KB has a strong answer (score >= 0.28) always prefer it, even for time-sensitive queries.
    """
    if query and _is_off_topic(query):
        return False
    if not kb_results:
        return True
    best_score = kb_results[0]["score"]
    # Time-sensitive: only search if KB doesn't have a solid specific answer
    if query and _is_time_sensitive(query) and best_score < 0.28:
        return True
    return best_score < WEB_SEARCH_THRESHOLD
