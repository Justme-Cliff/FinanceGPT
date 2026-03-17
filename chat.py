"""
FinanceGPT — Multi-Agent Reasoning Chat Interface
==================================================
4 specialized agents run in parallel on every query:
  1. KnowledgeAgent    — TF-IDF retrieval over all CSV Q&A pairs
  2. CalculationAgent  — detects and computes financial formulas
  3. ReasoningAgent    — chain-of-thought decomposition & scaffolding
  4. ModelAgent        — transformer generation with reasoned context

Persistent JSON conversation memory survives restarts.
"""

import math
import os
import sys
import textwrap
import time

import torch
from colorama import Fore, Style, init

from config import CHECKPOINT, TOKENIZER, GEN_CONFIG, DATA_DIR, CONVERSATION_HISTORY
from model import FinanceGPT
from tokenizer import BPETokenizer, SPECIAL
from knowledge_base import KnowledgeBase
from reasoning_engine import ReasoningEngine
from agents import AgentOrchestrator
from conversation_memory import ConversationMemory

# ── UI constants ───────────────────────────────────────────────────────

WIDTH = 84
_W = WIDTH + 2   # full box width including borders

def _make_banner(epochs_done: int = 0, best_ppl: float | None = None,
                 kb_pairs: int = 0, model_params: float = 0.0) -> str:
    """Build the two-panel welcome box dynamically after loading."""
    # left panel content (centred in ~42 chars)
    LP = 42
    RP = _W - LP - 3  # right panel width

    def lc(s: str) -> str:
        """Centre string inside left panel."""
        return s.center(LP)

    ppl_str  = f"{best_ppl:.2f}" if best_ppl else "n/a"
    ep_str   = str(epochs_done)
    kb_str   = f"{kb_pairs:,}" if kb_pairs else "—"
    mp_str   = f"{model_params:.1f}M" if model_params else "—"

    left_lines = [
        "",
        lc("Welcome to  FinanceGPT!"),
        lc(""),
        lc("   _______   "),
        lc("  / $   $ \\  "),
        lc(" | $ $ $ $ | "),
        lc(" |  $   $  | "),
        lc("  \\_______/  "),
        lc("    |   |    "),
        lc("    |___|    "),
        lc(""),
        lc("Multi-Agent Reasoning System"),
        lc(f"{mp_str} params  ·  31 datasets"),
        lc(f"Epochs: {ep_str}  ·  Best ppl: {ppl_str}"),
        lc(f"KB pairs: {kb_str}"),
        "",
    ]

    right_lines = [
        "",
        " Agents",
        " " + "─" * (RP - 2),
        "  ✦  Knowledge    TF-IDF retrieval",
        "  ✦  Calculation  Formula detection",
        "  ✦  Reasoning    Chain-of-thought",
        "  ✦  Model        Transformer gen",
        "",
        " Commands",
        " " + "─" * (RP - 2),
        "  /help    /agents   /history",
        "  /reset   /clear    /info   exit",
    ]

    # pad both panels to same height
    h = max(len(left_lines), len(right_lines))
    left_lines  += [""] * (h - len(left_lines))
    right_lines += [""] * (h - len(right_lines))

    top    = "╭─── FinanceGPT v1.0 " + "─" * (_W - 21) + "╮"
    sep    = "│" + " " * LP + "│" + " " * RP + "│"
    bottom = "╰" + "─" * _W + "╯"

    rows = [top]
    for l, r in zip(left_lines, right_lines):
        l_cell = l.ljust(LP)
        r_cell = r.ljust(RP)
        rows.append(f"│{l_cell}│{r_cell}│")
    rows.append(bottom)

    c = Fore.CYAN
    rst = Style.RESET_ALL
    return c + "\n".join(rows) + rst


HELP_TEXT = f"""
{Fore.YELLOW}  Commands:{Style.RESET_ALL}
  {Fore.CYAN}/help{Style.RESET_ALL}               show this message
  {Fore.CYAN}/agents{Style.RESET_ALL}             per-agent breakdown for last query
  {Fore.CYAN}/history{Style.RESET_ALL}            show recent conversation history
  {Fore.CYAN}/reset{Style.RESET_ALL}              clear session memory
  {Fore.CYAN}/clear{Style.RESET_ALL}              clear screen & restart view
  {Fore.CYAN}/info{Style.RESET_ALL}               model & knowledge base stats
  {Fore.CYAN}/stock TICKER{Style.RESET_ALL}       live stock price & mini analysis
  {Fore.CYAN}/portfolio{Style.RESET_ALL}          show portfolio with live P&L
  {Fore.CYAN}/portfolio add T N P{Style.RESET_ALL} add holding (ticker shares buy_price)
  {Fore.CYAN}/portfolio remove T{Style.RESET_ALL} remove holding
  {Fore.CYAN}exit{Style.RESET_ALL}               quit
"""

_DIVIDER = Fore.WHITE + Style.DIM + "─" * (_W + 2) + Style.RESET_ALL


def _wrap(text: str, indent: str = "  ") -> str:
    lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip() == "":
            lines.append("")
        else:
            wrapped = textwrap.wrap(paragraph, width=WIDTH - len(indent))
            lines.extend(indent + ln for ln in wrapped)
    return "\n".join(lines)


def _prompt() -> str:
    """Simple sandwiched prompt — divider above and below the input line."""
    print(_DIVIDER)
    sys.stdout.write(Fore.YELLOW + "❯ " + Style.RESET_ALL)
    sys.stdout.flush()
    try:
        raw = sys.stdin.readline().rstrip("\n")
    except (EOFError, KeyboardInterrupt):
        raise KeyboardInterrupt
    return raw.strip()


def _agent_summary(info: dict) -> str:
    agent = info.get("agent", "")
    ok = f"{Fore.GREEN}✓{Style.RESET_ALL}" if info.get("success") else f"{Fore.RED}✗{Style.RESET_ALL}"
    t  = f"{info.get('elapsed', 0):.2f}s"

    if "Knowledge" in agent:
        n = len(info.get("results", []))
        detail = f"retrieved {n} docs"
    elif "Calculation" in agent:
        calcs = info.get("calculations")
        detail = f"{len(calcs)} calculations" if calcs else "no math detected"
    elif "Reasoning" in agent:
        detail = f"type={info.get('question_type','?')}"
    elif "Model" in agent:
        r = info.get("response", "")
        detail = f"generated {len(r)} chars" if r else "empty output"
    elif "Web" in agent:
        n = len(info.get("results", []))
        detail = f"{n} results" if info.get("used") else "not triggered (KB had match)"
    else:
        detail = ""

    return f"  {ok} {agent:<20} {detail:<30} ({t})"


# ── Boot ───────────────────────────────────────────────────────────────

def _load_all(gen_config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{Fore.CYAN}  [1/5] Tokenizer …{Style.RESET_ALL}", end="", flush=True)
    tok = BPETokenizer()
    tok.load(TOKENIZER)
    print(f" {Fore.GREEN}✓{Style.RESET_ALL}  ({tok.vocab_size} tokens)")

    print(f"{Fore.CYAN}  [2/5] Model …{Style.RESET_ALL}", end="", flush=True)
    model, ckpt = FinanceGPT.load(CHECKPOINT)
    model.eval().to(device)
    print(f" {Fore.GREEN}✓{Style.RESET_ALL}  ({model.num_params()/1e6:.1f}M params, device={device})")

    print(f"{Fore.CYAN}  [3/5] Knowledge base …{Style.RESET_ALL}", end="", flush=True)
    kb = KnowledgeBase(DATA_DIR)
    print(f" {Fore.GREEN}✓{Style.RESET_ALL}  ({len(kb)} Q&A pairs indexed)")

    print(f"{Fore.CYAN}  [4/5] Reasoning engine …{Style.RESET_ALL}", end="", flush=True)
    engine = ReasoningEngine()
    print(f" {Fore.GREEN}✓{Style.RESET_ALL}")

    print(f"{Fore.CYAN}  [5/5] Spawning 4 agents …{Style.RESET_ALL}", end="", flush=True)
    orchestrator = AgentOrchestrator(kb, engine, model, tok, gen_config, device)
    print(f" {Fore.GREEN}✓{Style.RESET_ALL}  [Knowledge | Calculation | Reasoning | Model]")

    memory = ConversationMemory(CONVERSATION_HISTORY, load_turns=10)

    return model, tok, kb, orchestrator, memory, ckpt


# ── Main chat loop ─────────────────────────────────────────────────────

def chat_interface():
    init(autoreset=True)

    if not os.path.exists(CHECKPOINT) or not os.path.exists(TOKENIZER):
        print(Fore.RED + "\n  [ERROR] No trained model found.")
        print(Fore.YELLOW + "  Run:  python main.py /train\n")
        sys.exit(1)

    # Print a minimal loading header while assets boot
    print(f"\n{Fore.CYAN}  Loading FinanceGPT…{Style.RESET_ALL}\n")

    try:
        model, tok, kb, orchestrator, memory, ckpt = _load_all(GEN_CONFIG)
    except Exception as exc:
        print(Fore.RED + f"\n  [ERROR] {exc}\n")
        sys.exit(1)

    hist = ckpt.get("history", {})
    epochs_done = len(hist.get("val_epochs", []))
    best_val = min(
        (e["val_loss"] for e in hist.get("val_epochs", [])),
        default=None,
    )
    best_ppl = math.exp(min(best_val, 10)) if best_val else None

    # Now render the full welcome banner with real stats
    os.system("cls" if os.name == "nt" else "clear")
    print()
    print(_make_banner(
        epochs_done=epochs_done,
        best_ppl=best_ppl,
        kb_pairs=len(kb),
        model_params=model.num_params() / 1e6,
    ))
    print(f"\n{Fore.WHITE + Style.DIM}  Memory turns loaded: {memory.total_turns}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}  Ready — ask anything about finance.  Type /help for commands.{Style.RESET_ALL}")

    last_agent_info: dict = {}

    # helper: redraws the full welcome screen (used by /clear)
    def _redraw_home():
        os.system("cls" if os.name == "nt" else "clear")
        print()
        print(_make_banner(
            epochs_done=epochs_done,
            best_ppl=best_ppl,
            kb_pairs=len(kb),
            model_params=model.num_params() / 1e6,
        ))
        print(f"\n{Fore.WHITE + Style.DIM}  Memory turns loaded: {memory.total_turns}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}  Ready — ask anything about finance.  Type /help for commands.{Style.RESET_ALL}")

    while True:
        try:
            user_in = _prompt()
        except KeyboardInterrupt:
            print(Fore.YELLOW + "\n\n  Goodbye!\n")
            break

        if not user_in:
            continue

        cmd = user_in.lower().strip()

        # ── Time / Date quick answer ─────────────────────────────────
        if any(p in cmd for p in ("what time", "what's the time", "whats the time",
                                   "current time", "what date", "what's the date",
                                   "whats the date", "today's date", "todays date",
                                   "what day", "what year", "what month")):
            from datetime import datetime
            now = datetime.now()
            msg = f"It's {now.strftime('%A, %B %d, %Y')} — {now.strftime('%I:%M %p')}."
            print(f"\n{Fore.GREEN}  FinanceGPT ►{Style.RESET_ALL}")
            print(f"  {msg}\n")
            memory.add_turn(user_in, msg)
            continue

        # ── Commands ────────────────────────────────────────────────
        if cmd in ("exit", "quit", "bye", "/exit", "/quit"):
            print(_DIVIDER)
            print(Fore.YELLOW + "  Goodbye!\n")
            break

        if cmd == "/help":
            print(HELP_TEXT)
            continue

        if cmd == "/clear":
            _redraw_home()
            continue

        if cmd == "/reset":
            memory.clear_session()
            print(f"\n{Fore.YELLOW}  Session memory cleared.{Style.RESET_ALL}\n")
            continue

        if cmd == "/history":
            print(f"\n{Fore.CYAN}  Recent conversation:{Style.RESET_ALL}")
            print(memory.format_recent(5))
            print()
            continue

        if cmd == "/agents":
            if last_agent_info:
                print(f"\n{Fore.CYAN}  Agent activity (last query):{Style.RESET_ALL}")
                for key, info in last_agent_info.items():
                    if key == "web":
                        info = {**info, "agent": "WebSearchAgent"}
                    print(_agent_summary(info))
                print()
            else:
                print(f"\n{Fore.YELLOW}  No queries yet.{Style.RESET_ALL}\n")
            continue

        if cmd == "/info":
            print(f"\n{Fore.CYAN}  System info:{Style.RESET_ALL}")
            print(f"  Model params   : {model.num_params():,}")
            print(f"  Vocab size     : {tok.vocab_size}")
            print(f"  Context window : {model.config['max_seq_len']} tokens")
            print(f"  KB pairs       : {len(kb)}")
            print(f"  Conv. turns    : {memory.total_turns}")
            print()
            continue

        # ── /stock TICKER ────────────────────────────────────────────
        if cmd.startswith("/stock"):
            parts = user_in.split()
            if len(parts) < 2:
                print(f"\n{Fore.YELLOW}  Usage: /stock TICKER{Style.RESET_ALL}\n")
            else:
                from stock_tools import get_stock_info, format_stock_summary
                ticker = parts[1].upper()
                print(f"\n{Fore.CYAN}  ⟳ Fetching {ticker}…{Style.RESET_ALL}", end="", flush=True)
                data = get_stock_info(ticker)
                print(f"\r{' ' * 30}\r", end="")
                print(f"\n{Fore.GREEN}  FinanceGPT ►{Style.RESET_ALL}")
                print(_wrap(format_stock_summary(data)))
                print()
            continue

        # ── /portfolio ───────────────────────────────────────────────
        if cmd.startswith("/portfolio"):
            from portfolio import portfolio_cli, show_portfolio
            parts = user_in.split()
            if len(parts) == 1:
                print(show_portfolio())
            else:
                portfolio_cli(parts[1:])
            print()
            continue

        # ── Multi-agent query ───────────────────────────────────────

        # Auto-detect company/stock queries and route to right response
        live_prefix = ""
        try:
            from stock_tools import detect_company_query, get_stock_info, format_stock_summary, get_company_overview
            ticker, qtype = detect_company_query(user_in)

            if ticker and qtype == "overview":
                print(f"\n{Fore.CYAN}  ⟳ Looking up {ticker}…{Style.RESET_ALL}", end="", flush=True)
                overview = get_company_overview(ticker)
                print(f"\r{' ' * 30}\r", end="")
                print(f"\n{Fore.GREEN}  FinanceGPT ►{Style.RESET_ALL}")
                print(_wrap(overview))
                print(f"\n{Fore.CYAN + Style.DIM}  Source: Live (yfinance)  │  type: company_overview{Style.RESET_ALL}\n")
                memory.add_turn(user_in, overview)
                continue

            elif qtype == "unknown_company":
                msg = (f"I don't have '{ticker}' in my database. "
                       f"If it's a publicly traded company, try: /stock TICKER\n"
                       f"Or ask me something finance-related and I'll do my best!")
                print(f"\n{Fore.GREEN}  FinanceGPT ►{Style.RESET_ALL}")
                print(_wrap(msg))
                print()
                memory.add_turn(user_in, msg)
                continue

            elif ticker and qtype == "price":
                data = get_stock_info(ticker)
                if "error" not in data:
                    live_prefix = f"[Live data] {format_stock_summary(data)}\n\n"
        except Exception:
            pass

        print(f"\n{Fore.CYAN}  ⟳  Agents thinking…{Style.RESET_ALL}", end="", flush=True)

        t_start = time.perf_counter()

        try:
            history_ctx = memory.get_context(n_turns=5)
            query_with_context = live_prefix + user_in if live_prefix else user_in
            response, agent_info = orchestrator.process(query_with_context, history_ctx)
            # Prepend live data to response if fetched
            if live_prefix:
                response = live_prefix.strip() + "\n\n" + response
            elapsed = time.perf_counter() - t_start
            last_agent_info = agent_info

            # Clear "thinking" line
            print(f"\r{' ' * 30}\r", end="")

            # Print response box
            print(f"\n{Fore.GREEN}  FinanceGPT ►{Style.RESET_ALL}")
            print(_wrap(response))

            # Footer: sources + type + timing
            sources = [
                r["source"].replace("_", " ").title()
                for r in agent_info["knowledge"].get("results", [])[:2]
            ]
            web_used = agent_info.get("web", {}).get("used", False)
            if web_used:
                sources.append("🌐 Web")
            qtype = agent_info["reasoning"].get("question_type", "")
            src_str = f"  Sources: {', '.join(sources)}  │  " if sources else "  "
            print(f"\n{Fore.CYAN + Style.DIM}{src_str}type: {qtype}  │  {elapsed:.1f}s{Style.RESET_ALL}\n")

            # Persist turn
            memory.add_turn(user_in, response)

        except Exception as exc:
            print(f"\r{Fore.RED}  [ERROR] {exc}{Style.RESET_ALL}\n")
