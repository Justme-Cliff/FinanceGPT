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

BANNER = f"""
{Fore.CYAN}  ╔══════════════════════════════════════════════════════════════╗
  ║      FinanceGPT  ·  Multi-Agent Reasoning System              ║
  ║   Knowledge · Reasoning · Calculation · Generation            ║
  ╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}"""

HELP_TEXT = f"""
{Fore.YELLOW}  Commands:{Style.RESET_ALL}
  {Fore.CYAN}/help{Style.RESET_ALL}      show this message
  {Fore.CYAN}/agents{Style.RESET_ALL}    show what each agent did last turn
  {Fore.CYAN}/history{Style.RESET_ALL}   show recent conversation history
  {Fore.CYAN}/reset{Style.RESET_ALL}     clear this session's memory
  {Fore.CYAN}/clear{Style.RESET_ALL}     clear screen
  {Fore.CYAN}/info{Style.RESET_ALL}      model & knowledge base stats
  {Fore.CYAN}exit{Style.RESET_ALL}       quit
"""

WIDTH = 84


def _wrap(text: str, indent: str = "  ") -> str:
    lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip() == "":
            lines.append("")
        else:
            wrapped = textwrap.wrap(paragraph, width=WIDTH - len(indent))
            lines.extend(indent + ln for ln in wrapped)
    return "\n".join(lines)


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

    print(BANNER)
    print()

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

    print(f"\n{Fore.WHITE}  {'─'*62}")
    print(f"  Epochs trained : {epochs_done}")
    if best_val:
        print(f"  Best val ppl   : {math.exp(min(best_val, 10)):.2f}")
    print(f"  Memory turns   : {memory.total_turns} (loaded from disk)")
    print(f"  {'─'*62}")
    print(f"\n{Fore.GREEN}  Ready. Ask anything about finance. /help for commands.{Style.RESET_ALL}\n")

    last_agent_info: dict = {}

    while True:
        try:
            user_in = input(Fore.YELLOW + "  You ► " + Style.RESET_ALL).strip()
        except (EOFError, KeyboardInterrupt):
            print(Fore.YELLOW + "\n\n  Goodbye!\n")
            break

        if not user_in:
            continue

        cmd = user_in.lower().strip()

        # ── Commands ────────────────────────────────────────────────
        if cmd in ("exit", "quit", "bye"):
            print(Fore.YELLOW + "\n  Goodbye!\n")
            break

        if cmd == "/help":
            print(HELP_TEXT)
            continue

        if cmd == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            print(BANNER)
            print()
            continue

        if cmd == "/reset":
            memory.clear_session()
            print(Fore.YELLOW + "  Session memory cleared.\n")
            continue

        if cmd == "/history":
            print(f"\n{Fore.CYAN}  Recent conversation:{Style.RESET_ALL}")
            print(memory.format_recent(5))
            print()
            continue

        if cmd == "/agents":
            if last_agent_info:
                print(f"\n{Fore.CYAN}  Agent activity (last query):{Style.RESET_ALL}")
                for info in last_agent_info.values():
                    print(_agent_summary(info))
                print()
            else:
                print(Fore.YELLOW + "  No queries yet.\n")
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

        # ── Multi-agent query ───────────────────────────────────────
        print(f"{Fore.CYAN}  ⟳ Agents thinking…{Style.RESET_ALL}", end="", flush=True)
        t_start = time.perf_counter()

        try:
            history_ctx = memory.get_context(n_turns=5)
            response, agent_info = orchestrator.process(user_in, history_ctx)
            elapsed = time.perf_counter() - t_start
            last_agent_info = agent_info

            # Clear "thinking" line
            print(f"\r{' ' * 26}\r", end="")

            # Print response
            print(f"\n{Fore.GREEN}  FinanceGPT ►{Style.RESET_ALL}")
            print(_wrap(response))

            # Show source tags
            sources = [
                r["source"].replace("_", " ").title()
                for r in agent_info["knowledge"].get("results", [])[:2]
            ]
            qtype = agent_info["reasoning"].get("question_type", "")
            src_str = f"  Sources: {', '.join(sources)} | " if sources else "  "
            print(f"\n{Fore.CYAN}{src_str}Type: {qtype} | {elapsed:.1f}s{Style.RESET_ALL}\n")

            # Persist turn
            memory.add_turn(user_in, response)

        except Exception as exc:
            print(f"\r{Fore.RED}  [ERROR] {exc}{Style.RESET_ALL}\n")
