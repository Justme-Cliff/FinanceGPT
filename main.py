#!/usr/bin/env python3
"""
FinanceGPT — Entry Point
=========================
  python main.py /train                        train on ALL CSVs in data/
  python main.py /train data/economics.csv     train on one specific CSV
  python main.py /chat                         interactive chat session
  python main.py /audio                        voice mode (STT + TTS)
  python main.py /info                         show model & training stats
  python main.py /fetch stocks                 fetch live stock data → CSV
  python main.py /fetch crypto                 fetch live crypto data → CSV
  python main.py /fetch market                 fetch market indices → CSV
  python main.py /fetch sectors                fetch sector performance → CSV
  python main.py /fetch all                    fetch everything above
  python main.py /portfolio show               show portfolio with live P&L
  python main.py /portfolio add TICKER N PRICE add a holding
  python main.py /portfolio remove TICKER      remove a holding
  python main.py /stock TICKER                 quick live stock lookup
"""
import os
import sys


def usage():
    print(__doc__)
    sys.exit(0)


def cmd_info():
    import json, math
    from config import CHECKPOINT, TOKENIZER
    if not os.path.exists(CHECKPOINT):
        print("  No checkpoint found. Run /train first.")
        return
    import torch
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    hist = ckpt.get("history", {})
    val_ep = hist.get("val_epochs", [])
    print("\n  ── FinanceGPT Info ──────────────────────────────────")
    print(f"  d_model / heads / layers : {cfg['d_model']} / {cfg['n_heads']} / {cfg['n_layers']}")
    print(f"  Vocab size               : {cfg['vocab_size']}")
    print(f"  Context window           : {cfg['max_seq_len']} tokens")
    total_steps = len(hist.get("steps", []))
    print(f"  Total training steps     : {total_steps:,}")
    print(f"  Total epochs logged      : {len(val_ep)}")
    if val_ep:
        best = min(e["val_loss"] for e in val_ep)
        last = val_ep[-1]
        print(f"  Best val loss / ppl      : {best:.4f} / {math.exp(min(best,10)):.2f}")
        print(f"  Last train loss / ppl    : {last['train_loss']:.4f} / {last['train_ppl']:.2f}")
        print(f"  Last val   loss / ppl    : {last['val_loss']:.4f}  / {last['val_ppl']:.2f}")
    print()


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if len(sys.argv) < 2:
        usage()

    cmd = sys.argv[1].lower()

    if cmd == "/train":
        csv_file = sys.argv[2] if len(sys.argv) > 2 else None
        from trainer import train
        train(csv_file)

    elif cmd == "/chat":
        from chat import chat_interface
        chat_interface()

    elif cmd == "/audio":
        from audio_mode import audio_interface
        audio_interface()

    elif cmd == "/info":
        cmd_info()

    elif cmd == "/fetch":
        from fetcher import fetch_cli
        fetch_cli(sys.argv[2:])

    elif cmd == "/portfolio":
        from portfolio import portfolio_cli
        portfolio_cli(sys.argv[2:])

    elif cmd == "/stock":
        if len(sys.argv) < 3:
            print("  Usage: python main.py /stock TICKER")
            sys.exit(1)
        from stock_tools import get_stock_info, format_stock_summary
        ticker = sys.argv[2].upper()
        print(f"\n  Fetching {ticker}…")
        data = get_stock_info(ticker)
        print("\n" + format_stock_summary(data) + "\n")

    elif cmd in ("/agents", "/history", "/reset", "/clear", "/help"):
        print(f"\n  '{sys.argv[1]}' is an in-chat command.")
        print("  Start a chat session first:\n")
        print("    python main.py /chat\n")
        print(f"  Then type {sys.argv[1]} at the prompt.")

    else:
        print(f"  Unknown command: {sys.argv[1]}")
        usage()


if __name__ == "__main__":
    main()
