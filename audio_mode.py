"""
FinanceGPT — Voice Mode
========================
  STT : SpeechRecognition + Google free web API (no key needed)
  TTS : pyttsx3 (100% local, no internet required)

Usage:
  python main.py /audio

Required packages:
  pip install SpeechRecognition pyttsx3 pyaudio

In voice mode:
  - Say your question → model answers → answer is spoken aloud
  - Say "exit" or "quit" to stop
  - Say "clear" to reset session memory
  - Say "portfolio" to hear your portfolio summary
"""

import os
import sys
import time


# ── TTS ──────────────────────────────────────────────────────────────────

def _init_tts():
    try:
        import pyttsx3
        engine = pyttsx3.init()
        # Slightly slower rate for clarity
        engine.setProperty("rate", 165)
        engine.setProperty("volume", 1.0)
        # Try to pick a natural-sounding voice
        voices = engine.getProperty("voices")
        for v in voices:
            if "zira" in v.name.lower() or "david" in v.name.lower():
                engine.setProperty("voice", v.id)
                break
        return engine
    except ImportError:
        print("  [TTS] pyttsx3 not installed. Run: pip install pyttsx3")
        return None
    except Exception as e:
        print(f"  [TTS] Init failed: {e}")
        return None


def speak(engine, text: str):
    """Speak text aloud. Strips markdown/symbols first."""
    if engine is None:
        return
    clean = (text
             .replace("►", "").replace("✓", "").replace("✗", "")
             .replace("  ", " ").replace("\n", ". ").strip())
    try:
        engine.say(clean)
        engine.runAndWait()
    except Exception as e:
        print(f"  [TTS] Error: {e}")


# ── STT ──────────────────────────────────────────────────────────────────

def _init_stt():
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        r.pause_threshold  = 1.0   # seconds of silence before ending phrase
        r.energy_threshold = 300
        r.dynamic_energy_threshold = True
        return r, sr
    except ImportError:
        print("  [STT] SpeechRecognition not installed. Run: pip install SpeechRecognition pyaudio")
        return None, None


def listen(recognizer, sr_module) -> str | None:
    """Listen for one spoken phrase and return text, or None on failure."""
    try:
        with sr_module.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            print("  🎤 Listening…", end="", flush=True)
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=15)
        print("\r" + " " * 20 + "\r", end="")
        text = recognizer.recognize_google(audio)
        return text.strip()
    except sr_module.WaitTimeoutError:
        return None
    except sr_module.UnknownValueError:
        return None
    except sr_module.RequestError as e:
        print(f"\n  [STT] Network error: {e}")
        return None
    except Exception as e:
        print(f"\n  [STT] Error: {e}")
        return None


# ── Main audio loop ──────────────────────────────────────────────────────

def audio_interface():
    from colorama import Fore, Style, init
    init(autoreset=True)

    print(f"\n{Fore.CYAN}  FinanceGPT — Voice Mode{Style.RESET_ALL}")

    # Check model exists
    from config import CHECKPOINT, TOKENIZER
    if not os.path.exists(CHECKPOINT) or not os.path.exists(TOKENIZER):
        print(Fore.RED + "\n  [ERROR] No trained model found. Run: python main.py /train\n")
        sys.exit(1)

    # Init TTS / STT
    tts = _init_tts()
    recognizer, sr = _init_stt()
    if recognizer is None:
        sys.exit(1)

    # Load model stack (reuse chat loader)
    print(f"\n{Fore.CYAN}  Loading model…{Style.RESET_ALL}")
    from config import GEN_CONFIG, DATA_DIR, CONVERSATION_HISTORY
    from model import FinanceGPT
    from tokenizer import BPETokenizer
    from knowledge_base import KnowledgeBase
    from reasoning_engine import ReasoningEngine
    from agents import AgentOrchestrator
    from conversation_memory import ConversationMemory
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = BPETokenizer(); tok.load(TOKENIZER)
    model, _ = FinanceGPT.load(CHECKPOINT)
    model.eval().to(device)
    kb           = KnowledgeBase(DATA_DIR)
    engine       = ReasoningEngine()
    orchestrator = AgentOrchestrator(kb, engine, model, tok, GEN_CONFIG, device)
    memory       = ConversationMemory(CONVERSATION_HISTORY, load_turns=10)

    print(f"{Fore.GREEN}  ✓ Ready! Speak your finance question.{Style.RESET_ALL}")
    print(f"{Fore.WHITE + Style.DIM}  Say 'exit' to quit, 'portfolio' for portfolio, 'clear' to reset.{Style.RESET_ALL}\n")
    speak(tts, "Hello, I am Finance G P T. Ask me anything about finance.")

    while True:
        text = listen(recognizer, sr)

        if text is None:
            continue

        print(f"\n{Fore.YELLOW}  You: {text}{Style.RESET_ALL}")
        cmd = text.lower().strip()

        if cmd in ("exit", "quit", "bye", "stop"):
            speak(tts, "Goodbye! Happy investing.")
            print(f"{Fore.YELLOW}  Goodbye!\n{Style.RESET_ALL}")
            break

        if cmd == "clear":
            memory.clear_session()
            speak(tts, "Session memory cleared.")
            print(f"  {Fore.CYAN}Session cleared.{Style.RESET_ALL}")
            continue

        if any(p in cmd for p in ("what time", "what's the time", "current time",
                                   "what date", "today's date", "what day")):
            from datetime import datetime
            now = datetime.now()
            msg = f"It's {now.strftime('%A, %B %d, %Y')} at {now.strftime('%I:%M %p')}."
            print(f"\n{Fore.GREEN}  FinanceGPT ►{Style.RESET_ALL}\n  {msg}\n")
            speak(tts, msg)
            continue

        if "portfolio" in cmd:
            from portfolio import get_portfolio_summary
            summary = get_portfolio_summary()
            if not summary:
                msg = "Your portfolio is empty. Add holdings with the portfolio command."
            else:
                msg = summary
            print(f"\n{Fore.GREEN}  FinanceGPT ►{Style.RESET_ALL}\n  {msg}\n")
            speak(tts, msg)
            continue

        # Stock quick lookup
        from stock_tools import detect_ticker, get_stock_info, format_stock_summary
        ticker = detect_ticker(text)
        if ticker:
            data = get_stock_info(ticker)
            msg  = format_stock_summary(data)
            print(f"\n{Fore.GREEN}  FinanceGPT ►{Style.RESET_ALL}\n{msg}\n")
            speak(tts, msg)
            continue

        # Full agent pipeline
        print(f"  {Fore.CYAN}⟳ Thinking…{Style.RESET_ALL}", end="", flush=True)
        try:
            history_ctx = memory.get_context(n_turns=5)
            response, _ = orchestrator.process(text, history_ctx)
            print(f"\r{' ' * 20}\r", end="")
            print(f"\n{Fore.GREEN}  FinanceGPT ►{Style.RESET_ALL}")
            # Wrap for display
            import textwrap
            for line in response.split("\n"):
                print("  " + line)
            print()
            speak(tts, response)
            memory.add_turn(text, response)
        except Exception as e:
            print(f"\r{Fore.RED}  [ERROR] {e}{Style.RESET_ALL}\n")
            speak(tts, "Sorry, I ran into an error processing that.")
