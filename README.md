<div align="center">

<img src="docs/images/banner.svg" alt="FinanceGPT Banner" width="100%"/>

# FinanceGPT

**A fully self-contained finance AI — custom transformer, custom tokenizer, 4-agent reasoning system.**
**No OpenAI. No Anthropic. No external AI APIs. Runs entirely on your machine.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Topics](https://img.shields.io/badge/Finance_Domains-32-F97316?style=for-the-badge)](data/)
[![Parameters](https://img.shields.io/badge/Parameters-~6M-8B5CF6?style=for-the-badge)](model.py)
[![Status](https://img.shields.io/badge/Status-Active-22C55E?style=for-the-badge)]()

</div>

---

## Demo

<div align="center">
<img src="docs/images/demo.svg" alt="FinanceGPT Chat Demo" width="85%"/>
</div>

---

## What Is FinanceGPT?

FinanceGPT is a production-quality finance language model built from the ground up — every component is custom-built, including the neural network architecture, the tokenizer, and the multi-agent reasoning pipeline. It answers financial questions with structured, step-by-step reasoning grounded in a curated knowledge base of 1,200+ finance Q&A pairs across 32 domains.

The system is designed around a key insight: **retrieval + reasoning + generation beats generation alone.** Rather than relying solely on what the model memorised during training, every query passes through a 4-agent pipeline that retrieves relevant knowledge, detects and computes financial formulas, scaffolds chain-of-thought reasoning, and finally generates a grounded response.

| Component | Technology |
|---|---|
| Language Model | Decoder-only transformer — 4 layers · d_model=256 · 8 heads (RoPE · RMSNorm · SwiGLU) |
| Tokenizer | Custom BPE trained from scratch on finance text |
| Knowledge Retrieval | TF-IDF cosine similarity over 1,200+ Q&A pairs |
| Reasoning | Rule-based CoT scaffold with 8-type question classifier |
| Agent Orchestration | `ThreadPoolExecutor` — parallel Phase 1 agents |
| Memory | JSON-backed conversation history, persistent across sessions |
| Live Market Data | `yfinance` — real-time stock, crypto, and index prices (no API key) |
| Web Search Fallback | `duckduckgo-search` — fires when KB confidence is low (no API key) |
| Voice Mode | `pyttsx3` TTS (local) + `SpeechRecognition` STT (free) |
| Hardware | CPU and CUDA (mixed precision via `torch.autocast`) |

---

## System Architecture

### Agent Pipeline

```mermaid
flowchart TD
    A([User Query]) --> B & C

    subgraph Phase1 ["⚡ Phase 1 — Parallel Execution"]
        B["🔍 KnowledgeAgent\nTF-IDF search over 1,200+ Q&A pairs\nReturns top-6 relevant passages"]
        C["🧮 CalculationAgent\nDetects financial formulas in query\nComputes Sharpe · ROI · PV · EV/EBITDA"]
    end

    B --> D
    C --> D

    subgraph Phase2 ["🧠 Phase 2 — Reasoning"]
        D["💡 ReasoningAgent\nClassifies question into 8 types\nDecomposes compound questions\nBuilds chain-of-thought scaffold"]
    end

    D --> E

    subgraph Phase3 ["✨ Phase 3 — Generation"]
        E["🤖 ModelAgent\nFinanceGPT transformer\nGenerates response from enriched context\n(history + knowledge + scaffold + calc)"]
    end

    E --> F(["📋 Synthesis Layer\nCombines all agent outputs"])
    F --> G([Final Response])

    style Phase1 fill:#1e3a5f,stroke:#3b82f6,color:#fff
    style Phase2 fill:#3b1f5e,stroke:#8b5cf6,color:#fff
    style Phase3 fill:#1f3b2e,stroke:#22c55e,color:#fff
```

### Transformer Architecture

```mermaid
flowchart TD
    A["Input Tokens"] --> B

    B["Embedding Layer\nd_model = 256"]
    B --> C

    subgraph Block ["Transformer Block ×4"]
        direction TB
        C1["RMSNorm"] --> C2
        C2["Multi-Head Attention\n8 heads · Causal mask\nRoPE positional encoding"] --> C3
        C3["Residual Add"] --> C4
        C4["RMSNorm"] --> C5
        C5["SwiGLU MLP\nd_ff = 768\nsilu(gate·x) × up(x) → down"] --> C6
        C6["Residual Add"]
    end

    C --> C1
    C6 --> D

    D["RMSNorm"] --> E
    E["LM Head\nWeight-tied with embedding\n→ vocab logits"]

    style Block fill:#1e293b,stroke:#475569,color:#e2e8f0
```

---

## Features

<img src="docs/images/features.svg" alt="Feature Overview" width="100%"/>

- **4-agent parallel reasoning** — Knowledge and Calculation agents run simultaneously with zero sequential bottleneck
- **Live market data** — real-time stock prices, crypto, and indices via `yfinance` (no API key required)
- **Live data fetcher** — `/fetch` pulls live market data, generates a CSV, and auto-trains the model in one shot — no duplicates, fixed filenames
- **Web search fallback** — when a question isn't in the training data, DuckDuckGo (free, no key) fetches live web results and injects them into the model's context automatically
- **Portfolio tracker** — track your holdings locally with live P&L and % gain/loss per position
- **Voice mode** — `/audio` for full hands-free operation: speak your question, hear the answer
- **Live financial math** — automatically detects and computes Sharpe Ratio, ROI, Compound Interest, Present Value, and EV/EBITDA directly from your query
- **Chain-of-thought trained** — every answer in the training data follows `Step 1 → Step 2 → Therefore` format, so the model reasons, not just recalls
- **32 finance domains** — from quantitative finance and derivatives to ESG, startup funding, and financial crises
- **1,200+ curated Q&A pairs** — hand-structured with step-by-step reasoning throughout
- **Persistent conversation memory** — history is saved to JSON and reloaded automatically on next session
- **GPU + CPU support** — mixed precision (`torch.autocast`) on CUDA; graceful CPU fallback
- **Extensible by design** — add a new CSV, retrain, done. The tokenizer and KB update automatically

---

## Installation

### Requirements

- Python **3.10+**
- pip

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/financegpt.git
cd financegpt

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (first-time setup)
python main.py /train

# 4. Start chatting
python main.py /chat
```

### Dependencies

```
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
pandas>=2.0.0
tqdm>=4.65.0
colorama>=0.4.6
yfinance>=0.2.0       # live stock/crypto/index data
pyttsx3>=2.90         # local text-to-speech (voice mode)
SpeechRecognition>=3.10.0  # speech-to-text (voice mode)
pyaudio>=0.2.13       # microphone input (voice mode)
```

> **Windows note:** if `pyaudio` fails to install, run: `pip install pipwin && pipwin install pyaudio`

---

## Usage

### Training

```bash
# Train on all 32 CSV datasets (recommended)
python main.py /train

# Train on a single topic (fast iteration / fine-tuning)
python main.py /train data/quantitative_finance.csv
```

On first run, `/train` will:
1. Build the BPE tokenizer vocabulary from all finance text
2. Train the transformer for up to 25 epochs with cosine LR and early stopping
3. Save model weights and tokenizer to `checkpoints/`
4. Export four training plots to `training_plots/`

**Estimated training time:**

| Hardware | Time |
|---|---|
| CPU (modern) | 30–90 minutes |
| GPU (CUDA) | 5–15 minutes |

### Chatting

```bash
python main.py /chat
```

### Voice Mode

```bash
python main.py /audio
```

Speak your question — the model answers and reads the response aloud. Say `exit` to quit.

### Live Stock Lookup

```bash
# From the terminal
python main.py /stock AAPL
python main.py /stock TSLA

# Inside /chat — also auto-detects stock queries like "how is NVDA doing?"
/stock NVDA
```

### Live Data Fetcher

Pulls real-time market data, generates Q&A training CSVs, and **automatically fine-tunes the model** — all in one command:

```bash
python main.py /fetch stocks    # fetch top 20 S&P stocks → auto-train
python main.py /fetch crypto    # fetch top 10 crypto → auto-train
python main.py /fetch market    # fetch major indices → auto-train
python main.py /fetch sectors   # fetch sector ETFs → auto-train
python main.py /fetch all       # fetch everything → auto-train all

# Fetch only (skip training):
python main.py /fetch stocks --no-train
```

Each topic always writes to a **fixed filename** (`fetched_stocks.csv`, `fetched_crypto.csv`, etc.) — running `/fetch` multiple times overwrites the same file, never creates duplicates.

### Portfolio Tracker

```bash
# Add holdings
python main.py /portfolio add AAPL 10 150.00
python main.py /portfolio add TSLA 5 200.00

# View live P&L
python main.py /portfolio show

# Remove a holding
python main.py /portfolio remove AAPL

# Also available inside /chat:
/portfolio
/portfolio add NVDA 3 500.00
```

### Chat Commands

| Command | Description |
|---|---|
| `/agents` | Show per-agent breakdown for the last query — timing, retrieval results, calculation output |
| `/history` | Show recent conversation memory |
| `/reset` | Clear current session memory |
| `/clear` | Clear the terminal and redraw the welcome screen |
| `/info` | Show model stats: vocab size, parameter count, val loss, perplexity |
| `/stock TICKER` | Live stock price and mini analysis |
| `/portfolio` | Show portfolio with live P&L |
| `/portfolio add T N P` | Add a holding (ticker, shares, buy price) |
| `/portfolio remove T` | Remove a holding |
| `/help` | List all available commands |
| `exit` | Quit |

### Model Info

```bash
python main.py /info
```

Displays model architecture, training stats, best validation loss, and perplexity without starting a chat session.

---

## Example Conversation

<div align="center">
<img src="docs/images/example_sharpe.svg" alt="Example: Sharpe Ratio Calculation" width="80%"/>
</div>

```
────────────────────────────────────────────────────────────────────────────────────
❯ What is the Sharpe ratio and calculate it for a fund
  with 15% return, 3% risk-free rate, and 10% volatility?

  ⟳  Agents thinking…

  FinanceGPT ►
  Calculated:
    Sharpe Ratio = (15% − 3%) / 10% = 1.2000

  The Sharpe ratio measures how much excess return you receive
  per unit of risk taken. Step 1: Calculate excess return above
  the risk-free rate: 15% − 3% = 12%. Step 2: Divide by the
  portfolio's standard deviation (volatility): 12% ÷ 10% = 1.2.
  Step 3: Interpret — a Sharpe ratio above 1.0 is considered good;
  1.2 indicates the fund earns 1.2% of excess return for every 1%
  of volatility it accepts. Therefore, this fund demonstrates solid
  risk-adjusted performance relative to a passive benchmark.

  Sources: Quantitative Finance, Finance Fundamentals  │  type: calculation  │  1.8s
```

---

## Dataset — 31 Finance Domains

<div align="center">
<img src="docs/images/dataset_overview.svg" alt="Dataset Domain Overview" width="80%"/>
</div>

| Category | Domains |
|---|---|
| **Fundamentals** | Finance Fundamentals · Economics · Fundamental Analysis · Financial Ratios |
| **Markets** | Stock Market · Global Markets · Technical Analysis · Behavioral Finance |
| **Fixed Income** | Bonds & Fixed Income · Portfolio Theory |
| **Derivatives** | Options Trading · Quantitative Finance |
| **Strategies** | Investment Strategies · Hedge Funds · Risk Management |
| **Private Markets** | Private Equity · Startup & Entrepreneurship Finance |
| **Corporate** | Corporate Finance · Financial Modeling · Financial Crises |
| **Personal** | Personal Finance · Retirement Planning · Tax Strategies · Wealth Management |
| **Alternative** | Real Estate Investing · Commodities Trading · Forex Trading · Cryptocurrency |
| **Modern Finance** | Fintech & Blockchain · ESG & Sustainable Finance · Banking & Finance |

All Q&A pairs use **chain-of-thought answer format**:

```
"What is X?","To understand X, let's break it down:
Step 1: ...
Step 2: ...
Therefore, ..."
```

This trains the model to reason through problems rather than pattern-match to memorised surface answers.

---

## Reasoning System — Deep Dive

Every query passes through four stages, regardless of complexity.

### Stage 1 — Knowledge Retrieval

`KnowledgeAgent` performs TF-IDF cosine similarity search across all 1,200+ Q&A pairs and returns the top-6 most semantically relevant passages. This gives the model factual grounding even for questions outside its direct training distribution.

### Stage 2 — Financial Calculation

`CalculationAgent` scans the query for numeric inputs and formula keywords. If a supported formula is detected, it computes the answer directly and prepends it to the response — guaranteeing mathematical accuracy regardless of model output.

**Supported formulas:**

| Formula | Keywords Detected |
|---|---|
| Sharpe Ratio | `sharpe`, `risk-adjusted` |
| Compound Interest | `compound interest`, `compound`, `compounded` |
| Simple ROI | `roi`, `return on investment` |
| Present Value | `present value`, `pv`, `discount` |
| EV/EBITDA | `ev/ebitda`, `enterprise value` |

### Stage 3 — Reasoning Scaffold

`ReasoningAgent` classifies the query into one of 8 types and applies a matching chain-of-thought template:

| Type | Trigger Words | Scaffold Structure |
|---|---|---|
| `calculation` | calculate · compute · how much | Identify inputs → Apply formula → Interpret result |
| `definition` | what is · explain · define | Core meaning → Key components → Real-world example |
| `comparison` | compare · vs · difference between | Option A → Option B → When to use each |
| `causal` | why · because · what causes | Mechanism → Driving factors → Implications |
| `strategy` | should I · best way · recommend | Understand goals → Evaluate options → Risk considerations |
| `process` | how does · how to · steps to | Prerequisites → Execution → Expected outcome |
| `historical` | what happened · crisis · history of | Events → Causes → Lessons learned |
| `risk` | risk · hedge · protect · exposure | Identify → Quantify → Mitigate |

### Stage 4 — Generation

`ModelAgent` passes the full enriched context to the FinanceGPT transformer:

```
[conversation history] + [top-6 retrieved passages] + [CoT scaffold] + [computed result if any]
→ transformer generates grounded, structured response
```

---

## Training Plots

Four training plots are saved automatically to `training_plots/` after each run:

| Plot | What It Shows |
|---|---|
| `01_training_loss.png` | Raw step loss + smoothed curve + validation loss overlay |
| `02_perplexity.png` | Model perplexity over training steps |
| `03_train_vs_val.png` | Train vs. validation loss per epoch (overfitting diagnostic) |
| `04_learning_rate.png` | Cosine annealing LR schedule with warmup |

---

## Configuration

All hyperparameters live in `config.py` — the single source of truth.

```python
MODEL_CONFIG = {
    "d_model": 256,       # Embedding dimension
    "n_heads": 8,         # Attention heads  (d_k = 32)
    "n_layers": 4,        # Transformer blocks
    "d_ff": 768,          # FFN inner dim (3× d_model)
    "max_seq_len": 256,   # Context window (tokens)
    "dropout": 0.10,
}

TRAIN_CONFIG = {
    "epochs": 25,
    "batch_size": 32,
    "grad_accum": 2,      # Effective batch = 64
    "lr": 2e-4,
    "min_lr": 5e-6,
    "warmup_steps": 200,
    "grad_clip": 1.0,
    "label_smoothing": 0.05,
    "patience": 8,        # Early stopping patience
    "mixed_precision": True,
}

GEN_CONFIG = {
    "temperature": 0.82,        # Higher = more creative (0.5–1.0)
    "top_k": 50,
    "top_p": 0.92,
    "repetition_penalty": 1.3,  # Higher = less repetition
    "max_new_tokens": 220,
}
```

---

## Extending the Knowledge Base

Adding a new finance topic takes under 5 minutes:

**1. Create a CSV in `data/` with `question,answer` columns:**

```csv
question,answer
"What is X?","To understand X: Step 1: ... Step 2: ... Therefore, ..."
```

**2. Retrain:**

```bash
python main.py /train
```

The BPE tokenizer automatically extends its vocabulary to cover new terminology. The knowledge base reindexes on next chat startup. No code changes required.

---

## Project Structure

```
financegpt/
│
├── main.py                   # Entry point — all commands
├── config.py                 # All hyperparameters and file paths
├── model.py                  # FinanceGPT transformer (RoPE · RMSNorm · SwiGLU)
├── tokenizer.py              # BPE tokenizer — built from scratch
├── trainer.py                # Training loop (grad accum · mixed precision · cosine LR)
├── data_processor.py         # CSV loading + tokenized sliding-window dataset
│
├── knowledge_base.py         # TF-IDF retrieval over all CSVs
├── reasoning_engine.py       # Question classification + CoT scaffolding
├── agents.py                 # 4 parallel agents via ThreadPoolExecutor
├── conversation_memory.py    # JSON-backed persistent conversation history
├── chat.py                   # Multi-agent chat interface
│
├── stock_tools.py            # Live stock/crypto/index lookup via yfinance
├── fetcher.py                # /fetch — pulls live data → generates training CSVs
├── portfolio.py              # Local portfolio tracker with live P&L
├── audio_mode.py             # /audio — voice mode (STT + TTS)
│
├── data/                     # 32 CSV files — question,answer format
│   ├── finance_fundamentals.csv
│   ├── stock_market.csv
│   ├── investment_strategies.csv
│   ├── personal_finance.csv
│   ├── technical_analysis.csv
│   ├── fundamental_analysis.csv
│   ├── options_trading.csv
│   ├── real_estate_investing.csv
│   ├── tax_strategies.csv
│   ├── behavioral_finance.csv
│   ├── global_markets.csv
│   ├── banking_finance.csv
│   ├── portfolio_theory.csv
│   ├── corporate_finance.csv
│   ├── financial_modeling.csv
│   ├── commodities_trading.csv
│   ├── financial_crises.csv
│   ├── wealth_management.csv
│   ├── risk_management.csv
│   ├── forex_trading.csv
│   ├── startup_entrepreneurship_finance.csv
│   ├── financial_ratios.csv
│   ├── hedge_funds.csv
│   ├── retirement_planning.csv
│   ├── bonds_fixed_income.csv
│   ├── cryptocurrency.csv
│   ├── economics.csv
│   ├── esg_sustainable_finance.csv
│   ├── fintech_blockchain.csv
│   ├── private_equity.csv
│   ├── quantitative_finance.csv
│   └── conversation.csv          # casual conversation + greetings
│
├── checkpoints/              # gitignored — created after /train
│   ├── finance_gpt.pt        # Model weights
│   ├── tokenizer.json        # BPE vocab + merges
│   ├── portfolio.json        # Local portfolio holdings
│   └── conversation_history.json  # Persistent chat memory
│
├── training_plots/           # gitignored — PNG charts from each training run
├── docs/images/              # README images (banner, demo GIF, screenshots)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Debugging

**If the model gives a bad or incomplete answer:**

```bash
python main.py /chat
# Ask the question, then run:
/agents
```

The `/agents` output will tell you:

| Signal | Likely Cause | Fix |
|---|---|---|
| KB returned 0 docs | Topic not in any CSV | Add a CSV row for this topic and retrain |
| KB score < 0.1 | Query wording doesn't match training data | Rephrase, or add more varied questions to CSV |
| Question type misclassified | Trigger words not in query | Check `reasoning_engine.py` classifier rules |
| Model output is empty | Context window overflow | Reduce `max_new_tokens` or conversation length |
| Answer cuts off mid-sentence | `max_new_tokens` too low | Increase `GEN_CONFIG["max_new_tokens"]` in `config.py` |

---

## Acknowledgements

Architecture inspired by:

- [GPT-2](https://github.com/openai/gpt-2) — decoder-only transformer design
- [LLaMA](https://github.com/facebookresearch/llama) — RoPE, RMSNorm, and SwiGLU activations
- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy — clean, minimal implementation style

---

## License

MIT License — free to use, modify, and distribute.

---

<div align="center">

Built from scratch. No shortcuts. No APIs.

</div>
