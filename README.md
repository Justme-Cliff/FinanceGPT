# FinanceGPT

> A from-scratch finance AI with a 4-agent parallel reasoning system, persistent conversation memory, and a TF-IDF knowledge base — built entirely without external AI APIs.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Topics](https://img.shields.io/badge/Topics-31_Finance_Domains-orange)
![Parameters](https://img.shields.io/badge/Model-~10M_Parameters-purple)

---

## What Is This?

FinanceGPT is a complete, self-contained finance language model trained entirely from your own CSV data. It combines:

- A **custom GPT-style transformer** (RoPE, RMSNorm, SwiGLU activations)
- A **custom BPE tokenizer** trained from scratch on finance text
- A **4-agent parallel reasoning system** that retrieves knowledge, computes formulas, scaffolds chain-of-thought, and generates answers simultaneously
- A **TF-IDF knowledge base** that indexes 1,200+ Q&A pairs from 31 finance topics
- **Persistent conversation memory** that survives restarts

No OpenAI. No Anthropic. No external AI APIs. Everything runs on your machine.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FinanceGPT System                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   User Query                                                      │
│       │                                                           │
│       ├──────────────── Phase 1: Parallel ─────────────────────┐ │
│       │                                                         │ │
│       │   ┌─────────────────────┐  ┌──────────────────────┐   │ │
│       │   │   KnowledgeAgent    │  │  CalculationAgent    │   │ │
│       │   │  TF-IDF search over │  │  Detects financial   │   │ │
│       │   │  1,200+ Q&A pairs   │  │  formulas & computes │   │ │
│       │   │  from 31 CSV files  │  │  Sharpe, ROI, PV...  │   │ │
│       │   └──────────┬──────────┘  └──────────┬───────────┘   │ │
│       │              │                         │               │ │
│       └──────────────┴─────────────────────────┘               │ │
│                             │                                   │ │
│       ┌─────────────────────▼─────────────────────────────┐    │ │
│       │              ReasoningAgent                        │    │ │
│       │   Classifies question type (definition/calc/etc.)  │    │ │
│       │   Decomposes compound questions                    │    │ │
│       │   Builds chain-of-thought context + scaffold       │    │ │
│       └─────────────────────┬─────────────────────────────┘    │ │
│                             │                                   │ │
│       ┌─────────────────────▼─────────────────────────────┐    │ │
│       │                 ModelAgent                         │    │ │
│       │   FinanceGPT transformer generates response        │    │ │
│       │   using enriched, reasoned context                 │    │ │
│       └─────────────────────┬─────────────────────────────┘    │ │
│                             │                                   │ │
│       ┌─────────────────────▼─────────────────────────────┐    │ │
│       │               Synthesis Layer                      │    │ │
│       │   Combines calculations + model output + sources   │    │ │
│       └─────────────────────┬─────────────────────────────┘    │ │
│                             │                                   │ │
│                      Final Response                             │ │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Model Architecture

```
Input Tokens
     │
     ▼
┌─────────────┐
│  Embedding  │  d_model = 384
└──────┬──────┘
       │
       ▼
┌─────────────┐ ×8
│    Block    │  Pre-norm (RMSNorm → Attention → Residual → RMSNorm → MLP → Residual)
│  ┌────────┐ │
│  │  RoPE  │ │  Rotary Position Embedding — better long-range generalization
│  │  Attn  │ │  8 heads, causal mask
│  └────────┘ │
│  ┌────────┐ │
│  │ SwiGLU │ │  Gated activation: silu(gate(x)) * up(x)  →  down
│  │  MLP   │ │  d_ff = 1536 (4× d_model)
│  └────────┘ │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   RMSNorm   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   LM Head   │  Weight-tied with embedding
└─────────────┘
```

---

## Features

- **Reasoning, not recall** — answers follow `Step 1 → Step 2 → Therefore` patterns learned from training data
- **4 agents, parallel execution** — Knowledge + Calculation run simultaneously; no sequential bottleneck
- **Live financial math** — automatically detects and computes Sharpe Ratio, ROI, Compound Interest, Present Value, EV/EBITDA
- **Persistent memory** — conversation history saved to JSON, reloaded across sessions
- **31 finance domains** — from quantitative finance to ESG, hedge funds to startup finance
- **1,200+ curated Q&A pairs** — chain-of-thought style answers throughout
- **Zero dependencies on AI APIs** — runs fully offline after initial setup
- **GPU + CPU support** — uses `torch.autocast` for mixed precision on CUDA

---

## Installation

### Requirements

- Python 3.10+
- pip

### Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/financegpt.git
cd financegpt

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
pandas>=2.0.0
tqdm>=4.65.0
colorama>=0.4.6
```

---

## Usage

### Step 1 — Train

```bash
python main.py /train
```

Trains the transformer on all 31 CSV datasets. On first run this:
1. Builds the BPE tokenizer vocabulary from all finance text
2. Trains for up to 15 epochs with early stopping
3. Saves the model checkpoint and tokenizer to `checkpoints/`
4. Generates training plots in `training_plots/`

**Training time:** ~30–90 min on CPU, ~5–15 min on GPU

> To train on a single topic for quick testing:
> ```bash
> python main.py /train data/quantitative_finance.csv
> ```

### Step 2 — Chat

```bash
python main.py /chat
```

Starts the multi-agent chat interface. On startup, all 4 agents are initialized:

```
  [1/5] Tokenizer …  ✓  (12,847 tokens)
  [2/5] Model …      ✓  (10.2M params, device=cpu)
  [3/5] Knowledge base … ✓  (1,266 Q&A pairs indexed)
  [4/5] Reasoning engine … ✓
  [5/5] Spawning 4 agents … ✓  [Knowledge | Calculation | Reasoning | Model]
```

### Chat Commands

| Command | Description |
|---|---|
| `/agents` | Show what each agent did on the last query (timing, results) |
| `/history` | Show recent conversation history |
| `/reset` | Clear current session memory |
| `/clear` | Clear the terminal screen |
| `/info` | Model stats, vocab size, knowledge base size |
| `/help` | Show all commands |
| `exit` | Quit |

### Check Model Info

```bash
python main.py /info
```

---

## Example Conversation

```
  You ► What is the Sharpe ratio and calculate it for a fund with
        15% return, 3% risk-free rate, and 10% standard deviation?

  ⟳ Agents thinking…

  FinanceGPT ►
  **Calculated:**
    Sharpe Ratio: Sharpe = (15% - 3%) / 10% = 1.2000

  The Sharpe ratio measures risk-adjusted return. Step 1: Take the
  portfolio's excess return above the risk-free rate: 15% - 3% = 12%.
  Step 2: Divide by the portfolio's standard deviation (volatility):
  12% / 10% = 1.2. Step 3: Interpret the result — a Sharpe ratio of
  1.0 is considered good; 1.2 is above average, indicating the fund
  earns 1.2% of excess return per 1% of volatility taken. Therefore,
  this fund has solid risk-adjusted performance.

  Sources: Quantitative Finance, Finance Fundamentals | Type: calculation | 1.8s
```

---

## Dataset — 31 Finance Topics

| Category | Topics |
|---|---|
| **Fundamentals** | Finance Fundamentals, Economics, Fundamental Analysis, Financial Ratios |
| **Markets** | Stock Market, Global Markets, Technical Analysis, Behavioral Finance |
| **Fixed Income** | Bonds & Fixed Income, Portfolio Theory |
| **Derivatives** | Options Trading, Quantitative Finance |
| **Strategies** | Investment Strategies, Hedge Funds, Risk Management |
| **Private Markets** | Private Equity, Venture Capital (Startup Finance) |
| **Corporate** | Corporate Finance, Financial Modeling, Financial Crises |
| **Personal** | Personal Finance, Retirement Planning, Tax Strategies, Wealth Management |
| **Alternative** | Real Estate Investing, Commodities Trading, Forex Trading, Cryptocurrency |
| **Modern Finance** | Fintech & Blockchain, ESG & Sustainable Finance, Banking & Finance |

All Q&A pairs use chain-of-thought answer format:
> *"To understand X, let's break it down: Step 1: … Step 2: … Therefore, …"*

This trains the model to reason step-by-step rather than pattern-match to surface answers.

---

## Project Structure

```
financegpt/
│
├── main.py                  # Entry point — /train /chat /info
├── config.py                # All hyperparameters and file paths
├── model.py                 # FinanceGPT transformer (RoPE, RMSNorm, SwiGLU)
├── tokenizer.py             # BPE tokenizer — trained from scratch
├── trainer.py               # Training loop (grad accumulation, mixed precision)
├── data_processor.py        # CSV loading + tokenized sliding-window dataset
│
├── knowledge_base.py        # TF-IDF retrieval over all CSVs
├── reasoning_engine.py      # Question classification + CoT scaffolding
├── agents.py                # 4 parallel agents (ThreadPoolExecutor)
├── conversation_memory.py   # JSON-backed persistent conversation history
├── chat.py                  # Multi-agent chat interface
│
├── data/                    # 31 CSV files — question,answer format
│   ├── quantitative_finance.csv
│   ├── hedge_funds.csv
│   ├── private_equity.csv
│   └── ... (28 more)
│
├── requirements.txt
├── README.md
├── CLAUDE.md                # Context for Claude Code (gitignored)
└── .gitignore
```

> **Note:** `checkpoints/` and `training_plots/` are gitignored. After training, your model lives in `checkpoints/finance_gpt.pt`.

---

## Configuration

All settings are in `config.py`:

```python
MODEL_CONFIG = {
    "d_model": 384,       # embedding dimension
    "n_heads": 8,         # attention heads
    "n_layers": 8,        # transformer blocks
    "d_ff": 1536,         # feed-forward inner dim (4× d_model)
    "max_seq_len": 512,   # context window
    "dropout": 0.10,
}

TRAIN_CONFIG = {
    "epochs": 15,
    "batch_size": 16,
    "grad_accum": 4,      # effective batch size = 64
    "lr": 2e-4,
    "patience": 4,        # early stopping
    "mixed_precision": True,
}

GEN_CONFIG = {
    "temperature": 0.82,
    "top_k": 50,
    "top_p": 0.92,
    "repetition_penalty": 1.3,
    "max_new_tokens": 220,
}
```

---

## How the Reasoning System Works

Unlike a standard chatbot that pattern-matches a query to a memorized answer, FinanceGPT uses a 4-stage pipeline on every query:

### Stage 1 — Retrieve (KnowledgeAgent)
Searches all 1,200+ Q&A pairs using TF-IDF cosine similarity. Returns the top 6 most relevant answers to the query. This gives the model grounded, factual context even for questions it hasn't seen.

### Stage 2 — Calculate (CalculationAgent)
Parses the query for financial calculations. If it finds numbers + a formula pattern (Sharpe Ratio, ROI, Compound Interest, Present Value, EV/EBITDA), it computes the answer directly and prepends it to the response.

### Stage 3 — Reason (ReasoningAgent)
Classifies the question into one of 8 types:

| Type | Trigger | Scaffold |
|---|---|---|
| `calculation` | "calculate", "compute", "how much" | Identify inputs → apply formula → interpret |
| `definition` | "what is", "explain", "define" | Core meaning → components → example |
| `comparison` | "compare", "vs", "difference" | Option A → Option B → when to choose each |
| `causal` | "why", "because", "impact of" | Mechanism → drivers → implications |
| `strategy` | "should I", "best way", "recommend" | Goals → steps → risk considerations |
| `process` | "how does", "how to", "steps" | Setup → execution → outcome |
| `historical` | "what happened", "crisis", "history" | Events → causes → lessons |
| `risk` | "risk", "hedge", "protect" | Identify → quantify → mitigate |

This scaffold is prepended to the prompt so the model knows *how* to structure its answer.

### Stage 4 — Generate (ModelAgent)
The transformer receives the full enriched context: conversation history + retrieved knowledge + reasoning scaffold + any computed calculations. It generates a step-by-step answer grounded in all of the above.

---

## Adding Your Own Data

1. Create a CSV in `data/` with `question,answer` columns
2. Write answers in chain-of-thought format for best results:
   ```
   "What is X?","To understand X: Step 1: ... Step 2: ... Therefore, ..."
   ```
3. Retrain:
   ```bash
   python main.py /train
   ```
   The tokenizer automatically extends its vocabulary to cover new terminology.

---

## Training Output

After training completes, four plots are saved to `training_plots/`:

| Plot | Shows |
|---|---|
| `01_training_loss.png` | Raw loss + smoothed curve + validation loss |
| `02_perplexity.png` | Perplexity over training steps |
| `03_train_vs_val.png` | Train vs. validation loss per epoch |
| `04_learning_rate.png` | Cosine annealing LR schedule |

---

## License

MIT License — free to use, modify, and distribute.

---

## Acknowledgements

Architecture inspired by:
- [GPT-2](https://github.com/openai/gpt-2) — decoder-only transformer design
- [LLaMA](https://github.com/facebookresearch/llama) — RoPE, RMSNorm, SwiGLU activations
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) — clean minimal implementation style
