<div align="center">

# FinanceGPT

**A fully self-contained finance AI — custom transformer, custom tokenizer, 4-agent reasoning system.**
**No OpenAI. No Anthropic. No Python. No external AI APIs. Pure C.**

[![C](https://img.shields.io/badge/C-11-00599C?style=for-the-badge&logo=c&logoColor=white)]()
[![GCC](https://img.shields.io/badge/GCC-13%2B-F16822?style=for-the-badge)]()
[![AVX2](https://img.shields.io/badge/SIMD-AVX2-8B5CF6?style=for-the-badge)]()
[![OpenMP](https://img.shields.io/badge/OpenMP-multi--core-22C55E?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Parameters](https://img.shields.io/badge/Parameters-~19M-F97316?style=for-the-badge)]()

</div>

---

## What Is FinanceGPT?

FinanceGPT is a finance language model built entirely from scratch in C — no Python, no PyTorch, no external AI libraries. Every component is custom-built: the transformer, the BPE tokenizer, the math module, and the multi-agent reasoning pipeline.

It answers financial questions with structured, step-by-step reasoning grounded in a curated knowledge base of 1,200+ finance Q&A pairs across 31 domains.

| Component | Implementation |
|---|---|
| Language Model | Decoder-only transformer — 4 layers · d_model=512 · 8 heads · ~19M params (RoPE · RMSNorm · SwiGLU) |
| Math Module | Hand-written AVX2 SIMD + tiled GEMM + OpenMP — no BLAS, no external math libs |
| Tokenizer | Custom BPE loaded from `tokenizer.json` |
| Knowledge Retrieval | TF-IDF cosine similarity over 1,200+ Q&A pairs, inverted index, finance synonym expansion |
| Reasoning | Rule-based CoT scaffold with 8-type question classifier |
| Memory | JSON-backed conversation history, persistent across sessions |
| Hardware | CPU only — AVX2 SIMD (8 floats/cycle) × OpenMP (multi-core) |

---

## Why C?

| | Python + PyTorch | FinanceGPT C |
|---|---|---|
| Runtime dependencies | Python, PyTorch, NumPy, pip, CUDA drivers | None — one binary |
| Inference speed | ~1–5 tok/s on CPU | ~50–200 tok/s on CPU |
| Memory overhead | 500MB+ Python runtime | ~200MB model weights only |
| Deploy anywhere | Needs Python env | Copy binary + run |
| SIMD | Framework-managed | Direct AVX2 FMA intrinsics |
| Parallelism | GIL-limited | Full OpenMP across all cores |

---

## System Architecture

### Agent Pipeline

```
Your question
     │
     ├── [Agent 1] KnowledgeAgent     ──── parallel ────┐
     │     TF-IDF search over 1,200+ Q&A pairs          │
     │     Returns top-6 results by cosine similarity    │
     │                                                   │
     ├── [Agent 2] CalculationAgent   ──── parallel ────┤
     │     Detects numbers + formula keywords            │
     │     Computes: Sharpe · ROI · Compound · PV ·     │
     │               Mortgage · PE · Simple Interest     │
     │                                                   │
     ├── [Agent 3] ReasoningAgent  (uses Phase 1 output)│
     │     Classifies into 1 of 8 question types        │
     │     Builds chain-of-thought scaffold              │
     │     Assembles context: KB + calc + history        │
     │                                                   │
     └── [Agent 4] ModelAgent                           │
           Feeds enriched prompt into transformer       │
           Generates up to 280 new tokens               │
           top-k=50 · top-p=0.92 · temp=0.80 ──────────┘
```

### Transformer Architecture

```
Input tokens
     │
     ▼
Embedding [vocab × 512]  ← weight-tied with LM head
     │
     ▼  ×4 layers
┌─────────────────────────────────┐
│  RMSNorm                        │
│  Multi-Head Attention           │
│    8 heads · d_k=64             │
│    RoPE positional encoding     │
│    Causal mask                  │
│  + Residual                     │
│  RMSNorm                        │
│  SwiGLU FFN                     │
│    gate: [512→1536] · SiLU      │
│    up:   [512→1536]             │
│    down: [1536→512]             │
│  + Residual                     │
└─────────────────────────────────┘
     │
     ▼
RMSNorm → LM Head (weight-tied) → logits [vocab]
```

### Math Module (how it's fast)

**AVX2 SIMD** — 8 floats processed per CPU instruction:
```c
// Core GEMM inner loop — 8× scalar throughput
__m256 va = _mm256_set1_ps(a);
for (j=0; j <= jlim-8; j+=8)
    _mm256_storeu_ps(C+i*N+j, _mm256_fmadd_ps(va, _mm256_loadu_ps(B+k*N+j),
                                                    _mm256_loadu_ps(C+i*N+j)));
```

**Tiled GEMM** — TILE=64 keeps the working set in L1 cache instead of RAM (10–100× faster for large matrices).

**OpenMP** — every matrix multiply is split across all CPU cores:
```c
OMP_PARALLEL_FOR   // #pragma omp parallel for schedule(static)
for (int i = 0; i < M; i++) { ... }
```

On your machine: `OpenMP: 8 threads` at startup = 8× parallel throughput on every heavy operation.

---

## Parameters

```
d_model    = 512    embedding dimension
n_heads    = 8      attention heads  (d_k = 64)
n_layers   = 4      transformer blocks
d_ff       = 1536   FFN hidden size (3× d_model)
max_seq_len= 128    context window
vocab      ≈ 10,000 BPE tokens
```

| Component | Count |
|---|---|
| Embedding (weight-tied) | ~5.1M |
| QKV projections ×4 | 3.1M |
| Attention output ×4 | 1.0M |
| FFN gate + up ×4 | 6.3M |
| FFN down ×4 | 3.1M |
| RMSNorm weights | ~5k |
| **Total** | **~19M** |

---

## Quick Start

### Requirements

- GCC 11+ (or Clang)
- make
- Linux, macOS, or Windows with WSL

### Build

```bash
git clone https://github.com/yourusername/financegpt.git
cd financegpt
make
```

The Makefile auto-detects AVX2 and OpenMP — no manual flags needed.

### Train

```bash
./financegpt /train
```

Reads all 31 CSVs from `data/`, trains the transformer, saves `checkpoints/finance_gpt.bin`.

### Chat

```bash
./financegpt /chat
```

---

## Running on Windows (WSL)

```bash
# 1. Open WSL
wsl

# 2. Go to the project folder
cd /mnt/c/Users/yourname/path/to/financegpt

# 3. Build
make

# 4. Train
./financegpt /train

# 5. Chat
./financegpt /chat
```

---

## All Commands

### Terminal

```bash
./financegpt /train                      # train on all 31 CSVs
./financegpt /train data/economics.csv  # fine-tune on one topic
./financegpt /chat                       # interactive chat
./financegpt /info                       # model & KB stats (no training needed)
```

### Inside Chat

| Command | Description |
|---|---|
| `/agents` | Per-agent breakdown for last query — KB scores, calc output, question type |
| `/history` | Recent conversation turns |
| `/reset` | Clear session memory |
| `/clear` | Clear terminal screen |
| `/info` | Live model stats, vocab size, KB pair count |
| `/help` | List all commands |
| `exit` | Quit |

---

## Migrating from a Python Checkpoint

If you previously trained with the Python version and have `checkpoints/finance_gpt.pt`:

```bash
# Convert PyTorch weights to C binary format
python export_weights.py

# Then run as normal
./financegpt /chat
```

`export_weights.py` is the only Python file remaining — it's a one-time migration tool.

---

## Configuration

All hyperparameters are in `csrc/config.h` — the single source of truth.

```c
/* Model */
#define D_MODEL       512
#define N_HEADS       8
#define N_LAYERS      4
#define D_FF          1536
#define MAX_SEQ_LEN   128

/* Training */
#define TRAIN_EPOCHS        8
#define TRAIN_BATCH_SIZE    64
#define TRAIN_LR            2e-4f
#define TRAIN_PATIENCE      3      /* early stopping */
#define TRAIN_LABEL_SMOOTH  0.05f

/* Generation */
#define GEN_TEMPERATURE     0.80f
#define GEN_TOP_K           50
#define GEN_TOP_P           0.92f
#define GEN_REP_PENALTY     1.3f
#define GEN_MAX_NEW_TOKENS  280
```

Change a value, run `make`, done. No Python, no pip, no env.

---

## Dataset — 31 Finance Domains

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
| **Alternative** | Real Estate · Commodities · Forex · Cryptocurrency |
| **Modern Finance** | Fintech & Blockchain · ESG & Sustainable Finance · Banking & Finance |

All answers follow chain-of-thought format:
```
"Step 1: ... Step 2: ... Therefore, ..."
```

---

## Extending the Knowledge Base

```bash
# 1. Create data/new_topic.csv with question,answer columns
# 2. Retrain
./financegpt /train
```

The KB reindexes automatically at next chat startup. No code changes required.

---

## Reasoning System

### Question Types (8)

| Type | Trigger Words | Chain-of-Thought Scaffold |
|---|---|---|
| `calculation` | calculate · compute · how much | Identify inputs → Apply formula → Interpret |
| `definition` | what is · explain · define | Core meaning → Components → Example |
| `comparison` | compare · vs · difference | Option A → Option B → When to use each |
| `causal` | why · because · what causes | Mechanism → Drivers → Implications |
| `strategy` | should I · best way · recommend | Goals → Steps → Risk considerations |
| `process` | how does · how to · steps | Setup → Execution → Outcome |
| `historical` | what happened · crisis · history | Events → Causes → Lessons |
| `risk` | risk · hedge · protect | Identify → Quantify → Mitigate |

### Auto-Calculations

| Formula | Keywords |
|---|---|
| Sharpe Ratio | `sharpe`, `risk-adjusted` |
| Compound Interest | `compound`, `compounded` |
| Simple ROI | `roi`, `return on investment` |
| Present Value | `present value`, `pv`, `discount` |
| Mortgage Payment | `mortgage`, `loan payment` |
| Simple Interest | `simple interest` |
| P/E Ratio | `pe ratio`, `price to earnings` |
| Savings Goal | `savings goal`, `save for` |

---

## Debugging

Run `/agents` after any bad answer:

| Signal | Cause | Fix |
|---|---|---|
| KB returned 0 docs | Topic not in any CSV | Add a row to the relevant CSV and retrain |
| KB score < 0.1 | Query wording mismatch | Add more varied phrasings to the CSV |
| Wrong question type | Trigger word missing | Check `csrc/reasoning.c` classifier |
| Answer cuts off | `max_new_tokens` too low | Increase `GEN_MAX_NEW_TOKENS` in `csrc/config.h` |

---

## Project Structure

```
financegpt/
│
├── csrc/                     # All C source code
│   ├── main.c                # Entry point — /train /chat /info
│   ├── config.h              # All hyperparameters and paths
│   ├── compat.h              # Platform/compiler portability (AVX2, OpenMP, Windows/POSIX)
│   │
│   ├── math_ops.h/c          # AVX2 GEMM · RMSNorm · softmax · RoPE · sampling
│   ├── model.h/c             # Transformer: forward pass, backward pass, generate
│   ├── tokenizer.h/c         # BPE tokenizer with FNV-1a hash map
│   ├── trainer.h/c           # AdamW · cosine LR · early stopping · dataset
│   │
│   ├── knowledge_base.h/c    # TF-IDF index · inverted index · cosine similarity
│   ├── reasoning.h/c         # Question classifier · CoT scaffolds · prompt builder
│   ├── agents.h/c            # 4-agent orchestration · formula calculations · synthesis
│   ├── conversation.h/c      # JSON-backed persistent conversation history
│   ├── chat.h/c              # Interactive chat loop · ANSI colors · commands
│   │
│   ├── json.h/c              # Custom recursive-descent JSON parser + builder
│   ├── csv.h/c               # RFC-4180 CSV parser
│   └── arena.h/c             # Linear arena allocator (O(1) reset for inference)
│
├── data/                     # 31 CSV files — question,answer format
│   ├── finance_fundamentals.csv
│   ├── stock_market.csv
│   └── ... (29 more)
│
├── export_weights.py         # One-time tool: converts .pt checkpoint → .bin
├── Makefile                  # Cross-platform build (auto-detects AVX2, OpenMP)
├── README.md
└── .gitignore
```

```
checkpoints/                  # gitignored — created after /train
├── finance_gpt.bin           # Model weights (FGPT binary format)
├── tokenizer.json            # BPE vocab + merge rules
└── conversation_history.json # Persistent chat memory
```

---

## Acknowledgements

Architecture inspired by:

- [GPT-2](https://github.com/openai/gpt-2) — decoder-only transformer design
- [LLaMA](https://github.com/facebookresearch/llama) — RoPE, RMSNorm, SwiGLU
- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy — clean minimal style

---

## License

MIT License — free to use, modify, and distribute.

---

<div align="center">

Built from scratch. No shortcuts. No APIs. No Python.

</div>
