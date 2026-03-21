<div align="center">

# FinanceGPT

**A fully self-contained finance AI — custom transformer, custom tokenizer, 4-agent reasoning system.**
**No OpenAI. No Anthropic. No Python. No external AI APIs. Pure C.**

[![C](https://img.shields.io/badge/C-11-00599C?style=for-the-badge&logo=c&logoColor=white)]()
[![GCC](https://img.shields.io/badge/GCC-13%2B-F16822?style=for-the-badge)]()
[![AVX2](https://img.shields.io/badge/SIMD-AVX2-8B5CF6?style=for-the-badge)]()
[![OpenMP](https://img.shields.io/badge/OpenMP-multi--core-22C55E?style=for-the-badge)]()
[![OpenBLAS](https://img.shields.io/badge/BLAS-OpenBLAS-F97316?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Parameters](https://img.shields.io/badge/Parameters-~19M-F97316?style=for-the-badge)]()

</div>

---

## What Is FinanceGPT?

FinanceGPT is a finance language model built entirely from scratch in C — no Python, no PyTorch, no external AI libraries. Every component is hand-written: the transformer architecture, the BPE tokenizer, the SIMD math kernel, and the multi-agent reasoning pipeline.

It answers financial questions with structured, step-by-step reasoning grounded in a curated knowledge base of **3,000+ finance Q&A pairs across 50+ domains**. The system combines a neural language model with a BM25 retrieval engine and a rule-based calculation agent for precise numerical answers.

| Component | Implementation |
|---|---|
| Language Model | Decoder-only transformer — 4 layers · d_model=512 · 8 heads · ~19M params |
| Positional Encoding | RoPE (Rotary Position Embedding) — same as LLaMA/Mistral |
| Normalization | RMSNorm (pre-norm architecture) |
| Feed-Forward | SwiGLU activation — gate × up × silu → down projection |
| Math Kernel | Hand-written AVX2 SIMD + tiled GEMM + OpenMP + optional OpenBLAS |
| Tokenizer | Custom BPE trained from scratch on finance text |
| Knowledge Retrieval | BM25 scoring over 3,000+ Q&A pairs with inverted index + synonym expansion |
| Reasoning | 8-type question classifier with chain-of-thought scaffolding |
| Calculations | 18 built-in financial formulas (Sharpe, CAGR, Black-Scholes, mortgage, etc.) |
| Generation | KV cache — O(T) per step instead of O(T²), 10-20× faster inference |
| Memory | JSON-backed conversation history, persistent across sessions |

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
| Startup time | 3–10 seconds (Python import) | < 1 second |
| Binary size | Hundreds of MB | ~2MB |

---

## System Architecture

### Agent Pipeline

```
Your question
     │
     ├── [Agent 1] KnowledgeAgent        ──── parallel ─────┐
     │     BM25 search over 3,000+ Q&A pairs                │
     │     Inverted index + synonym expansion                │
     │     Returns top-6 results with relevance scores       │
     │                                                       │
     ├── [Agent 2] CalculationAgent      ──── parallel ─────┤
     │     Detects numbers in the query                      │
     │     Detects formula keywords (sharpe, cagr, roi...)   │
     │     Computes 18 financial formulas instantly          │
     │                                                       │
     ├── [Agent 3] ReasoningAgent  (uses Phase 1 output)    │
     │     Classifies question into 1 of 8 types            │
     │     Builds chain-of-thought reasoning scaffold        │
     │     Assembles: KB results + calc output + history     │
     │     Formats into structured prompt with User/Assistant│
     │                                                       │
     └── [Agent 4] ModelAgent                               │
           Feeds enriched prompt into transformer           │
           KV cache: processes 1 token/step (not full ctx) │
           Generates up to 280 new tokens                   │
           top-k=50 · top-p=0.92 · temp=0.80 ─────────────┘
                │
                ▼
         Synthesis Layer
         ┌────────────────────────────────────────────┐
         │  KB score ≥ 0.35 → use KB answer directly  │
         │  calc found + model > 30 chars → model      │
         │  KB score ≥ 0.15 + model > 50 → model      │
         │  model ≥ 30 chars → model                   │
         │  KB score > 0.05 → fallback to KB           │
         │  else → "not enough information"             │
         └────────────────────────────────────────────┘
                │
                ▼
          Final Answer
```

### Transformer Architecture

```
Input tokens [T]
     │
     ▼
Embedding [vocab × 512]  ← weight-tied with LM head
     │
     ▼  ×4 layers
┌────────────────────────────────────────┐
│  RMSNorm (pre-norm)                    │
│  Multi-Head Self-Attention             │
│    Q, K, V projections [512 → 512]     │
│    8 heads · d_k = 64 per head         │
│    RoPE applied to Q and K             │
│    Causal mask (lower triangular)      │
│    KV Cache during generation          │
│    Output projection [512 → 512]       │
│  + Residual connection                 │
│                                        │
│  RMSNorm (pre-norm)                    │
│  SwiGLU Feed-Forward Network           │
│    gate:  [512 → 1536] + SiLU         │
│    up:    [512 → 1536]                 │
│    act:   gate_output * up_output      │
│    down:  [1536 → 512]                 │
│  + Residual connection                 │
└────────────────────────────────────────┘
     │
     ▼
RMSNorm → LM Head (weight-tied with embedding) → logits [vocab]
```

### KV Cache — How Generation Became 10-20× Faster

Without KV cache, generating each new token requires re-running the full transformer on the entire growing context. For a 280-token response this means 280 forward passes of increasing cost — O(T²) total work.

With KV cache, K and V matrices are computed **once per token** and stored. Each new generation step only processes the single new token and reads the cached K/V for attention. This is O(T) — constant work per step regardless of context length.

```
Without KV cache (old):
  step 1: forward([prompt])             → logits
  step 2: forward([prompt, tok1])       → logits
  step 3: forward([prompt, tok1, tok2]) → logits
  ...280 full passes, each longer than the last

With KV cache (current):
  prefill: process all prompt tokens → fill K/V cache
  step 1: forward_one(tok1, pos=N)   → logits  (O(N) dot products)
  step 2: forward_one(tok2, pos=N+1) → logits  (O(N+1) dot products)
  ...only the new token processed each step
```

### BM25 Knowledge Retrieval

The knowledge base uses BM25 scoring (the same algorithm powering Elasticsearch and Solr) instead of plain TF-IDF:

```
BM25(t, d) = IDF(t) × (tf × (k1 + 1)) / (tf + k1 × (1 - b + b × |d|/avgdl))
             k1 = 1.5  (term saturation — mentioning a word 10× ≠ 10× better)
             b  = 0.75 (length normalization — short docs not penalized)
```

The index also applies:
- **Finance synonym expansion** — `stocks` → also searches `equities`, `shares`, `equity`
- **Bigram indexing** — `compound_interest` treated as a single term
- **Query normalization** — `what's` → `what is`, `tell me about` → `what is`
- **Exact match boost** — identical question gets 2× score bonus

### Math Module — How It's Fast

**AVX2 SIMD** — 8 floats processed per CPU instruction:
```c
// Core GEMM inner loop — 8× scalar throughput
__m256 va = _mm256_set1_ps(a[k]);
for (j = 0; j <= N-8; j += 8)
    _mm256_storeu_ps(C + i*N + j,
        _mm256_fmadd_ps(va, _mm256_loadu_ps(B + k*N + j),
                            _mm256_loadu_ps(C + i*N + j)));
```

**Tiled GEMM** — TILE=64 keeps the hot working set in L1 cache rather than reading from RAM on every access. For large matrices this is 10-100× faster than naive triple-loop.

**OpenMP** — every matrix multiply parallelized across all CPU cores:
```c
#pragma omp parallel for schedule(static)
for (int i = 0; i < M; i++) { ... }
```

**OpenBLAS** (auto-detected) — if installed, replaces the hand-written GEMM with professionally tuned BLAS routines. Typically 3-5× faster than even the AVX2 fallback.

**Pre-allocated workspace slab** — `model_forward` slices a single pre-allocated buffer instead of calling `malloc/free` 11 times per forward pass. Eliminates allocation overhead entirely during inference.

**Gradient accumulation** — `optimizer_zero_grad()` (which zeros 75MB of gradient arrays) runs only every 8 steps instead of every step. 8× fewer zeroing operations during training.

---

## Model Parameters

```
d_model     = 512     embedding dimension
n_heads     = 8       attention heads  (d_k = 64 per head)
n_layers    = 4       transformer blocks
d_ff        = 1536    FFN hidden size (3× d_model, SwiGLU)
max_seq_len = 512     context window (tokens)
vocab       ≈ 10,000  BPE subword tokens
```

| Component | Parameters |
|---|---|
| Token embedding (weight-tied with LM head) | vocab × 512 ≈ 5.1M |
| QKV projections (4 layers × 3 × 512²) | 3.1M |
| Attention output projections (4 layers) | 1.0M |
| FFN gate + up projections (4 layers × 2) | 6.3M |
| FFN down projections (4 layers) | 3.1M |
| RMSNorm weights (ln1, ln2, ln_f) | ~5K |
| **Total** | **~19M** |

---

## Quick Start

### Requirements

- GCC 11+ (or Clang 13+)
- make
- Linux, macOS, or Windows with WSL2

Optional (auto-detected, improves speed):
- OpenBLAS: `sudo apt install libopenblas-dev`
- OpenMP: included with GCC by default

### Build

```bash
git clone https://github.com/yourusername/financegpt.git
cd financegpt
make
```

The Makefile automatically detects and enables:
- **AVX2** — SIMD intrinsics (8 floats/cycle)
- **OpenMP** — multi-core parallelism
- **OpenBLAS** — professional BLAS (if installed)

Build output example:
```
AVX2: enabled
OpenMP: enabled
OpenBLAS: enabled

  Build complete: ./financegpt
```

### Train

```bash
./financegpt /train
```

What this does:
1. Loads all CSVs from `data/` (50+ files, 3,000+ Q&A pairs)
2. Trains or extends the BPE tokenizer to ~10,000 vocab tokens
3. Tokenizes all training text and builds sliding-window dataset
4. Trains the transformer for up to 8 epochs with cosine LR decay
5. Validates every epoch, saves the best checkpoint automatically
6. Early stopping (patience=3) prevents overfitting

Training progress is shown with Unicode progress bars:
```
  Epoch 1/8
  [████████████████████░░░░░░░░░░░░░░░░] 57%  step 420 | loss 3.241 | lr 1.82e-04
```

Estimated training time on Ryzen 3 5300U (8 threads, OpenBLAS):
- ~15–30 minutes for the full dataset

### Chat

```bash
./financegpt /chat
```

```
  FinanceGPT ready. Type your question or /help

  You: what is the sharpe ratio?

  FinanceGPT: The Sharpe ratio measures risk-adjusted return.
  Step 1: Calculate excess return = Portfolio Return − Risk-Free Rate
  Step 2: Divide by the standard deviation of portfolio returns
  Therefore, Sharpe = (Rp − Rf) / σp. A Sharpe above 1.0 is
  generally considered good; above 2.0 is excellent.
```

---

## Running on Windows (WSL)

```bash
# 1. Open WSL terminal
wsl

# 2. Navigate to the project
cd /mnt/c/Users/yourname/path/to/financegpt

# 3. Install build tools (first time only)
sudo apt update && sudo apt install gcc make libopenblas-dev

# 4. Build
make

# 5. Train the model
./financegpt /train

# 6. Start chatting
./financegpt /chat
```

Or run directly from PowerShell / CMD without entering WSL:
```powershell
wsl.exe -e bash -c "cd /mnt/c/path/to/financegpt && ./financegpt /chat"
```

---

## All Commands

### Terminal Commands

```bash
./financegpt /train                       # train on all CSVs in data/
./financegpt /train data/economics.csv   # fine-tune on one specific file
./financegpt /info                        # show model stats (no training needed)
./financegpt /chat                        # start interactive chat
```

### In-Chat Commands

| Command | Description |
|---|---|
| `/agents` | Per-agent breakdown for last query: KB scores, calc output, question type, timing |
| `/history` | Show recent conversation turns from this session |
| `/reset` | Clear session memory (keeps persistent history file) |
| `/clear` | Clear the terminal screen |
| `/info` | Live stats: model params, vocab size, KB doc count, val loss |
| `/help` | List all available commands |
| `exit` | Quit |

---

## Auto-Calculations (18 Formulas)

The CalculationAgent detects numbers and formula keywords in your question and computes the answer before the model even runs. Results are prepended to the response.

| Formula | Trigger Keywords | Example |
|---|---|---|
| Sharpe Ratio | `sharpe`, `risk-adjusted` | "sharpe ratio with 12% return, 4% risk-free, 8% std" |
| Compound Interest | `compound`, `compounded` | "compound $1000 at 7% for 20 years" |
| Simple Interest | `simple interest` | "simple interest on $5000 at 5% for 3 years" |
| ROI | `roi`, `return on investment` | "roi on $2000 investment worth $2800" |
| Present Value | `present value`, `pv`, `discount` | "pv of $10000 in 5 years at 6%" |
| Mortgage Payment | `mortgage`, `loan payment` | "mortgage on $300000 at 4.5% for 30 years" |
| Savings Goal | `save per month`, `monthly saving` | "save $500/month at 6% for 10 years" |
| Debt Payoff | `debt payoff`, `pay off` | "pay off $15000 debt at 18% with $500/month" |
| CAGR | `cagr`, `compound annual growth` | "cagr from $10000 to $25000 in 8 years" |
| P/E Ratio | `p/e`, `price to earnings` | "p/e with stock at $150, eps $8" |
| P/B Ratio | `price to book`, `p/b ratio` | "p/b with price $45, book value $30" |
| EV/EBITDA | `ev/ebitda`, `enterprise value` | "ev/ebitda with ev $500M, ebitda $80M" |
| Debt-to-Equity | `debt to equity`, `d/e ratio` | "d/e ratio with $200M debt, $400M equity" |
| Current Ratio | `current ratio`, `liquidity ratio` | "current ratio with $500K assets, $300K liabilities" |
| Dividend Yield | `dividend yield` | "dividend yield with $3 annual div, $60 stock price" |
| Break-even | `break even`, `fixed costs` | "break-even with $50000 fixed costs, $25 price, $10 variable" |
| Dollar-Cost Averaging | `dca`, `dollar cost averaging` | "dca $200/month for 24 months at $50/share" |
| Rule of 72 | `rule of 72`, `double my money` | "rule of 72 at 8% interest rate" |
| Real Return | `real return`, `inflation adjusted` | "real return with 10% nominal, 3% inflation" |
| Portfolio Expected Return | `expected return`, `weighted average return` | "expected return 60% stocks at 9%, 40% bonds at 4%" |

---

## Reasoning System

### Question Types (8)

The ReasoningAgent classifies every question into one of 8 types and applies a matching chain-of-thought scaffold to guide the model's answer structure.

| Type | Trigger Words | Reasoning Scaffold Applied |
|---|---|---|
| `calculation` | calculate · compute · how much · formula | Identify inputs → Apply formula → Compute → Interpret |
| `definition` | what is · explain · define · describe | Core meaning → Components → Real-world example |
| `comparison` | compare · vs · difference · better than | Option A details → Option B details → When to use each |
| `causal` | why · reason · cause · effect of · impact | Mechanism → Key drivers → Practical implications |
| `strategy` | should I · best way · recommend · plan | Goals → Constraints → Steps → Risk considerations |
| `process` | how does · how to · steps · walk me through | Setup → Execution → Outcome |
| `historical` | what happened · crisis · crash · history | Events in sequence → Root causes → Lessons learned |
| `risk` | risk · hedge · protect · downside · volatility | Identify risks → Quantify → Mitigation strategies |

### Conversation Context Format

The model receives structured context so follow-up questions work correctly:

```
[Conversation History]
User: What is compound interest?
Assistant: Compound interest is interest earned on both principal and
           accumulated interest. Step 1: ...

[Retrieved Knowledge]
Source: Finance fundamentals
Q: How does compound interest differ from simple interest?
A: Step 1: Simple interest calculates only on principal...
---

=== Reasoning approach ===
This is a comparison question. I will outline each option's key
characteristics, highlight the critical differences...

[Question]
User: How does it compare to simple interest?
Assistant:
```

---

## Configuration

All hyperparameters live in `csrc/config.h` — the single source of truth. Change a value, run `make`, done.

```c
/* ── Model architecture ─────────────────────────────────────────── */
#define D_MODEL       512      /* embedding dimension */
#define N_HEADS       8        /* attention heads (d_k = D_MODEL/N_HEADS = 64) */
#define N_LAYERS      4        /* transformer blocks */
#define D_FF          1536     /* FFN hidden size (3 × D_MODEL, SwiGLU) */
#define MAX_SEQ_LEN   512      /* context window in tokens */

/* ── Training ───────────────────────────────────────────────────── */
#define TRAIN_EPOCHS         8
#define TRAIN_BATCH_SIZE     64
#define TRAIN_GRAD_ACCUM     8      /* gradient accumulation steps */
#define TRAIN_LR             2e-4f  /* peak learning rate */
#define TRAIN_MIN_LR         5e-6f  /* cosine decay floor */
#define TRAIN_WARMUP_STEPS   100    /* linear LR warmup */
#define TRAIN_GRAD_CLIP      1.0f   /* gradient norm clipping */
#define TRAIN_LABEL_SMOOTH   0.05f  /* label smoothing */
#define TRAIN_PATIENCE       3      /* early stopping patience */

/* ── Generation ─────────────────────────────────────────────────── */
#define GEN_TEMPERATURE      0.80f  /* higher = more creative */
#define GEN_TOP_K            50     /* top-k filtering */
#define GEN_TOP_P            0.92f  /* nucleus sampling threshold */
#define GEN_REP_PENALTY      1.3f   /* repetition penalty */
#define GEN_MAX_NEW_TOKENS   280    /* max tokens to generate */

/* ── Knowledge base ─────────────────────────────────────────────── */
#define KB_TOP_K             6      /* results returned per query */
#define KB_MIN_SCORE         0.05f  /* minimum BM25 score to include */
#define KB_DIRECT_SCORE      0.20f  /* threshold for direct KB answer */

/* ── Optimizer (AdamW) ──────────────────────────────────────────── */
#define ADAM_BETA1           0.9f
#define ADAM_BETA2           0.95f
#define ADAM_WEIGHT_DECAY    0.1f
```

---

## Dataset — 50+ Finance Domains

The knowledge base covers over 3,000 curated Q&A pairs across 50+ CSV files. All answers follow chain-of-thought format: `Step 1: ... Step 2: ... Therefore, ...`

| Category | Files |
|---|---|
| **Core Fundamentals** | finance_fundamentals · economics · financial_literacy_basics · financial_ratios |
| **Markets & Analysis** | stock_market · technical_analysis · fundamental_analysis · stock_analysis_fundamentals · global_markets · global_investing |
| **Investment** | investment_strategies · portfolio_theory · advanced_investing · behavioral_finance · financial_psychology |
| **Fixed Income** | bonds_fixed_income · fixed_income_advanced |
| **Derivatives** | options_trading · derivatives_futures · quantitative_finance |
| **Risk & Hedging** | risk_management · hedge_funds |
| **Personal Finance** | personal_finance · everyday_banking · debt_management · home_buying_guide · insurance_basics · career_salary_finance |
| **Retirement & Wealth** | retirement_planning · wealth_management · estate_planning |
| **Tax** | tax_strategies · tax_advanced |
| **Real Estate** | real_estate_investing · real_estate_advanced |
| **Corporate & Startup** | corporate_finance · financial_modeling · startup_entrepreneurship_finance · small_business_finance · private_equity |
| **Macro & Banking** | macroeconomics_advanced · economic_indicators · banking_finance · banking_systems |
| **Alternative Assets** | cryptocurrency · cryptocurrency_advanced · commodities_trading · forex_trading |
| **Modern Finance** | fintech_blockchain · esg_sustainable_finance |
| **Calculations** | calculations_advanced |
| **Conversational** | conversational_questions · conversation · comparisons_finance |

---

## Extending the Knowledge Base

```bash
# 1. Create a new CSV with question,answer headers
cat > data/my_topic.csv << 'EOF'
question,answer
"What is X?","Step 1: X is... Step 2: It works by... Therefore, ..."
EOF

# 2. Retrain — tokenizer extends automatically, KB reindexes at chat startup
./financegpt /train
```

Answer format should always follow chain-of-thought:
```
Step 1: [observation or definition]
Step 2: [reasoning or calculation]
Step 3: [implication or conclusion]
Therefore, [final answer in plain English]
```

---

## Debugging Bad Answers

Run `/agents` after any bad answer to see exactly what each agent returned:

| Signal | Likely Cause | Fix |
|---|---|---|
| KB returned 0 docs | Topic not in any CSV | Add rows to the relevant CSV and retrain |
| KB score < 0.05 | Query wording mismatch | Add more varied phrasings to the CSV |
| KB score 0.05–0.15 | Weak retrieval | Add more specific Q&A pairs on this topic |
| Wrong question type | Trigger word not in classifier | Check `csrc/reasoning.c` keyword tables |
| Answer cuts off mid-sentence | `max_new_tokens` too low | Increase `GEN_MAX_NEW_TOKENS` in `csrc/config.h` |
| Answer is repetitive | `rep_penalty` too low | Increase `GEN_REP_PENALTY` (try 1.4–1.5) |
| Answer is too random | Temperature too high | Lower `GEN_TEMPERATURE` (try 0.6–0.7) |
| Calculation wrong | Numbers not detected | Check `find_numbers()` in `csrc/agents.c` |

---

## Project Structure

```
financegpt/
│
├── csrc/                        # All C source code
│   ├── main.c                   # Entry point — /train /chat /info dispatch
│   ├── config.h                 # All hyperparameters and file paths
│   ├── compat.h                 # Platform portability (AVX2, OpenMP, Windows/POSIX)
│   │
│   ├── math_ops.h / math_ops.c  # AVX2 GEMM · tiled GEMM · RMSNorm · softmax
│   │                            # RoPE precompute/apply · sampling · attention
│   │
│   ├── model.h / model.c        # Transformer: forward, backward, generate
│   │                            # KV cache (model_forward_one, kv_cache_*)
│   │                            # AdamW optimizer · checkpoint save/load
│   │
│   ├── tokenizer.h / tokenizer.c # BPE tokenizer: train, encode, decode
│   │                             # FNV-1a hash map for O(1) token lookup
│   │
│   ├── trainer.h / trainer.c    # Training loop: gradient accumulation
│   │                            # Cosine LR schedule · early stopping
│   │                            # Dataset: sliding-window tokenization
│   │                            # Unicode progress bars
│   │
│   ├── knowledge_base.h / .c    # BM25 retrieval · inverted index
│   │                            # Finance synonym expansion · bigrams
│   │                            # Query normalization · exact match boost
│   │
│   ├── reasoning.h / reasoning.c # 8-type question classifier
│   │                             # Chain-of-thought scaffolds
│   │                             # Context + prompt builder
│   │                             # Conversation history formatter
│   │
│   ├── agents.h / agents.c      # 4-agent orchestration pipeline
│   │                            # 20 financial formula calculations
│   │                            # 6-branch synthesis logic
│   │
│   ├── conversation.h / .c      # JSON-backed persistent chat history
│   │                            # Multi-session management
│   │
│   ├── chat.h / chat.c          # Interactive chat loop · ANSI colors
│   │                            # All in-chat commands (/agents, /info...)
│   │
│   ├── json.h / json.c          # Custom recursive-descent JSON parser + builder
│   ├── csv.h / csv.c            # RFC-4180 CSV parser with flexible column names
│   └── arena.h / arena.c        # Linear arena allocator (O(1) reset)
│
├── data/                        # 50+ CSV files — question,answer format
│   ├── finance_fundamentals.csv
│   ├── stock_market.csv
│   └── ... (48 more)
│
├── export_weights.py            # One-time tool: converts .pt → .bin format
├── Makefile                     # Cross-platform (auto-detects AVX2, OpenMP, OpenBLAS)
├── plan.md                      # Improvement roadmap
├── README.md
└── .gitignore
```

```
checkpoints/                     # gitignored — created after /train
├── finance_gpt.bin              # Model weights (FGPT binary format, ~76MB)
├── tokenizer.json               # BPE vocab (~10K tokens) + merge rules
└── conversation_history.json    # Persistent chat memory across sessions
```

---

## Binary Weight Format

The model saves weights in a custom binary format (`FGPT`):

```
Header:
  magic[4]      = "FGPT"
  version       = 1
  vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len
  n_tensors

Per tensor:
  name_len (uint32) + name (char[name_len])
  n_dims (uint32) + dims (uint32[n_dims])
  n_elems (uint32) + data (float32[n_elems])
```

Named tensors: `embed`, `ln_f`, `blocks.N.ln1_w`, `blocks.N.attn_qkv`, `blocks.N.attn_proj`, `blocks.N.ln2_w`, `blocks.N.ffn_gate`, `blocks.N.ffn_up`, `blocks.N.ffn_down`.

---

## Migrating from a Python Checkpoint

If you previously trained with the Python version and have `checkpoints/finance_gpt.pt`:

```bash
# Convert PyTorch weights to FGPT binary format
python export_weights.py

# Then run as normal
./financegpt /chat
```

`export_weights.py` is the only Python file remaining — it's a one-time migration tool.

---

## Acknowledgements

Architecture inspired by:

- [GPT-2](https://github.com/openai/gpt-2) — decoder-only transformer design
- [LLaMA](https://github.com/facebookresearch/llama) — RoPE, RMSNorm, SwiGLU
- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy — clean minimal transformer implementation

---

## License

MIT License — free to use, modify, and distribute.

---

<div align="center">

Built from scratch. No shortcuts. No APIs. No Python.

</div>
