# FinanceGPT — Speed, Accuracy & Quality Plan

**Goal:** Faster inference, more accurate answers, better training, more data, 10 training plots.

**Current state:**
- 4-layer transformer, d_model=512, ~19M params, 512-token context
- KV cache in place (O(T) generation)
- 50+ CSVs, ~3,200 Q&A pairs
- BM25 retrieval, 20 formulas, structured prompts
- Training time: ~1.5–2.5 hours on Ryzen 3 5300U

---

## Priority 1 — 10 Training Plots (was 4)

**What:** The trainer currently generates 4 PNG plots. Expand to 10 for full visibility into training health.

**Current 4 plots:**
1. `01_training_loss.png` — raw loss + smoothed curve
2. `02_perplexity.png` — perplexity over steps
3. `03_train_vs_val.png` — train vs val loss per epoch
4. `04_learning_rate.png` — cosine LR schedule

**6 new plots to add:**

| # | File | What it shows | Why useful |
|---|---|---|---|
| 5 | `05_gradient_norm.png` | Gradient norm per step | Detects exploding/vanishing gradients |
| 6 | `06_token_accuracy.png` | % correct next-token predictions | Direct quality metric beyond loss |
| 7 | `07_loss_per_epoch.png` | Bar chart of train + val loss per epoch | Shows convergence rate at a glance |
| 8 | `08_loss_delta.png` | Improvement Δloss between epochs | Detects when training stalls early |
| 9 | `09_loss_stability.png` | Rolling std of loss (window=50 steps) | Shows if training is noisy or stable |
| 10 | `10_train_val_gap.png` | Overfitting gap = train_loss − val_loss | Early warning for overfitting |

**How to implement:**
- In `csrc/trainer.c`: already tracks `TrainHistory` struct with `losses[]`, `val_epochs[]`, `lrs[]`
- Add `grad_norms[]` array to `TrainHistory` and record per step
- Add `token_accs[]` array — compute as `exp(-loss)` per token or count exact matches
- `generate_plots()` function: add 6 new gnuplot/custom PNG writers after the existing 4
- The PNG writer is already in `trainer.c` — just add 6 more calls with different data

**Files to change:** `csrc/trainer.h` (struct fields), `csrc/trainer.c` (recording + plot generation)

**Effort:** Low — 1 day.

---

## Priority 2 — More + Better Training Data (biggest quality win)

### 2a — Generate 20+ new CSV files

Target: reach 6,000+ total Q&A pairs across 70+ files.

**New CSVs to generate (topics not yet covered or thin):**

| File | Topic | Target Rows |
|---|---|---|
| `options_greeks.csv` | Delta, Gamma, Theta, Vega, Rho explained + calculated | 40 |
| `fixed_income_math.csv` | Duration, convexity, YTM, bond pricing calculations | 40 |
| `portfolio_optimization.csv` | Markowitz, efficient frontier, Sharpe maximization | 40 |
| `financial_statement_analysis.csv` | Balance sheet, income stmt, cash flow analysis | 40 |
| `mergers_acquisitions.csv` | M&A process, LBO, synergies, deal structures | 35 |
| `factor_investing.csv` | Value, momentum, quality, size, low-vol factors | 35 |
| `economic_cycles.csv` | Business cycle phases, sector rotation, recession signals | 35 |
| `central_banking.csv` | Monetary policy tools, QE, rate decisions, Fed | 35 |
| `financial_crises_detailed.csv` | 2008, dot-com, 1929 — mechanisms and lessons | 35 |
| `startup_valuation.csv` | Pre-money/post-money, term sheets, cap tables, dilution | 35 |
| `crypto_defi.csv` | DeFi protocols, yield farming, staking, tokenomics | 35 |
| `insurance_advanced.csv` | Life, liability, annuities, actuarial basics | 30 |
| `commodities_advanced.csv` | Contango, backwardation, commodity cycles, storage | 30 |
| `international_finance.csv` | Balance of payments, FX intervention, carry trade | 30 |
| `esg_metrics.csv` | Carbon credits, ESG scoring, green bonds, impact | 30 |
| `alternative_investments.csv` | Infrastructure, timberland, art, collectibles | 30 |
| `financial_regulations.csv` | Basel III, Dodd-Frank, MiFID, fiduciary duty | 30 |
| `fintech_advanced.csv` | Open banking, BNPL, embedded finance, RegTech | 30 |
| `wealth_planning.csv` | Asset protection, family office, philanthropy, trusts | 30 |
| `income_investing.csv` | Dividend growth, DRIP, covered calls, high-yield | 30 |

**Format rule — every answer MUST follow:**
```
Step 1: [define or identify the key concept]
Step 2: [show the mechanism, formula, or process]
Step 3: [give a concrete numerical example or real-world case]
Therefore, [plain-English summary of the answer]
```

### 2b — Audit and fix existing CSVs

Many existing answers don't follow the chain-of-thought format consistently. The model learns inconsistency and blends styles badly.

**Script to write (Python, one-time):**
```python
# audit_csvs.py — finds answers NOT following Step 1/Step 2/Therefore format
import csv, glob
for path in glob.glob("data/*.csv"):
    with open(path) as f:
        for row in csv.DictReader(f):
            ans = row.get("answer","")
            if "Step 1" not in ans and "Therefore" not in ans:
                print(f"{path}: {row['question'][:60]}")
```

Fix every flagged row before retraining. Format inconsistency is a direct quality hit.

### 2c — Add hard negative examples

Add rows where the answer explicitly says what something is NOT. The model currently over-predicts positive/affirmative answers:

```
"Is crypto a safe long-term investment?",
"Step 1: Cryptocurrency is not a safe haven asset — it has no intrinsic cash flows.
Step 2: Bitcoin has experienced 80%+ drawdowns multiple times (2018, 2022).
Step 3: Volatility of 60-100% annually dwarfs stocks (15-20%) and bonds (5-8%).
Therefore, crypto is high-risk speculation, not a safe long-term investment for most people."
```

### 2d — Add multi-turn conversation examples

Create `conversation_pairs.csv` with follow-up questions that explicitly reference a prior answer:

```
"Given that compound interest doubles money faster than simple interest, how long does it take to triple?",
"Step 1: To triple, we solve: (1 + r)^t = 3. Step 2: Using Rule of 72 extended: t = log(3)/log(1+r)..."
```

This teaches the model how to handle follow-up questions that say "given that..." or "as you mentioned...".

**Effort:** Medium — 2–3 days of CSV generation.

---

## Priority 3 — Flash Attention (biggest training speed win)

**What:** The current attention implementation allocates a full `[n_heads, T, T]` matrix. At T=512:
- Memory: 8 × 512 × 512 × 4 bytes = **8MB per forward pass**
- Reading/writing 8MB to RAM every layer is the main training bottleneck on CPU

Flash Attention computes attention in **tiles** that fit in L1 cache (32KB), never materializing the full T×T matrix.

**Impact on training:** 2–4× faster training for T=512. Memory usage drops from O(T²) to O(T).

**How to implement in C:**

```c
/* Flash Attention — tiled causal attention, tile size = 32 */
#define FA_TILE 32

void flash_attention_forward(float* out, const float* q, const float* k, const float* v,
                              int n_heads, int T, int d_k, int d_model) {
    float scale = 1.0f / sqrtf((float)d_k);
    int n_tiles  = (T + FA_TILE - 1) / FA_TILE;

    /* For each query tile: compute attention only over key tiles that are causal */
    for (int h = 0; h < n_heads; h++) {
        for (int qi = 0; qi < n_tiles; qi++) {
            /* Running softmax statistics per query row (online softmax) */
            float m_prev[FA_TILE], l_prev[FA_TILE];  /* max, normalizer */
            float acc[FA_TILE * d_k];                /* running output */
            /* ... tile-level QK^T, online softmax, V aggregation ... */
        }
    }
}
```

**Key algorithm change:** Use online softmax (Milakov & Gimelshein 2018) — compute max and sum incrementally as you scan key tiles. Never need the full row simultaneously.

**Files to change:** `csrc/math_ops.h/c` — add `flash_attention_forward()`, then update `model.c` to call it instead of `attention_forward()` during training.

**Effort:** Medium-hard — 3 days.

---

## Priority 4 — Int8 Quantization (inference speed)

**What:** Store all weight matrices as `int8_t` instead of `float32`. At inference, dequantize on the fly.

**Impact:**
- Model file: 76MB → 19MB (4× smaller)
- Memory bandwidth: reading 19MB instead of 76MB per generation step
- On Ryzen 3 5300U (40 GB/s bandwidth): saves ~1.4ms per step × 280 steps = ~400ms per response
- Roughly 1.5–2× faster inference on top of the KV cache speedup

**Quality loss:** <1% perplexity increase for a 19M param model. Acceptable.

**Implementation:**

```c
/* Per-tensor quantization stored in model */
typedef struct {
    int8_t* data;    /* quantized weights */
    float   scale;   /* max(abs(w)) / 127 */
    int      n;      /* number of elements */
} QuantTensor;

/* Quantize: find scale, convert */
void quantize_tensor(QuantTensor* qt, const float* w, int n) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) if (fabsf(w[i]) > max_abs) max_abs = fabsf(w[i]);
    qt->scale = max_abs / 127.0f;
    for (int i = 0; i < n; i++) qt->data[i] = (int8_t)roundf(w[i] / qt->scale);
}

/* Dequantize row on the fly during GEMM */
static inline float dequant(int8_t v, float scale) { return (float)v * scale; }
```

**GEMM with int8 weights:**
```c
void matmul_q8_f32(const float* A, const QuantTensor* B, float* C, int M, int K, int N) {
    /* Dequantize one row of B at a time into a float buffer, then multiply */
    float row_buf[K];
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) row_buf[k] = dequant(B->data[n*K+k], B->scale);
        for (int m = 0; m < M; m++) C[m*N+n] = vec_dot_f32(A+m*K, row_buf, K);
    }
}
```

**Files to change:** `csrc/model.h/c` — add `QuantModel` variant, `csrc/math_ops.h/c` — add `matmul_q8_f32`.

Add CLI flag: `./financegpt /quantize` — loads full model, quantizes, saves `finance_gpt_q8.bin`.
Then `/chat` auto-loads q8 if it exists.

**Effort:** Medium — 2–3 days.

---

## Priority 5 — Fused Kernels (free speed, no quality loss)

**What:** Several operations are done in separate passes that could be fused into one. Each extra pass = extra RAM read/write = wasted bandwidth.

### 5a — Fused RMSNorm + Linear

Currently: `rms_norm(xn, x, w)` writes `xn` to RAM, then `matmul(xn, W, out)` reads it back.

```c
/* Fused: normalize x into local register, immediately multiply with W */
void rms_norm_matmul_f32(float* out, const float* x, const float* weight,
                          const float* W, int T, int dm, int out_dim) {
    for (int t = 0; t < T; t++) {
        /* 1. Compute RMS norm in registers */
        float rms = 0.0f;
        for (int i = 0; i < dm; i++) rms += x[t*dm+i] * x[t*dm+i];
        float scale = 1.0f / sqrtf(rms / dm + RMSNORM_EPS);
        /* 2. Immediately use normalized value in dot products — no write to RAM */
        for (int j = 0; j < out_dim; j++) {
            float acc = 0.0f;
            for (int i = 0; i < dm; i++)
                acc += (x[t*dm+i] * scale * weight[i]) * W[j*dm+i];
            out[t*out_dim+j] = acc;
        }
    }
}
```

Apply to: pre-attention norm + QKV projection (fused), pre-FFN norm + gate/up projection (fused).

### 5b — Fused SwiGLU

Currently three separate passes: `silu(gate)`, `mul(gate, up)`, then down projection.

```c
/* Fused SwiGLU: compute gate, up, silu(gate)*up in one pass, feed to down matmul */
void swiglu_fused_f32(float* act, const float* gate, const float* up, int n) {
    /* SIMD: silu(gate) * up, result in act[] — one read of gate+up, one write of act */
}
```

**Files to change:** `csrc/math_ops.h/c` (new fused functions), `csrc/model.c` (call them).

**Effort:** Low-medium — 1–2 days.

---

## Priority 6 — Grouped Query Attention (GQA)

**What:** Instead of 8 independent K/V heads (current), use 2 K/V groups shared across 4 Q heads each. This is how LLaMA 2/3 and Mistral work.

**Impact:**
- KV projection compute: 4× less (2 KV heads vs 8)
- KV cache size: 4× smaller (critical for 512 context + long conversations)
- Quality: small loss, mostly imperceptible for <1B parameter models

**Architecture change:**
```c
/* config.h additions */
#define N_KV_HEADS  2    /* K/V heads — shared across Q heads */
#define N_Q_HEADS   8    /* Q heads — full */
#define N_GROUPS    (N_Q_HEADS / N_KV_HEADS)  /* 4 Q heads per KV group */

/* model.h — block changes */
typedef struct {
    float* attn_q;    /* [d_model, d_model]              — Q projection */
    float* attn_kv;   /* [2 * N_KV_HEADS * d_k, d_model] — K and V projection */
    float* attn_proj; /* [d_model, d_model] */
    /* FFN unchanged */
} Block;
```

**Attention forward with GQA:**
```c
/* Each Q head h uses K/V head (h / N_GROUPS) */
for (int h = 0; h < N_Q_HEADS; h++) {
    int kv_h = h / N_GROUPS;  /* which K/V head to use */
    /* scores: q[h] dot k[kv_h] */
    /* output: scores weighted sum of v[kv_h] */
}
```

**Requires retraining from scratch.** Do this LAST — after all other changes are working.

**Files to change:** `csrc/config.h`, `csrc/model.h/c`, `csrc/math_ops.c` (attention_forward GQA variant).

**Effort:** Hard — 3–4 days + full retrain.

---

## Priority 7 — Better Training: Cosine Warmup Restart (SGDR)

**What:** Replace the current single cosine decay with Cosine Annealing with Warm Restarts (SGDR). The LR restarts every N steps, letting the optimizer escape local minima multiple times.

```
Current LR schedule (cosine decay):
  |‾‾‾\
  |    \____________________

SGDR (cosine restarts):
  |‾‾\  /‾‾\  /‾‾\  /‾‾\
  |   \/    \/    \/    \___
```

**Implementation:**
```c
/* In trainer.c — replace cosine_lr() */
float cosine_restart_lr(int step, float lr_max, float lr_min, int T0, float T_mult) {
    /* Find which restart cycle we're in */
    int t_cur = step, T_i = T0;
    while (t_cur >= T_i) { t_cur -= T_i; T_i = (int)(T_i * T_mult); }
    return lr_min + 0.5f * (lr_max - lr_min) * (1.0f + cosf((float)M_PI * t_cur / T_i));
}
/* T0=200 steps, T_mult=2.0 → restarts at steps 200, 600, 1400, 3000... */
```

**Add to config.h:**
```c
#define TRAIN_LR_RESTART_STEPS  200     /* first restart at step 200 */
#define TRAIN_LR_RESTART_MULT   2.0f    /* each restart is 2× longer */
```

**Quality impact:** Lower final loss, better generalization. Models trained with SGDR typically achieve 5-15% better perplexity than plain cosine decay.

**Files to change:** `csrc/trainer.c` — `cosine_lr()` function, `csrc/config.h`.

**Effort:** Very low — half a day.

---

## Priority 8 — Re-ranking with Model Embeddings (retrieval quality)

**What:** After BM25 returns top-20 candidates, re-rank them using dot-product similarity between the query's transformer embedding and each document's embedding.

This is the single biggest quality improvement for KB-backed answers.

**How:**
1. At KB build time: for each Q&A pair, run `model_forward_one()` on the question tokens, take the final hidden state at the last token as the embedding (dim=512). Store it.
2. At query time: embed the query the same way (one forward pass). Compute dot product against all 20 BM25 candidates. Re-rank by combined score: `0.4 * bm25_score + 0.6 * embedding_similarity`.

```c
/* In knowledge_base.h — add to KbDoc */
typedef struct {
    char*  question;
    char*  answer;
    char*  source;
    float  embedding[D_MODEL];   /* model embedding of the question */
    int    has_embedding;        /* 1 after kb_embed_all() is called */
} KbDoc;

/* New function — call after model is loaded */
void kb_embed_all(KnowledgeBase* kb, Model* m, Tokenizer* tok);
```

**Re-ranking in kb_search():**
```c
/* After BM25 top-20: */
float emb_score = vec_dot_f32(query_embedding, docs[i].embedding, D_MODEL);
float final_score = 0.4f * bm25_score + 0.6f * emb_score;
```

**Caveat:** Embeddings are only valid after the model is trained. `kb_embed_all()` is called once in `/chat` startup after `model_load()`.

**Files to change:** `csrc/knowledge_base.h/c` (add embedding field + embed_all), `csrc/agents.c` (pass model to KB for re-ranking).

**Effort:** Medium — 2 days.

---

## Priority 9 — Better Tokenizer: Finance-Aware Merges

**What:** The current BPE tokenizer is trained on the full dataset with standard merges. Many important finance terms get split into subwords that lose meaning:
- `sharpe` → `sharp` + `e`
- `ebitda` → `eb` + `it` + `da`
- `401k` → `401` + `k`
- `reit` → `re` + `it`

**Fix:** Add a pre-tokenizer that treats finance terms as atomic units before BPE:

```c
/* In tokenizer.c — pre_tokenize() */
static const char* FINANCE_ATOMS[] = {
    "sharpe","ebitda","wacc","capm","reit","etf","401k","roth","ira",
    "nasdaq","nyse","s&p","vix","cds","clo","mbs","abs","spac","ipo",
    "dcf","npv","irr","eps","roe","roa","p/e","p/b","ev/ebitda",
    "cagr","ytm","dca","ltv","dscr","fcf","ebit","nopat",
    NULL
};
/* Before BPE merges: ensure these tokens are never split */
```

**Impact:** Better token alignment with finance concepts → lower loss on domain-specific text → more accurate answers.

**Files to change:** `csrc/tokenizer.c` — `pretokenize()` function.

**Effort:** Low — 1 day.

---

## Priority 10 — Larger Model Variant (d_model=768, 6 layers)

**What:** For users willing to accept longer training (~4-6 hours), provide a larger model config that is significantly more capable.

```c
/* config_large.h — separate config for the big model */
#define D_MODEL       768      /* was 512  */
#define N_HEADS       12       /* was 8    */
#define N_LAYERS      6        /* was 4    */
#define D_FF          2048     /* was 1536 */
#define MAX_SEQ_LEN   512      /* same     */
```

Parameter count:
- Embedding: 768 × 10000 = 7.68M
- Per layer: QKV=1.77M + proj=0.59M + FFN gate+up=3.15M + FFN down=1.57M = 7.08M × 6 = 42.5M
- Total: **~50M params**

**Quality gain:** Roughly 15-20% lower perplexity vs the 19M model on the same data. Bigger jump than adding more data at the 19M scale.

**Speed impact:** ~2.5× slower training, ~2.5× slower inference (before quant). With int8 quant (Priority 4), inference returns to near the 19M model speed.

**Implementation:** Just a separate config. No code changes — all dimensions are parameterized.

Add CLI flag: `./financegpt /train large` to train with the large config.

**Files to change:** `csrc/config.h` — add `MODEL_SIZE` define, `csrc/main.c` — parse `large` argument.

**Effort:** Low — half a day of code, plus the training time.

---

## Summary

| # | Feature | Speed Impact | Quality Impact | Effort |
|---|---|---|---|---|
| 1 | 10 training plots | None | Better visibility | Low |
| 2 | 20 new CSVs + format audit | None | **High** | Medium |
| 3 | Flash Attention | **+2–4× training** | None | Med-Hard |
| 4 | Int8 Quantization | **+1.5–2× inference** | ~1% loss | Medium |
| 5 | Fused kernels | +10–20% overall | None | Low-Med |
| 6 | GQA | +15% inference | Small loss | Hard |
| 7 | SGDR learning rate | None | +5–15% lower loss | Very Low |
| 8 | Re-ranking | Negligible | **High** | Medium |
| 9 | Finance-aware BPE | None | Medium | Low |
| 10 | Larger model (50M) | −2.5× | **Very High** | Low + training |

## Recommended execution order

1. **Priority 7** (SGDR) — half a day, retrain with better LR immediately
2. **Priority 1** (10 plots) — 1 day, visual feedback for every future run
3. **Priority 2** (CSVs) — generate + audit, retrain on more + better data
4. **Priority 5** (fused kernels) — free speed, no risk
5. **Priority 9** (finance BPE) — retrain tokenizer once
6. **Priority 3** (Flash Attention) — unlock fast 512-context training
7. **Priority 4** (int8 quant) — fast inference, do after model is stable
8. **Priority 8** (re-ranking) — needs stable model first
9. **Priority 10** (large model) — do last, after everything else is tuned
10. **Priority 6** (GQA) — requires full retrain, do with Priority 10

## What NOT to do

- **Don't add internet access** — the model should be self-contained
- **Don't go beyond 6 layers or 768 d_model** — memory bandwidth bottleneck on Ryzen 3
- **Don't use FP16** — Ryzen 3 5300U has no native FP16 units; it runs as FP32 anyway
- **Don't add Python dependencies** — zero-dependency C is the whole point
- **Don't skip the CSV audit (2b)** — format inconsistency is silently killing quality

## Files the next Claude needs to read first

1. `csrc/config.h` — all hyperparameters
2. `csrc/trainer.h/c` — training loop, history struct, plot generation
3. `csrc/model.h/c` — forward pass, KV cache, generate
4. `csrc/math_ops.h/c` — attention, GEMM, RoPE
5. `csrc/knowledge_base.c` — BM25 search
6. `csrc/agents.c` — orchestration and synthesis
7. `Makefile` — build flags
