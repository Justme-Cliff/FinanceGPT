#ifndef CONFIG_H
#define CONFIG_H

/* FinanceGPT — Central configuration (mirrors config.py) */

/* ── Model architecture ─────────────────────────────────────────── */
#define D_MODEL       512
#define N_HEADS       8
#define N_LAYERS      4
#define D_FF          1536
#define MAX_SEQ_LEN   512
#define D_K           (D_MODEL / N_HEADS)   /* 64 */
#define DROPOUT       0.10f

/* ── Training ───────────────────────────────────────────────────── */
#define TRAIN_EPOCHS         10
#define TRAIN_BATCH_SIZE     64
#define TRAIN_GRAD_ACCUM     1
#define TRAIN_LR             1e-3f
#define TRAIN_MIN_LR         5e-5f
#define TRAIN_WARMUP_STEPS   0
#define TRAIN_GRAD_CLIP      1.0f
#define TRAIN_LABEL_SMOOTH   0.0f
#define TRAIN_VAL_SPLIT      0.10f
#define TRAIN_BLOCK_SIZE     64
#define TRAIN_STRIDE         256
#define TRAIN_PATIENCE       10
#define TRAIN_LR_RESTART_STEPS   150
#define TRAIN_LR_RESTART_MULT    2.0f

/* ── Generation ─────────────────────────────────────────────────── */
#define GEN_TEMPERATURE      0.80f
#define GEN_TOP_K            50
#define GEN_TOP_P            0.92f
#define GEN_REP_PENALTY      1.3f
#define GEN_MAX_NEW_TOKENS   280

/* ── Tokenizer ──────────────────────────────────────────────────── */
#define VOCAB_TARGET         10000
#define TOK_PAD              0
#define TOK_UNK              1
#define TOK_BOS              2
#define TOK_EOS              3
#define TOK_SEP              4
#define EOW_CHAR             "\xe2\x96\x81"  /* UTF-8 for ▁ */

/* ── Knowledge base ─────────────────────────────────────────────── */
#define KB_TOP_K             6
#define KB_MIN_SCORE         0.05f
#define KB_DIRECT_SCORE      0.20f

/* ── Paths ──────────────────────────────────────────────────────── */
#define CHECKPOINT_PATH      "checkpoints/finance_gpt.bin"
#define TOKENIZER_PATH       "checkpoints/tokenizer.json"
#define CONV_HISTORY_PATH    "checkpoints/conversation_history.json"
#define DATA_DIR             "data"
#define PLOTS_DIR            "training_plots"

/* ── AdamW optimizer ────────────────────────────────────────────── */
#define ADAM_BETA1           0.9f
#define ADAM_BETA2           0.95f
#define ADAM_EPS             1e-8f
#define ADAM_WEIGHT_DECAY    0.1f

/* ── RoPE ───────────────────────────────────────────────────────── */
#define ROPE_THETA           10000.0f

/* ── RMSNorm ────────────────────────────────────────────────────── */
#define RMSNORM_EPS          1e-6f

/* ── Conversation memory ────────────────────────────────────────── */
#define MAX_HISTORY_TURNS    40
#define LOAD_HISTORY_TURNS   10
#define MAX_SESSIONS         10

/* ── TF-IDF knowledge base ──────────────────────────────────────── */
#define KB_MAX_DOCS          8192
#define KB_MAX_VOCAB         65536
#define KB_MAX_BIGRAMS       32768

#endif /* CONFIG_H */
