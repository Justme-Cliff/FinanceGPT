"""Central configuration for FinanceGPT."""

# ── Model ──────────────────────────────────────────────────────────────
MODEL_CONFIG = {
    "vocab_size": None,       # filled at runtime from tokenizer
    "d_model": 384,           # embedding dimension (was 256)
    "n_heads": 8,             # attention heads
    "n_layers": 8,            # transformer blocks (was 6)
    "d_ff": 1536,             # feed-forward inner dim (4 × d_model)
    "max_seq_len": 512,       # context window (was 256)
    "dropout": 0.10,
}

# ── Training ───────────────────────────────────────────────────────────
TRAIN_CONFIG = {
    "epochs": 15,
    "batch_size": 16,
    "grad_accum": 4,          # effective batch = 64
    "lr": 2e-4,
    "min_lr": 2e-5,
    "warmup_steps": 400,
    "grad_clip": 1.0,
    "label_smoothing": 0.10,
    "val_split": 0.10,
    "block_size": 512,
    "patience": 4,            # early-stopping epochs without val improvement
    "mixed_precision": True,  # use torch.autocast when CUDA available
}

# ── Generation ─────────────────────────────────────────────────────────
GEN_CONFIG = {
    "temperature": 0.82,
    "top_k": 50,
    "top_p": 0.92,            # nucleus sampling
    "repetition_penalty": 1.3,
    "max_new_tokens": 220,
}

# ── Paths ──────────────────────────────────────────────────────────────
CHECKPOINT           = "checkpoints/finance_gpt.pt"
TOKENIZER            = "checkpoints/tokenizer.json"
HISTORY              = "checkpoints/history.json"
CONVERSATION_HISTORY = "checkpoints/conversation_history.json"
PLOTS_DIR            = "training_plots"
DATA_DIR             = "data"
