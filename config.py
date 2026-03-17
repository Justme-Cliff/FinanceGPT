"""Central configuration for FinanceGPT."""

# ── Model ──────────────────────────────────────────────────────────────
MODEL_CONFIG = {
    "vocab_size": None,       # filled at runtime from tokenizer
    "d_model": 512,           # 256→512: 4× more representational capacity
    "n_heads": 8,             # attention heads (d_k = 64)
    "n_layers": 6,            # 4→6: deeper reasoning, +50% depth
    "d_ff": 1536,             # 3× d_model
    "max_seq_len": 512,       # 256→512: longer context = richer answers
    "dropout": 0.10,
}

# ── Training ───────────────────────────────────────────────────────────
TRAIN_CONFIG = {
    "epochs": 20,
    "batch_size": 16,         # smaller batch for larger model memory footprint
    "grad_accum": 4,          # effective batch = 64 (same as before)
    "lr": 1e-4,               # lower LR for larger model — more stable
    "min_lr": 5e-6,
    "warmup_steps": 400,      # longer warmup for larger model
    "grad_clip": 1.0,
    "label_smoothing": 0.05,
    "val_split": 0.10,
    "block_size": 512,        # matches max_seq_len
    "patience": 8,            # early-stopping epochs without val improvement
    "mixed_precision": True,  # use torch.autocast when CUDA available
}

# ── Generation ─────────────────────────────────────────────────────────
GEN_CONFIG = {
    "temperature": 0.80,
    "top_k": 50,
    "top_p": 0.92,            # nucleus sampling
    "repetition_penalty": 1.3,
    "max_new_tokens": 280,    # more tokens for richer answers
}

# ── Paths ──────────────────────────────────────────────────────────────
CHECKPOINT           = "checkpoints/finance_gpt.pt"
TOKENIZER            = "checkpoints/tokenizer.json"
HISTORY              = "checkpoints/history.json"
CONVERSATION_HISTORY = "checkpoints/conversation_history.json"
PORTFOLIO_FILE       = "checkpoints/portfolio.json"
PLOTS_DIR            = "training_plots"
DATA_DIR             = "data"
