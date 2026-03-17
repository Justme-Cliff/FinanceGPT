"""Central configuration for FinanceGPT."""

# ── Model ──────────────────────────────────────────────────────────────
MODEL_CONFIG = {
    "vocab_size": None,       # filled at runtime from tokenizer
    "d_model": 512,           # 4× more capacity than original
    "n_heads": 8,             # attention heads (d_k = 64)
    "n_layers": 6,            # deep reasoning
    "d_ff": 1536,             # 3× d_model
    "max_seq_len": 256,       # 512→256: attention is O(n²) — halving this ~3× faster
    "dropout": 0.10,
}

# ── Training ───────────────────────────────────────────────────────────
TRAIN_CONFIG = {
    "epochs": 12,             # early stopping handles the rest
    "batch_size": 32,         # 16→32: 2× fewer optimizer steps per epoch
    "grad_accum": 2,          # effective batch = 64
    "lr": 1.5e-4,             # slightly higher for faster convergence
    "min_lr": 5e-6,
    "warmup_steps": 150,
    "grad_clip": 1.0,
    "label_smoothing": 0.05,
    "val_split": 0.10,
    "block_size": 256,        # matches max_seq_len
    "patience": 5,            # stop when converged
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
