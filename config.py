"""Central configuration for FinanceGPT."""

# ── Model ──────────────────────────────────────────────────────────────
MODEL_CONFIG = {
    "vocab_size": None,       # filled at runtime from tokenizer
    "d_model": 256,           # reduced 384→256 for CPU speed (~3× fewer MLP FLOPs)
    "n_heads": 8,             # attention heads (d_k = 32)
    "n_layers": 4,            # reduced 8→4 for CPU speed
    "d_ff": 768,              # 3× d_model
    "max_seq_len": 256,       # context window
    "dropout": 0.10,
}

# ── Training ───────────────────────────────────────────────────────────
TRAIN_CONFIG = {
    "epochs": 25,
    "batch_size": 32,
    "grad_accum": 2,          # effective batch = 64
    "lr": 2e-4,
    "min_lr": 5e-6,
    "warmup_steps": 200,
    "grad_clip": 1.0,
    "label_smoothing": 0.05,
    "val_split": 0.10,
    "block_size": 256,        # matches max_seq_len
    "patience": 8,            # early-stopping epochs without val improvement
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
