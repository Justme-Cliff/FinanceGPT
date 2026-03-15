"""Dataset loading and preprocessing for FinanceGPT."""

import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset

from tokenizer import BPETokenizer


# Column-name aliases
Q_COLS = {"question", "q", "query", "input", "prompt", "ask"}
A_COLS = {"answer", "a", "response", "output", "reply", "text"}

# Light question augmentations (keeps training set diverse)
_Q_PREFIXES = [
    "Q: {q} <SEP> A:",
    "Question: {q} <SEP> Answer:",
    "{q} <SEP>",
]


def _fmt(q: str, a: str) -> str:
    tmpl = random.choice(_Q_PREFIXES)
    return tmpl.replace("{q}", q.strip()) + " " + a.strip()


def _load_csv(path: str) -> list[str]:
    df = pd.read_csv(path, dtype=str).fillna("")
    cols_low = {c.lower(): c for c in df.columns}
    q_col = next((cols_low[c] for c in cols_low if c in Q_COLS), None)
    a_col = next((cols_low[c] for c in cols_low if c in A_COLS), None)
    texts = []
    if q_col and a_col:
        for _, row in df.iterrows():
            q, a = row[q_col].strip(), row[a_col].strip()
            if q and a:
                texts.append(_fmt(q, a))
    else:
        for _, row in df.iterrows():
            line = " ".join(row.values).strip()
            if line:
                texts.append(line)
    return texts


def load_all_csv(data_dir: str = "data", specific: str = None) -> list[str]:
    paths = (
        [specific]
        if specific
        else [
            os.path.join(data_dir, f)
            for f in sorted(os.listdir(data_dir))
            if f.endswith(".csv")
        ]
    )
    texts = []
    for p in paths:
        if not os.path.exists(p):
            print(f"  [warn] {p} not found, skipping.")
            continue
        batch = _load_csv(p)
        texts.extend(batch)
        print(f"  Loaded {len(batch):>4} samples  ← {os.path.basename(p)}")
    return texts


class FinanceDataset(Dataset):
    """Sliding-window token dataset with train/val split support."""

    def __init__(self, tokens: torch.Tensor, block_size: int):
        self.data       = tokens
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def make_datasets(texts: list[str],
                  tokenizer: BPETokenizer,
                  block_size: int,
                  val_split: float = 0.10):
    """Tokenise all texts and split into train / validation datasets."""
    all_ids: list[int] = []
    for t in texts:
        all_ids.extend(tokenizer.encode(t, add_special=True))

    data = torch.tensor(all_ids, dtype=torch.long)
    n_val = max(block_size + 1, int(len(data) * val_split))

    # Shuffle at block level (not token level) to preserve context
    n_blocks = len(data) // block_size
    idx      = list(range(n_blocks))
    random.shuffle(idx)
    val_blocks  = set(idx[:max(1, int(n_blocks * val_split))])
    train_ids, val_ids = [], []
    for i in range(n_blocks):
        seg = data[i * block_size : (i + 1) * block_size + 1].tolist()
        if len(seg) < block_size + 1:
            continue
        if i in val_blocks:
            val_ids.extend(seg[:-1])
        else:
            train_ids.extend(seg[:-1])

    t_data = torch.tensor(train_ids, dtype=torch.long)
    v_data = torch.tensor(val_ids,   dtype=torch.long)

    train_ds = FinanceDataset(t_data, block_size)
    val_ds   = FinanceDataset(v_data, block_size)

    print(f"  Train tokens: {len(t_data):,}  |  Val tokens: {len(v_data):,}")
    print(f"  Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")
    return train_ds, val_ds
