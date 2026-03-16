"""Dataset loading and preprocessing for FinanceGPT."""

import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

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
    df = pd.read_csv(path, dtype=str, on_bad_lines="skip", quoting=0).fillna("")
    cols_low = {c.lower(): c for c in df.columns}
    q_col = next((cols_low[c] for c in cols_low if c in Q_COLS), None)
    a_col = next((cols_low[c] for c in cols_low if c in A_COLS), None)
    texts = []
    if q_col and a_col:
        # Vectorised access is ~10× faster than iterrows for large DataFrames
        qs = df[q_col].astype(str).str.strip()
        as_ = df[a_col].astype(str).str.strip()
        for q, a in zip(qs, as_):
            if q and a and q != "nan" and a != "nan":
                texts.append(_fmt(q, a))
    else:
        for row in df.itertuples(index=False):
            line = " ".join(str(v) for v in row).strip()
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
    pbar = tqdm(paths, desc="  Loading CSVs", ncols=82, unit="file")
    for p in pbar:
        if not os.path.exists(p):
            tqdm.write(f"  [warn] {p} not found, skipping.")
            continue
        batch = _load_csv(p)
        texts.extend(batch)
        pbar.set_postfix(file=os.path.basename(p), samples=len(batch))
    return texts


class FinanceDataset(Dataset):
    """Sliding-window block dataset with configurable stride.

    stride=block_size // 2 gives 50% overlap → 2× more samples and no
    boundary bleed between blocks.
    """

    def __init__(self, tokens: torch.Tensor, block_size: int, stride: int = None):
        self.data       = tokens
        self.block_size = block_size
        self.stride     = stride if stride is not None else block_size

    def __len__(self):
        # +1 so the last full window is always included
        return max(0, (len(self.data) - self.block_size - 1) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        chunk = self.data[start : start + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def make_datasets(texts: list[str],
                  tokenizer: BPETokenizer,
                  block_size: int,
                  val_split: float = 0.10):
    """Tokenise all texts and split into train / validation datasets.

    Uses stride = block_size // 2 (50 % overlap) so every token appears in
    ~2 windows.  This gives 2× more training samples from the same data and
    helps the model see each Q-A pair in different positional contexts.
    """
    all_ids: list[int] = []
    # Pre-encode in one pass — avoids repeated list-extend overhead
    pbar = tqdm(texts, desc="  Tokenising", ncols=82, unit="sample")
    for t in pbar:
        all_ids.extend(tokenizer.encode(t, add_special=True))

    data   = torch.tensor(all_ids, dtype=torch.long)
    split  = int(len(data) * (1 - val_split))
    # Overlapping stride produces 2× more training windows at no extra cost
    stride = max(1, block_size // 2)

    train_ds = FinanceDataset(data[:split], block_size, stride)
    # Validation uses full stride (no overlap) — speed and no data leak
    val_ds   = FinanceDataset(data[split:], block_size, block_size)

    print(f"  Tokens:  {split:,} train | {len(data) - split:,} val")
    print(f"  Samples: {len(train_ds):,} train | {len(val_ds):,} val")
    return train_ds, val_ds
