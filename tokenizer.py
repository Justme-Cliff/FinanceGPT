"""
Byte-Pair Encoding (BPE) tokenizer trained from scratch on finance text.

Why BPE?
--------
* Handles finance sub-words well: "EBITDA", "P/E", "NAV", "401(k)", "%".
* Balances vocabulary size vs. sequence length better than word-level.
* The merge list is saved alongside the vocabulary so the same splits
  are reproduced exactly at inference time.
"""

import json
import os
import re
from collections import Counter
from typing import List, Dict, Tuple


SPECIAL = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "<SEP>": 4}
EOW = "▁"   # end-of-word marker (replaces trailing char)


def _pre_tokenize(text: str) -> List[str]:
    """Split text into 'words' before BPE is applied.
    Keeps finance tokens like '$', '%', '/', numbers intact.
    """
    return re.findall(
        r"\d+(?:[.,]\d+)*%?|[$€£]?\d+(?:[.,]\d+)*|[a-zA-Z]+(?:['\/\-][a-zA-Z]+)*|[^\w\s]",
        text.lower(),
    )


class BPETokenizer:
    """Train-from-scratch BPE tokenizer with finance-aware pre-tokenisation."""

    def __init__(self):
        self.vocab:    Dict[str, int] = {}
        self.inv:      Dict[int, str] = {}
        self.merges:   Dict[Tuple[str, str], str] = {}   # (a,b) → ab
        self.vocab_size: int = 0

    # ── Training ───────────────────────────────────────────────────────

    def train(self, texts: List[str], vocab_size: int = 10_000):
        print("  Building BPE vocabulary…")

        # 1. Word-frequency table (each word stored as char sequence + EOW)
        word_freq: Counter = Counter()
        for text in texts:
            for word in _pre_tokenize(text):
                word_freq[word + EOW] += 1

        # 2. Initial char vocabulary
        splits: Dict[str, List[str]] = {
            w: list(w) for w in word_freq
        }
        char_vocab: set = set()
        for chars in splits.values():
            char_vocab.update(chars)

        # 3. BPE merges
        merge_list: List[Tuple[str, str]] = []
        current_vocab = set(char_vocab) | set(SPECIAL.keys())

        target = vocab_size - len(SPECIAL)
        while len(current_vocab) < target:
            pairs: Counter = Counter()
            for word, freq in word_freq.items():
                syms = splits[word]
                for a, b in zip(syms, syms[1:]):
                    pairs[(a, b)] += freq

            if not pairs:
                break

            best = max(pairs, key=pairs.get)
            merged = "".join(best)
            merge_list.append(best)
            current_vocab.add(merged)

            # Apply merge across all words
            for word in splits:
                syms = splits[word]
                new: List[str] = []
                i = 0
                while i < len(syms):
                    if (i < len(syms) - 1
                            and syms[i] == best[0]
                            and syms[i + 1] == best[1]):
                        new.append(merged)
                        i += 2
                    else:
                        new.append(syms[i])
                        i += 1
                splits[word] = new

        # 4. Build final vocab
        self.vocab = dict(SPECIAL)
        idx = len(SPECIAL)
        for tok in sorted(current_vocab):
            if tok not in self.vocab:
                self.vocab[tok] = idx
                idx += 1
        self.inv       = {v: k for k, v in self.vocab.items()}
        self.merges    = {pair: "".join(pair) for pair in merge_list}
        self.vocab_size = len(self.vocab)
        print(f"  BPE vocab size: {self.vocab_size}  |  merges: {len(self.merges)}")

    # ── Encoding ───────────────────────────────────────────────────────

    def _bpe_word(self, word: str) -> List[str]:
        """Apply stored BPE merges to a single word (already has EOW appended)."""
        syms = list(word)
        changed = True
        while changed and len(syms) > 1:
            changed = False
            new: List[str] = []
            i = 0
            while i < len(syms):
                if (i < len(syms) - 1
                        and (syms[i], syms[i + 1]) in self.merges):
                    new.append(self.merges[(syms[i], syms[i + 1])])
                    i += 2
                    changed = True
                else:
                    new.append(syms[i])
                    i += 1
            syms = new
        return syms

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        unk = SPECIAL["<UNK>"]
        ids: List[int] = []
        for word in _pre_tokenize(text):
            for sym in self._bpe_word(word + EOW):
                ids.append(self.vocab.get(sym, unk))
        if add_special:
            ids = [SPECIAL["<BOS>"]] + ids + [SPECIAL["<EOS>"]]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        parts: List[str] = []
        for i in ids:
            tok = self.inv.get(i, "<UNK>")
            if skip_special and tok in SPECIAL:
                if tok == "<EOS>":
                    break
                continue
            parts.append(tok)

        # Reconstruct text from BPE pieces
        text = "".join(parts).replace(EOW, " ").strip()
        # Light cleanup
        text = re.sub(r" ([.,!?;:])", r"\1", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # ── Persistence ────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "vocab":      self.vocab,
                    "merges":     [list(k) for k in self.merges],
                    "vocab_size": self.vocab_size,
                },
                f,
            )

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab      = data["vocab"]
        self.inv        = {v: k for k, v in self.vocab.items()}
        self.merges     = {tuple(pair): "".join(pair) for pair in data["merges"]}
        self.vocab_size = data["vocab_size"]

    def __len__(self):
        return self.vocab_size
