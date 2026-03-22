#!/usr/bin/env python3
"""
train_tokenizer.py -- Train BPE tokenizer from CSV data and save tokenizer.json.
Produces the exact format expected by csrc/tokenizer.c.

Usage:
    python train_tokenizer.py

Output:
    checkpoints/tokenizer.json
"""

import os
import csv
import glob
import json
from collections import Counter

# ── Constants (must match config.h) ────────────────────────────────────
VOCAB_TARGET    = 10000
DATA_DIR        = "data"
TOKENIZER_PATH  = "checkpoints/tokenizer.json"
EOW             = "\u2581"   # ▁  appended to each word during BPE encode

SPECIAL_TOKENS = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "<SEP>": 4}

# Finance atoms (must match csrc/tokenizer.c FINANCE_ATOMS[])
FINANCE_ATOMS = sorted([
    "ebitda","ebit","ebitdac","cagr","wacc","capm","roe","roa","roce",
    "eps","bvps","fcf","ocf","dcf","ltv","ltm","ntm",
    "etf","reit","spac","cdo","cds","clo","mbs","abs","cmbs",
    "ipo","apo","spo","dpo",
    "libor","sofr","shibor","euribor","ffr",
    "sharpe","sortino","treynor","calmar","omega",
    "p/e","p/b","p/s","p/fcf","ev/ebitda","ev/ebit","ev/sales",
    "401k","403b","457b","ira","roth","hsa","fsa",
    "defi","nft","dao","dex","cex","amm","tvl",
    "esg","sri","unpri",
    "gdp","cpi","ppi","pce","ism","pmi","nfp",
], key=len, reverse=True)   # longest-first for greedy matching


# ── Pre-tokenizer (mirrors csrc/tokenizer.c pre_tokenize) ──────────────
def pre_tokenize(text):
    """Split text into words the same way the C runtime will."""
    words = []
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break

        # Dollar-prefixed number: $1,234.56%
        if text[i] == '$' and i + 1 < n and text[i+1].isdigit():
            j = i + 1
            while j < n and (text[j].isdigit() or text[j] in ',.'):
                j += 1
            if j < n and text[j] == '%':
                j += 1
            words.append(text[i:j])
            i = j
            continue

        # Plain number, maybe with trailing %
        if text[i].isdigit():
            j = i
            while j < n and (text[j].isdigit() or text[j] in ',.'):
                j += 1
            if j < n and text[j] == '%':
                j += 1
                words.append(text[i:j])
                i = j
                continue
            # Check for a finance atom that starts with digits (e.g. 401k)
            if j < n and text[j].isalpha():
                cand = text[i:j].lower()
                k = j
                while k < n and text[k].isalpha():
                    cand += text[k].lower()
                    k += 1
                if cand in FINANCE_ATOMS:
                    words.append(cand)
                    i = k
                    continue
            words.append(text[i:j])
            i = j
            continue

        # Finance atom (alpha start)
        matched = False
        low = text[i:].lower()
        for atom in FINANCE_ATOMS:
            if low.startswith(atom):
                end = i + len(atom)
                if end >= n or not (text[end].isalnum() or text[end] == '_'):
                    words.append(atom)
                    i = end
                    matched = True
                    break
        if matched:
            continue

        # Alphabetic run (apostrophe / hyphen / slash allowed inside)
        if text[i].isalpha():
            j = i
            while j < n and (text[j].isalpha() or text[j] in "'/-"):
                j += 1
            words.append(text[i:j].lower())
            i = j
            continue

        # Any other single non-whitespace character
        words.append(text[i])
        i += 1

    return words


# ── CSV loader ──────────────────────────────────────────────────────────
def load_texts(data_dir):
    Q_NAMES = {"question","q","query","input","prompt","ask"}
    A_NAMES = {"answer","a","response","output","reply","text"}
    texts = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        try:
            with open(path, encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue
                hmap = {h.lower().strip(): h for h in reader.fieldnames if h}
                qcol = next((hmap[h] for h in Q_NAMES if h in hmap), None)
                acol = next((hmap[h] for h in A_NAMES if h in hmap), None)
                if not qcol or not acol:
                    continue
                for row in reader:
                    q = (row.get(qcol) or "").strip()
                    a = (row.get(acol) or "").strip()
                    if q and a and q != "nan" and a != "nan":
                        texts.append(f"Q: {q} <SEP> A: {a}")
        except Exception as e:
            print(f"  Warning: {path}: {e}")
    return texts


# ── BPE trainer ─────────────────────────────────────────────────────────
def get_pair_stats(vocab):
    pairs = Counter()
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs


def merge_pair(vocab, pair):
    merged = pair[0] + pair[1]
    new_vocab = {}
    for symbols, freq in vocab.items():
        lst = list(symbols)
        i = 0
        out = []
        while i < len(lst):
            if i < len(lst) - 1 and lst[i] == pair[0] and lst[i+1] == pair[1]:
                out.append(merged)
                i += 2
            else:
                out.append(lst[i])
                i += 1
        new_vocab[tuple(out)] = freq
    return new_vocab


def train_bpe(texts, target_vocab):
    print("  Pre-tokenizing...")
    word_freq = Counter()
    for text in texts:
        for word in pre_tokenize(text):
            word_freq[word] += 1

    print(f"  Unique words: {len(word_freq)}")

    # Each word becomes a tuple of individual characters + EOW at the end
    bpe_vocab = {}
    base_chars = set()
    for word, freq in word_freq.items():
        chars = tuple(list(word) + [EOW])
        bpe_vocab[chars] = freq
        base_chars.update(chars)

    # Start with special tokens + all base characters
    vocab = dict(SPECIAL_TOKENS)
    for ch in sorted(base_chars):
        if ch not in vocab:
            vocab[ch] = len(vocab)

    merges = []
    n_needed = target_vocab - len(vocab)
    print(f"  Base vocab: {len(vocab)}  ->  need {n_needed} merges to reach {target_vocab}")

    for i in range(n_needed):
        if i % 1000 == 0 and i > 0:
            print(f"  Merge {i}/{n_needed}...")
        stats = get_pair_stats(bpe_vocab)
        if not stats:
            break
        best = stats.most_common(1)[0][0]
        bpe_vocab = merge_pair(bpe_vocab, best)
        merges.append(list(best))
        merged_tok = best[0] + best[1]
        if merged_tok not in vocab:
            vocab[merged_tok] = len(vocab)

    print(f"  Final vocab size: {len(vocab)}  |  Merges: {len(merges)}")
    return vocab, merges


# ── Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  FinanceGPT -- BPE Tokenizer Trainer")
    print(f"  Target vocab size : {VOCAB_TARGET}")
    print(f"  Data dir          : {DATA_DIR}")
    print(f"  Output            : {TOKENIZER_PATH}")
    print("=" * 60)

    print("\n[1/3] Loading CSVs...")
    texts = load_texts(DATA_DIR)
    if not texts:
        print("  ERROR: no Q&A pairs found in data/")
        return
    print(f"  Loaded {len(texts)} Q&A pairs")

    print("\n[2/3] Training BPE...")
    vocab, merges = train_bpe(texts, VOCAB_TARGET)

    print("\n[3/3] Saving tokenizer.json...")
    os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)
    with open(TOKENIZER_PATH, "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab, "merges": merges}, f, ensure_ascii=False)

    kb = os.path.getsize(TOKENIZER_PATH) / 1024
    print(f"  Saved {TOKENIZER_PATH} ({kb:.0f} KB)")
    print("\nDone!  Now run:  .\\financegpt.exe /train")


if __name__ == "__main__":
    main()
