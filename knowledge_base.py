"""
Knowledge Base — TF-IDF retrieval over all CSV Q&A pairs.
Built entirely from scratch (no external libraries beyond stdlib + pandas for CSV loading).
"""

import os
import csv
import math
import re
from collections import defaultdict


# Stop words to ignore during indexing
_STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'in', 'on', 'at',
    'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into',
    'through', 'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
    'neither', 'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just',
    'that', 'this', 'it', 'its', 'if', 'as', 'also', 'how', 'what', 'when',
    'where', 'which', 'who', 'whom', 'why', 'then', 'there', 'here', 'some',
    'any', 'all', 'no', 'more', 'most', 'other', 'such', 'between', 'each',
    'over', 'under', 'while', 'because', 'since', 'before', 'after', 'does',
    'use', 'used', 'using', 'make', 'made', 'making', 'get', 'set', 'put',
}

# Question/Answer column name aliases
_Q_COLS = {'question', 'q', 'query', 'input', 'prompt', 'ask'}
_A_COLS = {'answer', 'a', 'response', 'output', 'reply', 'text'}


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, remove stopwords."""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+(?:['\-][a-z]+)*", text)
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


class KnowledgeBase:
    """
    Builds a TF-IDF index over all CSV Q&A pairs and supports
    cosine-similarity search over questions + answers combined.
    """

    def __init__(self, data_dir: str = "data"):
        self.qa_pairs: list[tuple[str, str, str]] = []   # (question, answer, source)
        self.doc_tfidf: list[dict[str, float]] = []       # per-doc TF-IDF vectors
        self.idf: dict[str, float] = {}

        self._load_all_csvs(data_dir)
        self._build_index()

    # ── Loading ────────────────────────────────────────────────────────

    def _load_all_csvs(self, data_dir: str):
        if not os.path.isdir(data_dir):
            return
        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(".csv"):
                continue
            source = fname[:-4]   # strip .csv
            fpath = os.path.join(data_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        q_key = next((k for k in row if k.strip().lower() in _Q_COLS), None)
                        a_key = next((k for k in row if k.strip().lower() in _A_COLS), None)
                        if q_key and a_key:
                            q = row[q_key].strip()
                            a = row[a_key].strip()
                            if q and a:
                                self.qa_pairs.append((q, a, source))
            except Exception:
                pass

    # ── Indexing ───────────────────────────────────────────────────────

    def _build_index(self):
        N = len(self.qa_pairs)
        if N == 0:
            return

        # Document frequency
        df: dict[str, int] = defaultdict(int)
        all_tf: list[dict[str, float]] = []

        for q, a, _ in self.qa_pairs:
            doc_text = q + " " + a
            tokens = _tokenize(doc_text)
            raw_tf: dict[str, int] = defaultdict(int)
            for t in tokens:
                raw_tf[t] += 1
            total = max(len(tokens), 1)
            # Sublinear TF: log(1 + count) — always non-negative
            tf = {t: math.log(1.0 + c) for t, c in raw_tf.items()}
            all_tf.append(tf)
            for t in raw_tf:
                df[t] += 1

        # IDF with smoothing
        self.idf = {
            t: math.log((N + 1) / (count + 1)) + 1.0
            for t, count in df.items()
        }

        # Pre-compute TF-IDF vectors (normalized)
        self.doc_tfidf = []
        for tf in all_tf:
            vec = {t: tf[t] * self.idf.get(t, 0.0) for t in tf}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self.doc_tfidf.append({t: v / norm for t, v in vec.items()})

    # ── Search ─────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top_k most relevant Q&A pairs for the query."""
        if not self.qa_pairs:
            return []

        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        # Build query TF-IDF vector
        raw_qtf: dict[str, int] = defaultdict(int)
        for t in q_tokens:
            raw_qtf[t] += 1
        total = len(q_tokens)
        q_vec = {}
        for t, c in raw_qtf.items():
            idf_val = self.idf.get(t, 0.0)
            if idf_val > 0:
                q_vec[t] = math.log(1.0 + c) * idf_val

        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        q_vec = {t: v / q_norm for t, v in q_vec.items()}

        # Cosine similarity against all docs
        scores = []
        for i, doc_vec in enumerate(self.doc_tfidf):
            dot = sum(q_vec.get(t, 0.0) * doc_vec.get(t, 0.0) for t in q_vec)
            scores.append((dot, i))

        scores.sort(key=lambda x: -x[0])

        results = []
        for score, idx in scores[:top_k]:
            if score <= 0.0:
                break
            q, a, source = self.qa_pairs[idx]
            results.append({
                "question": q,
                "answer": a,
                "source": source,
                "score": round(score, 4),
            })
        return results

    def __len__(self):
        return len(self.qa_pairs)
