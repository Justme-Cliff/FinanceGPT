"""
Knowledge Base — TF-IDF retrieval over all CSV Q&A pairs.
Built entirely from scratch (no external libraries beyond stdlib + csv).

Optimisations vs original:
  1. Inverted index  — search only touches docs that share a term with the query
                       instead of iterating all N docs.  ~10-50× faster at scale.
  2. Bigram tokens   — "credit score" is indexed as one feature, improving phrase
                       matching for compound finance terms.
  3. Disk cache      — TF-IDF index pickled to checkpoints/kb_cache.pkl.
                       If no CSV has changed since last run, load takes ~0.1 s
                       instead of several seconds of rebuilding.
  4. Query expansion — search also runs on a normalised form of the query so
                       "what's a loan" and "What is a loan?" both hit the same doc.
"""

import csv
import hashlib
import math
import os
import pickle
import re
from collections import defaultdict


# ── Stop words ─────────────────────────────────────────────────────────
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

# Column-name aliases recognised in CSV files
_Q_COLS = {'question', 'q', 'query', 'input', 'prompt', 'ask'}
_A_COLS = {'answer', 'a', 'response', 'output', 'reply', 'text'}

# Conversational normalisation rules (same as reasoning_engine)
_REPHRASE_RULES = [
    (r"\bwhat'?s\b", "what is"),
    (r"\bhow'?s\b", "how does"),
    (r"\bi don'?t understand\b", "explain"),
    (r"\bhelp me understand\b", "explain"),
    (r"\bcan you explain\b", "explain"),
    (r"\btell me about\b", "what is"),
    (r"\bplease\b", ""),
    (r"\bkind of\b", ""),
    (r"\bsort of\b", ""),
]

_CACHE_VERSION = 3   # bump to invalidate old caches after format changes


# ── Tokeniser ───────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, remove stop words, add bigrams."""
    text = text.lower()
    words = re.findall(r"[a-z0-9]+(?:['\-][a-z]+)*", text)
    unigrams = [t for t in words if t not in _STOPWORDS and len(t) > 1]
    # Bigrams improve matching of compound terms like "credit score", "stock market"
    bigrams = [f"{unigrams[i]}_{unigrams[i+1]}" for i in range(len(unigrams) - 1)]
    return unigrams + bigrams


def _normalize_query(query: str) -> str:
    """Map informal phrasings to canonical forms that match training data."""
    q = query.strip()
    for pattern, replacement in _REPHRASE_RULES:
        q = re.sub(pattern, replacement, q, flags=re.IGNORECASE)
    return re.sub(r" {2,}", " ", q).strip()


# ── Cache helpers ───────────────────────────────────────────────────────

def _dir_fingerprint(data_dir: str) -> str:
    """MD5 of all CSV filenames + mtimes + sizes.  Changes whenever any file does."""
    h = hashlib.md5()
    h.update(str(_CACHE_VERSION).encode())
    if not os.path.isdir(data_dir):
        return h.hexdigest()
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".csv"):
            path = os.path.join(data_dir, fname)
            stat = os.stat(path)
            h.update(f"{fname}:{stat.st_mtime:.1f}:{stat.st_size}".encode())
    return h.hexdigest()


# ── Main class ──────────────────────────────────────────────────────────

class KnowledgeBase:
    """
    TF-IDF index over all CSV Q&A pairs with cosine-similarity search.

    The inverted index means search time is proportional to the number of
    documents that share *at least one token* with the query, not the total
    number of documents — a 10-50× speedup for typical queries at scale.
    """

    def __init__(self, data_dir: str = "data", cache_path: str | None = None):
        self.qa_pairs:  list[tuple[str, str, str]] = []
        self.idf:       dict[str, float] = {}
        self.doc_tfidf: list[dict[str, float]] = []
        self._inv_idx:  dict[str, list[tuple[int, float]]] = {}

        cache_path = cache_path or os.path.join("checkpoints", "kb_cache.pkl")

        if self._load_cache(data_dir, cache_path):
            return  # fast path — everything loaded from disk

        self._load_all_csvs(data_dir)
        self._build_index()
        self._save_cache(data_dir, cache_path)

    # ── Cache I/O ─────────────────────────────────────────────────────

    def _load_cache(self, data_dir: str, cache_path: str) -> bool:
        try:
            if not os.path.exists(cache_path):
                return False
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            if cache.get("fingerprint") != _dir_fingerprint(data_dir):
                return False
            self.qa_pairs  = cache["qa_pairs"]
            self.idf       = cache["idf"]
            self.doc_tfidf = cache["doc_tfidf"]
            self._inv_idx  = cache["inv_idx"]
            return True
        except Exception:
            return False

    def _save_cache(self, data_dir: str, cache_path: str) -> None:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "fingerprint": _dir_fingerprint(data_dir),
                        "qa_pairs":    self.qa_pairs,
                        "idf":         self.idf,
                        "doc_tfidf":   self.doc_tfidf,
                        "inv_idx":     self._inv_idx,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        except Exception:
            pass  # caching is best-effort

    # ── CSV loading ───────────────────────────────────────────────────

    def _load_all_csvs(self, data_dir: str) -> None:
        if not os.path.isdir(data_dir):
            return
        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(".csv"):
                continue
            source = fname[:-4]
            fpath  = os.path.join(data_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        q_key = next(
                            (k for k in row if k.strip().lower() in _Q_COLS), None
                        )
                        a_key = next(
                            (k for k in row if k.strip().lower() in _A_COLS), None
                        )
                        if q_key and a_key:
                            q = row[q_key].strip()
                            a = row[a_key].strip()
                            if q and a:
                                self.qa_pairs.append((q, a, source))
            except Exception:
                pass

    # ── Index building ────────────────────────────────────────────────

    def _build_index(self) -> None:
        N = len(self.qa_pairs)
        if N == 0:
            return

        df: dict[str, int] = defaultdict(int)
        all_tf: list[dict[str, float]] = []

        for q, a, _ in self.qa_pairs:
            tokens  = _tokenize(q + " " + a)
            raw_tf: dict[str, int] = defaultdict(int)
            for t in tokens:
                raw_tf[t] += 1
            tf = {t: math.log(1.0 + c) for t, c in raw_tf.items()}
            all_tf.append(tf)
            for t in raw_tf:
                df[t] += 1

        # Smoothed IDF
        self.idf = {
            t: math.log((N + 1) / (count + 1)) + 1.0
            for t, count in df.items()
        }

        # Normalised TF-IDF vectors
        self.doc_tfidf = []
        for tf in all_tf:
            vec  = {t: tf[t] * self.idf.get(t, 0.0) for t in tf}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self.doc_tfidf.append({t: v / norm for t, v in vec.items()})

        # Inverted index: term → [(doc_id, tfidf_value), ...]
        inv: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for doc_id, vec in enumerate(self.doc_tfidf):
            for term, val in vec.items():
                inv[term].append((doc_id, val))
        self._inv_idx = dict(inv)

    # ── Search ────────────────────────────────────────────────────────

    def _query_vec(self, query: str) -> dict[str, float]:
        """Build a normalised TF-IDF vector for an arbitrary query string."""
        tokens = _tokenize(query)
        if not tokens:
            return {}
        raw: dict[str, int] = defaultdict(int)
        for t in tokens:
            raw[t] += 1
        vec = {}
        for t, c in raw.items():
            idf_val = self.idf.get(t, 0.0)
            if idf_val > 0:
                vec[t] = math.log(1.0 + c) * idf_val
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Return top_k most relevant Q&A pairs for *query*.

        Uses the inverted index to skip documents that share no terms with the
        query, then optionally boosts exact / substring question matches.
        Both the original query and a normalised variant are searched and merged.
        """
        if not self.qa_pairs:
            return []

        # Build score maps for original query and its normalised form
        q_vec      = self._query_vec(query)
        q_norm_str = _normalize_query(query)
        q_vec_norm = self._query_vec(q_norm_str) if q_norm_str != query else q_vec

        candidate_scores: dict[int, float] = {}

        for vec in (q_vec, q_vec_norm):
            for t, q_val in vec.items():
                for doc_id, doc_val in self._inv_idx.get(t, []):
                    candidate_scores[doc_id] = (
                        candidate_scores.get(doc_id, 0.0) + q_val * doc_val
                    )

        if not candidate_scores:
            return []

        # Exact / substring question-match boost
        query_canon = query.lower().strip().rstrip("?").strip()
        norm_canon  = q_norm_str.lower().strip().rstrip("?").strip()

        for doc_id in list(candidate_scores):
            stored_q = self.qa_pairs[doc_id][0].lower().strip().rstrip("?").strip()
            if stored_q in (query_canon, norm_canon):
                candidate_scores[doc_id] += 0.5
            elif (
                query_canon in stored_q
                or stored_q in query_canon
                or norm_canon in stored_q
                or stored_q in norm_canon
            ):
                candidate_scores[doc_id] += 0.15

        sorted_docs = sorted(candidate_scores.items(), key=lambda x: -x[1])

        results = []
        for doc_id, score in sorted_docs[:top_k]:
            if score <= 0.0:
                break
            q, a, source = self.qa_pairs[doc_id]
            results.append(
                {
                    "question": q,
                    "answer":   a,
                    "source":   source,
                    "score":    round(score, 4),
                }
            )
        return results

    def __len__(self) -> int:
        return len(self.qa_pairs)
