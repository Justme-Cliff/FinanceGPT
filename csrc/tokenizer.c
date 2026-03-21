#include "tokenizer.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>

/* ── Hash map constants ────────────────────────────────────────────── */

#define HT_EMPTY  (-1)
#define HT_LOAD   0.60   /* max fill ratio before the table is too small */

/* FNV-1a 32-bit hash */
static uint32_t str_hash(const char* s) {
    uint32_t h = 2166136261u;
    for (; *s; s++)
        h = (h ^ (unsigned char)*s) * 16777619u;
    return h;
}

/* Insert vocab[id] into the hash table.
   The table must not be full (caller guarantees this via HT_LOAD). */
static void ht_insert(Tokenizer* t, const char* key, int id) {
    uint32_t h = str_hash(key) % (uint32_t)t->ht_cap;
    while (t->ht_keys[h] != HT_EMPTY &&
           strcmp(t->vocab[t->ht_keys[h]], key) != 0)
        h = (h + 1) % (uint32_t)t->ht_cap;
    t->ht_keys[h] = id;
    t->ht_vals[h] = id;
}

/* Look up a token string; returns the token id or TOK_UNK_ID if absent. */
int tok_id(Tokenizer* t, const char* key) {
    if (!t || !key) return TOK_UNK_ID;
    uint32_t h = str_hash(key) % (uint32_t)t->ht_cap;
    int probes = 0;
    while (t->ht_keys[h] != HT_EMPTY && probes < t->ht_cap) {
        int slot_id = t->ht_keys[h];
        if (slot_id >= 0 && slot_id < t->vocab_size &&
            strcmp(t->vocab[slot_id], key) == 0)
            return t->ht_vals[h];
        h = (h + 1) % (uint32_t)t->ht_cap;
        probes++;
    }
    return TOK_UNK_ID;
}

/* ── Loader ─────────────────────────────────────────────────────────── */

/*
 * Expected tokenizer.json layout (produced by tokenizer.py):
 *
 *   {
 *     "vocab":  { "<PAD>": 0, "<UNK>": 1, "<BOS>": 2, … },
 *     "merges": [ ["piece_a", "piece_b"], … ]
 *   }
 *
 * vocab is a JSON object whose keys are token strings and values are integer ids.
 * merges is a JSON array of 2-element arrays.
 */
Tokenizer* tok_load(const char* path) {
    JsonNode* root = json_parse_file(path);
    if (!root) {
        fprintf(stderr, "tok_load: cannot parse '%s'\n", path);
        return NULL;
    }

    Tokenizer* t = (Tokenizer*)xcalloc(1, sizeof(Tokenizer));

    /* ── 1. Vocabulary ────────────────────────────────────────────── */
    JsonNode* vocab_node = json_get(root, "vocab");
    if (!vocab_node || vocab_node->type != JSON_OBJECT) {
        fprintf(stderr, "tok_load: missing or invalid 'vocab' in %s\n", path);
        json_free(root);
        free(t);
        return NULL;
    }

    /* First pass: find the maximum id so we can allocate vocab[]. */
    int max_id = 0;
    for (int i = 0; i < vocab_node->arr.count; i++) {
        JsonNode* entry = vocab_node->arr.items[i];
        int id = (int)json_num(entry, -1.0);
        if (id > max_id) max_id = id;
    }

    t->vocab_size = max_id + 1;
    t->vocab      = (char**)xcalloc(t->vocab_size, sizeof(char*));

    /* Second pass: fill in vocab[id] = token string. */
    for (int i = 0; i < vocab_node->arr.count; i++) {
        JsonNode* entry = vocab_node->arr.items[i];
        if (!entry || !entry->key) continue;
        int id = (int)json_num(entry, -1.0);
        if (id < 0 || id >= t->vocab_size) continue;
        t->vocab[id] = xstrdup(entry->key);
    }

    /* Fill any gaps with a placeholder so we never dereference NULL. */
    for (int i = 0; i < t->vocab_size; i++)
        if (!t->vocab[i]) t->vocab[i] = xstrdup("<UNK>");

    /* ── 2. Hash table ────────────────────────────────────────────── */
    t->ht_cap  = (int)((double)t->vocab_size / HT_LOAD) + 128;
    t->ht_keys = (int*)xmalloc(t->ht_cap * sizeof(int));
    t->ht_vals = (int*)xmalloc(t->ht_cap * sizeof(int));
    for (int i = 0; i < t->ht_cap; i++) {
        t->ht_keys[i] = HT_EMPTY;
        t->ht_vals[i] = HT_EMPTY;
    }
    for (int id = 0; id < t->vocab_size; id++)
        ht_insert(t, t->vocab[id], id);

    /* ── 3. Merge rules ───────────────────────────────────────────── */
    JsonNode* merges_node = json_get(root, "merges");
    t->n_merges = merges_node ? json_len(merges_node) : 0;
    t->merges   = (BpeMerge*)xmalloc((t->n_merges + 1) * sizeof(BpeMerge));

    for (int i = 0; i < t->n_merges; i++) {
        JsonNode* pair = json_get_index(merges_node, i);
        /* Pair can be a 2-element array ["a","b"] or an object */
        const char* a = "";
        const char* b = "";
        if (pair && pair->type == JSON_ARRAY && json_len(pair) >= 2) {
            a = json_str(json_get_index(pair, 0), "");
            b = json_str(json_get_index(pair, 1), "");
        }
        t->merges[i].pair_a     = xstrdup(a);
        t->merges[i].pair_b     = xstrdup(b);
        t->merges[i].merge_rank = i;

        /* merged = concat(a, b) */
        size_t la = strlen(a), lb = strlen(b);
        t->merges[i].merged = (char*)xmalloc(la + lb + 1);
        memcpy(t->merges[i].merged, a, la);
        memcpy(t->merges[i].merged + la, b, lb);
        t->merges[i].merged[la + lb] = '\0';
    }

    json_free(root);
    return t;
}

void tok_free(Tokenizer* t) {
    if (!t) return;
    for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    free(t->vocab);
    free(t->ht_keys);
    free(t->ht_vals);
    for (int i = 0; i < t->n_merges; i++) {
        free(t->merges[i].pair_a);
        free(t->merges[i].pair_b);
        free(t->merges[i].merged);
    }
    free(t->merges);
    free(t);
}

/* ── Pre-tokenizer ───────────────────────────────────────────────────
 *
 * Mirrors the Python regex-based pre-tokenizer in tokenizer.py.
 * Groups the input into "words" that are then BPE-encoded independently:
 *
 *   • Optionally leading '$'  followed by digits/commas/periods/trailing '%'
 *   • Alphabetic runs (may include apostrophe, hyphen, slash)
 *   • Any other single non-whitespace character (punctuation, symbols, …)
 *
 * All alphabetic characters are lowercased (matches Python behaviour).
 * Whitespace is consumed as a word separator and is NOT emitted.
 *
 * Returns a heap-allocated array of heap-allocated strings.
 * *out_n is set to the number of words.
 * Caller must free each string then the array.
 * ─────────────────────────────────────────────────────────────────── */
static char** pre_tokenize(const char* text, int* out_n) {
    int    cap   = 64;
    char** words = (char**)xmalloc(cap * sizeof(char*));
    int    n     = 0;

    /* Scratch buffer — longest realistic token well under 1 KB */
    char   buf[2048];

    const char* p = text;
    while (*p) {
        /* Skip whitespace */
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;

        /* Grow the output array if needed */
        if (n >= cap) {
            cap *= 2;
            words = (char**)xrealloc(words, cap * sizeof(char*));
        }

        size_t len = 0;

        /* ── Dollar-sign-prefixed number ──────────────────────────── */
        if (*p == '$' && isdigit((unsigned char)p[1])) {
            buf[len++] = '$';
            p++;
            while (*p && (isdigit((unsigned char)*p) || *p == ',' || *p == '.')) {
                if (len < sizeof(buf) - 2) buf[len++] = *p;
                p++;
            }
            if (*p == '%') { if (len < sizeof(buf) - 2) buf[len++] = '%'; p++; }
        }
        /* ── Plain number (may have trailing %) ───────────────────── */
        else if (isdigit((unsigned char)*p)) {
            while (*p && (isdigit((unsigned char)*p) || *p == ',' || *p == '.')) {
                if (len < sizeof(buf) - 2) buf[len++] = *p;
                p++;
            }
            if (*p == '%') { if (len < sizeof(buf) - 2) buf[len++] = '%'; p++; }
        }
        /* ── Alphabetic word (finance chars: apostrophe, hyphen, /) ─ */
        else if (isalpha((unsigned char)*p)) {
            while (*p && (isalpha((unsigned char)*p) ||
                          *p == '\'' || *p == '-'    || *p == '/')) {
                if (len < sizeof(buf) - 2)
                    buf[len++] = (char)tolower((unsigned char)*p);
                p++;
            }
        }
        /* ── Single non-space character ───────────────────────────── */
        else {
            buf[len++] = *p++;
        }

        if (len == 0) { p++; continue; } /* safety: skip unknown */

        buf[len] = '\0';
        words[n++] = xstrdup(buf);
    }

    *out_n = n;
    return words;
}

/* ── Symbol list (dynamic array of char*) ────────────────────────────
 *
 * During BPE encoding each pre-token word is broken into individual
 * UTF-8 characters (with the EOW marker appended to the last character)
 * and then merge rules are applied until no more rules match.
 * ─────────────────────────────────────────────────────────────────── */
typedef struct {
    char** syms;
    int    n;
    int    cap;
} SymList;

static void sl_init(SymList* sl, int initial_cap) {
    sl->cap  = initial_cap > 0 ? initial_cap : 8;
    sl->syms = (char**)xmalloc(sl->cap * sizeof(char*));
    sl->n    = 0;
}

static void sl_push(SymList* sl, char* s) {
    if (sl->n >= sl->cap) {
        sl->cap *= 2;
        sl->syms = (char**)xrealloc(sl->syms, sl->cap * sizeof(char*));
    }
    sl->syms[sl->n++] = s;
}

static void sl_free(SymList* sl) {
    for (int i = 0; i < sl->n; i++) free(sl->syms[i]);
    free(sl->syms);
    sl->n   = 0;
    sl->cap = 0;
}

/*
 * Build a SymList from a word string (already has EOW appended by the
 * caller).  Each symbol is a single UTF-8 code point.
 */
static SymList symlist_from_word(const char* word) {
    size_t word_len = strlen(word);
    SymList sl;
    sl_init(&sl, (int)word_len + 4);

    const char* p = word;
    while (*p) {
        unsigned char c = (unsigned char)*p;
        int bytes = 1;
        if      (c >= 0xF0) bytes = 4;
        else if (c >= 0xE0) bytes = 3;
        else if (c >= 0xC0) bytes = 2;

        char tmp[8] = {0};
        int copied  = 0;
        while (copied < bytes && *p) tmp[copied++] = *p++;
        tmp[copied] = '\0';

        sl_push(&sl, xstrdup(tmp));
    }
    return sl;
}

/*
 * Apply BPE merges to a SymList.
 *
 * Algorithm (greedy, priority-ordered):
 *   Repeat until no merge applies:
 *     Scan all adjacent pairs in the symbol list.
 *     Find the pair that matches the merge rule with the lowest rank
 *     (i.e., earliest in the merges array).
 *     Apply that merge (collapse the two symbols into one).
 *
 * This exactly replicates the Python BPE logic.
 */
static void apply_merges(SymList* sl, const BpeMerge* merges, int n_merges) {
    int changed = 1;
    while (changed && sl->n > 1) {
        changed      = 0;
        int best_rank = n_merges; /* sentinel — no merge found yet */
        int best_pos  = -1;

        /* Scan all adjacent pairs */
        for (int i = 0; i < sl->n - 1; i++) {
            /* Binary-search would be faster but n_merges ≤ 65536 and this
               runs per word, so linear scan is acceptable. */
            for (int m = 0; m < best_rank; m++) {
                if (strcmp(sl->syms[i],     merges[m].pair_a) == 0 &&
                    strcmp(sl->syms[i + 1], merges[m].pair_b) == 0) {
                    best_rank = m;
                    best_pos  = i;
                    break; /* can't improve further for this position */
                }
            }
        }

        if (best_pos < 0) break; /* no applicable merge */

        /* Replace syms[best_pos] with the merged string */
        free(sl->syms[best_pos]);
        sl->syms[best_pos] = xstrdup(merges[best_rank].merged);

        /* Remove syms[best_pos + 1] by shifting the rest left */
        free(sl->syms[best_pos + 1]);
        for (int i = best_pos + 1; i < sl->n - 1; i++)
            sl->syms[i] = sl->syms[i + 1];
        sl->n--;

        changed = 1;
    }
}

/* ── tok_encode ──────────────────────────────────────────────────────
 *
 * 1. Pre-tokenize text into words.
 * 2. For each word, append EOW (▁) and split into UTF-8 code points.
 * 3. Apply BPE merges.
 * 4. Map each resulting symbol to an id via the hash table.
 * 5. Optionally wrap with BOS/EOS.
 * ─────────────────────────────────────────────────────────────────── */
int* tok_encode(Tokenizer* t, const char* text, int* out_len, int add_special) {
    if (!t || !text) { *out_len = 0; return (int*)xcalloc(1, sizeof(int)); }

    int    n_words;
    char** words = pre_tokenize(text, &n_words);

    /* Allocate generously; we grow as needed. */
    int cap = n_words * 6 + 8;
    int* ids = (int*)xmalloc(cap * sizeof(int));
    int  n   = 0;

#define IDS_PUSH(id) do {                                               \
    if (n >= cap) {                                                     \
        cap *= 2;                                                       \
        ids = (int*)xrealloc(ids, cap * sizeof(int));                   \
    }                                                                   \
    ids[n++] = (id);                                                    \
} while(0)

    if (add_special) IDS_PUSH(TOK_BOS_ID);

    for (int wi = 0; wi < n_words; wi++) {
        const char* word = words[wi];
        size_t wlen  = strlen(word);
        size_t ewlen = TOK_EOW_LEN;

        /* Build word + EOW */
        char* word_eow = (char*)xmalloc(wlen + ewlen + 1);
        memcpy(word_eow, word, wlen);
        memcpy(word_eow + wlen, TOK_EOW, ewlen);
        word_eow[wlen + ewlen] = '\0';

        /* Split into UTF-8 symbols and apply BPE merges */
        SymList sl = symlist_from_word(word_eow);
        free(word_eow);
        apply_merges(&sl, t->merges, t->n_merges);

        /* Convert symbols to ids */
        for (int si = 0; si < sl.n; si++) {
            int id = tok_id(t, sl.syms[si]);
            IDS_PUSH(id);
        }
        sl_free(&sl);
    }

    if (add_special) IDS_PUSH(TOK_EOS_ID);

    /* Free pre-tokenized words */
    for (int wi = 0; wi < n_words; wi++) free(words[wi]);
    free(words);

#undef IDS_PUSH

    *out_len = n;
    return ids;
}

/* ── tok_decode ──────────────────────────────────────────────────────
 *
 * 1. Concatenate all token strings (skipping/stopping at specials).
 * 2. Replace every occurrence of the EOW marker (▁) with a space,
 *    except when it appears at the very start of the output (to avoid
 *    a leading space).
 * 3. Trim trailing whitespace.
 * ─────────────────────────────────────────────────────────────────── */
char* tok_decode(Tokenizer* t, const int* ids, int n, int skip_special) {
    if (!t || !ids || n <= 0) return xstrdup("");

    /* ── Phase 1: concatenate token strings ────────────────────── */
    size_t cap = 4096;
    char*  buf = (char*)xmalloc(cap);
    size_t len = 0;

    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id < 0 || id >= t->vocab_size) continue;

        if (skip_special) {
            if (id == TOK_PAD_ID || id == TOK_BOS_ID ||
                id == TOK_SEP_ID || id == TOK_UNK_ID)
                continue;
            if (id == TOK_EOS_ID) break;
        }

        const char* tok = t->vocab[id];
        size_t tl = strlen(tok);

        /* Ensure capacity (leave room for null terminator) */
        while (len + tl + 2 > cap) { cap *= 2; buf = (char*)xrealloc(buf, cap); }
        memcpy(buf + len, tok, tl);
        len += tl;
    }
    buf[len] = '\0';

    /* ── Phase 2: replace ▁ (EOW) with spaces ───────────────────── */
    const char* eow    = TOK_EOW;
    size_t      ewlen  = TOK_EOW_LEN;

    size_t cap2 = len + 64;
    char*  out  = (char*)xmalloc(cap2);
    size_t olen = 0;

    for (size_t i = 0; i < len; ) {
        /* Check for EOW marker */
        if (len - i >= ewlen && memcmp(buf + i, eow, ewlen) == 0) {
            /* Add a space unless this is the very beginning of the output */
            if (olen > 0) {
                if (olen + 2 >= cap2) { cap2 *= 2; out = (char*)xrealloc(out, cap2); }
                out[olen++] = ' ';
            }
            i += ewlen;
        } else {
            if (olen + 2 >= cap2) { cap2 *= 2; out = (char*)xrealloc(out, cap2); }
            out[olen++] = buf[i++];
        }
    }
    out[olen] = '\0';
    free(buf);

    /* ── Phase 3: trim trailing whitespace ──────────────────────── */
    while (olen > 0 && isspace((unsigned char)out[olen - 1]))
        out[--olen] = '\0';

    return out;
}
