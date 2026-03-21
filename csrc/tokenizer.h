#ifndef TOKENIZER_H
#define TOKENIZER_H
#include "compat.h"
#include "json.h"

/*
 * BPE tokenizer — C port of tokenizer.py
 *
 * Special token ids (must match tokenizer.json / Python constants):
 *   <PAD> = 0   <UNK> = 1   <BOS> = 2   <EOS> = 3   <SEP> = 4
 *
 * End-of-word marker: Unicode U+2581 LOWER ONE EIGHTH BLOCK (▁)
 * Encoded in UTF-8 as: 0xE2 0x96 0x81
 */

#define TOK_MAX_VOCAB   65536
#define TOK_MAX_MERGES  65536

#define TOK_EOW         "\xe2\x96\x81"   /* UTF-8 encoding of ▁ */
#define TOK_EOW_LEN     3                /* byte length of TOK_EOW */

#define TOK_PAD_ID      0
#define TOK_UNK_ID      1
#define TOK_BOS_ID      2
#define TOK_EOS_ID      3
#define TOK_SEP_ID      4

/*
 * One BPE merge rule.
 * When pair (pair_a, pair_b) appears adjacently in the symbol sequence,
 * replace both with merged (= concat(pair_a, pair_b)).
 * merge_rank is the 0-based index in the merges array — lower = higher priority.
 */
typedef struct {
    char* pair_a;     /* left symbol            */
    char* pair_b;     /* right symbol           */
    char* merged;     /* concatenated result    */
    int   merge_rank; /* index in merges array  */
} BpeMerge;

/*
 * Main tokenizer struct.
 *
 * vocab[id]  — token string for a given id.
 * Hash map (open-addressing, linear probing) maps token string -> id
 * for O(1) lookup during encoding.
 * merges[]   — BPE merge rules in priority order (index 0 = highest priority).
 */
typedef struct {
    char**    vocab;       /* vocab[id] = token string; size = vocab_size  */
    int       vocab_size;

    BpeMerge* merges;      /* merge rules; n_merges entries                */
    int       n_merges;

    /* Hash table: open addressing, linear probing */
    int       ht_cap;      /* must be > vocab_size / HT_LOAD               */
    int*      ht_keys;     /* ht_keys[slot] = vocab index, or HT_EMPTY     */
    int*      ht_vals;     /* ht_vals[slot] = token id                     */
} Tokenizer;

/* ── Lifecycle ──────────────────────────────────────────────────────── */

/* Load tokenizer from a tokenizer.json file produced by tokenizer.py.
   Returns NULL on failure. */
Tokenizer* tok_load  (const char* path);

/* Free all memory owned by the tokenizer. */
void       tok_free  (Tokenizer* t);

/* ── Lookup ─────────────────────────────────────────────────────────── */

/* Look up a token string and return its id.
   Returns TOK_UNK_ID if the token is not in the vocabulary. */
int  tok_id  (Tokenizer* t, const char* token);

/* ── Encode / Decode ────────────────────────────────────────────────── */

/*
 * Encode a UTF-8 text string into a sequence of token ids.
 *
 *   add_special  — if non-zero, prepend BOS and append EOS.
 *   *out_len     — set to the number of ids returned.
 *
 * Returns a heap-allocated int array.  Caller must free().
 */
int*  tok_encode (Tokenizer* t, const char* text, int* out_len, int add_special);

/*
 * Decode a sequence of token ids back to a UTF-8 string.
 *
 *   skip_special — if non-zero, skip PAD / BOS / SEP / UNK tokens
 *                  and stop at the first EOS token.
 *
 * Returns a heap-allocated string.  Caller must free().
 */
char* tok_decode (Tokenizer* t, const int* ids, int n, int skip_special);

#endif /* TOKENIZER_H */
