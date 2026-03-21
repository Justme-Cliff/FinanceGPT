#ifndef KNOWLEDGE_BASE_H
#define KNOWLEDGE_BASE_H
#include "compat.h"
#include "config.h"

/* BM25 knowledge base over CSV Q&A pairs.
   Exact port of knowledge_base.py — scratch-built, no external libs. */

typedef struct {
    char* question;
    char* answer;
    char* source;    /* CSV filename without .csv extension */
} KbDoc;

typedef struct {
    /* Documents */
    KbDoc* docs;
    int    n_docs;
    int    cap_docs;

    /* Vocabulary (unique terms) */
    char** terms;     /* terms[term_id] = string */
    int    n_terms;
    int    ht_cap;
    int*   ht_keys;   /* term_id or -1 */
    int*   ht_vals;

    /* IDF values */
    float* idf;       /* idf[term_id] */

    /* TF-IDF matrix: sparse representation via inverted index */
    /* For each term: list of (doc_id, tf_idf_value) */
    int**   inv_doc_ids;  /* [n_terms] -> list of doc_ids */
    float** inv_tfidf;    /* [n_terms] -> list of tfidf values */
    int*    inv_len;      /* [n_terms] -> length of each list */
    int*    inv_cap;      /* [n_terms] -> capacity */

    /* Per-document L2 norm (for cosine similarity) */
    float* doc_norms;    /* [n_docs] */

    /* BM25 length statistics */
    float  avgdl;        /* average document length */
    int*   doc_len;      /* [n_docs] token count per document */
} KnowledgeBase;

typedef struct {
    char*  question;
    char*  answer;
    char*  source;
    float  score;
} KbResult;

/* Load all CSVs from data_dir and build TF-IDF index */
KnowledgeBase* kb_create (const char* data_dir);
void           kb_free   (KnowledgeBase* kb);
int            kb_size   (const KnowledgeBase* kb);

/* Search: returns top_k results sorted by score descending.
   Results array has top_k entries; caller frees with kb_results_free. */
KbResult* kb_search (KnowledgeBase* kb, const char* query, int top_k);
void      kb_results_free(KbResult* results, int n);

/* Normalize query using rephrase rules */
void kb_normalize_query(const char* in, char* out, int out_cap);

#endif /* KNOWLEDGE_BASE_H */
