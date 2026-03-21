#include "knowledge_base.h"
#include "csv.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>

#ifdef PLATFORM_WINDOWS
#  include <windows.h>
#else
#  include <dirent.h>
#endif

#define BM25_K1  1.5f
#define BM25_B   0.75f

/* ─────────────────────────────────────────────────────────────────────────
   Stopwords
   ───────────────────────────────────────────────────────────────────────── */
static const char* STOPWORDS[] = {
    "a","an","the","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could",
    "should","may","might","shall","can","in","on","at","to",
    "for","of","with","by","from","up","about","into","and",
    "but","or","nor","not","that","this","it","its","if","as",
    "also","how","what","when","where","which","who","why","then",
    "there","some","any","all","no","more","most","other","over",
    "under","use","used","make","get","set","put","than","too",
    "very","just","each","both","either",
    NULL
};

static int is_stopword(const char* w) {
    for (int i = 0; STOPWORDS[i]; i++)
        if (strcmp(w, STOPWORDS[i]) == 0) return 1;
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────
   Finance synonyms (~50 most important)
   ───────────────────────────────────────────────────────────────────────── */
typedef struct { const char* word; const char* syns[6]; } SynEntry;
static const SynEntry SYNONYMS[] = {
    {"stocks",      {"equities","shares","equity","stock",NULL,NULL}},
    {"bonds",       {"fixed","income","debt","treasuries","securities",NULL}},
    {"roi",         {"return","profit","gain","investment",NULL,NULL}},
    {"sharpe",      {"risk","adjusted","return","ratio",NULL,NULL}},
    {"crypto",      {"cryptocurrency","bitcoin","blockchain","digital",NULL,NULL}},
    {"bitcoin",     {"btc","crypto","cryptocurrency","digital",NULL,NULL}},
    {"inflation",   {"cpi","prices","purchasing","power","rate",NULL}},
    {"recession",   {"downturn","contraction","gdp","negative",NULL,NULL}},
    {"compound",    {"interest","growth","reinvest","exponential",NULL,NULL}},
    {"etf",         {"fund","index","diversification","passive",NULL,NULL}},
    {"hedge",       {"protection","risk","management","offset",NULL,NULL}},
    {"portfolio",   {"diversification","allocation","assets","weights",NULL,NULL}},
    {"dividend",    {"income","yield","payout","shareholders",NULL,NULL}},
    {"mortgage",    {"home","loan","real","estate","interest",NULL}},
    {"reit",        {"real","estate","investment","trust","property",NULL}},
    {"pe",          {"price","earnings","valuation","multiple","ratio",NULL}},
    {"dcf",         {"discounted","cash","flow","valuation","intrinsic",NULL}},
    {"fed",         {"federal","reserve","fomc","central","bank",NULL}},
    {"volatility",  {"risk","vix","standard","deviation","fluctuation",NULL}},
    {"ira",         {"retirement","account","tax","traditional","roth",NULL}},
    {"401k",        {"retirement","employer","match","tax","deferred",NULL}},
    {"roth",        {"ira","retirement","tax","free","growth",NULL}},
    {"options",     {"calls","puts","derivatives","strike","premium",NULL}},
    {"futures",     {"derivatives","contracts","commodities","leverage",NULL,NULL}},
    {"forex",       {"currency","exchange","fx","trading","pairs",NULL}},
    {"liquidity",   {"cash","flow","assets","quick","ratio",NULL}},
    {"leverage",    {"debt","margin","borrowed","ratio","multiplier",NULL}},
    {"beta",        {"volatility","market","correlation","risk","measure",NULL}},
    {"alpha",       {"excess","return","performance","market","beat",NULL}},
    {"ebitda",      {"earnings","operating","income","depreciation","amortization",NULL}},
    {"eps",         {"earnings","per","share","profit","quarter",NULL}},
    {"roe",         {"return","equity","profitability","ratio",NULL,NULL}},
    {"roa",         {"return","assets","efficiency","profitability",NULL,NULL}},
    {"wacc",        {"cost","capital","weighted","average","discount",NULL}},
    {"npv",         {"net","present","value","discounted","cash","flow"}},
    {"irr",         {"internal","rate","return","hurdle","project",NULL}},
    {"gdp",         {"economy","growth","output","recession","expansion",NULL}},
    {"cpi",         {"inflation","prices","consumer","index","purchasing",NULL}},
    {"pe_ratio",    {"price","earnings","valuation","multiple",NULL,NULL}},
    {"buyback",     {"repurchase","shares","treasury","stock","return",NULL}},
    {"ipo",         {"initial","public","offering","listing","shares",NULL}},
    {"spac",        {"blank","check","merger","acquisition","listing",NULL}},
    {"venture",     {"startup","equity","seed","funding","capital",NULL}},
    {"private",     {"equity","buyout","leveraged","illiquid","fund",NULL}},
    {"commodity",   {"gold","oil","silver","wheat","raw",NULL}},
    {"rebalance",   {"allocation","drift","portfolio","adjust","weights",NULL}},
    {"drawdown",    {"loss","peak","trough","recovery","risk",NULL}},
    {"yield",       {"interest","rate","bond","income","return",NULL}},
    {"spread",      {"difference","bid","ask","yield","premium",NULL}},
    {"arbitrage",   {"mispricing","risk","free","profit","exploit",NULL}},
    {NULL,          {NULL,NULL,NULL,NULL,NULL,NULL}}
};

/* ─────────────────────────────────────────────────────────────────────────
   Hash map (open addressing, FNV-1a) for term -> id
   ───────────────────────────────────────────────────────────────────────── */
static uint32_t fnv1a(const char* s) {
    uint32_t h = 2166136261u;
    for (; *s; s++) h = (h ^ (unsigned char)*s) * 16777619u;
    return h;
}

static int term_lookup(KnowledgeBase* kb, const char* w) {
    uint32_t h = fnv1a(w) % (uint32_t)kb->ht_cap;
    while (kb->ht_keys[h] != -1) {
        if (strcmp(kb->terms[kb->ht_vals[h]], w) == 0) return kb->ht_vals[h];
        h = (h + 1) % (uint32_t)kb->ht_cap;
    }
    return -1;
}

/* Rehash into a new table of size new_cap */
static void ht_rehash(KnowledgeBase* kb, int new_cap) {
    int* new_keys = (int*)xmalloc((size_t)new_cap * sizeof(int));
    int* new_vals = (int*)xmalloc((size_t)new_cap * sizeof(int));
    for (int i = 0; i < new_cap; i++) new_keys[i] = -1;

    for (int i = 0; i < kb->ht_cap; i++) {
        if (kb->ht_keys[i] == -1) continue;
        int   id = kb->ht_vals[i];
        uint32_t h = fnv1a(kb->terms[id]) % (uint32_t)new_cap;
        while (new_keys[h] != -1) h = (h + 1) % (uint32_t)new_cap;
        new_keys[h] = id;
        new_vals[h] = id;
    }
    free(kb->ht_keys);
    free(kb->ht_vals);
    kb->ht_keys = new_keys;
    kb->ht_vals = new_vals;
    kb->ht_cap  = new_cap;
}

/* Insert term; return its id (existing or new). Returns -1 on overflow. */
static int term_insert(KnowledgeBase* kb, const char* w) {
    int id = term_lookup(kb, w);
    if (id >= 0) return id;

    /* Grow hash table when load > 60% */
    if (kb->n_terms >= (int)((double)kb->ht_cap * 0.6))
        ht_rehash(kb, kb->ht_cap * 2 + 16);

    if (kb->n_terms >= KB_MAX_VOCAB) return -1;  /* hard cap */

    id = kb->n_terms++;
    kb->terms[id] = xstrdup(w);

    uint32_t h = fnv1a(w) % (uint32_t)kb->ht_cap;
    while (kb->ht_keys[h] != -1) h = (h + 1) % (uint32_t)kb->ht_cap;
    kb->ht_keys[h] = id;
    kb->ht_vals[h] = id;
    return id;
}

/* ─────────────────────────────────────────────────────────────────────────
   Tokenizer: lowercase, split on non-alnum, remove stopwords,
              generate bigrams, expand synonyms
   ───────────────────────────────────────────────────────────────────────── */
static char** tokenize_text(const char* text, int* out_n, int include_bigrams) {
    int   cap    = 128;
    char** tokens = (char**)xmalloc((size_t)cap * sizeof(char*));
    int    n      = 0;

    /* -- Unigrams -------------------------------------------------------- */
    const char* p = text;
    while (*p) {
        while (*p && !isalnum((unsigned char)*p)) p++;
        if (!*p) break;
        const char* start = p;
        while (*p && isalnum((unsigned char)*p)) p++;
        size_t len = (size_t)(p - start);
        if (len < 2 || len > 40) continue;

        char w[64] = {0};
        size_t copy = len < 63 ? len : 63;
        for (size_t i = 0; i < copy; i++)
            w[i] = (char)tolower((unsigned char)start[i]);
        w[copy] = '\0';

        if (is_stopword(w)) continue;

        if (n >= cap - 2) { cap *= 2; tokens = (char**)xrealloc(tokens, (size_t)cap * sizeof(char*)); }
        tokens[n++] = xstrdup(w);
    }

    /* -- Bigrams --------------------------------------------------------- */
    if (include_bigrams) {
        int orig_n = n;
        for (int i = 0; i < orig_n - 1; i++) {
            if (n >= cap - 2) { cap *= 2; tokens = (char**)xrealloc(tokens, (size_t)cap * sizeof(char*)); }
            size_t la = strlen(tokens[i]);
            size_t lb = strlen(tokens[i + 1]);
            char* bg  = (char*)xmalloc(la + lb + 2);
            memcpy(bg, tokens[i], la);
            bg[la] = '_';
            memcpy(bg + la + 1, tokens[i + 1], lb + 1);
            tokens[n++] = bg;
        }
    }

    /* -- Synonym expansion ---------------------------------------------- */
    int orig_n2 = n;
    for (int i = 0; i < orig_n2; i++) {
        for (int si = 0; SYNONYMS[si].word; si++) {
            if (strcmp(tokens[i], SYNONYMS[si].word) != 0) continue;
            for (int j = 0; j < 6 && SYNONYMS[si].syns[j]; j++) {
                if (n >= cap - 1) { cap *= 2; tokens = (char**)xrealloc(tokens, (size_t)cap * sizeof(char*)); }
                tokens[n++] = xstrdup(SYNONYMS[si].syns[j]);
            }
            break;
        }
    }

    *out_n = n;
    return tokens;
}

/* ─────────────────────────────────────────────────────────────────────────
   Inverted index helpers
   ───────────────────────────────────────────────────────────────────────── */
static void inv_append(KnowledgeBase* kb, int term_id, int doc_id, float tfidf) {
    if (term_id < 0 || term_id >= KB_MAX_VOCAB) return;
    if (kb->inv_len[term_id] >= kb->inv_cap[term_id]) {
        int nc = kb->inv_cap[term_id] * 2 + 4;
        kb->inv_doc_ids[term_id] = (int*)  xrealloc(kb->inv_doc_ids[term_id], (size_t)nc * sizeof(int));
        kb->inv_tfidf[term_id]   = (float*)xrealloc(kb->inv_tfidf[term_id],   (size_t)nc * sizeof(float));
        kb->inv_cap[term_id]     = nc;
    }
    kb->inv_doc_ids[term_id][kb->inv_len[term_id]] = doc_id;
    kb->inv_tfidf  [term_id][kb->inv_len[term_id]] = tfidf;
    kb->inv_len[term_id]++;
}

/* ─────────────────────────────────────────────────────────────────────────
   Query normalization  (mirrors _REPHRASE_RULES in reasoning_engine.py)
   ───────────────────────────────────────────────────────────────────────── */
void kb_normalize_query(const char* in, char* out, int out_cap) {
    strncpy(out, in, (size_t)out_cap - 1);
    out[out_cap - 1] = '\0';

    static const struct { const char* from; const char* to; } rules[] = {
        {"what's",                "what is"},
        {"whats",                 "what is"},
        {"how's",                 "how does"},
        {"hows",                  "how does"},
        {"i don't understand",    "explain"},
        {"help me understand",    "explain"},
        {"can you explain",       "explain"},
        {"tell me about",         "what is"},
        {"please ",               ""},
        {"kind of ",              ""},
        {"sort of ",              ""},
        {NULL, NULL}
    };

    for (int r = 0; rules[r].from; r++) {
        /* Work on a lowercase copy to find matches case-insensitively */
        char lower[4096];
        strncpy(lower, out, sizeof(lower) - 1);
        lower[sizeof(lower) - 1] = '\0';
        for (int i = 0; lower[i]; i++) lower[i] = (char)tolower((unsigned char)lower[i]);

        char* pos = strstr(lower, rules[r].from);
        if (!pos) continue;

        size_t offset = (size_t)(pos - lower);
        size_t flen   = strlen(rules[r].from);
        size_t tlen   = strlen(rules[r].to);
        size_t cur    = strlen(out);

        char result[4096];
        memcpy(result, out, offset);
        memcpy(result + offset, rules[r].to, tlen);
        memcpy(result + offset + tlen, out + offset + flen, cur - offset - flen + 1);
        strncpy(out, result, (size_t)out_cap - 1);
        out[out_cap - 1] = '\0';
    }

    /* Trim trailing whitespace */
    int len = (int)strlen(out);
    while (len > 0 && isspace((unsigned char)out[len - 1])) out[--len] = '\0';
}

/* ─────────────────────────────────────────────────────────────────────────
   kb_create  — load CSVs, build TF-IDF inverted index
   ───────────────────────────────────────────────────────────────────────── */
KnowledgeBase* kb_create(const char* data_dir) {
    KnowledgeBase* kb = (KnowledgeBase*)xcalloc(1, sizeof(KnowledgeBase));

    kb->cap_docs    = KB_MAX_DOCS;
    kb->docs        = (KbDoc*)xmalloc((size_t)kb->cap_docs * sizeof(KbDoc));

    kb->terms       = (char**)xcalloc(KB_MAX_VOCAB, sizeof(char*));
    kb->ht_cap      = KB_MAX_VOCAB * 2 + 16;
    kb->ht_keys     = (int*)xmalloc((size_t)kb->ht_cap * sizeof(int));
    kb->ht_vals     = (int*)xmalloc((size_t)kb->ht_cap * sizeof(int));
    for (int i = 0; i < kb->ht_cap; i++) kb->ht_keys[i] = -1;

    kb->inv_doc_ids = (int**)  xcalloc(KB_MAX_VOCAB, sizeof(int*));
    kb->inv_tfidf   = (float**)xcalloc(KB_MAX_VOCAB, sizeof(float*));
    kb->inv_len     = (int*)   xcalloc(KB_MAX_VOCAB, sizeof(int));
    kb->inv_cap     = (int*)   xcalloc(KB_MAX_VOCAB, sizeof(int));

    /* Recognized column name aliases */
    static const char* Q_NAMES[] = {"question","q","query","input","prompt","ask",NULL};
    static const char* A_NAMES[] = {"answer","a","response","output","reply","text",NULL};

    /* ── Step 1: Load all CSVs ─────────────────────────────────────── */
    printf("  [KB] Loading CSVs from %s/\n", data_dir);

#ifdef PLATFORM_WINDOWS
    char pattern[512];
    snprintf(pattern, sizeof(pattern), "%s\\*.csv", data_dir);
    WIN32_FIND_DATAA ffd;
    HANDLE hFind = FindFirstFileA(pattern, &ffd);
    if (hFind == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "  [KB] No CSVs found in %s\n", data_dir);
        return kb;
    }
    do {
        char path[512];
        snprintf(path, sizeof(path), "%s\\%s", data_dir, ffd.cFileName);
        const char* fname = ffd.cFileName;
#else
    DIR* dir = opendir(data_dir);
    if (!dir) { fprintf(stderr, "  [KB] Cannot open %s\n", data_dir); return kb; }
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (!strstr(entry->d_name, ".csv")) continue;
        char path[512];
        snprintf(path, sizeof(path), "%s/%s", data_dir, entry->d_name);
        const char* fname = entry->d_name;
#endif
        CsvTable* t = csv_load(path);
        if (!t) { continue; }

        /* Locate q/a columns */
        int q_col = -1, a_col = -1;
        for (int qi = 0; Q_NAMES[qi] && q_col < 0; qi++) q_col = csv_col(t, Q_NAMES[qi]);
        for (int ai = 0; A_NAMES[ai] && a_col < 0; ai++) a_col = csv_col(t, A_NAMES[ai]);
        if (q_col < 0 || a_col < 0) { csv_free(t); continue; }

        /* Source = filename without .csv */
        char source[128] = {0};
        strncpy(source, fname, sizeof(source) - 1);
        char* dot = strrchr(source, '.');
        if (dot) *dot = '\0';

        for (int r = 0; r < t->n_rows && kb->n_docs < kb->cap_docs; r++) {
            const char* q = csv_get(t, r, q_col);
            const char* a = csv_get(t, r, a_col);
            if (!q || !*q || !a || !*a) continue;
            if (strcmp(q, "nan") == 0 || strcmp(a, "nan") == 0) continue;

            KbDoc* doc    = &kb->docs[kb->n_docs++];
            doc->question = xstrdup(q);
            doc->answer   = xstrdup(a);
            doc->source   = xstrdup(source);
        }
        csv_free(t);
#ifdef PLATFORM_WINDOWS
    } while (FindNextFileA(hFind, &ffd));
    FindClose(hFind);
#else
    }
    closedir(dir);
#endif

    printf("  [KB] Loaded %d documents\n", kb->n_docs);
    if (kb->n_docs == 0) return kb;

    /* ── Step 2: Build per-doc term frequencies & vocabulary ────────── */

    typedef struct { int* term_ids; float* tf; int n; int n_toks; } DocTerms;
    DocTerms* all_dt = (DocTerms*)xcalloc((size_t)kb->n_docs, sizeof(DocTerms));

    /* document-frequency counter (per vocab slot) */
    int* doc_freq = (int*)xcalloc(KB_MAX_VOCAB, sizeof(int));

    for (int d = 0; d < kb->n_docs; d++) {
        const char* q    = kb->docs[d].question;
        const char* a    = kb->docs[d].answer;
        size_t       qlen = strlen(q);
        size_t       alen = strlen(a);
        char*        combined = (char*)xmalloc(qlen + alen + 2);
        memcpy(combined, q, qlen);
        combined[qlen] = ' ';
        memcpy(combined + qlen + 1, a, alen + 1);

        int    n_toks;
        char** toks = tokenize_text(combined, &n_toks, 1);
        free(combined);

        /* Local term-count table (linear scan; docs are small) */
        int   tmp_cap   = 64;
        int*  term_ids  = (int*)xmalloc((size_t)tmp_cap * sizeof(int));
        int*  term_cnts = (int*)xcalloc((size_t)tmp_cap, sizeof(int));
        int   n_unique  = 0;

        for (int ti = 0; ti < n_toks; ti++) {
            int id = term_insert(kb, toks[ti]);
            free(toks[ti]);
            if (id < 0) continue;

            int found = -1;
            for (int j = 0; j < n_unique; j++) {
                if (term_ids[j] == id) { found = j; break; }
            }
            if (found < 0) {
                if (n_unique >= tmp_cap) {
                    int old_cap = tmp_cap;
                    tmp_cap *= 2;
                    term_ids  = (int*)xrealloc(term_ids,  (size_t)tmp_cap * sizeof(int));
                    term_cnts = (int*)xrealloc(term_cnts, (size_t)tmp_cap * sizeof(int));
                    memset(term_cnts + old_cap, 0, (size_t)(tmp_cap - old_cap) * sizeof(int));
                }
                term_ids[n_unique]  = id;
                term_cnts[n_unique] = 1;
                n_unique++;
                doc_freq[id]++;
            } else {
                term_cnts[found]++;
            }
        }
        free(toks);

        /* TF = count / total_tokens */
        float* tf    = (float*)xmalloc((size_t)n_unique * sizeof(float));
        float  total = (float)(n_toks > 0 ? n_toks : 1);
        for (int j = 0; j < n_unique; j++) tf[j] = (float)term_cnts[j] / total;
        free(term_cnts);

        all_dt[d].term_ids = term_ids;
        all_dt[d].tf       = tf;
        all_dt[d].n        = n_unique;
        all_dt[d].n_toks   = n_toks;
    }

    /* ── Step 3: Compute IDF, build inverted index, compute doc norms ── */
    kb->idf = (float*)xmalloc((size_t)kb->n_terms * sizeof(float));
    for (int t = 0; t < kb->n_terms; t++) {
        /* Smoothed IDF: log((N+1)/(df+1)) + 1  (sklearn default) */
        kb->idf[t] = logf((float)(kb->n_docs + 1) / (float)(doc_freq[t] + 1)) + 1.0f;
    }

    /* Allocate and populate doc_len; compute avgdl */
    kb->doc_len = (int*)xmalloc((size_t)kb->n_docs * sizeof(int));
    {
        long long total_len = 0;
        for (int d = 0; d < kb->n_docs; d++) {
            kb->doc_len[d] = all_dt[d].n_toks;
            total_len += all_dt[d].n_toks;
        }
        kb->avgdl = (kb->n_docs > 0) ? (float)total_len / (float)kb->n_docs : 1.0f;
    }

    kb->doc_norms = (float*)xcalloc((size_t)kb->n_docs, sizeof(float));

    for (int d = 0; d < kb->n_docs; d++) {
        DocTerms* dt = &all_dt[d];
        for (int j = 0; j < dt->n; j++) {
            int   tid = dt->term_ids[j];
            float tv  = dt->tf[j] * kb->idf[tid];
            inv_append(kb, tid, d, tv);
            kb->doc_norms[d] += tv * tv;
        }
        free(dt->term_ids);
        free(dt->tf);
    }

    for (int d = 0; d < kb->n_docs; d++)
        kb->doc_norms[d] = sqrtf(kb->doc_norms[d] + 1e-10f);

    free(all_dt);
    free(doc_freq);

    printf("  [KB] Index built: %d terms, %d docs\n", kb->n_terms, kb->n_docs);
    return kb;
}

/* ─────────────────────────────────────────────────────────────────────────
   kb_free
   ───────────────────────────────────────────────────────────────────────── */
void kb_free(KnowledgeBase* kb) {
    if (!kb) return;
    for (int d = 0; d < kb->n_docs; d++) {
        free(kb->docs[d].question);
        free(kb->docs[d].answer);
        free(kb->docs[d].source);
    }
    free(kb->docs);

    for (int t = 0; t < kb->n_terms; t++) free(kb->terms[t]);
    free(kb->terms);
    free(kb->ht_keys);
    free(kb->ht_vals);
    free(kb->idf);

    for (int t = 0; t < kb->n_terms; t++) {
        free(kb->inv_doc_ids[t]);
        free(kb->inv_tfidf[t]);
    }
    free(kb->inv_doc_ids);
    free(kb->inv_tfidf);
    free(kb->inv_len);
    free(kb->inv_cap);
    free(kb->doc_norms);
    free(kb->doc_len);
    free(kb);
}

int kb_size(const KnowledgeBase* kb) { return kb ? kb->n_docs : 0; }

/* ─────────────────────────────────────────────────────────────────────────
   kb_search — cosine-similarity TF-IDF retrieval with exact-match boost
   ───────────────────────────────────────────────────────────────────────── */
KbResult* kb_search(KnowledgeBase* kb, const char* query, int top_k) {
    KbResult* results = (KbResult*)xcalloc((size_t)top_k, sizeof(KbResult));
    if (kb->n_docs == 0 || top_k <= 0) return results;

    /* Normalize query */
    char norm_q[4096];
    kb_normalize_query(query, norm_q, sizeof(norm_q));

    /* Tokenize query */
    int    n_qtoks;
    char** qtoks = tokenize_text(norm_q, &n_qtoks, 1);

    float* scores = (float*)xcalloc((size_t)kb->n_docs, sizeof(float));

    /* Query vector norm accumulator */
    float q_norm = 0.0f;

    float avgdl = (kb->avgdl > 0.0f) ? kb->avgdl : 1.0f;

    for (int qi = 0; qi < n_qtoks; qi++) {
        int tid = term_lookup(kb, qtoks[qi]);
        free(qtoks[qi]);
        if (tid < 0) continue;

        float qtf    = 1.0f / (float)(n_qtoks > 0 ? n_qtoks : 1);
        float qtfidf = qtf * kb->idf[tid];
        q_norm += qtfidf * qtfidf;

        for (int i = 0; i < kb->inv_len[tid]; i++) {
            int   doc_id  = kb->inv_doc_ids[tid][i];
            float dtfidf  = kb->inv_tfidf[tid][i];
            /* Recover raw tf from stored tfidf value */
            float tf      = dtfidf / kb->idf[tid];
            float dl      = (float)kb->doc_len[doc_id];
            /* BM25 per-term score */
            float bm25    = kb->idf[tid]
                            * (tf * (BM25_K1 + 1.0f))
                            / (tf + BM25_K1 * (1.0f - BM25_B + BM25_B * (dl / avgdl)));
            scores[doc_id] += qtfidf * bm25;
        }
    }
    free(qtoks);

    q_norm = sqrtf(q_norm + 1e-10f);

    /* Normalise to cosine similarity */
    for (int d = 0; d < kb->n_docs; d++)
        scores[d] /= q_norm * kb->doc_norms[d];

    /* Exact / substring boost */
    char q_lower[2048];
    strncpy(q_lower, norm_q, sizeof(q_lower) - 1);
    q_lower[sizeof(q_lower) - 1] = '\0';
    for (int i = 0; q_lower[i]; i++) q_lower[i] = (char)tolower((unsigned char)q_lower[i]);

    for (int d = 0; d < kb->n_docs; d++) {
        char doc_q_lower[2048];
        strncpy(doc_q_lower, kb->docs[d].question, sizeof(doc_q_lower) - 1);
        doc_q_lower[sizeof(doc_q_lower) - 1] = '\0';
        for (int i = 0; doc_q_lower[i]; i++)
            doc_q_lower[i] = (char)tolower((unsigned char)doc_q_lower[i]);

        if (strcmp(q_lower, doc_q_lower) == 0)
            scores[d] = scores[d] * 2.0f + 1.0f;        /* exact match */
        else if (strstr(doc_q_lower, q_lower) || strstr(q_lower, doc_q_lower))
            scores[d] *= 1.5f;                            /* substring match */
    }

    /* Partial selection sort to extract top_k */
    char* used    = (char*)xcalloc((size_t)kb->n_docs, 1);
    int   found   = 0;
    int   k_clamp = top_k < kb->n_docs ? top_k : kb->n_docs;

    for (int i = 0; i < k_clamp; i++) {
        float best    = -1e38f;
        int   best_id = -1;
        for (int d = 0; d < kb->n_docs; d++) {
            if (!used[d] && scores[d] > best) { best = scores[d]; best_id = d; }
        }
        if (best_id < 0) break;
        used[best_id] = 1;

        int d = best_id;
        results[found].question = xstrdup(kb->docs[d].question);
        results[found].answer   = xstrdup(kb->docs[d].answer);
        results[found].source   = xstrdup(kb->docs[d].source);
        results[found].score    = best;
        found++;
    }

    free(scores);
    free(used);
    return results;
}

/* ─────────────────────────────────────────────────────────────────────────
   kb_results_free
   ───────────────────────────────────────────────────────────────────────── */
void kb_results_free(KbResult* results, int n) {
    if (!results) return;
    for (int i = 0; i < n; i++) {
        free(results[i].question);
        free(results[i].answer);
        free(results[i].source);
    }
    free(results);
}
