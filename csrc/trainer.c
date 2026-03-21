#include "trainer.h"
#include "csv.h"
#include "math_ops.h"
#include "tokenizer.h"
#include "model.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#ifdef PLATFORM_WINDOWS
#include <windows.h>
#include <io.h>
#else
#include <dirent.h>
#endif

/* ── Dataset ─────────────────────────────────────────────────────── */
Dataset* dataset_create(int* tokens, size_t n_tokens, int block_size, int stride) {
    Dataset* d = (Dataset*)xcalloc(1, sizeof(Dataset));
    d->tokens    = tokens;
    d->n_tokens  = n_tokens;
    d->block_size = block_size;
    d->stride    = stride > 0 ? stride : block_size;
    /* n_samples = max(0, (n_tokens - block_size - 1) / stride + 1) */
    if (n_tokens > (size_t)(block_size + 1))
        d->n_samples = (n_tokens - block_size - 1) / d->stride + 1;
    else
        d->n_samples = 0;
    return d;
}

void dataset_free(Dataset* d) {
    if (!d) return;
    free(d->tokens);
    free(d);
}

void dataset_get(const Dataset* d, size_t idx, int* x_out, int* y_out) {
    size_t start = idx * d->stride;
    for (int i = 0; i <= d->block_size; i++) {
        int tok = (start + i < d->n_tokens) ? d->tokens[start + i] : TOK_PAD_ID;
        if (i < d->block_size) x_out[i] = tok;
        if (i > 0)             y_out[i-1] = tok;
    }
}

/* ── History ─────────────────────────────────────────────────────── */
TrainHistory* history_create(void) {
    TrainHistory* h = (TrainHistory*)xcalloc(1, sizeof(TrainHistory));
    h->cap_steps  = 1024;
    h->losses     = (float*)xmalloc(h->cap_steps * sizeof(float));
    h->lrs        = (float*)xmalloc(h->cap_steps * sizeof(float));
    h->steps      = (int*)xmalloc(h->cap_steps * sizeof(int));
    h->cap_epochs = 64;
    h->val_epochs = (EpochRecord*)xmalloc(h->cap_epochs * sizeof(EpochRecord));
    return h;
}

void history_free(TrainHistory* h) {
    if (!h) return;
    free(h->losses); free(h->lrs); free(h->steps); free(h->val_epochs);
    free(h);
}

void history_add_step(TrainHistory* h, int step, float loss, float lr) {
    if (h->n_steps >= h->cap_steps) {
        h->cap_steps *= 2;
        h->losses = (float*)xrealloc(h->losses, h->cap_steps * sizeof(float));
        h->lrs    = (float*)xrealloc(h->lrs,    h->cap_steps * sizeof(float));
        h->steps  = (int*)xrealloc(h->steps,    h->cap_steps * sizeof(int));
    }
    h->losses[h->n_steps] = loss;
    h->lrs[h->n_steps]    = lr;
    h->steps[h->n_steps]  = step;
    h->n_steps++;
}

void history_add_epoch(TrainHistory* h, EpochRecord rec) {
    if (h->n_epochs >= h->cap_epochs) {
        h->cap_epochs *= 2;
        h->val_epochs = (EpochRecord*)xrealloc(h->val_epochs, h->cap_epochs * sizeof(EpochRecord));
    }
    h->val_epochs[h->n_epochs++] = rec;
}

/* ── AdamW ───────────────────────────────────────────────────────── */
/* Initialize moment buffers to zero */
void optimizer_init(Model* m) {
    const ModelConfig* c = &m->cfg;
    size_t embed_sz = (size_t)c->vocab_size * c->d_model;

    m->m_embed = (float*)xcalloc(embed_sz, sizeof(float));
    m->v_embed = (float*)xcalloc(embed_sz, sizeof(float));
    m->m_ln_f  = (float*)xcalloc(c->d_model, sizeof(float));
    m->v_ln_f  = (float*)xcalloc(c->d_model, sizeof(float));

    m->m_blocks = (Block*)xcalloc(c->n_layers, sizeof(Block));
    m->v_blocks = (Block*)xcalloc(c->n_layers, sizeof(Block));
    for (int l = 0; l < c->n_layers; l++) {
        Block* mb = &m->m_blocks[l];
        Block* vb = &m->v_blocks[l];

        /* m buffers */
        mb->ln1_w    = (float*)xcalloc(c->d_model, sizeof(float));
        mb->attn_qkv = (float*)xcalloc(3 * c->d_model * c->d_model, sizeof(float));
        mb->attn_proj= (float*)xcalloc(c->d_model * c->d_model, sizeof(float));
        mb->ln2_w    = (float*)xcalloc(c->d_model, sizeof(float));
        mb->ffn_gate = (float*)xcalloc(c->d_ff * c->d_model, sizeof(float));
        mb->ffn_up   = (float*)xcalloc(c->d_ff * c->d_model, sizeof(float));
        mb->ffn_down = (float*)xcalloc(c->d_model * c->d_ff, sizeof(float));

        /* v buffers — separate, distinct memory */
        vb->ln1_w    = (float*)xcalloc(c->d_model, sizeof(float));
        vb->attn_qkv = (float*)xcalloc(3 * c->d_model * c->d_model, sizeof(float));
        vb->attn_proj= (float*)xcalloc(c->d_model * c->d_model, sizeof(float));
        vb->ln2_w    = (float*)xcalloc(c->d_model, sizeof(float));
        vb->ffn_gate = (float*)xcalloc(c->d_ff * c->d_model, sizeof(float));
        vb->ffn_up   = (float*)xcalloc(c->d_ff * c->d_model, sizeof(float));
        vb->ffn_down = (float*)xcalloc(c->d_model * c->d_ff, sizeof(float));
    }
}

/* Apply AdamW update to one parameter array */
static void adamw_update(float* param, float* grad, float* m_buf, float* v_buf,
                          size_t n, float lr, float beta1, float beta2, float eps,
                          float weight_decay, int t, int is_bias) {
    float bc1 = 1.0f - powf(beta1, (float)t);
    float bc2 = 1.0f - powf(beta2, (float)t);
    for (size_t i = 0; i < n; i++) {
        float g = grad[i];
        m_buf[i] = beta1 * m_buf[i] + (1.0f - beta1) * g;
        v_buf[i] = beta2 * v_buf[i] + (1.0f - beta2) * g * g;
        float m_hat = m_buf[i] / bc1;
        float v_hat = v_buf[i] / bc2;
        float update = lr * m_hat / (sqrtf(v_hat) + eps);
        if (!is_bias && weight_decay > 0.0f)
            update += lr * weight_decay * param[i];
        param[i] -= update;
    }
}

void optimizer_step(Model* m, float lr, float beta1, float beta2,
                    float eps, float weight_decay, int step) {
    if (!m->grad_embed) return;
    const ModelConfig* c = &m->cfg;

    int t = step + 1;  /* 1-indexed for bias correction */

    /* Gradient clipping: compute global norm */
    float gnorm = 0.0f;
    {
        size_t embed_sz = (size_t)c->vocab_size * c->d_model;
        for (size_t i = 0; i < embed_sz; i++) gnorm += m->grad_embed[i] * m->grad_embed[i];
        for (int i = 0; i < c->d_model; i++) gnorm += m->grad_ln_f[i] * m->grad_ln_f[i];
        for (int l = 0; l < c->n_layers; l++) {
            Block* g = &m->grad_blocks[l];
            for (int i = 0; i < c->d_model; i++) gnorm += g->ln1_w[i]*g->ln1_w[i] + g->ln2_w[i]*g->ln2_w[i];
            for (int i = 0; i < 3*c->d_model*c->d_model; i++) gnorm += g->attn_qkv[i]*g->attn_qkv[i];
            for (int i = 0; i < c->d_model*c->d_model; i++) gnorm += g->attn_proj[i]*g->attn_proj[i];
            for (int i = 0; i < c->d_ff*c->d_model; i++) gnorm += g->ffn_gate[i]*g->ffn_gate[i] + g->ffn_up[i]*g->ffn_up[i];
            for (int i = 0; i < c->d_model*c->d_ff; i++) gnorm += g->ffn_down[i]*g->ffn_down[i];
        }
    }
    gnorm = sqrtf(gnorm);
    float clip = TRAIN_GRAD_CLIP;
    float scale = gnorm > clip ? clip / gnorm : 1.0f;
    if (scale < 1.0f) {
        /* Scale all gradients */
        size_t embed_sz = (size_t)c->vocab_size * c->d_model;
        for (size_t i = 0; i < embed_sz; i++) m->grad_embed[i] *= scale;
        for (int i = 0; i < c->d_model; i++) m->grad_ln_f[i] *= scale;
        for (int l = 0; l < c->n_layers; l++) {
            Block* g = &m->grad_blocks[l];
            for (int i = 0; i < c->d_model; i++) { g->ln1_w[i]*=scale; g->ln2_w[i]*=scale; }
            for (int i = 0; i < 3*c->d_model*c->d_model; i++) g->attn_qkv[i]*=scale;
            for (int i = 0; i < c->d_model*c->d_model; i++) g->attn_proj[i]*=scale;
            for (int i = 0; i < c->d_ff*c->d_model; i++) { g->ffn_gate[i]*=scale; g->ffn_up[i]*=scale; }
            for (int i = 0; i < c->d_model*c->d_ff; i++) g->ffn_down[i]*=scale;
        }
    }

    /* Update all parameters */
    adamw_update(m->embed, m->grad_embed, m->m_embed, m->v_embed,
                 (size_t)c->vocab_size * c->d_model, lr, beta1, beta2, eps, weight_decay, t, 0);
    adamw_update(m->ln_f, m->grad_ln_f, m->m_ln_f, m->v_ln_f,
                 c->d_model, lr, beta1, beta2, eps, 0.0f, t, 1);  /* no WD for norms */

    for (int l = 0; l < c->n_layers; l++) {
        Block* p  = &m->blocks[l];
        Block* g  = &m->grad_blocks[l];
        Block* mb = &m->m_blocks[l];
        Block* vb = &m->v_blocks[l];
        adamw_update(p->ln1_w,    g->ln1_w,    mb->ln1_w,    vb->ln1_w,    c->d_model,               lr, beta1, beta2, eps, 0,            t, 1);
        adamw_update(p->attn_qkv, g->attn_qkv, mb->attn_qkv, vb->attn_qkv, 3*c->d_model*c->d_model,  lr, beta1, beta2, eps, weight_decay, t, 0);
        adamw_update(p->attn_proj,g->attn_proj, mb->attn_proj,vb->attn_proj, c->d_model*c->d_model,   lr, beta1, beta2, eps, weight_decay, t, 0);
        adamw_update(p->ln2_w,    g->ln2_w,    mb->ln2_w,    vb->ln2_w,    c->d_model,               lr, beta1, beta2, eps, 0,            t, 1);
        adamw_update(p->ffn_gate, g->ffn_gate, mb->ffn_gate, vb->ffn_gate, c->d_ff*c->d_model,       lr, beta1, beta2, eps, weight_decay, t, 0);
        adamw_update(p->ffn_up,   g->ffn_up,   mb->ffn_up,   vb->ffn_up,   c->d_ff*c->d_model,       lr, beta1, beta2, eps, weight_decay, t, 0);
        adamw_update(p->ffn_down, g->ffn_down, mb->ffn_down, vb->ffn_down, c->d_model*c->d_ff,       lr, beta1, beta2, eps, weight_decay, t, 0);
    }
}

void optimizer_zero_grad(Model* m) {
    if (!m->grad_embed) return;
    const ModelConfig* c = &m->cfg;
    vec_zero_f32(m->grad_embed, (size_t)c->vocab_size * c->d_model);
    vec_zero_f32(m->grad_ln_f,  c->d_model);
    for (int l = 0; l < c->n_layers; l++) {
        Block* g = &m->grad_blocks[l];
        vec_zero_f32(g->ln1_w,    c->d_model);
        vec_zero_f32(g->attn_qkv, 3*c->d_model*c->d_model);
        vec_zero_f32(g->attn_proj,c->d_model*c->d_model);
        vec_zero_f32(g->ln2_w,    c->d_model);
        vec_zero_f32(g->ffn_gate, c->d_ff*c->d_model);
        vec_zero_f32(g->ffn_up,   c->d_ff*c->d_model);
        vec_zero_f32(g->ffn_down, c->d_model*c->d_ff);
    }
}

float lr_schedule(int step, int total_steps, float lr, float min_lr, int warmup) {
    if (step < warmup)
        return lr * (float)step / (float)(warmup > 0 ? warmup : 1);
    float p = (float)(step - warmup) / (float)(total_steps - warmup > 0 ? total_steps - warmup : 1);
    return min_lr + (lr - min_lr) * 0.5f * (1.0f + cosf(3.14159265f * p));
}

/* ── Data loading ────────────────────────────────────────────────── */
static const char* Q_PREFIXES[] = {
    "Q: %s <SEP> A: %s",
    "Question: %s <SEP> Answer: %s",
    "%s <SEP> %s",
    NULL
};

static char* format_qa(const char* q, const char* a, uint64_t* rng) {
    int idx = (int)(rng_float(rng) * 3);
    const char* tmpl = Q_PREFIXES[idx];
    size_t n = strlen(q) + strlen(a) + 64;
    char* out = (char*)xmalloc(n);
    snprintf(out, n, tmpl, q, a);
    return out;
}

static char** load_texts(const char* data_dir, const char* specific_csv, int* out_n) {
    int cap = 256;
    char** texts = (char**)xmalloc(cap * sizeof(char*));
    int n = 0;
    uint64_t rng = 12345ULL;

    const char* Q_NAMES[] = {"question","q","query","input","prompt","ask",NULL};
    const char* A_NAMES[] = {"answer","a","response","output","reply","text",NULL};

    if (specific_csv) {
        /* Single file */
        CsvTable* t = csv_load(specific_csv);
        if (t) {
            int qc = -1, ac = -1;
            for (int qi = 0; Q_NAMES[qi] && qc < 0; qi++) qc = csv_col(t, Q_NAMES[qi]);
            for (int ai = 0; A_NAMES[ai] && ac < 0; ai++) ac = csv_col(t, A_NAMES[ai]);
            if (qc >= 0 && ac >= 0) {
                for (int r = 0; r < t->n_rows; r++) {
                    const char* q = csv_get(t, r, qc);
                    const char* a = csv_get(t, r, ac);
                    if (!q || !*q || !a || !*a || !strcmp(q,"nan") || !strcmp(a,"nan")) continue;
                    if (n >= cap) { cap *= 2; texts = (char**)xrealloc(texts, cap * sizeof(char*)); }
                    texts[n++] = format_qa(q, a, &rng);
                }
            }
            csv_free(t);
        }
    } else {
#ifdef PLATFORM_WINDOWS
        char pattern[512];
        snprintf(pattern, sizeof(pattern), "%s\\*.csv", data_dir);
        WIN32_FIND_DATAA ffd;
        HANDLE hFind = FindFirstFileA(pattern, &ffd);
        if (hFind != INVALID_HANDLE_VALUE) {
            do {
                char path[512];
                snprintf(path, sizeof(path), "%s\\%s", data_dir, ffd.cFileName);
                CsvTable* t = csv_load(path);
                if (!t) continue;
                int qc = -1, ac = -1;
                for (int qi = 0; Q_NAMES[qi] && qc < 0; qi++) qc = csv_col(t, Q_NAMES[qi]);
                for (int ai = 0; A_NAMES[ai] && ac < 0; ai++) ac = csv_col(t, A_NAMES[ai]);
                if (qc >= 0 && ac >= 0) {
                    for (int r = 0; r < t->n_rows; r++) {
                        const char* q = csv_get(t, r, qc);
                        const char* a = csv_get(t, r, ac);
                        if (!q||!*q||!a||!*a||!strcmp(q,"nan")||!strcmp(a,"nan")) continue;
                        if (n >= cap) { cap *= 2; texts = (char**)xrealloc(texts, cap * sizeof(char*)); }
                        texts[n++] = format_qa(q, a, &rng);
                    }
                }
                csv_free(t);
            } while (FindNextFileA(hFind, &ffd));
            FindClose(hFind);
        }
#else
        DIR* dir = opendir(data_dir);
        if (dir) {
            struct dirent* ent;
            while ((ent = readdir(dir)) != NULL) {
                if (!strstr(ent->d_name, ".csv")) continue;
                char path[512];
                snprintf(path, sizeof(path), "%s/%s", data_dir, ent->d_name);
                CsvTable* t = csv_load(path);
                if (!t) continue;
                int qc = -1, ac = -1;
                for (int qi = 0; Q_NAMES[qi] && qc < 0; qi++) qc = csv_col(t, Q_NAMES[qi]);
                for (int ai = 0; A_NAMES[ai] && ac < 0; ai++) ac = csv_col(t, A_NAMES[ai]);
                if (qc >= 0 && ac >= 0) {
                    for (int r = 0; r < t->n_rows; r++) {
                        const char* q = csv_get(t, r, qc);
                        const char* a = csv_get(t, r, ac);
                        if (!q||!*q||!a||!*a||!strcmp(q,"nan")||!strcmp(a,"nan")) continue;
                        if (n >= cap) { cap *= 2; texts = (char**)xrealloc(texts, cap * sizeof(char*)); }
                        texts[n++] = format_qa(q, a, &rng);
                    }
                }
                csv_free(t);
            }
            closedir(dir);
        }
#endif
    }

    *out_n = n;
    return texts;
}

/* ── Evaluate ────────────────────────────────────────────────────── */
float evaluate(Model* m, Tokenizer* tok, Dataset* val_ds,
               int block_size, float label_smoothing) {
    (void)tok;
    float total_loss = 0.0f;
    int   count      = 0;

    int* x = (int*)xmalloc(block_size * sizeof(int));
    int* y = (int*)xmalloc(block_size * sizeof(int));
    float* logits = (float*)xmalloc((size_t)block_size * m->cfg.vocab_size * sizeof(float));

    /* Sample up to 200 batches for speed */
    size_t n = val_ds->n_samples;
    size_t step = n > 200 ? n / 200 : 1;

    for (size_t i = 0; i < n; i += step) {
        dataset_get(val_ds, i, x, y);
        model_forward(m, x, block_size, logits);
        float loss = cross_entropy_f32(logits, y, NULL, block_size, m->cfg.vocab_size, -1, label_smoothing);
        total_loss += loss;
        count++;
    }

    free(x); free(y); free(logits);
    return count > 0 ? total_loss / count : 0.0f;
}

/* ── Progress bar ────────────────────────────────────────────────── */
static void print_bar(int cur, int total, const char* suffix) {
    const int W = 36;
    int filled = (total > 0) ? (int)((long long)cur * W / total) : 0;
    printf("\r  [");
    for (int i = 0; i < W; i++) printf(i < filled ? "\xe2\x96\x88" : "\xe2\x96\x91");
    printf("] %3d%%  %s", total > 0 ? (int)((long long)cur * 100 / total) : 0, suffix);
    fflush(stdout);
}

/* ── Main train function ─────────────────────────────────────────── */
void train(const char* csv_file) {
    const char sep_line[] = "======================================================================";
    printf("\n%s\n  FinanceGPT -- C Training Session\n", sep_line);

    double t0 = now_sec();

    /* ── 1. Load data ─────────────────────────────────────────── */
    printf("\n[1/5] Loading data...\n");
    int n_texts;
    char** texts = load_texts(DATA_DIR, csv_file, &n_texts);
    if (n_texts == 0) { printf("  No data found. Aborting.\n"); return; }
    printf("  Total samples: %d\n", n_texts);

    /* ── 2. Tokenizer ─────────────────────────────────────────── */
    printf("\n[2/5] Tokenizer...\n");
    Tokenizer* tok = NULL;

    /* Check if tokenizer.json exists — if so, load it. */
    FILE* tf = fopen(TOKENIZER_PATH, "rb");
    if (tf) {
        fclose(tf);
        tok = tok_load(TOKENIZER_PATH);
        printf("  Loaded tokenizer: %d tokens\n", tok ? tok->vocab_size : 0);
    }

    if (!tok) {
        printf("  No tokenizer found at %s\n", TOKENIZER_PATH);
        printf("  Please run the Python trainer first to generate the tokenizer, then re-run.\n");
        printf("  Or convert: python export_weights.py\n");
        for (int i = 0; i < n_texts; i++) free(texts[i]);
        free(texts);
        return;
    }

    /* ── 3. Tokenize all texts ────────────────────────────────── */
    printf("\n[3/5] Building dataset...\n");
    size_t total_tokens = 0;
    size_t all_cap = 65536;
    int* all_ids = (int*)xmalloc(all_cap * sizeof(int));

    for (int i = 0; i < n_texts; i++) {
        int len;
        int* ids = tok_encode(tok, texts[i], &len, 1);
        while (total_tokens + (size_t)len + 4 > all_cap) {
            all_cap *= 2;
            all_ids = (int*)xrealloc(all_ids, all_cap * sizeof(int));
        }
        memcpy(all_ids + total_tokens, ids, len * sizeof(int));
        total_tokens += len;
        free(ids);
        free(texts[i]);

        char suf[64];
        snprintf(suf, sizeof(suf), "%d/%d samples", i + 1, n_texts);
        print_bar(i + 1, n_texts, suf);
    }
    free(texts);
    printf("\n");

    printf("  Total tokens: %zu\n", total_tokens);

    /* Split train/val */
    size_t split = (size_t)(total_tokens * (1.0f - TRAIN_VAL_SPLIT));
    int block_size = TRAIN_BLOCK_SIZE;
    int stride     = block_size / 2;

    int* train_toks = (int*)xmalloc(split * sizeof(int));
    int* val_toks   = (int*)xmalloc((total_tokens - split) * sizeof(int));
    memcpy(train_toks, all_ids,          split * sizeof(int));
    memcpy(val_toks,   all_ids + split,  (total_tokens - split) * sizeof(int));
    free(all_ids);

    Dataset* train_ds = dataset_create(train_toks, split,                block_size, stride);
    Dataset* val_ds   = dataset_create(val_toks,   total_tokens - split, block_size, block_size);
    printf("  Samples: %zu train | %zu val\n", train_ds->n_samples, val_ds->n_samples);

    /* ── 4. Model ─────────────────────────────────────────────── */
    printf("\n[4/5] Model...\n");
    Model* m = NULL;
    FILE* mf = fopen(CHECKPOINT_PATH, "rb");
    if (mf) { fclose(mf); m = model_load(CHECKPOINT_PATH); printf("  Resumed from checkpoint\n"); }
    if (!m) {
        ModelConfig cfg = {
            .vocab_size  = tok->vocab_size,
            .d_model     = D_MODEL,
            .n_heads     = N_HEADS,
            .n_layers    = N_LAYERS,
            .d_ff        = D_FF,
            .max_seq_len = MAX_SEQ_LEN,
            .d_k         = D_K,
        };
        m = model_create(cfg);
        printf("  New model created\n");
    }
    printf("  Parameters: %zu\n", model_n_params(m));
    optimizer_init(m);

    /* ── 5. Training loop ─────────────────────────────────────── */
    int total_steps   = (int)train_ds->n_samples * TRAIN_EPOCHS;
    int global_step   = 0;
    float best_val    = 1e38f;
    int   no_improve  = 0;
    TrainHistory* hist = history_create();

    printf("\n[5/5] Training (%d epochs, %zu steps/epoch)...\n",
           TRAIN_EPOCHS, train_ds->n_samples);
    printf("%s\n\n", sep_line);

    int*   x_buf = (int*)xmalloc(block_size * sizeof(int));
    int*   y_buf = (int*)xmalloc(block_size * sizeof(int));
    Activations* acts = activations_create(&m->cfg, block_size);

    /* Shuffle indices */
    size_t n_train = train_ds->n_samples;
    size_t* order  = (size_t*)xmalloc(n_train * sizeof(size_t));
    for (size_t i = 0; i < n_train; i++) order[i] = i;

    for (int epoch = 1; epoch <= TRAIN_EPOCHS; epoch++) {
        double ep_t0 = now_sec();

        /* Shuffle (Fisher-Yates) */
        uint64_t rng2 = (uint64_t)time(NULL) + epoch;
        for (size_t i = n_train - 1; i > 0; i--) {
            size_t j = (size_t)(rng_splitmix64(&rng2) % (i + 1));
            size_t tmp = order[i]; order[i] = order[j]; order[j] = tmp;
        }

        float ep_loss_sum = 0.0f;
        int   ep_steps    = 0;
        float cur_lr      = TRAIN_LR;

        optimizer_zero_grad(m);

        for (size_t si = 0; si < n_train; si++) {
            dataset_get(train_ds, order[si], x_buf, y_buf);

            float loss = model_train_step(m, x_buf, y_buf, block_size,
                                           1, TRAIN_LABEL_SMOOTH, acts);

            ep_loss_sum += loss;
            ep_steps++;
            global_step++;
            cur_lr = lr_schedule(global_step, total_steps, TRAIN_LR, TRAIN_MIN_LR, TRAIN_WARMUP_STEPS);
            history_add_step(hist, global_step, loss, cur_lr);

            /* Optimizer step every TRAIN_GRAD_ACCUM samples */
            if ((si + 1) % TRAIN_GRAD_ACCUM == 0 || si == n_train - 1) {
                optimizer_step(m, cur_lr, ADAM_BETA1, ADAM_BETA2, ADAM_EPS, ADAM_WEIGHT_DECAY, global_step);
                optimizer_zero_grad(m);
            }

            /* Progress bar every 10 steps */
            if (si % 10 == 0) {
                float avg = ep_loss_sum / ep_steps;
                char suf[80];
                snprintf(suf, sizeof(suf),
                         "%zu/%zu  loss=%.4f  ppl=%.1f  lr=%.1e",
                         si, n_train, avg,
                         expf(avg < 10.0f ? avg : 10.0f), cur_lr);
                print_bar((int)si, (int)n_train, suf);
            }
        }

        float avg_train = ep_steps > 0 ? ep_loss_sum / ep_steps : 0.0f;
        float val_loss  = evaluate(m, tok, val_ds, block_size, TRAIN_LABEL_SMOOTH);
        double elapsed  = now_sec() - ep_t0;

        EpochRecord rec = {
            .train_loss = avg_train,
            .val_loss   = val_loss,
            .epoch      = epoch,
            .step       = global_step,
            .train_ppl  = expf(avg_train < 10.0f ? avg_train : 10.0f),
            .val_ppl    = expf(val_loss   < 10.0f ? val_loss  : 10.0f),
        };
        history_add_epoch(hist, rec);

        printf("\n  Epoch %d/%d | train=%.4f ppl=%.1f | val=%.4f ppl=%.1f | %.0fs\n",
               epoch, TRAIN_EPOCHS,
               avg_train, rec.train_ppl,
               val_loss,  rec.val_ppl,
               elapsed);

        if (val_loss < best_val - 1e-4f) {
            best_val   = val_loss;
            no_improve = 0;
            model_save(m, CHECKPOINT_PATH);
            printf("  [*] New best val=%.4f -- checkpoint saved\n", best_val);
        } else {
            no_improve++;
            if (no_improve >= TRAIN_PATIENCE) {
                printf("  Early stopping (no improvement for %d epochs)\n", TRAIN_PATIENCE);
                break;
            }
        }
    }

    /* Final save */
    model_save(m, CHECKPOINT_PATH);

    double total_time = now_sec() - t0;
    printf("\n%s\n", sep_line);
    printf("  Training complete!  Total time: %.1fs\n", total_time);
    printf("  Best val loss: %.4f\n", best_val);
    printf("  Checkpoint: %s\n", CHECKPOINT_PATH);
    printf("%s\n\n", sep_line);

    free(x_buf); free(y_buf); free(order);
    activations_free(acts, &m->cfg);
    history_free(hist);
    dataset_free(train_ds);
    dataset_free(val_ds);
    tok_free(tok);
    model_free(m);
}
