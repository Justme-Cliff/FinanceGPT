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
#include <sys/stat.h>
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
    h->grad_norms = (float*)xmalloc(h->cap_steps * sizeof(float));
    h->steps      = (int*)xmalloc(h->cap_steps * sizeof(int));
    h->cap_epochs = 64;
    h->val_epochs = (EpochRecord*)xmalloc(h->cap_epochs * sizeof(EpochRecord));
    return h;
}

void history_free(TrainHistory* h) {
    if (!h) return;
    free(h->losses); free(h->lrs); free(h->grad_norms); free(h->steps); free(h->val_epochs);
    free(h);
}

void history_add_step(TrainHistory* h, int step, float loss, float lr, float gnorm) {
    if (h->n_steps >= h->cap_steps) {
        h->cap_steps *= 2;
        h->losses     = (float*)xrealloc(h->losses,     h->cap_steps * sizeof(float));
        h->lrs        = (float*)xrealloc(h->lrs,        h->cap_steps * sizeof(float));
        h->grad_norms = (float*)xrealloc(h->grad_norms, h->cap_steps * sizeof(float));
        h->steps      = (int*)xrealloc(h->steps,        h->cap_steps * sizeof(int));
    }
    h->losses[h->n_steps]     = loss;
    h->lrs[h->n_steps]        = lr;
    h->grad_norms[h->n_steps] = gnorm;
    h->steps[h->n_steps]      = step;
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
    /* Linear warmup */
    if (step < warmup)
        return lr * (float)step / (float)(warmup > 0 ? warmup : 1);
    /* Single-cycle cosine decay from peak LR down to min_LR over full training */
    int decay_steps = total_steps - warmup;
    if (decay_steps <= 0) return min_lr;
    float p = (float)(step - warmup) / (float)decay_steps;
    if (p > 1.0f) p = 1.0f;
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
    for (int i = 0; i < W; i++) printf(i < filled ? "#" : "-");
    printf("] %3d%%  %-90s", total > 0 ? (int)((long long)cur * 100 / total) : 0, suffix);
    fflush(stdout);
}

/* ── SVG plot generation ─────────────────────────────────────────── */
/* Write a minimal inline SVG line chart to a file.
   xs/ys: n data points, title/xlabel/ylabel: labels */
static void write_svg(const char* path,
                      const float* xs, const float* ys, int n,
                      const float* ys2,        /* optional second series (NULL = none) */
                      const char* title,
                      const char* xlabel, const char* ylabel,
                      const char* legend1, const char* legend2) {
    if (n <= 0) return;
    FILE* f = fopen(path, "w");
    if (!f) { fprintf(stderr, "  [plot] cannot open %s\n", path); return; }

    /* Find data bounds */
    float xmin = xs[0], xmax = xs[0];
    float ymin = ys[0], ymax = ys[0];
    for (int i = 1; i < n; i++) {
        if (xs[i] < xmin) xmin = xs[i];
        if (xs[i] > xmax) xmax = xs[i];
        if (ys[i] < ymin) ymin = ys[i];
        if (ys[i] > ymax) ymax = ys[i];
    }
    if (ys2) {
        for (int i = 0; i < n; i++) {
            if (ys2[i] < ymin) ymin = ys2[i];
            if (ys2[i] > ymax) ymax = ys2[i];
        }
    }
    float xrange = xmax - xmin; if (xrange < 1e-9f) xrange = 1.0f;
    float yrange = ymax - ymin; if (yrange < 1e-9f) yrange = 1.0f;

    /* SVG canvas */
    int W = 900, H = 520;
    int ml = 80, mr = 30, mt = 50, mb = 60;
    int pw = W - ml - mr, ph = H - mt - mb;

    fprintf(f, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    fprintf(f, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%d\" height=\"%d\">\n", W, H);
    fprintf(f, "<rect width=\"%d\" height=\"%d\" fill=\"#1e1e2e\"/>\n", W, H);
    /* Chart area background */
    fprintf(f, "<rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"#13131f\" stroke=\"#444\" stroke-width=\"1\"/>\n", ml, mt, pw, ph);

    /* Grid lines (5 horizontal) */
    for (int gi = 0; gi <= 4; gi++) {
        float gy = mt + ph * gi / 4.0f;
        float val = ymax - yrange * gi / 4.0f;
        fprintf(f, "<line x1=\"%d\" y1=\"%.1f\" x2=\"%d\" y2=\"%.1f\" stroke=\"#333\" stroke-width=\"1\" stroke-dasharray=\"4,4\"/>\n",
                ml, gy, ml+pw, gy);
        fprintf(f, "<text x=\"%d\" y=\"%.1f\" fill=\"#aaa\" font-size=\"11\" font-family=\"monospace\" text-anchor=\"end\">%.4g</text>\n",
                ml-5, gy+4, val);
    }
    /* X axis ticks (5) */
    for (int gi = 0; gi <= 4; gi++) {
        float gx = ml + pw * gi / 4.0f;
        float val = xmin + xrange * gi / 4.0f;
        fprintf(f, "<line x1=\"%.1f\" y1=\"%d\" x2=\"%.1f\" y2=\"%d\" stroke=\"#555\" stroke-width=\"1\"/>\n",
                gx, mt+ph, gx, mt+ph+5);
        fprintf(f, "<text x=\"%.1f\" y=\"%d\" fill=\"#aaa\" font-size=\"11\" font-family=\"monospace\" text-anchor=\"middle\">%.0f</text>\n",
                gx, mt+ph+18, val);
    }

    /* Build polyline for series 1 */
    fprintf(f, "<polyline fill=\"none\" stroke=\"#7ec8e3\" stroke-width=\"2\" points=\"");
    for (int i = 0; i < n; i++) {
        float px2 = ml + (xs[i] - xmin) / xrange * pw;
        float py2 = mt + ph - (ys[i] - ymin) / yrange * ph;
        fprintf(f, "%.1f,%.1f ", px2, py2);
    }
    fprintf(f, "\"/>\n");

    /* Series 2 (optional) */
    if (ys2) {
        fprintf(f, "<polyline fill=\"none\" stroke=\"#f28b82\" stroke-width=\"2\" stroke-dasharray=\"6,3\" points=\"");
        for (int i = 0; i < n; i++) {
            float px2 = ml + (xs[i] - xmin) / xrange * pw;
            float py2 = mt + ph - (ys2[i] - ymin) / yrange * ph;
            fprintf(f, "%.1f,%.1f ", px2, py2);
        }
        fprintf(f, "\"/>\n");
    }

    /* Title */
    fprintf(f, "<text x=\"%d\" y=\"%d\" fill=\"#fff\" font-size=\"15\" font-family=\"sans-serif\" font-weight=\"bold\" text-anchor=\"middle\">%s</text>\n",
            W/2, mt-18, title);
    /* X label */
    fprintf(f, "<text x=\"%d\" y=\"%d\" fill=\"#ccc\" font-size=\"12\" font-family=\"sans-serif\" text-anchor=\"middle\">%s</text>\n",
            W/2, H-8, xlabel);
    /* Y label (rotated) */
    fprintf(f, "<text transform=\"rotate(-90)\" x=\"%d\" y=\"%d\" fill=\"#ccc\" font-size=\"12\" font-family=\"sans-serif\" text-anchor=\"middle\">%s</text>\n",
            -(H/2), 16, ylabel);

    /* Legend */
    if (legend1) {
        fprintf(f, "<rect x=\"%d\" y=\"%d\" width=\"12\" height=\"3\" fill=\"#7ec8e3\"/>\n", ml+10, mt+12);
        fprintf(f, "<text x=\"%d\" y=\"%d\" fill=\"#ccc\" font-size=\"11\" font-family=\"sans-serif\">%s</text>\n", ml+26, mt+16, legend1);
    }
    if (ys2 && legend2) {
        fprintf(f, "<rect x=\"%d\" y=\"%d\" width=\"12\" height=\"3\" fill=\"#f28b82\"/>\n", ml+120, mt+12);
        fprintf(f, "<text x=\"%d\" y=\"%d\" fill=\"#ccc\" font-size=\"11\" font-family=\"sans-serif\">%s</text>\n", ml+136, mt+16, legend2);
    }

    fprintf(f, "</svg>\n");
    fclose(f);
    printf("  [plot] %s\n", path);
}

/* Compute smoothed array (exponential moving average, alpha=0.05) */
static float* smooth_ema(const float* src, int n, float alpha) {
    float* out = (float*)xmalloc(n * sizeof(float));
    if (n == 0) return out;
    out[0] = src[0];
    for (int i = 1; i < n; i++)
        out[i] = alpha * src[i] + (1.0f - alpha) * out[i-1];
    return out;
}

/* Compute perplexity array from loss array */
static float* loss_to_ppl(const float* losses, int n) {
    float* out = (float*)xmalloc(n * sizeof(float));
    for (int i = 0; i < n; i++)
        out[i] = expf(losses[i] < 10.0f ? losses[i] : 10.0f);
    return out;
}

/* Convert int steps to float steps for SVG */
static float* steps_to_float(const int* steps, int n) {
    float* out = (float*)xmalloc(n * sizeof(float));
    for (int i = 0; i < n; i++) out[i] = (float)steps[i];
    return out;
}

/* Convert epoch records to float arrays */
static void epochs_to_arrays(const EpochRecord* recs, int n,
                               float** out_steps, float** out_train, float** out_val,
                               float** out_train_ppl, float** out_val_ppl) {
    *out_steps     = (float*)xmalloc(n * sizeof(float));
    *out_train     = (float*)xmalloc(n * sizeof(float));
    *out_val       = (float*)xmalloc(n * sizeof(float));
    *out_train_ppl = (float*)xmalloc(n * sizeof(float));
    *out_val_ppl   = (float*)xmalloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        (*out_steps)[i]     = (float)recs[i].step;
        (*out_train)[i]     = recs[i].train_loss;
        (*out_val)[i]       = recs[i].val_loss;
        (*out_train_ppl)[i] = recs[i].train_ppl;
        (*out_val_ppl)[i]   = recs[i].val_ppl;
    }
}

void generate_plots(const TrainHistory* h, const char* plots_dir) {
    if (!h || h->n_steps == 0) return;
    int n = h->n_steps;
    int ne = h->n_epochs;

    /* Make sure plots_dir exists */
#ifdef PLATFORM_WINDOWS
    CreateDirectoryA(plots_dir, NULL);
#else
    mkdir(plots_dir, 0755);
#endif

    char path[512];
    float* xf = steps_to_float(h->steps, n);
    float* ppl = loss_to_ppl(h->losses, n);
    float* smooth = smooth_ema(h->losses, n, 0.05f);
    float* smooth_ppl = loss_to_ppl(smooth, n);

    /* Plot 1: Raw training loss */
    snprintf(path, sizeof(path), "%s/01_training_loss.svg", plots_dir);
    write_svg(path, xf, h->losses, n, smooth, "Training Loss", "Step", "Loss", "Raw Loss", "Smoothed");

    /* Plot 2: Perplexity */
    snprintf(path, sizeof(path), "%s/02_perplexity.svg", plots_dir);
    write_svg(path, xf, ppl, n, smooth_ppl, "Training Perplexity", "Step", "Perplexity", "Raw PPL", "Smoothed");

    /* Plot 3: Learning rate schedule */
    snprintf(path, sizeof(path), "%s/03_learning_rate.svg", plots_dir);
    write_svg(path, xf, h->lrs, n, NULL, "Learning Rate Schedule (SGDR)", "Step", "LR", "LR", NULL);

    /* Plot 4: Gradient norms */
    snprintf(path, sizeof(path), "%s/04_grad_norms.svg", plots_dir);
    write_svg(path, xf, h->grad_norms, n, NULL, "Gradient Norms", "Step", "Grad Norm", "\xe2\x80\x96\xe2\x88\x87\xe2\x80\x96", NULL);

    /* Epoch-level plots */
    if (ne > 0) {
        float *ep_steps, *ep_train, *ep_val, *ep_train_ppl, *ep_val_ppl;
        epochs_to_arrays(h->val_epochs, ne, &ep_steps, &ep_train, &ep_val, &ep_train_ppl, &ep_val_ppl);

        /* Plot 5: Train vs Val Loss per epoch */
        snprintf(path, sizeof(path), "%s/05_train_vs_val_loss.svg", plots_dir);
        write_svg(path, ep_steps, ep_train, ne, ep_val, "Train vs Validation Loss", "Step", "Loss", "Train Loss", "Val Loss");

        /* Plot 6: Train vs Val Perplexity per epoch */
        snprintf(path, sizeof(path), "%s/06_train_vs_val_ppl.svg", plots_dir);
        write_svg(path, ep_steps, ep_train_ppl, ne, ep_val_ppl, "Train vs Validation Perplexity", "Step", "Perplexity", "Train PPL", "Val PPL");

        /* Plot 7: Generalization gap (val - train loss) per epoch */
        float* gap = (float*)xmalloc(ne * sizeof(float));
        for (int i = 0; i < ne; i++) gap[i] = ep_val[i] - ep_train[i];
        snprintf(path, sizeof(path), "%s/07_generalization_gap.svg", plots_dir);
        write_svg(path, ep_steps, gap, ne, NULL, "Generalization Gap (Val - Train Loss)", "Step", "Gap", "Val - Train", NULL);
        free(gap);

        /* Plot 8: Val loss only (larger view) */
        snprintf(path, sizeof(path), "%s/08_val_loss.svg", plots_dir);
        write_svg(path, ep_steps, ep_val, ne, NULL, "Validation Loss", "Step", "Loss", "Val Loss", NULL);

        free(ep_steps); free(ep_train); free(ep_val); free(ep_train_ppl); free(ep_val_ppl);
    }

    /* Plot 9: Smoothed loss only */
    snprintf(path, sizeof(path), "%s/09_smoothed_loss.svg", plots_dir);
    write_svg(path, xf, smooth, n, NULL, "Smoothed Training Loss (EMA alpha=0.05)", "Step", "Loss", "EMA Loss", NULL);

    /* Plot 10: Loss histogram (distribution of per-step losses) */
    {
        /* Build histogram with 50 bins */
        float lmin = h->losses[0], lmax = h->losses[0];
        for (int i = 1; i < n; i++) {
            if (h->losses[i] < lmin) lmin = h->losses[i];
            if (h->losses[i] > lmax) lmax = h->losses[i];
        }
        int nbins = 50;
        float* bin_x = (float*)xcalloc(nbins, sizeof(float));
        float* bin_y = (float*)xcalloc(nbins, sizeof(float));
        float brange = lmax - lmin; if (brange < 1e-9f) brange = 1.0f;
        for (int i = 0; i < nbins; i++) bin_x[i] = lmin + brange * (i + 0.5f) / nbins;
        for (int i = 0; i < n; i++) {
            int b = (int)((h->losses[i] - lmin) / brange * nbins);
            if (b < 0) b = 0;
            if (b >= nbins) b = nbins - 1;
            bin_y[b] += 1.0f;
        }
        snprintf(path, sizeof(path), "%s/10_loss_histogram.svg", plots_dir);
        write_svg(path, bin_x, bin_y, nbins, NULL, "Loss Distribution Histogram", "Loss Value", "Count", "Frequency", NULL);
        free(bin_x); free(bin_y);
    }

    free(xf); free(ppl); free(smooth); free(smooth_ppl);
    printf("  [plots] 10 SVG training plots saved to %s/\n", plots_dir);
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

        if (i % 20 == 0 || i == n_texts - 1) {
            char suf[64];
            snprintf(suf, sizeof(suf), "%d/%d samples", i + 1, n_texts);
            print_bar(i + 1, n_texts, suf);
        }
    }
    free(texts);
    printf("\n");

    printf("  Total tokens: %zu\n", total_tokens);

    /* Split train/val */
    size_t split = (size_t)(total_tokens * (1.0f - TRAIN_VAL_SPLIT));
    int block_size = TRAIN_BLOCK_SIZE;
    int stride     = TRAIN_STRIDE;

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
            history_add_step(hist, global_step, loss, cur_lr, 0.0f);

            /* Optimizer step every TRAIN_GRAD_ACCUM samples */
            if ((si + 1) % TRAIN_GRAD_ACCUM == 0 || si == n_train - 1) {
                optimizer_step(m, cur_lr, ADAM_BETA1, ADAM_BETA2, ADAM_EPS, ADAM_WEIGHT_DECAY, global_step);
                optimizer_zero_grad(m);
            }

            /* Progress bar every 10 steps */
            if (si % 10 == 0 && si > 0) {
                float avg = ep_loss_sum / ep_steps;
                double elapsed = now_sec() - ep_t0;
                double secs_per_step = elapsed / (double)si;
                double eta_sec = secs_per_step * (double)(n_train - si);
                int eta_m = (int)(eta_sec / 60.0);
                int eta_s = (int)(eta_sec) % 60;
                char suf[120];
                snprintf(suf, sizeof(suf),
                         "%zu/%zu  loss=%.4f  ppl=%.1f  lr=%.1e  eta=%dm%02ds",
                         si, n_train, avg,
                         expf(avg < 10.0f ? avg : 10.0f), cur_lr,
                         eta_m, eta_s);
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

    generate_plots(hist, PLOTS_DIR);

    free(x_buf); free(y_buf); free(order);
    activations_free(acts, &m->cfg);
    history_free(hist);
    dataset_free(train_ds);
    dataset_free(val_ds);
    tok_free(tok);
    model_free(m);
}
