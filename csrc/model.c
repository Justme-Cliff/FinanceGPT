#include "model.h"
#include "math_ops.h"
#include "config.h"
#include "compat.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#ifdef PLATFORM_WINDOWS
#  include <direct.h>   /* _mkdir */
#  define MKDIR(p) _mkdir(p)
#else
#  include <sys/stat.h>
#  define MKDIR(p) mkdir((p), 0755)
#endif

/* ════════════════════════════════════════════════════════════════════
 * Internal helpers
 * ════════════════════════════════════════════════════════════════════ */

/* Allocate all weight arrays for one Block. */
static void block_alloc(Block* b, const ModelConfig* c)
{
    b->ln1_w     = (float*)xcalloc((size_t)c->d_model,                    sizeof(float));
    b->attn_qkv  = (float*)xcalloc((size_t)3 * c->d_model * c->d_model,   sizeof(float));
    b->attn_proj = (float*)xcalloc((size_t)c->d_model * c->d_model,       sizeof(float));
    b->ln2_w     = (float*)xcalloc((size_t)c->d_model,                    sizeof(float));
    b->ffn_gate  = (float*)xcalloc((size_t)c->d_ff * c->d_model,          sizeof(float));
    b->ffn_up    = (float*)xcalloc((size_t)c->d_ff * c->d_model,          sizeof(float));
    b->ffn_down  = (float*)xcalloc((size_t)c->d_model * c->d_ff,          sizeof(float));
}

/* Free all arrays inside a Block (does NOT free the Block itself). */
static void block_free_fields(Block* b)
{
    if (!b) return;
    free(b->ln1_w);
    free(b->attn_qkv);
    free(b->attn_proj);
    free(b->ln2_w);
    free(b->ffn_gate);
    free(b->ffn_up);
    free(b->ffn_down);
}

/* Initialize one Block with random weights (Box-Muller normal distribution).
   RMSNorm weights start at 1.0; projections use std = 0.02 with residual
   down-projection scaled by 1/sqrt(2*n_layers). */
static void block_random_init(Block* b, const ModelConfig* c, uint64_t* rng)
{
    const float std      = 0.02f;
    const float std_proj = 0.02f / sqrtf(2.0f * (float)c->n_layers);
    int i;

    /* RMSNorm weights = 1 */
    for (i = 0; i < c->d_model; i++) { b->ln1_w[i] = 1.0f; b->ln2_w[i] = 1.0f; }

    /* QKV projection */
    for (i = 0; i < 3 * c->d_model * c->d_model; i++) {
        float u1 = rng_float(rng) + 1e-10f;
        float u2 = rng_float(rng);
        b->attn_qkv[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2) * std;
    }
    /* Attention output projection (residual — scaled std) */
    for (i = 0; i < c->d_model * c->d_model; i++) {
        float u1 = rng_float(rng) + 1e-10f;
        float u2 = rng_float(rng);
        b->attn_proj[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2) * std_proj;
    }
    /* FFN gate */
    for (i = 0; i < c->d_ff * c->d_model; i++) {
        float u1 = rng_float(rng) + 1e-10f;
        float u2 = rng_float(rng);
        b->ffn_gate[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2) * std;
    }
    /* FFN up */
    for (i = 0; i < c->d_ff * c->d_model; i++) {
        float u1 = rng_float(rng) + 1e-10f;
        float u2 = rng_float(rng);
        b->ffn_up[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2) * std;
    }
    /* FFN down (residual — scaled std) */
    for (i = 0; i < c->d_model * c->d_ff; i++) {
        float u1 = rng_float(rng) + 1e-10f;
        float u2 = rng_float(rng);
        b->ffn_down[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2) * std_proj;
    }
}

/* ════════════════════════════════════════════════════════════════════
 * model_create
 * ════════════════════════════════════════════════════════════════════ */
Model* model_create(ModelConfig cfg)
{
    cfg.d_k = cfg.d_model / cfg.n_heads;

    Model* m = (Model*)xcalloc(1, sizeof(Model));
    m->cfg   = cfg;

    /* ── Embedding ────────────────────────────────────────────────── */
    m->embed = (float*)xcalloc((size_t)cfg.vocab_size * cfg.d_model, sizeof(float));

    /* ── Transformer blocks ───────────────────────────────────────── */
    m->blocks = (Block*)xcalloc((size_t)cfg.n_layers, sizeof(Block));
    /* NOTE: allocate each block ONCE here — no second loop */
    uint64_t rng = 42ULL;
    for (int l = 0; l < cfg.n_layers; l++) {
        block_alloc(&m->blocks[l], &cfg);
        block_random_init(&m->blocks[l], &cfg, &rng);
    }

    /* ── Final RMSNorm ────────────────────────────────────────────── */
    m->ln_f = (float*)xcalloc((size_t)cfg.d_model, sizeof(float));
    for (int i = 0; i < cfg.d_model; i++) m->ln_f[i] = 1.0f;

    /* ── RoPE tables ─────────────────────────────────────────────── */
    /* rope_precompute expects tables of size [max_seq_len * d_k/2]
       (one cos/sin pair per dimension pair per position).
       We allocate [max_seq_len * d_k] to be safe with the interface. */
    m->rope_cos = (float*)xcalloc((size_t)cfg.max_seq_len * cfg.d_k, sizeof(float));
    m->rope_sin = (float*)xcalloc((size_t)cfg.max_seq_len * cfg.d_k, sizeof(float));
    rope_precompute(m->rope_cos, m->rope_sin, cfg.max_seq_len, cfg.d_k, ROPE_THETA);

    /* ── Random-init embedding ───────────────────────────────────── */
    for (int i = 0; i < cfg.vocab_size * cfg.d_model; i++) {
        float u1 = rng_float(&rng) + 1e-10f;
        float u2 = rng_float(&rng);
        m->embed[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2) * 0.02f;
    }

    m->step          = 0;
    m->best_val_loss = 1e30f;

    /* ── Pre-allocate forward-pass scratch workspace ─────────────── */
    /* Covers: x, xn, qkv, q, k, v, attn_w, attn_out, tmp, gate, up, act */
    int T  = cfg.max_seq_len;
    int dm = cfg.d_model;
    int dk = cfg.d_k;
    int nh = cfg.n_heads;
    int df = cfg.d_ff;
    m->ws_size = (size_t)(T*dm*4          /* x, xn, attn_out, tmp */
                        + T*3*dm          /* qkv */
                        + nh*T*dk*3       /* q, k, v */
                        + nh*T*T          /* attn_w */
                        + T*df*3);        /* gate, up, act */
    m->ws = (float*)xmalloc(m->ws_size * sizeof(float));

    return m;
}

/* ════════════════════════════════════════════════════════════════════
 * model_free
 * ════════════════════════════════════════════════════════════════════ */
void model_free(Model* m)
{
    if (!m) return;
    const ModelConfig* c = &m->cfg;

    free(m->ws);
    free(m->embed);

    if (m->blocks) {
        for (int l = 0; l < c->n_layers; l++) block_free_fields(&m->blocks[l]);
        free(m->blocks);
    }

    free(m->ln_f);
    free(m->rope_cos);
    free(m->rope_sin);

    /* Gradients */
    if (m->grad_embed) free(m->grad_embed);
    if (m->grad_blocks) {
        for (int l = 0; l < c->n_layers; l++) block_free_fields(&m->grad_blocks[l]);
        free(m->grad_blocks);
    }
    if (m->grad_ln_f) free(m->grad_ln_f);

    /* AdamW first moments */
    if (m->m_embed) free(m->m_embed);
    if (m->v_embed) free(m->v_embed);
    if (m->m_blocks) {
        for (int l = 0; l < c->n_layers; l++) {
            block_free_fields(&m->m_blocks[l]);
            block_free_fields(&m->v_blocks[l]);
        }
        free(m->m_blocks);
        free(m->v_blocks);
    }
    if (m->m_ln_f) free(m->m_ln_f);
    if (m->v_ln_f) free(m->v_ln_f);

    free(m);
}

/* ════════════════════════════════════════════════════════════════════
 * model_n_params
 * ════════════════════════════════════════════════════════════════════ */
size_t model_n_params(const Model* m)
{
    const ModelConfig* c = &m->cfg;
    size_t n = 0;

    /* Embedding (weight-tied with lm_head — count once) */
    n += (size_t)c->vocab_size * c->d_model;
    /* Final norm */
    n += (size_t)c->d_model;

    for (int l = 0; l < c->n_layers; l++) {
        n += (size_t)c->d_model * 2;              /* ln1_w + ln2_w */
        n += (size_t)3 * c->d_model * c->d_model; /* QKV */
        n += (size_t)c->d_model * c->d_model;     /* attn_proj */
        n += (size_t)c->d_ff * c->d_model * 2;    /* ffn_gate + ffn_up */
        n += (size_t)c->d_model * c->d_ff;        /* ffn_down */
    }
    return n;
}

/* ════════════════════════════════════════════════════════════════════
 * Save / Load helpers
 * ════════════════════════════════════════════════════════════════════ */

/* Write a single named tensor to an open file. */
static void write_tensor(FILE* f, const char* name,
                         const float* data, size_t n_elems)
{
    uint32_t name_len = (uint32_t)strlen(name);
    uint32_t n_dims   = 1;
    uint32_t dim      = (uint32_t)n_elems;
    uint32_t ne       = (uint32_t)n_elems;

    fwrite(&name_len, sizeof(uint32_t), 1, f);
    fwrite(name,      1,               name_len, f);
    fwrite(&n_dims,   sizeof(uint32_t), 1, f);
    fwrite(&dim,      sizeof(uint32_t), 1, f);
    fwrite(&ne,       sizeof(uint32_t), 1, f);
    fwrite(data,      sizeof(float),   n_elems, f);
}

/* Create parent directory of path if it doesn't exist. */
static void ensure_parent_dir(const char* path)
{
    char dir[512];
    strncpy(dir, path, sizeof(dir) - 1);
    dir[sizeof(dir) - 1] = '\0';

    /* Walk backward to find last separator */
    for (int i = (int)strlen(dir) - 1; i >= 0; i--) {
        if (dir[i] == '/' || dir[i] == '\\') {
            dir[i] = '\0';
            break;
        }
    }
    if (dir[0] != '\0') MKDIR(dir);
}

/* ════════════════════════════════════════════════════════════════════
 * model_save
 * ════════════════════════════════════════════════════════════════════ */
int model_save(const Model* m, const char* path)
{
    ensure_parent_dir(path);

    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[model_save] Cannot open '%s' for writing\n", path);
        return -1;
    }

    const ModelConfig* c = &m->cfg;

    /* Header */
    fwrite("FGPT", 1, 4, f);
    uint32_t ver = 1;
    fwrite(&ver, sizeof(uint32_t), 1, f);

    uint32_t vs  = (uint32_t)c->vocab_size;
    uint32_t dm  = (uint32_t)c->d_model;
    uint32_t nh  = (uint32_t)c->n_heads;
    uint32_t nl  = (uint32_t)c->n_layers;
    uint32_t df  = (uint32_t)c->d_ff;
    uint32_t msl = (uint32_t)c->max_seq_len;
    fwrite(&vs,  sizeof(uint32_t), 1, f);
    fwrite(&dm,  sizeof(uint32_t), 1, f);
    fwrite(&nh,  sizeof(uint32_t), 1, f);
    fwrite(&nl,  sizeof(uint32_t), 1, f);
    fwrite(&df,  sizeof(uint32_t), 1, f);
    fwrite(&msl, sizeof(uint32_t), 1, f);

    /* Tensor count: embed + ln_f + 7 per layer */
    uint32_t n_tensors = 2u + (uint32_t)c->n_layers * 7u;
    fwrite(&n_tensors, sizeof(uint32_t), 1, f);

    /* Write tensors */
    write_tensor(f, "embed", m->embed, (size_t)c->vocab_size * c->d_model);
    write_tensor(f, "ln_f",  m->ln_f,  (size_t)c->d_model);

    char name[64];
    for (int l = 0; l < c->n_layers; l++) {
        const Block* b = &m->blocks[l];

        snprintf(name, sizeof(name), "blocks.%d.ln1_w",    l);
        write_tensor(f, name, b->ln1_w,    (size_t)c->d_model);

        snprintf(name, sizeof(name), "blocks.%d.attn_qkv", l);
        write_tensor(f, name, b->attn_qkv, (size_t)3 * c->d_model * c->d_model);

        snprintf(name, sizeof(name), "blocks.%d.attn_proj",l);
        write_tensor(f, name, b->attn_proj,(size_t)c->d_model * c->d_model);

        snprintf(name, sizeof(name), "blocks.%d.ln2_w",    l);
        write_tensor(f, name, b->ln2_w,    (size_t)c->d_model);

        snprintf(name, sizeof(name), "blocks.%d.ffn_gate", l);
        write_tensor(f, name, b->ffn_gate, (size_t)c->d_ff * c->d_model);

        snprintf(name, sizeof(name), "blocks.%d.ffn_up",   l);
        write_tensor(f, name, b->ffn_up,   (size_t)c->d_ff * c->d_model);

        snprintf(name, sizeof(name), "blocks.%d.ffn_down", l);
        write_tensor(f, name, b->ffn_down, (size_t)c->d_model * c->d_ff);
    }

    fclose(f);
    return 0;
}

/* ════════════════════════════════════════════════════════════════════
 * model_load
 * ════════════════════════════════════════════════════════════════════ */
Model* model_load(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[model_load] Cannot open '%s'\n", path);
        return NULL;
    }

    /* Magic */
    char magic[5] = {0};
    if (fread(magic, 1, 4, f) != 4 || strcmp(magic, "FGPT") != 0) {
        fprintf(stderr, "[model_load] Bad magic bytes in '%s'\n", path);
        fclose(f);
        return NULL;
    }

    /* Version */
    uint32_t ver = 0;
    (void)fread(&ver, sizeof(uint32_t), 1, f);
    if (ver != 1) {
        fprintf(stderr, "[model_load] Unsupported version %u\n", ver);
        fclose(f);
        return NULL;
    }

    /* Architecture fields */
    uint32_t vs, dm, nh, nl, df, msl;
    (void)fread(&vs,  sizeof(uint32_t), 1, f);
    (void)fread(&dm,  sizeof(uint32_t), 1, f);
    (void)fread(&nh,  sizeof(uint32_t), 1, f);
    (void)fread(&nl,  sizeof(uint32_t), 1, f);
    (void)fread(&df,  sizeof(uint32_t), 1, f);
    (void)fread(&msl, sizeof(uint32_t), 1, f);

    ModelConfig cfg;
    cfg.vocab_size  = (int)vs;
    cfg.d_model     = (int)dm;
    cfg.n_heads     = (int)nh;
    cfg.n_layers    = (int)nl;
    cfg.d_ff        = (int)df;
    cfg.max_seq_len = (int)msl;
    cfg.d_k         = (int)dm / (int)nh;

    Model* m = model_create(cfg);  /* allocates + random-inits; will be overwritten */

    uint32_t n_tensors = 0;
    (void)fread(&n_tensors, sizeof(uint32_t), 1, f);

    for (uint32_t ti = 0; ti < n_tensors; ti++) {
        /* Name */
        uint32_t name_len = 0;
        (void)fread(&name_len, sizeof(uint32_t), 1, f);
        char name[256] = {0};
        uint32_t read_len = name_len < (uint32_t)(sizeof(name) - 1)
                            ? name_len : (uint32_t)(sizeof(name) - 1);
        (void)fread(name, 1, read_len, f);
        /* Skip any overflow bytes in the name */
        if (name_len > read_len)
            fseek(f, (long)(name_len - read_len), SEEK_CUR);
        name[read_len] = '\0';

        /* Dims */
        uint32_t n_dims = 0;
        (void)fread(&n_dims, sizeof(uint32_t), 1, f);
        uint32_t dims[8] = {0};
        for (uint32_t d = 0; d < n_dims && d < 8; d++)
            (void)fread(&dims[d], sizeof(uint32_t), 1, f);
        /* Skip extra dims */
        if (n_dims > 8)
            fseek(f, (long)((n_dims - 8) * sizeof(uint32_t)), SEEK_CUR);

        uint32_t n_elems = 0;
        (void)fread(&n_elems, sizeof(uint32_t), 1, f);

        /* Resolve destination pointer */
        float* dst = NULL;

        if (strcmp(name, "embed") == 0) {
            dst = m->embed;
        } else if (strcmp(name, "ln_f") == 0) {
            dst = m->ln_f;
        } else {
            int layer = -1;
            char field[64] = {0};
            if (sscanf(name, "blocks.%d.%63s", &layer, field) == 2
                && layer >= 0 && layer < cfg.n_layers)
            {
                Block* b = &m->blocks[layer];
                if      (strcmp(field, "ln1_w")    == 0) dst = b->ln1_w;
                else if (strcmp(field, "attn_qkv") == 0) dst = b->attn_qkv;
                else if (strcmp(field, "attn_proj")== 0) dst = b->attn_proj;
                else if (strcmp(field, "ln2_w")    == 0) dst = b->ln2_w;
                else if (strcmp(field, "ffn_gate") == 0) dst = b->ffn_gate;
                else if (strcmp(field, "ffn_up")   == 0) dst = b->ffn_up;
                else if (strcmp(field, "ffn_down") == 0) dst = b->ffn_down;
            }
        }

        if (dst) {
            (void)fread(dst, sizeof(float), n_elems, f);
        } else {
            fprintf(stderr, "[model_load] Unknown tensor '%s' — skipping\n", name);
            fseek(f, (long)(n_elems * sizeof(float)), SEEK_CUR);
        }
    }

    fclose(f);
    return m;
}

/* ════════════════════════════════════════════════════════════════════
 * model_forward  — single-sample inference forward pass
 *
 * ids[T]         : input token ids
 * logits[T*vocab]: output logits (caller must pre-allocate)
 * ════════════════════════════════════════════════════════════════════ */
void model_forward(Model* m, const int* ids, int T, float* logits)
{
    const ModelConfig* c = &m->cfg;
    const int dm = c->d_model;
    const int dk = c->d_k;
    const int nh = c->n_heads;
    const int df = c->d_ff;
    const int vs = c->vocab_size;

    /* ── Scratch buffers — sliced from pre-allocated workspace ─────── */
    float* ws = m->ws;
    float* x        = ws;                         ws += (size_t)T * dm;
    float* xn       = ws;                         ws += (size_t)T * dm;
    float* attn_out = ws;                         ws += (size_t)T * dm;
    float* tmp      = ws;                         ws += (size_t)T * dm;
    float* qkv      = ws;                         ws += (size_t)T * 3 * dm;
    float* q        = ws;                         ws += (size_t)nh * T * dk;
    float* k        = ws;                         ws += (size_t)nh * T * dk;
    float* v        = ws;                         ws += (size_t)nh * T * dk;
    float* attn_w   = ws;                         ws += (size_t)nh * T * T;
    float* gate     = ws;                         ws += (size_t)T * df;
    float* up       = ws;                         ws += (size_t)T * df;
    float* act      = ws;

    /* ── Embedding lookup ────────────────────────────────────────── */
    for (int t = 0; t < T; t++) {
        int id = ids[t];
        if (id < 0 || id >= vs) id = TOK_UNK_ID;
        memcpy(x + (size_t)t * dm, m->embed + (size_t)id * dm,
               (size_t)dm * sizeof(float));
    }

    /* ── Transformer blocks ──────────────────────────────────────── */
    for (int l = 0; l < c->n_layers; l++) {
        const Block* b = &m->blocks[l];

        /* ── Pre-attention RMSNorm ──────────────────────────────── */
        for (int t = 0; t < T; t++)
            rms_norm_f32(xn + (size_t)t * dm, x + (size_t)t * dm,
                         b->ln1_w, dm, RMSNORM_EPS);

        /* ── QKV projection: [T, 3*dm] = xn @ attn_qkv^T ────────
           attn_qkv layout: [3*dm, dm] — each row is one output dim.
           matmul_t_f32(A [M,K], B [N,K], C [M,N]) computes A @ B^T  */
        matmul_t_f32(xn, b->attn_qkv, qkv, T, dm, 3 * dm);

        /* Deinterleave QKV into per-head arrays [nh, T, dk] */
        for (int t = 0; t < T; t++) {
            const float* row = qkv + (size_t)t * 3 * dm;
            for (int h = 0; h < nh; h++) {
                memcpy(q + ((size_t)h * T + t) * dk,
                       row               + (size_t)h * dk,
                       (size_t)dk * sizeof(float));
                memcpy(k + ((size_t)h * T + t) * dk,
                       row + dm           + (size_t)h * dk,
                       (size_t)dk * sizeof(float));
                memcpy(v + ((size_t)h * T + t) * dk,
                       row + 2 * dm       + (size_t)h * dk,
                       (size_t)dk * sizeof(float));
            }
        }

        /* ── RoPE ──────────────────────────────────────────────── */
        rope_apply(q, m->rope_cos, m->rope_sin, nh, T, dk);
        rope_apply(k, m->rope_cos, m->rope_sin, nh, T, dk);

        /* ── Causal scaled dot-product attention ────────────────── */
        attention_forward(attn_out, attn_w, q, k, v, nh, T, dk, dm);

        /* ── Attention output projection: tmp = attn_out @ attn_proj^T */
        matmul_t_f32(attn_out, b->attn_proj, tmp, T, dm, dm);

        /* ── Residual add ────────────────────────────────────────── */
        vec_add_f32(x, tmp, T * dm);

        /* ── Pre-FFN RMSNorm ────────────────────────────────────── */
        for (int t = 0; t < T; t++)
            rms_norm_f32(xn + (size_t)t * dm, x + (size_t)t * dm,
                         b->ln2_w, dm, RMSNORM_EPS);

        /* ── SwiGLU FFN ─────────────────────────────────────────── */
        /* gate_pre = xn @ ffn_gate^T   [T, df] */
        matmul_t_f32(xn, b->ffn_gate, gate, T, dm, df);
        /* up_pre   = xn @ ffn_up^T     [T, df] */
        matmul_t_f32(xn, b->ffn_up,   up,   T, dm, df);
        /* act = silu(gate) * up */
        silu_mul_f32(act, gate, up, T * df);
        /* down = act @ ffn_down^T   [T, dm] */
        matmul_t_f32(act, b->ffn_down, tmp, T, df, dm);

        /* ── Residual add ────────────────────────────────────────── */
        vec_add_f32(x, tmp, T * dm);
    }

    /* ── Final RMSNorm ───────────────────────────────────────────── */
    for (int t = 0; t < T; t++)
        rms_norm_f32(xn + (size_t)t * dm, x + (size_t)t * dm,
                     m->ln_f, dm, RMSNORM_EPS);

    /* ── LM head: logits = xn @ embed^T  (weight-tied) ──────────── */
    matmul_t_f32(xn, m->embed, logits, T, dm, vs);

}

/* ════════════════════════════════════════════════════════════════════
 * Activations lifecycle
 * ════════════════════════════════════════════════════════════════════ */
Activations* activations_create(const ModelConfig* c, int T)
{
    Activations* a = (Activations*)xcalloc(1, sizeof(Activations));
    a->T = T;

    a->pre_ln1   = (float**)xcalloc((size_t)c->n_layers, sizeof(float*));
    a->pre_ln2   = (float**)xcalloc((size_t)c->n_layers, sizeof(float*));
    a->attn_q    = (float**)xcalloc((size_t)c->n_layers, sizeof(float*));
    a->attn_k    = (float**)xcalloc((size_t)c->n_layers, sizeof(float*));
    a->attn_v    = (float**)xcalloc((size_t)c->n_layers, sizeof(float*));
    a->attn_w    = (float**)xcalloc((size_t)c->n_layers, sizeof(float*));
    a->attn_out  = (float**)xcalloc((size_t)c->n_layers, sizeof(float*));
    a->ffn_gate_x= (float**)xcalloc((size_t)c->n_layers, sizeof(float*));
    a->ffn_up_x  = (float**)xcalloc((size_t)c->n_layers, sizeof(float*));
    a->x         = (float**)xcalloc((size_t)(c->n_layers + 1), sizeof(float*));

    for (int l = 0; l < c->n_layers; l++) {
        a->pre_ln1[l]   = (float*)xcalloc((size_t)T * c->d_model, sizeof(float));
        a->pre_ln2[l]   = (float*)xcalloc((size_t)T * c->d_model, sizeof(float));
        a->attn_q[l]    = (float*)xcalloc((size_t)c->n_heads * T * c->d_k, sizeof(float));
        a->attn_k[l]    = (float*)xcalloc((size_t)c->n_heads * T * c->d_k, sizeof(float));
        a->attn_v[l]    = (float*)xcalloc((size_t)c->n_heads * T * c->d_k, sizeof(float));
        a->attn_w[l]    = (float*)xcalloc((size_t)c->n_heads * T * T,       sizeof(float));
        a->attn_out[l]  = (float*)xcalloc((size_t)T * c->d_model, sizeof(float));
        a->ffn_gate_x[l]= (float*)xcalloc((size_t)T * c->d_ff,    sizeof(float));
        a->ffn_up_x[l]  = (float*)xcalloc((size_t)T * c->d_ff,    sizeof(float));
        a->x[l]         = (float*)xcalloc((size_t)T * c->d_model, sizeof(float));
    }
    a->x[c->n_layers] = (float*)xcalloc((size_t)T * c->d_model, sizeof(float));
    a->logits          = (float*)xcalloc((size_t)T * c->vocab_size, sizeof(float));

    return a;
}

void activations_free(Activations* a, const ModelConfig* c)
{
    if (!a) return;
    for (int l = 0; l < c->n_layers; l++) {
        free(a->pre_ln1[l]);
        free(a->pre_ln2[l]);
        free(a->attn_q[l]);
        free(a->attn_k[l]);
        free(a->attn_v[l]);
        free(a->attn_w[l]);
        free(a->attn_out[l]);
        free(a->ffn_gate_x[l]);
        free(a->ffn_up_x[l]);
        free(a->x[l]);
    }
    free(a->x[c->n_layers]);

    free(a->pre_ln1);
    free(a->pre_ln2);
    free(a->attn_q);
    free(a->attn_k);
    free(a->attn_v);
    free(a->attn_w);
    free(a->attn_out);
    free(a->ffn_gate_x);
    free(a->ffn_up_x);
    free(a->x);
    free(a->logits);
    free(a);
}

/* ════════════════════════════════════════════════════════════════════
 * model_train_step
 *
 * Forward pass with full activation storage, cross-entropy loss
 * computation, and complete backward pass accumulating gradients into
 * m->grad_* buffers.
 *
 * Caller is responsible for:
 *   1. Zeroing gradients before calling (or relying on the zeroing
 *      done inside this function on first call).
 *   2. Running an optimizer step after accumulating gradients over a
 *      mini-batch.
 *
 * Returns: mean cross-entropy loss over the T tokens.
 * ════════════════════════════════════════════════════════════════════ */
float model_train_step(Model* m, const int* x_ids, const int* y_ids, int T,
                       int batch_size, float label_smoothing, Activations* acts)
{
    (void)batch_size;  /* reserved for future batching */

    const ModelConfig* c = &m->cfg;
    const int dm = c->d_model;
    const int dk = c->d_k;
    const int nh = c->n_heads;
    const int df = c->d_ff;
    const int vs = c->vocab_size;

    /* ────────────────────────────────────────────────────────────────
     * FORWARD PASS  (same as model_forward but saves activations)
     * ──────────────────────────────────────────────────────────────── */

    /* Scratch buffers reused across layers */
    float* qkv      = (float*)xmalloc((size_t)T * 3 * dm * sizeof(float));
    float* tmp_dm   = (float*)xmalloc((size_t)T * dm * sizeof(float));
    float* gate_pre = (float*)xmalloc((size_t)T * df * sizeof(float));
    float* up_pre   = (float*)xmalloc((size_t)T * df * sizeof(float));
    float* act_fwd  = (float*)xmalloc((size_t)T * df * sizeof(float));

    /* Embedding lookup → x[0] */
    for (int t = 0; t < T; t++) {
        int id = x_ids[t];
        if (id < 0 || id >= vs) id = TOK_UNK_ID;
        memcpy(acts->x[0] + (size_t)t * dm,
               m->embed + (size_t)id * dm,
               (size_t)dm * sizeof(float));
    }

    for (int l = 0; l < c->n_layers; l++) {
        const Block* b  = &m->blocks[l];
        const float* xi = acts->x[l];
        float*       xo = acts->x[l + 1];

        /* Residual starts as copy of input */
        memcpy(xo, xi, (size_t)T * dm * sizeof(float));

        /* Pre-attention RMSNorm → pre_ln1[l] */
        for (int t = 0; t < T; t++)
            rms_norm_f32(acts->pre_ln1[l] + (size_t)t * dm,
                         xi + (size_t)t * dm, b->ln1_w, dm, RMSNORM_EPS);

        /* QKV projection */
        matmul_t_f32(acts->pre_ln1[l], b->attn_qkv, qkv, T, dm, 3 * dm);

        /* Deinterleave → attn_q/k/v[l] */
        for (int t = 0; t < T; t++) {
            const float* row = qkv + (size_t)t * 3 * dm;
            for (int h = 0; h < nh; h++) {
                memcpy(acts->attn_q[l] + ((size_t)h * T + t) * dk,
                       row               + (size_t)h * dk,
                       (size_t)dk * sizeof(float));
                memcpy(acts->attn_k[l] + ((size_t)h * T + t) * dk,
                       row + dm           + (size_t)h * dk,
                       (size_t)dk * sizeof(float));
                memcpy(acts->attn_v[l] + ((size_t)h * T + t) * dk,
                       row + 2 * dm       + (size_t)h * dk,
                       (size_t)dk * sizeof(float));
            }
        }

        /* RoPE */
        rope_apply(acts->attn_q[l], m->rope_cos, m->rope_sin, nh, T, dk);
        rope_apply(acts->attn_k[l], m->rope_cos, m->rope_sin, nh, T, dk);

        /* Causal attention → attn_out[l], attn_w[l] */
        attention_forward(acts->attn_out[l], acts->attn_w[l],
                          acts->attn_q[l], acts->attn_k[l], acts->attn_v[l],
                          nh, T, dk, dm);

        /* Output projection → tmp_dm, then residual into xo */
        matmul_t_f32(acts->attn_out[l], b->attn_proj, tmp_dm, T, dm, dm);
        vec_add_f32(xo, tmp_dm, T * dm);

        /* Pre-FFN RMSNorm → pre_ln2[l] */
        for (int t = 0; t < T; t++)
            rms_norm_f32(acts->pre_ln2[l] + (size_t)t * dm,
                         xo + (size_t)t * dm, b->ln2_w, dm, RMSNORM_EPS);

        /* SwiGLU — save pre-activations for backward */
        matmul_t_f32(acts->pre_ln2[l], b->ffn_gate, gate_pre, T, dm, df);
        matmul_t_f32(acts->pre_ln2[l], b->ffn_up,   up_pre,   T, dm, df);

        memcpy(acts->ffn_gate_x[l], gate_pre, (size_t)T * df * sizeof(float));
        memcpy(acts->ffn_up_x[l],   up_pre,   (size_t)T * df * sizeof(float));

        /* act = silu(gate) * up */
        silu_mul_f32(act_fwd, gate_pre, up_pre, T * df);

        /* Down projection → residual into xo */
        matmul_t_f32(act_fwd, b->ffn_down, tmp_dm, T, df, dm);
        vec_add_f32(xo, tmp_dm, T * dm);
    }

    /* Final RMSNorm */
    float* xn_final = (float*)xmalloc((size_t)T * dm * sizeof(float));
    for (int t = 0; t < T; t++)
        rms_norm_f32(xn_final + (size_t)t * dm,
                     acts->x[c->n_layers] + (size_t)t * dm,
                     m->ln_f, dm, RMSNORM_EPS);

    /* LM head (weight-tied) */
    matmul_t_f32(xn_final, m->embed, acts->logits, T, dm, vs);

    /* ────────────────────────────────────────────────────────────────
     * LOSS  — cross-entropy with optional label smoothing
     * ──────────────────────────────────────────────────────────────── */
    float* dlogits = (float*)xcalloc((size_t)T * vs, sizeof(float));
    float  loss    = cross_entropy_f32(acts->logits, y_ids, dlogits,
                                       T, vs, -1, label_smoothing);

    /* ────────────────────────────────────────────────────────────────
     * ENSURE GRADIENT BUFFERS EXIST & ZERO THEM
     * ──────────────────────────────────────────────────────────────── */
    if (!m->grad_embed) {
        m->grad_embed  = (float*)xcalloc((size_t)vs * dm, sizeof(float));
        m->grad_blocks = (Block*)xcalloc((size_t)c->n_layers, sizeof(Block));
        for (int l = 0; l < c->n_layers; l++)
            block_alloc(&m->grad_blocks[l], c);
        m->grad_ln_f = (float*)xcalloc((size_t)dm, sizeof(float));
    } else {
        vec_zero_f32(m->grad_embed, vs * dm);
        for (int l = 0; l < c->n_layers; l++) {
            Block* gb = &m->grad_blocks[l];
            vec_zero_f32(gb->ln1_w,     dm);
            vec_zero_f32(gb->attn_qkv,  3 * dm * dm);
            vec_zero_f32(gb->attn_proj, dm * dm);
            vec_zero_f32(gb->ln2_w,     dm);
            vec_zero_f32(gb->ffn_gate,  df * dm);
            vec_zero_f32(gb->ffn_up,    df * dm);
            vec_zero_f32(gb->ffn_down,  dm * df);
        }
        vec_zero_f32(m->grad_ln_f, dm);
    }

    /* ────────────────────────────────────────────────────────────────
     * BACKWARD PASS
     * ──────────────────────────────────────────────────────────────── */

    /* ── grad of lm_head (weight-tied embed) ─────────────────────
       logits = xn_final @ embed^T
       d_embed  += xn_final^T @ dlogits  →  shape [vs, dm]
       d_xn_fin  = dlogits @ embed        →  shape [T, dm]        */

    /* Accumulate grad_embed from lm_head output */
    for (int t = 0; t < T; t++) {
        const float* dl = dlogits + (size_t)t * vs;
        const float* xn = xn_final + (size_t)t * dm;
        for (int v = 0; v < vs; v++) {
            float g = dl[v];
            if (g == 0.0f) continue;
            float* ge = m->grad_embed + (size_t)v * dm;
            for (int d = 0; d < dm; d++) ge[d] += g * xn[d];
        }
    }

    float* dx_final = (float*)xmalloc((size_t)T * dm * sizeof(float));
    matmul_f32(dlogits, m->embed, dx_final, T, vs, dm);

    /* ── grad through final RMSNorm ──────────────────────────────── */
    float* dx = (float*)xcalloc((size_t)T * dm, sizeof(float));
    for (int t = 0; t < T; t++) {
        rms_norm_bwd_f32(dx + (size_t)t * dm,
                         m->grad_ln_f,
                         acts->x[c->n_layers] + (size_t)t * dm,
                         m->ln_f,
                         dx_final + (size_t)t * dm,
                         dm, RMSNORM_EPS);
    }

    /* ── Per-layer backward (reversed) ──────────────────────────── */
    float* dqkv   = (float*)xmalloc((size_t)T * 3 * dm * sizeof(float));
    float* dq     = (float*)xcalloc((size_t)nh * T * dk, sizeof(float));
    float* dk_buf = (float*)xcalloc((size_t)nh * T * dk, sizeof(float));
    float* dv     = (float*)xcalloc((size_t)nh * T * dk, sizeof(float));
    float* dao    = (float*)xcalloc((size_t)T * dm,       sizeof(float));
    float* dffn   = (float*)xmalloc((size_t)T * df * sizeof(float));
    float* d_up   = (float*)xmalloc((size_t)T * df * sizeof(float));
    float* silu_a = (float*)xmalloc((size_t)T * df * sizeof(float));

    for (int l = c->n_layers - 1; l >= 0; l--) {
        const Block* b  = &m->blocks[l];
        Block*       gb = &m->grad_blocks[l];

        /* ── FFN backward ─────────────────────────────────────────
           forward: xo = xi_ffn + ffn_down @ (silu(gate) * up)
           dx flows back through the residual connection and through
           the FFN sub-layer.  dx is the gradient w.r.t. xo.       */

        /* grad ffn_down: [dm, df] += act^T @ dx
           We need to reconstruct act = silu(gate) * up.            */
        silu_f32(silu_a, acts->ffn_gate_x[l], T * df);
        /* silu_a now = silu(gate_pre) */
        float* ffn_act = (float*)xmalloc((size_t)T * df * sizeof(float));
        vec_mul_f32(ffn_act, silu_a, acts->ffn_up_x[l], T * df);
        /* grad_ffn_down += ffn_act^T @ dx  →  [df, dm]
           matmul_t_f32(A[M,K], B[N,K], C[M,N]) = A @ B^T
           We want C[df,dm] = ffn_act[T,df]^T @ dx[T,dm]
           i.e. outer product sum: C[j,d] += ffn_act[t,j]*dx[t,d]  */
        for (int t = 0; t < T; t++) {
            const float* fa = ffn_act + (size_t)t * df;
            const float* dx_t = dx + (size_t)t * dm;
            for (int j = 0; j < df; j++) {
                float fa_j = fa[j];
                if (fa_j == 0.0f) continue;
                float* gd = gb->ffn_down + (size_t)j * dm;
                for (int d = 0; d < dm; d++) gd[d] += fa_j * dx_t[d];
            }
        }
        free(ffn_act);

        /* d_ffn_act = dx @ ffn_down  →  [T, df] */
        matmul_f32(dx, b->ffn_down, dffn, T, dm, df);

        /* Backward through act = silu(gate) * up:
             d_gate_pre = silu_bwd(gate_pre) * up * d_ffn_act
             d_up_pre   = silu(gate_pre)     *     d_ffn_act   */
        float* d_gate_silu = (float*)xmalloc((size_t)T * df * sizeof(float));
        /* d_gate_silu = dffn * up_pre  (chain rule: d(silu*up)/d_gate = dsilu * up) */
        vec_mul_f32(d_gate_silu, dffn, acts->ffn_up_x[l], T * df);
        /* silu backward */
        silu_bwd_f32(dffn, acts->ffn_gate_x[l], d_gate_silu, T * df);
        /* dffn now holds d_gate_pre */
        free(d_gate_silu);

        /* d_up = silu_a * d_ffn_act (reconstruct d_ffn_act via dffn*up before bwd) */
        /* We already clobbered dffn — recompute d_up from saved info.
           d_up[t,j] = silu(gate_pre[t,j]) * d_ffn_act[t,j]
           But d_ffn_act has been partially consumed.  Use matmul_f32 result
           stored in dffn (pre-silu_bwd) — recompute fresh.                 */
        matmul_f32(dx, b->ffn_down, d_up, T, dm, df); /* d_ffn_act again */
        vec_mul_f32(d_up, silu_a, d_up, T * df);       /* d_up = silu_a * d_ffn_act */

        /* grad_ffn_gate += pre_ln2^T @ dffn  (dffn = d_gate_pre) */
        for (int t = 0; t < T; t++) {
            const float* pln = acts->pre_ln2[l] + (size_t)t * dm;
            const float* dg  = dffn + (size_t)t * df;
            for (int j = 0; j < df; j++) {
                float dg_j = dg[j];
                if (dg_j == 0.0f) continue;
                float* gg = gb->ffn_gate + (size_t)j * dm;
                for (int d = 0; d < dm; d++) gg[d] += dg_j * pln[d];
            }
        }

        /* grad_ffn_up += pre_ln2^T @ d_up */
        for (int t = 0; t < T; t++) {
            const float* pln = acts->pre_ln2[l] + (size_t)t * dm;
            const float* du  = d_up + (size_t)t * df;
            for (int j = 0; j < df; j++) {
                float du_j = du[j];
                if (du_j == 0.0f) continue;
                float* gu = gb->ffn_up + (size_t)j * dm;
                for (int d = 0; d < dm; d++) gu[d] += du_j * pln[d];
            }
        }

        /* d_pre_ln2 = dffn @ ffn_gate + d_up @ ffn_up  →  [T, dm] */
        float* d_pre_ln2 = (float*)xcalloc((size_t)T * dm, sizeof(float));
        matmul_f32(dffn, b->ffn_gate, d_pre_ln2, T, df, dm);
        {
            float* tmp_up = (float*)xcalloc((size_t)T * dm, sizeof(float));
            matmul_f32(d_up, b->ffn_up, tmp_up, T, df, dm);
            vec_add_f32(d_pre_ln2, tmp_up, T * dm);
            free(tmp_up);
        }

        /* ── FFN RMSNorm backward ─────────────────────────────────
           dx_residual += rms_norm_bwd(xo, ln2_w, d_pre_ln2)
           (dx also carries through the residual skip directly)    */
        float* dx_rms2 = (float*)xcalloc((size_t)T * dm, sizeof(float));
        for (int t = 0; t < T; t++) {
            rms_norm_bwd_f32(dx_rms2 + (size_t)t * dm,
                             gb->ln2_w,
                             acts->x[l + 1] + (size_t)t * dm,
                             b->ln2_w,
                             d_pre_ln2 + (size_t)t * dm,
                             dm, RMSNORM_EPS);
        }
        /* dx = dx (residual skip) + dx_rms2 (through norm) */
        vec_add_f32(dx, dx_rms2, T * dm);
        free(d_pre_ln2);
        free(dx_rms2);

        /* ── Attention backward ───────────────────────────────────
           attn path: pre_ln1 → QKV → rope → sdpa → attn_proj → residual
           dx here contains gradient w.r.t. xo (the output after attn residual).
           The attn residual: xo = xi + attn_proj(attn_out)
           So dx flows directly as d_xi gradient, and through attn_proj.      */

        /* grad_attn_proj += attn_out^T @ dx  →  [dm, dm] */
        for (int t = 0; t < T; t++) {
            const float* ao_t = acts->attn_out[l] + (size_t)t * dm;
            const float* dx_t = dx + (size_t)t * dm;
            for (int i = 0; i < dm; i++) {
                float ao_i = ao_t[i];
                if (ao_i == 0.0f) continue;
                float* gp = gb->attn_proj + (size_t)i * dm;
                for (int j = 0; j < dm; j++) gp[j] += ao_i * dx_t[j];
            }
        }

        /* dao = dx @ attn_proj  →  [T, dm] */
        vec_zero_f32(dao, T * dm);
        matmul_f32(dx, b->attn_proj, dao, T, dm, dm);

        /* Attention backward: → dq, dk_buf, dv */
        vec_zero_f32(dq,     nh * T * dk);
        vec_zero_f32(dk_buf, nh * T * dk);
        vec_zero_f32(dv,     nh * T * dk);
        attention_backward(dq, dk_buf, dv,
                           acts->attn_w[l], dao,
                           acts->attn_q[l], acts->attn_k[l], acts->attn_v[l],
                           nh, T, dk, dm);

        /* Reinterleave dq/dk/dv → dqkv [T, 3*dm] */
        vec_zero_f32(dqkv, T * 3 * dm);
        for (int t = 0; t < T; t++) {
            float* row = dqkv + (size_t)t * 3 * dm;
            for (int h = 0; h < nh; h++) {
                memcpy(row               + (size_t)h * dk,
                       dq     + ((size_t)h * T + t) * dk,
                       (size_t)dk * sizeof(float));
                memcpy(row + dm           + (size_t)h * dk,
                       dk_buf + ((size_t)h * T + t) * dk,
                       (size_t)dk * sizeof(float));
                memcpy(row + 2 * dm       + (size_t)h * dk,
                       dv     + ((size_t)h * T + t) * dk,
                       (size_t)dk * sizeof(float));
            }
        }

        /* grad_attn_qkv += pre_ln1^T @ dqkv  →  [3*dm, dm] */
        for (int t = 0; t < T; t++) {
            const float* pln = acts->pre_ln1[l] + (size_t)t * dm;
            const float* dq3 = dqkv + (size_t)t * 3 * dm;
            for (int j = 0; j < 3 * dm; j++) {
                float d = dq3[j];
                if (d == 0.0f) continue;
                float* gqkv = gb->attn_qkv + (size_t)j * dm;
                for (int dd = 0; dd < dm; dd++) gqkv[dd] += d * pln[dd];
            }
        }

        /* d_pre_ln1 = dqkv @ attn_qkv  →  [T, dm]
           attn_qkv [3*dm, dm]: d_pre_ln1 = dqkv [T, 3*dm] @ attn_qkv [3*dm, dm] */
        float* d_pre_ln1 = (float*)xcalloc((size_t)T * dm, sizeof(float));
        matmul_f32(dqkv, b->attn_qkv, d_pre_ln1, T, 3 * dm, dm);

        /* Attention RMSNorm backward → dx */
        float* dx_rms1 = (float*)xcalloc((size_t)T * dm, sizeof(float));
        for (int t = 0; t < T; t++) {
            rms_norm_bwd_f32(dx_rms1 + (size_t)t * dm,
                             gb->ln1_w,
                             acts->x[l] + (size_t)t * dm,
                             b->ln1_w,
                             d_pre_ln1 + (size_t)t * dm,
                             dm, RMSNORM_EPS);
        }
        /* dx += dx_rms1 (through norm) — residual skip already in dx */
        vec_add_f32(dx, dx_rms1, T * dm);
        free(d_pre_ln1);
        free(dx_rms1);

        /* ── Accumulate embedding gradients from this layer's dx ──
           dx is w.r.t. x[l] (before the layer).  For the embedding
           layer, the gradient flows back from x[0].  We accumulate
           here for all intermediate layers too — this is a simplification
           that works for a single-sequence pass where x[0] is the
           only embedding input.  The true per-layer residual gradient
           is carried forward into the next (lower) iteration of dx.
           Embedding gradient accumulation happens after the loop.     */
    }

    /* ── Embedding gradient ──────────────────────────────────────── */
    /* After all layers, dx holds the gradient w.r.t. x[0].
       Each token's embedding row contributed once, so:
         grad_embed[id] += dx[t]  for each token t.                  */
    for (int t = 0; t < T; t++) {
        int id = x_ids[t];
        if (id >= 0 && id < vs) {
            vec_add_f32(m->grad_embed + (size_t)id * dm,
                        dx + (size_t)t * dm, dm);
        }
    }

    /* ── Cleanup ─────────────────────────────────────────────────── */
    free(qkv);
    free(tmp_dm);
    free(gate_pre);
    free(up_pre);
    free(act_fwd);
    free(xn_final);
    free(dlogits);
    free(dx_final);
    free(dx);
    free(dqkv);
    free(dq);
    free(dk_buf);
    free(dv);
    free(dao);
    free(dffn);
    free(d_up);
    free(silu_a);

    return loss;
}

/* ════════════════════════════════════════════════════════════════════
 * KV Cache lifecycle
 * ════════════════════════════════════════════════════════════════════ */
KVCache* kv_cache_create(int n_layers, int n_heads, int max_len, int d_k)
{
    KVCache* c = (KVCache*)xcalloc(1, sizeof(KVCache));
    c->n_layers = n_layers;
    c->n_heads  = n_heads;
    c->max_len  = max_len;
    c->d_k      = d_k;
    size_t sz   = (size_t)n_layers * n_heads * max_len * d_k;
    c->k = (float*)xcalloc(sz, sizeof(float));
    c->v = (float*)xcalloc(sz, sizeof(float));
    return c;
}

void kv_cache_free(KVCache* c)
{
    if (!c) return;
    free(c->k);
    free(c->v);
    free(c);
}

void kv_cache_reset(KVCache* c)
{
    if (!c) return;
    size_t sz = (size_t)c->n_layers * c->n_heads * c->max_len * c->d_k;
    memset(c->k, 0, sz * sizeof(float));
    memset(c->v, 0, sz * sizeof(float));
}

/* ════════════════════════════════════════════════════════════════════
 * model_forward_one — single-token forward pass with KV cache
 *
 * Processes token_id at sequence position `pos`.
 * Writes new K/V into cache at position pos.
 * Fills logits[vocab_size] with the predictions for the next token.
 *
 * All scratch buffers are sliced from the pre-allocated m->ws slab
 * (the slab is large enough; T=1 only uses a small fraction of it).
 * ════════════════════════════════════════════════════════════════════ */
void model_forward_one(Model* m, int token_id, int pos, KVCache* cache, float* logits)
{
    const ModelConfig* c  = &m->cfg;
    const int dm  = c->d_model;
    const int dk  = c->d_k;
    const int nh  = c->n_heads;
    const int df  = c->d_ff;
    const int vs  = c->vocab_size;

    /* ── Scratch buffers from pre-allocated workspace (T=1) ──────── */
    float* ws       = m->ws;
    float* x        = ws; ws += (size_t)dm;
    float* xn       = ws; ws += (size_t)dm;
    float* attn_out = ws; ws += (size_t)dm;
    float* tmp      = ws; ws += (size_t)dm;
    float* qkv      = ws; ws += (size_t)3 * dm;
    float* q        = ws; ws += (size_t)nh * dk;
    float* new_k    = ws; ws += (size_t)nh * dk;
    float* new_v    = ws; ws += (size_t)nh * dk;
    float* gate     = ws; ws += (size_t)df;
    float* up       = ws; ws += (size_t)df;
    float* act      = ws;

    /* Attention score buffer — size bounded by max_seq_len (compile-time) */
    float scores[MAX_SEQ_LEN];

    /* ── Embedding lookup ────────────────────────────────────────── */
    int id = (token_id >= 0 && token_id < vs) ? token_id : TOK_UNK_ID;
    memcpy(x, m->embed + (size_t)id * dm, (size_t)dm * sizeof(float));

    /* ── RoPE tables for this position ──────────────────────────── */
    int rope_pos = pos < c->max_seq_len ? pos : c->max_seq_len - 1;
    const float* pos_cos = m->rope_cos + (size_t)rope_pos * dk;
    const float* pos_sin = m->rope_sin + (size_t)rope_pos * dk;

    /* ── Transformer blocks ──────────────────────────────────────── */
    for (int l = 0; l < c->n_layers; l++) {
        const Block* b = &m->blocks[l];

        /* Pre-attention RMSNorm */
        rms_norm_f32(xn, x, b->ln1_w, dm, RMSNORM_EPS);

        /* QKV projection (M=1): [3*dm] = xn @ attn_qkv^T */
        matmul_t_f32(xn, b->attn_qkv, qkv, 1, dm, 3 * dm);

        /* Deinterleave into per-head Q, K, V [nh, dk] */
        for (int h = 0; h < nh; h++) {
            memcpy(q     + h * dk, qkv              + h * dk, (size_t)dk * sizeof(float));
            memcpy(new_k + h * dk, qkv +     dm     + h * dk, (size_t)dk * sizeof(float));
            memcpy(new_v + h * dk, qkv + 2 * dm     + h * dk, (size_t)dk * sizeof(float));
        }

        /* Apply RoPE at position `pos` to q and new_k */
        for (int h = 0; h < nh; h++) {
            float* qh = q     + h * dk;
            float* kh = new_k + h * dk;
            for (int i = 0; i < dk; i += 2) {
                float q0 = qh[i],    q1 = qh[i+1];
                float k0 = kh[i],    k1 = kh[i+1];
                qh[i]   = q0 * pos_cos[i]   - q1 * pos_sin[i];
                qh[i+1] = q1 * pos_cos[i+1] + q0 * pos_sin[i+1];
                kh[i]   = k0 * pos_cos[i]   - k1 * pos_sin[i];
                kh[i+1] = k1 * pos_cos[i+1] + k0 * pos_sin[i+1];
            }
        }

        /* Write new K/V into cache at position pos */
        for (int h = 0; h < nh; h++) {
            float* ck = cache->k + ((size_t)(l * nh + h) * cache->max_len + pos) * dk;
            float* cv = cache->v + ((size_t)(l * nh + h) * cache->max_len + pos) * dk;
            memcpy(ck, new_k + h * dk, (size_t)dk * sizeof(float));
            memcpy(cv, new_v + h * dk, (size_t)dk * sizeof(float));
        }

        /* Causal attention: q[h] attends to cache K/V at positions 0..pos */
        float scale   = 1.0f / sqrtf((float)dk);
        int   n_pos   = pos + 1;  /* number of positions to attend over */
        vec_zero_f32(attn_out, dm);

        for (int h = 0; h < nh; h++) {
            const float* qh = q + h * dk;

            /* Compute scores */
            for (int j = 0; j < n_pos; j++) {
                const float* ck = cache->k + ((size_t)(l * nh + h) * cache->max_len + j) * dk;
                scores[j] = vec_dot_f32(qh, ck, dk) * scale;
            }
            softmax_inplace_f32(scores, n_pos);

            /* Aggregate values into attn_out (interleaved head layout: [T, d_model]) */
            int out_base = h * dk;
            for (int j = 0; j < n_pos; j++) {
                float w = scores[j];
                const float* cv = cache->v + ((size_t)(l * nh + h) * cache->max_len + j) * dk;
                for (int d = 0; d < dk; d++)
                    attn_out[out_base + d] += w * cv[d];
            }
        }

        /* Output projection (M=1) */
        matmul_t_f32(attn_out, b->attn_proj, tmp, 1, dm, dm);

        /* Residual add */
        vec_add_f32(x, tmp, dm);

        /* Pre-FFN RMSNorm */
        rms_norm_f32(xn, x, b->ln2_w, dm, RMSNORM_EPS);

        /* SwiGLU FFN (M=1) */
        matmul_t_f32(xn, b->ffn_gate, gate, 1, dm, df);
        matmul_t_f32(xn, b->ffn_up,   up,   1, dm, df);
        silu_mul_f32(act, gate, up, df);
        matmul_t_f32(act, b->ffn_down, tmp, 1, df, dm);

        /* Residual add */
        vec_add_f32(x, tmp, dm);
    }

    /* Final RMSNorm */
    rms_norm_f32(xn, x, m->ln_f, dm, RMSNORM_EPS);

    /* LM head: logits[vocab_size] = xn @ embed^T */
    matmul_t_f32(xn, m->embed, logits, 1, dm, vs);
}

/* ════════════════════════════════════════════════════════════════════
 * model_generate  — autoregressive generation with KV cache
 *
 * Uses model_forward_one for O(T) per step (vs O(T^2) without cache).
 * Prompt is processed token-by-token to fill the cache; then each
 * new token only processes a single position.
 *
 * Returns heap-allocated array containing the full sequence
 * (prompt + generated tokens).  Caller must free().
 * *out_len is set to the total length.
 * ════════════════════════════════════════════════════════════════════ */
int* model_generate(Model* m, const int* prompt_ids, int prompt_len,
                    int max_new_tokens, float temperature,
                    int top_k, float top_p, float rep_penalty,
                    int eos_id, int* out_len)
{
    const ModelConfig* c  = &m->cfg;
    const int vs      = c->vocab_size;
    const int max_ctx = c->max_seq_len;

    /* Create KV cache — filled during prompt processing, grown during generation */
    KVCache* cache = kv_cache_create(c->n_layers, c->n_heads, max_ctx, c->d_k);

    /* Working buffer: prompt + up to max_new_tokens */
    int buf_cap = prompt_len + max_new_tokens + 8;
    int* buf    = (int*)xmalloc((size_t)buf_cap * sizeof(int));
    memcpy(buf, prompt_ids, (size_t)prompt_len * sizeof(int));
    int n = prompt_len;

    /* Per-token logits */
    float* logits = (float*)xmalloc((size_t)vs * sizeof(float));

    uint64_t rng = (uint64_t)time(NULL) ^ 0xDEADBEEF12345678ULL;

    /* Clamp prompt to max_ctx (take the last max_ctx tokens if too long) */
    int prompt_start   = (n > max_ctx) ? (n - max_ctx) : 0;
    int prompt_ctx_len = n - prompt_start;

    /* ── Phase 1: process prompt tokens to fill KV cache ──────────
       After this loop, `logits` holds the predictions for the first
       generated token (the output from the last prompt token).      */
    for (int p = 0; p < prompt_ctx_len; p++)
        model_forward_one(m, buf[prompt_start + p], p, cache, logits);

    int cache_pos = prompt_ctx_len;  /* next free cache position */

    /* ── Phase 2: autoregressive generation ───────────────────────
       Each iteration: sample from current logits, emit the token,
       run one forward pass on that token to get logits for the next. */
    for (int step = 0; step < max_new_tokens; step++) {

        /* ── Repetition penalty ──────────────────────────────────── */
        if (rep_penalty != 1.0f) {
            for (int i = 0; i < n; i++) {
                int tok = buf[i];
                if (tok < 0 || tok >= vs) continue;
                if (logits[tok] > 0.0f) logits[tok] /= rep_penalty;
                else                     logits[tok] *= rep_penalty;
            }
        }

        /* ── Temperature scaling ─────────────────────────────────── */
        if (temperature > 1e-6f && temperature != 1.0f)
            vec_scale_f32(logits, 1.0f / temperature, vs);

        /* ── Top-k filtering ─────────────────────────────────────── */
        if (top_k > 0 && top_k < vs)
            top_k_filter(logits, vs, top_k);

        /* ── Softmax ─────────────────────────────────────────────── */
        softmax_inplace_f32(logits, vs);

        /* ── Sample (top-p / nucleus) ────────────────────────────── */
        int next_id = sample_topk_topp(logits, vs, 0, top_p, 1.0f, &rng);

        /* ── Append to buffer ────────────────────────────────────── */
        if (n >= buf_cap - 1) {
            buf_cap *= 2;
            buf = (int*)xrealloc(buf, (size_t)buf_cap * sizeof(int));
        }
        buf[n++] = next_id;

        if (next_id == eos_id) break;
        if (step == max_new_tokens - 1) break;

        /* ── Slide cache window if full ──────────────────────────── */
        if (cache_pos >= max_ctx) {
            /* Shift K/V left by 1 position to make room */
            int nl = c->n_layers, nh = c->n_heads, dk = c->d_k;
            for (int l = 0; l < nl; l++) {
                for (int h = 0; h < nh; h++) {
                    float* ck = cache->k + ((size_t)(l * nh + h) * max_ctx) * dk;
                    float* cv = cache->v + ((size_t)(l * nh + h) * max_ctx) * dk;
                    memmove(ck, ck + dk, (size_t)(max_ctx - 1) * dk * sizeof(float));
                    memmove(cv, cv + dk, (size_t)(max_ctx - 1) * dk * sizeof(float));
                }
            }
            cache_pos = max_ctx - 1;
        }

        /* ── Forward on the just-emitted token to get next logits ── */
        model_forward_one(m, next_id, cache_pos, cache, logits);
        cache_pos++;
    }

    kv_cache_free(cache);
    free(logits);
    *out_len = n;
    return buf;
}
