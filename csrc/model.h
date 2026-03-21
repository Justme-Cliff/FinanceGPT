#ifndef MODEL_H
#define MODEL_H
#include "compat.h"
#include "config.h"
#include "math_ops.h"
#include "tokenizer.h"
#include <stdint.h>

/* ── Weight binary format ─────────────────────────────────────────
   File layout:
     magic:       char[4] = "FGPT"
     version:     uint32_t = 1
     vocab_size:  uint32_t
     d_model:     uint32_t
     n_heads:     uint32_t
     n_layers:    uint32_t
     d_ff:        uint32_t
     max_seq_len: uint32_t
     n_tensors:   uint32_t
     for each tensor:
       name_len: uint32_t
       name:     char[name_len]  (no null terminator in file)
       n_dims:   uint32_t
       dims:     uint32_t[n_dims]
       n_elems:  uint32_t
       data:     float32[n_elems]
   ─────────────────────────────────────────────────────────────── */

typedef struct {
    int vocab_size;
    int d_model;
    int n_heads;
    int n_layers;
    int d_ff;
    int max_seq_len;
    int d_k;          /* d_model / n_heads */
} ModelConfig;

/* All weight tensors for one transformer block */
typedef struct {
    /* Attention */
    float* ln1_w;        /* [d_model] RMSNorm weight */
    float* attn_qkv;     /* [3*d_model, d_model] QKV projection */
    float* attn_proj;    /* [d_model, d_model] output projection */
    /* FFN */
    float* ln2_w;        /* [d_model] RMSNorm weight */
    float* ffn_gate;     /* [d_ff, d_model] SwiGLU gate */
    float* ffn_up;       /* [d_ff, d_model] SwiGLU up */
    float* ffn_down;     /* [d_model, d_ff] SwiGLU down */
} Block;

typedef struct {
    ModelConfig cfg;

    /* Embedding (also tied to lm_head) */
    float* embed;       /* [vocab_size, d_model] */

    /* Transformer blocks */
    Block* blocks;      /* [n_layers] */

    /* Final norm */
    float* ln_f;        /* [d_model] */

    /* RoPE tables */
    float* rope_cos;    /* [max_seq_len, d_k/2] precomputed cos values */
    float* rope_sin;    /* [max_seq_len, d_k/2] precomputed sin values */

    /* Gradient arrays (NULL during inference) */
    float* grad_embed;
    Block* grad_blocks;
    float* grad_ln_f;

    /* Optimizer state (AdamW) */
    float* m_embed;     /* first moment  */
    float* v_embed;     /* second moment */
    Block* m_blocks;
    Block* v_blocks;
    float* m_ln_f;
    float* v_ln_f;

    /* Training state */
    int   step;
    float best_val_loss;

    /* Pre-allocated scratch workspace for forward pass (avoids malloc per call) */
    float* ws;          /* single slab covering all scratch buffers */
    size_t ws_size;     /* size in floats */
} Model;

/* ── Lifecycle ─────────────────────────────────────────────────── */
Model* model_create  (ModelConfig cfg);
void   model_free    (Model* m);

/* Save/load binary weights */
int    model_save    (const Model* m, const char* path);
Model* model_load    (const char* path);

/* Number of parameters */
size_t model_n_params(const Model* m);

/* ── Forward pass (inference, batch=1) ─────────────────────────── */
/* Input: token ids[T], Output: logits[T * vocab_size] */
void model_forward   (Model* m, const int* ids, int T, float* logits);

/* ── Forward pass for training (stores activations for backward) ── */
typedef struct {
    /* Per-layer activations — needed for backward */
    float** pre_ln1;    /* [n_layers][T * d_model] input to first RMSNorm */
    float** pre_ln2;    /* [n_layers][T * d_model] input to second RMSNorm */
    float** attn_q;     /* [n_layers][n_heads * T * d_k] */
    float** attn_k;     /* [n_layers][n_heads * T * d_k] */
    float** attn_v;     /* [n_layers][n_heads * T * d_k] */
    float** attn_w;     /* [n_layers][n_heads * T * T] attention weights (post-softmax) */
    float** attn_out;   /* [n_layers][T * d_model] attention output before proj */
    float** ffn_gate_x; /* [n_layers][T * d_ff] gate pre-activation */
    float** ffn_up_x;   /* [n_layers][T * d_ff] up pre-activation */
    float** x;          /* residual stream [n_layers+1][T * d_model] */
    float*  logits;     /* [T * vocab_size] */
    int     T;
} Activations;

Activations* activations_create (const ModelConfig* cfg, int T);
void         activations_free   (Activations* a, const ModelConfig* cfg);

/* Returns cross-entropy loss for the sequence.
   x_ids: input token ids [T], y_ids: target token ids [T] */
float model_train_step (Model* m, const int* x_ids, const int* y_ids, int T,
                        int batch_size, float label_smoothing,
                        Activations* acts);

/* ── KV Cache (for fast autoregressive generation) ──────────────── */
/* Stores K and V for all seen positions so generation is O(T), not O(T^2). */
typedef struct {
    float* k;       /* [n_layers * n_heads * max_len * d_k] */
    float* v;       /* [n_layers * n_heads * max_len * d_k] */
    int    max_len;
    int    n_layers;
    int    n_heads;
    int    d_k;
} KVCache;

KVCache* kv_cache_create(int n_layers, int n_heads, int max_len, int d_k);
void     kv_cache_free  (KVCache* c);
void     kv_cache_reset (KVCache* c);

/* Single-token forward pass with KV cache.
   Processes token_id at sequence position pos, writes K/V into cache,
   returns logits[vocab_size] in pre-allocated output buffer. */
void model_forward_one(Model* m, int token_id, int pos, KVCache* cache, float* logits);

/* ── Generation ─────────────────────────────────────────────────── */
/* Returns heap-allocated array of token ids (prompt + generated).
   Caller must free().  *out_len set to total length. */
int* model_generate  (Model* m, const int* prompt_ids, int prompt_len,
                      int max_new_tokens, float temperature,
                      int top_k, float top_p, float rep_penalty,
                      int eos_id, int* out_len);

#endif /* MODEL_H */
