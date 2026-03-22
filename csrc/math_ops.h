#ifndef MATH_OPS_H
#define MATH_OPS_H
#include "compat.h"
#include <stdint.h>

/*
 * Custom math module — hand-optimized for CPU inference + training.
 * Uses AVX2 (8 floats/cycle) + OpenMP (multi-core) for near-BLAS throughput.
 * All functions are "plain C with optional SIMD" — safe fallback always exists.
 */

/* ── GEMM: C = A @ B  (row-major, no transpose) ────────────────── */
/* A: [M x K], B: [K x N], C: [M x N] */
void matmul_f32(const float* A, const float* B, float* C, int M, int K, int N);
/* C = A @ B^T */
void matmul_t_f32(const float* A, const float* B, float* C, int M, int K, int N);
/* C += A @ B */
void matmul_acc_f32(const float* A, const float* B, float* C, int M, int K, int N);
/* C[M,N] += A^T @ B  where A is stored row-major as [K,M], B is [K,N]
   Uses OpenMP for M-level parallelism and AVX2 for the N inner loop. */
void matmul_at_acc_f32(const float* A, const float* B, float* C, int K, int M, int N);

/* ── Element-wise ops ───────────────────────────────────────────── */
void vec_add_f32   (float* dst, const float* src, int n);             /* dst += src */
void vec_scale_f32 (float* dst, float scale, int n);                  /* dst *= scale */
void vec_mul_f32   (float* dst, const float* a, const float* b, int n); /* dst = a * b */
void vec_copy_f32  (float* dst, const float* src, int n);
void vec_zero_f32  (float* dst, int n);
float vec_dot_f32  (const float* a, const float* b, int n);           /* dot product */
float vec_sum_f32  (const float* a, int n);
float vec_max_f32  (const float* a, int n);

/* ── Activations ────────────────────────────────────────────────── */
void silu_f32      (float* dst, const float* src, int n);  /* x * sigmoid(x) */
void silu_bwd_f32  (float* dxout, const float* x, const float* dout, int n);
/* Fused SwiGLU: dst[i] = gate[i] * sigmoid(gate[i]) * up[i]  (one pass) */
void silu_mul_f32  (float* dst, const float* gate, const float* up, int n);
void gelu_f32      (float* dst, const float* src, int n);

/* ── Normalization ──────────────────────────────────────────────── */
/* RMSNorm forward: out = x / rms(x) * weight */
void rms_norm_f32  (float* out, const float* x, const float* weight,
                    int n, float eps);
/* RMSNorm backward: dx = d/dx[rms_norm] */
void rms_norm_bwd_f32(float* dx, float* dweight,
                      const float* x, const float* weight,
                      const float* dout, int n, float eps);

/* ── Softmax ────────────────────────────────────────────────────── */
void softmax_f32         (float* out, const float* in, int n);
void softmax_inplace_f32 (float* x, int n);
void log_softmax_f32     (float* out, const float* in, int n);

/* ── Cross-entropy ──────────────────────────────────────────────── */
/* targets: int32 class indices, ignore_index=-1, label_smoothing in [0,1) */
float cross_entropy_f32(const float* logits, const int* targets, float* grad,
                        int batch, int vocab, int ignore_idx, float smoothing);

/* ── Sampling ───────────────────────────────────────────────────── */
int  sample_topk_topp(const float* probs, int vocab,
                      int top_k, float top_p, float temperature, uint64_t* rng);
void top_k_filter(float* logits, int vocab, int k);
void top_p_filter(float* probs,  int vocab, float p);

/* ── RoPE ───────────────────────────────────────────────────────── */
/* Precompute cos/sin tables: freqs[seq][d//2] (interleaved cos,sin) */
void rope_precompute(float* cos_table, float* sin_table,
                     int seq_len, int d_k, float theta);
/* Apply RoPE in-place to q or k: shape [n_heads, T, d_k] */
void rope_apply(float* qk, const float* cos_t, const float* sin_t,
                int n_heads, int T, int d_k);

/* ── Causal attention ───────────────────────────────────────────── */
/* q,k,v: [n_heads, T, d_k], out: [T, d_model]
   mask: lower-triangular (1=keep, 0=mask)
   This is the full scaled dot-product attention */
void attention_forward(float* out, float* attn_weights,
                       const float* q, const float* k, const float* v,
                       int n_heads, int T, int d_k, int d_model);
void attention_backward(float* dq, float* dk, float* dv,
                        const float* attn_weights,
                        const float* dout,
                        const float* q, const float* k, const float* v,
                        int n_heads, int T, int d_k, int d_model);

/* ── Random ─────────────────────────────────────────────────────── */
uint64_t rng_splitmix64(uint64_t* state);
float    rng_float     (uint64_t* rng);  /* uniform [0, 1) */

#endif /* MATH_OPS_H */
