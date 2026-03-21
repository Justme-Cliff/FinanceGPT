#include "math_ops.h"
#include "config.h"
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#ifdef HAVE_OPENBLAS
#  include <cblas.h>
#endif

/* ── RNG ─────────────────────────────────────────────────────────── */
uint64_t rng_splitmix64(uint64_t* s) {
    uint64_t z = (*s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
float rng_float(uint64_t* rng) {
    return (float)(rng_splitmix64(rng) >> 11) * (1.0f / (float)(1ULL << 53));
}

/* ── Vec helpers ─────────────────────────────────────────────────── */
void vec_zero_f32(float* dst, int n) { memset(dst, 0, (size_t)n * sizeof(float)); }
void vec_copy_f32(float* dst, const float* src, int n) { memcpy(dst, src, (size_t)n * sizeof(float)); }

void vec_add_f32(float* dst, const float* src, int n) {
    int i = 0;
#ifdef HAVE_AVX2
    for (; i <= n - 8; i += 8) {
        __m256 a = _mm256_loadu_ps(dst + i);
        __m256 b = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(a, b));
    }
#endif
    for (; i < n; i++) dst[i] += src[i];
}

void vec_scale_f32(float* dst, float s, int n) {
    int i = 0;
#ifdef HAVE_AVX2
    __m256 vs = _mm256_set1_ps(s);
    for (; i <= n - 8; i += 8) {
        __m256 a = _mm256_loadu_ps(dst + i);
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(a, vs));
    }
#endif
    for (; i < n; i++) dst[i] *= s;
}

void vec_mul_f32(float* dst, const float* a, const float* b, int n) {
    int i = 0;
#ifdef HAVE_AVX2
    for (; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(va, vb));
    }
#endif
    for (; i < n; i++) dst[i] = a[i] * b[i];
}

float vec_dot_f32(const float* a, const float* b, int n) {
    float s = 0.0f;
    int i = 0;
#ifdef HAVE_AVX2
    __m256 acc = _mm256_setzero_ps();
    for (; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    /* horizontal sum */
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    s = _mm_cvtss_f32(lo);
#endif
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

float vec_sum_f32(const float* a, int n) {
    float s = 0.0f;
    int i = 0;
#ifdef HAVE_AVX2
    __m256 acc = _mm256_setzero_ps();
    for (; i <= n - 8; i += 8)
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(a + i));
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    s = _mm_cvtss_f32(lo);
#endif
    for (; i < n; i++) s += a[i];
    return s;
}

float vec_max_f32(const float* a, int n) {
    float m = a[0];
    for (int i = 1; i < n; i++) if (a[i] > m) m = a[i];
    return m;
}

/* ── GEMM ────────────────────────────────────────────────────────── */
/* Uses OpenBLAS when available (fastest), else hand-written AVX2 GEMM */

#ifdef HAVE_OPENBLAS

void matmul_f32(const float* A, const float* B, float* C, int M, int K, int N) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
}
void matmul_t_f32(const float* A, const float* B, float* C, int M, int K, int N) {
    /* C = A @ B^T  —  B is [N,K] stored row-major, so transpose B */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, A, K, B, K, 0.0f, C, N);
}
void matmul_acc_f32(const float* A, const float* B, float* C, int M, int K, int N) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 1.0f, C, N);
}

#else  /* built-in tiled AVX2 GEMM fallback */

#define TILE 64
void matmul_f32(const float* A, const float* B, float* C, int M, int K, int N) {
    vec_zero_f32(C, M * N);
    OMP_PARALLEL_FOR
    for (int i = 0; i < M; i++) {
        for (int kk = 0; kk < K; kk += TILE) {
            int klim = kk + TILE < K ? kk + TILE : K;
            for (int jj = 0; jj < N; jj += TILE) {
                int jlim = jj + TILE < N ? jj + TILE : N;
                for (int k = kk; k < klim; k++) {
                    float a = A[i * K + k];
                    int j = jj;
#ifdef HAVE_AVX2
                    __m256 va = _mm256_set1_ps(a);
                    for (; j <= jlim - 8; j += 8) {
                        __m256 vc = _mm256_loadu_ps(C + i * N + j);
                        __m256 vb = _mm256_loadu_ps(B + k * N + j);
                        _mm256_storeu_ps(C + i * N + j, _mm256_fmadd_ps(va, vb, vc));
                    }
#endif
                    for (; j < jlim; j++)
                        C[i * N + j] += a * B[k * N + j];
                }
            }
        }
    }
}
void matmul_t_f32(const float* A, const float* B, float* C, int M, int K, int N) {
    OMP_PARALLEL_FOR
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            C[i * N + j] = vec_dot_f32(A + i * K, B + j * K, K);
}
void matmul_acc_f32(const float* A, const float* B, float* C, int M, int K, int N) {
    OMP_PARALLEL_FOR
    for (int i = 0; i < M; i++) {
        for (int kk = 0; kk < K; kk += TILE) {
            int klim = kk + TILE < K ? kk + TILE : K;
            for (int k = kk; k < klim; k++) {
                float a = A[i * K + k];
                int j = 0;
#ifdef HAVE_AVX2
                __m256 va = _mm256_set1_ps(a);
                for (; j <= N - 8; j += 8) {
                    __m256 vc = _mm256_loadu_ps(C + i * N + j);
                    __m256 vb = _mm256_loadu_ps(B + k * N + j);
                    _mm256_storeu_ps(C + i * N + j, _mm256_fmadd_ps(va, vb, vc));
                }
#endif
                for (; j < N; j++) C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}

#endif /* HAVE_OPENBLAS */

/* ── SiLU  x * sigmoid(x) ────────────────────────────────────────── */
void silu_f32(float* dst, const float* src, int n) {
    for (int i = 0; i < n; i++) {
        float x = src[i];
        dst[i] = x / (1.0f + fast_exp(-x));
    }
}
void silu_bwd_f32(float* dxout, const float* x, const float* dout, int n) {
    for (int i = 0; i < n; i++) {
        float sig = 1.0f / (1.0f + fast_exp(-x[i]));
        dxout[i] = dout[i] * sig * (1.0f + x[i] * (1.0f - sig));
    }
}
void gelu_f32(float* dst, const float* src, int n) {
    const float c = 0.7978845608f; /* sqrt(2/pi) */
    for (int i = 0; i < n; i++) {
        float x = src[i];
        float t = tanhf(c * (x + 0.044715f * x * x * x));
        dst[i] = 0.5f * x * (1.0f + t);
    }
}

/* ── RMSNorm ─────────────────────────────────────────────────────── */
void rms_norm_f32(float* out, const float* x, const float* w, int n, float eps) {
    float ss = 0.0f;
    int i = 0;
#ifdef HAVE_AVX2
    __m256 acc = _mm256_setzero_ps();
    for (; i <= n - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    ss = _mm_cvtss_f32(lo);
#endif
    for (; i < n; i++) ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / (float)n + eps);
    i = 0;
#ifdef HAVE_AVX2
    __m256 vs = _mm256_set1_ps(scale);
    for (; i <= n - 8; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(w + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(vx, vs), vw));
    }
#endif
    for (; i < n; i++) out[i] = x[i] * scale * w[i];
}

void rms_norm_bwd_f32(float* dx, float* dweight,
                      const float* x, const float* w,
                      const float* dout, int n, float eps) {
    /* compute rms */
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float var    = ss / (float)n + eps;
    float rms    = sqrtf(var);
    float inv    = 1.0f / rms;
    float inv3   = inv / var;   /* d/dx of 1/rms = -1/(rms^3 * n) — stored as inv/var for reuse */

    float dot = 0.0f; /* sum(dout * w * x) */
    for (int i = 0; i < n; i++) dot += dout[i] * w[i] * x[i];

    for (int i = 0; i < n; i++) {
        float norm_xi = x[i] * inv;
        dweight[i]   += dout[i] * norm_xi;
        dx[i]        += inv * w[i] * dout[i]
                      - inv3 * w[i] * dout[i] * (1.0f / (float)n) * dot;
    }
}

/* ── Softmax (numerically stable) ───────────────────────────────── */
void softmax_f32(float* out, const float* in, int n) {
    float m = vec_max_f32(in, n);
    float s = 0.0f;
    for (int i = 0; i < n; i++) { out[i] = fast_exp(in[i] - m); s += out[i]; }
    float inv = 1.0f / s;
    vec_scale_f32(out, inv, n);
}
void softmax_inplace_f32(float* x, int n) {
    float m = vec_max_f32(x, n);
    float s = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = fast_exp(x[i] - m); s += x[i]; }
    vec_scale_f32(x, 1.0f / s, n);
}
void log_softmax_f32(float* out, const float* in, int n) {
    float m = vec_max_f32(in, n);
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += fast_exp(in[i] - m);
    float log_s = fast_log(s);
    for (int i = 0; i < n; i++) out[i] = in[i] - m - log_s;
}

/* ── Cross-entropy with label smoothing ─────────────────────────── */
float cross_entropy_f32(const float* logits, const int* targets, float* grad,
                        int batch, int vocab, int ignore_idx, float smoothing) {
    float total_loss = 0.0f;
    int   count      = 0;

    if (grad) vec_zero_f32(grad, batch * vocab);

    for (int b = 0; b < batch; b++) {
        int t = targets[b];
        if (t == ignore_idx) continue;
        count++;
        const float* row = logits + (size_t)b * vocab;
        float* grow      = grad   ? grad + (size_t)b * vocab : NULL;

        /* log-softmax */
        float m = vec_max_f32(row, vocab);
        float s = 0.0f;
        for (int v = 0; v < vocab; v++) s += fast_exp(row[v] - m);
        float log_s = m + fast_log(s);

        /* label-smoothed loss */
        float smooth_term = smoothing / (float)vocab;
        float hard_term   = 1.0f - smoothing;

        float loss = 0.0f;
        for (int v = 0; v < vocab; v++) {
            float lsm = row[v] - log_s;            /* log prob */
            float p   = fast_exp(lsm);             /* prob */
            float tgt = (v == t) ? hard_term : 0.0f;
            tgt += smooth_term;
            loss -= tgt * lsm;
            if (grow) grow[v] = (p - tgt) / (float)batch;
        }
        total_loss += loss;
    }
    return count > 0 ? total_loss / (float)count : 0.0f;
}

/* ── Top-k + nucleus sampling ───────────────────────────────────── */
void top_k_filter(float* logits, int vocab, int k) {
    if (k <= 0 || k >= vocab) return;
    /* Find k-th largest with partial selection sort */
    float* tmp = (float*)xmalloc((size_t)vocab * sizeof(float));
    memcpy(tmp, logits, (size_t)vocab * sizeof(float));
    for (int i = 0; i < k; i++) {
        for (int j = i + 1; j < vocab; j++) {
            if (tmp[j] > tmp[i]) { float t = tmp[i]; tmp[i] = tmp[j]; tmp[j] = t; }
        }
    }
    float threshold = tmp[k - 1];
    free(tmp);
    for (int i = 0; i < vocab; i++)
        if (logits[i] < threshold) logits[i] = -1e38f;
}

void top_p_filter(float* probs, int vocab, float p) {
    if (p >= 1.0f) return;
    /* Sort indices by probability descending, zero out tail beyond cumulative p */
    int* idx = (int*)xmalloc((size_t)vocab * sizeof(int));
    for (int i = 0; i < vocab; i++) idx[i] = i;
    for (int i = 0; i < vocab - 1; i++) {
        int m = i;
        for (int j = i + 1; j < vocab; j++)
            if (probs[idx[j]] > probs[idx[m]]) m = j;
        int t = idx[i]; idx[i] = idx[m]; idx[m] = t;
        float cum = 0.0f;
        for (int kk = 0; kk <= i; kk++) cum += probs[idx[kk]];
        if (cum >= p) {
            /* zero out everything after i */
            for (int kk = i + 1; kk < vocab; kk++) probs[idx[kk]] = 0.0f;
            break;
        }
    }
    free(idx);
}

int sample_topk_topp(const float* probs_in, int vocab,
                     int top_k, float top_p, float temperature, uint64_t* rng) {
    (void)temperature; /* caller applies temperature to logits before softmax */
    float* probs = (float*)xmalloc((size_t)vocab * sizeof(float));
    memcpy(probs, probs_in, (size_t)vocab * sizeof(float));

    if (top_k > 0 && top_k < vocab) {
        /* zero out tokens outside top-k */
        float* tmp = (float*)xmalloc((size_t)vocab * sizeof(float));
        memcpy(tmp, probs, (size_t)vocab * sizeof(float));
        for (int i = 0; i < top_k; i++) {
            int m = i;
            for (int j = i + 1; j < vocab; j++) if (tmp[j] > tmp[m]) m = j;
            float t = tmp[i]; tmp[i] = tmp[m]; tmp[m] = t;
        }
        float thresh = tmp[top_k - 1];
        free(tmp);
        for (int i = 0; i < vocab; i++) if (probs[i] < thresh) probs[i] = 0.0f;
    }

    if (top_p > 0.0f && top_p < 1.0f) {
        top_p_filter(probs, vocab, top_p);
    }

    /* Renormalize */
    float s = vec_sum_f32(probs, vocab);
    if (s < 1e-10f) { free(probs); return 0; }
    vec_scale_f32(probs, 1.0f / s, vocab);

    /* Multinomial sample */
    float r = rng_float(rng);
    float cum = 0.0f;
    int result = vocab - 1;
    for (int i = 0; i < vocab; i++) {
        cum += probs[i];
        if (r < cum) { result = i; break; }
    }
    free(probs);
    return result;
}

/* ── RoPE ────────────────────────────────────────────────────────── */
void rope_precompute(float* cos_t, float* sin_t, int seq_len, int d_k, float theta) {
    /* cos_t, sin_t: [seq_len, d_k] — each pair (2i, 2i+1) shares the same angle */
    for (int pos = 0; pos < seq_len; pos++) {
        for (int i = 0; i < d_k / 2; i++) {
            float freq  = 1.0f / powf(theta, (float)(2 * i) / (float)d_k);
            float angle = (float)pos * freq;
            cos_t[pos * d_k + 2 * i]     = cosf(angle);
            cos_t[pos * d_k + 2 * i + 1] = cosf(angle);
            sin_t[pos * d_k + 2 * i]     = sinf(angle);
            sin_t[pos * d_k + 2 * i + 1] = sinf(angle);
        }
    }
}

/* Apply RoPE in-place to qk: [n_heads, T, d_k] */
void rope_apply(float* qk, const float* cos_t, const float* sin_t,
                int n_heads, int T, int d_k) {
    for (int h = 0; h < n_heads; h++) {
        for (int t = 0; t < T; t++) {
            float* row       = qk    + (h * T + t) * d_k;
            const float* c   = cos_t + t * d_k;
            const float* s   = sin_t + t * d_k;
            for (int i = 0; i < d_k; i += 2) {
                float x0   = row[i];
                float x1   = row[i + 1];
                row[i]     = x0 * c[i]     - x1 * s[i];
                row[i + 1] = x1 * c[i + 1] + x0 * s[i + 1];
            }
        }
    }
}

/* ── Causal Scaled Dot-Product Attention ─────────────────────────── */
/*
 * q,k,v: [n_heads, T, d_k]   (contiguous)
 * out:   [T, n_heads * d_k]  (= [T, d_model])
 * attn_weights: [n_heads, T, T]  (saved for backward)
 */
void attention_forward(float* out, float* attn_weights,
                       const float* q, const float* k, const float* v,
                       int n_heads, int T, int d_k, int d_model) {
    float scale = 1.0f / sqrtf((float)d_k);

    vec_zero_f32(out, T * d_model);

    OMP_PARALLEL_FOR
    for (int h = 0; h < n_heads; h++) {
        const float* qh = q + h * T * d_k;
        const float* kh = k + h * T * d_k;
        const float* vh = v + h * T * d_k;
        float*       wh = attn_weights + h * T * T;

        /* QK^T * scale + causal mask, then softmax per row */
        for (int i = 0; i < T; i++) {
            for (int j = 0; j <= i; j++) {        /* causal: j <= i */
                wh[i * T + j] = vec_dot_f32(qh + i * d_k, kh + j * d_k, d_k) * scale;
            }
            for (int j = i + 1; j < T; j++) wh[i * T + j] = -1e38f; /* mask future */
            softmax_inplace_f32(wh + i * T, T);
        }

        /* attn @ V — scatter into output by head offset */
        for (int i = 0; i < T; i++) {
            for (int j = 0; j <= i; j++) {
                float w        = wh[i * T + j];
                int   out_base = i * d_model + h * d_k;
                for (int d = 0; d < d_k; d++)
                    out[out_base + d] += w * vh[j * d_k + d];
            }
        }
    }
}

void attention_backward(float* dq, float* dk, float* dv,
                        const float* attn_weights,
                        const float* dout,
                        const float* q, const float* k, const float* v,
                        int n_heads, int T, int d_k, int d_model) {
    float scale = 1.0f / sqrtf((float)d_k);

    for (int h = 0; h < n_heads; h++) {
        const float* wh  = attn_weights + h * T * T;
        const float* vh  = v + h * T * d_k;
        const float* qh  = q + h * T * d_k;
        const float* kh  = k + h * T * d_k;
        float*       dqh = dq + h * T * d_k;
        float*       dkh = dk + h * T * d_k;
        float*       dvh = dv + h * T * d_k;

        /* dV: for each key position j, accumulate over all queries i >= j */
        for (int j = 0; j < T; j++) {
            for (int i = j; i < T; i++) {
                float w        = wh[i * T + j];
                int   db       = i * d_model + h * d_k;
                for (int d = 0; d < d_k; d++)
                    dvh[j * d_k + d] += w * dout[db + d];
            }
        }

        /* dAttn: dout @ V^T  — shape [T, T] */
        float* dattn = (float*)xmalloc((size_t)T * T * sizeof(float));
        for (int i = 0; i < T; i++) {
            int db = i * d_model + h * d_k;
            for (int j = 0; j <= i; j++) {
                float s = 0.0f;
                for (int d = 0; d < d_k; d++) s += dout[db + d] * vh[j * d_k + d];
                dattn[i * T + j] = s;
            }
            for (int j = i + 1; j < T; j++) dattn[i * T + j] = 0.0f;
        }

        /* Softmax backward: dS[i,j] = p[i,j] * (dA[i,j] - sum_j(dA[i,j]*p[i,j])) */
        float* ds = (float*)xmalloc((size_t)T * T * sizeof(float));
        for (int i = 0; i < T; i++) {
            float dot = 0.0f;
            for (int j = 0; j <= i; j++) dot += dattn[i * T + j] * wh[i * T + j];
            for (int j = 0; j <= i; j++)
                ds[i * T + j] = wh[i * T + j] * (dattn[i * T + j] - dot);
            for (int j = i + 1; j < T; j++) ds[i * T + j] = 0.0f;
        }

        /* dQ: ds @ K * scale */
        for (int i = 0; i < T; i++) {
            for (int j = 0; j <= i; j++) {
                float s = ds[i * T + j] * scale;
                for (int d = 0; d < d_k; d++) dqh[i * d_k + d] += s * kh[j * d_k + d];
            }
        }

        /* dK: ds^T @ Q * scale — for each key j, sum over queries i >= j */
        for (int j = 0; j < T; j++) {
            for (int i = j; i < T; i++) {
                float s = ds[i * T + j] * scale;
                for (int d = 0; d < d_k; d++) dkh[j * d_k + d] += s * qh[i * d_k + d];
            }
        }

        free(dattn);
        free(ds);
    }
}
