"""
FinanceGPT – Transformer language model with Rotary Position Embedding (RoPE).

Architecture highlights
-----------------------
* Pre-norm (GPT-2 / LLaMA style): LayerNorm before each sub-layer.
* RoPE instead of learned positional embeddings – better generalisation at
  longer sequence lengths and no fixed positional table to fine-tune.
* Causal self-attention with a persistent causal mask buffer.
* Weight-tied input embedding ↔ output projection.
* Scaled residual initialisation (GPT-2 recipe).
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Rotary Position Embedding ──────────────────────────────────────────

def _rope_freqs(dim: int, seq_len: int, theta: float = 10_000.0, device=None):
    """Return complex RoPE frequencies of shape (seq_len, dim//2)."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device).float() / dim)
    )
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                        # (T, dim/2)
    return torch.polar(torch.ones_like(freqs), freqs)       # complex


def _apply_rope(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor):
    """Apply RoPE to query and key tensors.

    q, k : (B, n_heads, T, d_k)
    freqs_cis : (T, d_k//2)  complex
    """
    def rotate(x):
        # x: (B, H, T, d_k) → complex (B, H, T, d_k/2)
        xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        xc = xc * freqs_cis.unsqueeze(0).unsqueeze(0)
        return torch.view_as_real(xc).flatten(-2).type_as(x)

    return rotate(q), rotate(k)


# ── Building blocks ────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalisation (faster than LayerNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, max_seq_len: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.c_attn  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.c_proj  = nn.Linear(d_model, d_model,     bias=False)
        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Causal mask – not a parameter
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len))
            .view(1, 1, max_seq_len, max_seq_len),
        )

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B,H,T,dk)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        q, k = _apply_rope(q, k, freqs_cis[:T])

        scale = 1.0 / math.sqrt(self.d_k)
        att   = (q @ k.transpose(-2, -1)) * scale
        att   = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att   = F.softmax(att, dim=-1)
        att   = self.attn_drop(att)

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff,   d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: gate(x) * sigmoid(gate(x)) ⊙ up(x)
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float, max_seq_len: int):
        super().__init__()
        self.ln1  = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_seq_len)
        self.ln2  = RMSNorm(d_model)
        self.mlp  = MLP(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), freqs_cis)
        x = x + self.mlp(self.ln2(x))
        return x


# ── Main model ─────────────────────────────────────────────────────────

class FinanceGPT(nn.Module):
    """Decoder-only transformer trained on finance Q&A text."""

    def __init__(self, config: dict):
        super().__init__()
        c = config
        self.config = c

        self.embed   = nn.Embedding(c["vocab_size"], c["d_model"])
        self.drop    = nn.Dropout(c["dropout"])
        self.blocks  = nn.ModuleList(
            [Block(c["d_model"], c["n_heads"], c["d_ff"],
                   c["dropout"], c["max_seq_len"])
             for _ in range(c["n_layers"])]
        )
        self.ln_f    = RMSNorm(c["d_model"])
        self.lm_head = nn.Linear(c["d_model"], c["vocab_size"], bias=False)
        self.lm_head.weight = self.embed.weight   # weight tying

        # Pre-compute RoPE frequencies (not a parameter)
        freqs = _rope_freqs(c["d_model"] // c["n_heads"],
                            c["max_seq_len"])
        self.register_buffer("freqs_cis", freqs)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)
            if name.endswith(("c_proj.weight", "down.weight")):
                nn.init.normal_(p, mean=0.0,
                                std=0.02 / math.sqrt(2 * self.config["n_layers"]))

    # ------------------------------------------------------------------
    def forward(self, idx: torch.Tensor,
                targets: torch.Tensor = None,
                label_smoothing: float = 0.0):
        B, T = idx.shape
        assert T <= self.config["max_seq_len"], \
            f"Sequence length {T} > max_seq_len {self.config['max_seq_len']}"

        x = self.drop(self.embed(idx))
        for block in self.blocks:
            x = block(x, self.freqs_cis)
        x      = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                label_smoothing=label_smoothing,
            )
        return logits, loss

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx: torch.Tensor,
                 max_new_tokens: int = 220,
                 temperature: float = 0.82,
                 top_k: int = 50,
                 top_p: float = 0.92,
                 repetition_penalty: float = 1.3,
                 stop_ids: list = None) -> torch.Tensor:
        """Autoregressive generation with top-k + nucleus sampling and
        repetition penalty."""
        stop_ids = set(stop_ids or [])

        for _ in range(max_new_tokens):
            ctx     = idx[:, -self.config["max_seq_len"]:]
            logits, _ = self(ctx)
            logits  = logits[:, -1, :].clone()          # (1, V)

            # Repetition penalty (multiply logits down for seen tokens)
            if repetition_penalty != 1.0:
                seen = idx[0].tolist()
                for tok in set(seen):
                    if logits[0, tok] > 0:
                        logits[0, tok] /= repetition_penalty
                    else:
                        logits[0, tok] *= repetition_penalty

            logits = logits / temperature

            # Top-k
            if top_k:
                kth, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < kth[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)

            # Nucleus (top-p)
            if 0.0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_probs, dim=-1)
                # remove tokens once cumulative prob > top_p
                mask = (cum - sorted_probs) > top_p
                sorted_probs[mask] = 0.0
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                next_id = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
            else:
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_id), dim=1)
            if next_id.item() in stop_ids:
                break

        return idx

    # ------------------------------------------------------------------
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str, extra: dict = None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ckpt = {"config": self.config, "model": self.state_dict()}
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)

    @classmethod
    def load(cls, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m    = cls(ckpt["config"])
        m.load_state_dict(ckpt["model"])
        return m, ckpt
