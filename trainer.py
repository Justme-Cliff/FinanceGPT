"""
FinanceGPT trainer
==================
Features
--------
* Gradient accumulation (effective batch >> physical batch).
* torch.autocast mixed-precision on CUDA.
* Train + validation loss tracked every epoch.
* Label smoothing to prevent overconfident predictions.
* Early stopping on validation loss plateau.
* Cumulative training: loads existing checkpoint, extends vocabulary,
  and continues from previous step count.
* 4 publication-quality training plots.
"""

import contextlib
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import MODEL_CONFIG, TRAIN_CONFIG, CHECKPOINT, TOKENIZER, PLOTS_DIR
from data_processor import load_all_csv, make_datasets
from model import FinanceGPT
from tokenizer import BPETokenizer


# ── Helpers ────────────────────────────────────────────────────────────

def _lr(step: int, total: int, cfg: dict) -> float:
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * step / max(1, cfg["warmup_steps"])
    p = (step - cfg["warmup_steps"]) / max(1, total - cfg["warmup_steps"])
    return cfg["min_lr"] + (cfg["lr"] - cfg["min_lr"]) * 0.5 * (1 + math.cos(math.pi * p))


def _smooth(y, w):
    return np.convolve(y, np.ones(w) / w, mode="valid") if len(y) >= w else y


# ── Plotting ───────────────────────────────────────────────────────────

_DARK = {
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor":   "#30363d", "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e",     "ytick.color":    "#8b949e",
    "text.color":  "#c9d1d9",     "grid.color":     "#21262d",
    "grid.linestyle": "--",       "grid.alpha":     0.6,
    "font.family": "monospace",
}

COLORS = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657"]


def save_plots(history: dict):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.rcParams.update(_DARK)

    steps  = np.array(history["steps"])
    losses = np.array(history["losses"])
    lrs    = np.array(history["lrs"])

    val_steps  = np.array([e["step"]     for e in history.get("val_epochs", [])])
    val_losses = np.array([e["val_loss"] for e in history.get("val_epochs", [])])
    perps      = np.minimum(np.exp(losses), 500)

    w = max(5, len(losses) // 30)

    # 1 ── Training loss ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(steps, losses, color=COLORS[0], lw=1.0, alpha=0.35, label="raw")
    if len(losses) >= w:
        ax.plot(steps[w - 1:], _smooth(losses, w),
                color=COLORS[0], lw=2.2, label=f"smooth (w={w})")
    if len(val_losses):
        ax.plot(val_steps, val_losses, "o--",
                color=COLORS[2], lw=1.8, ms=5, label="val loss")
    ax.set_title("FinanceGPT — Training Loss", fontsize=14, pad=12)
    ax.set_xlabel("Global step"); ax.set_ylabel("Cross-entropy loss")
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "01_training_loss.png"), dpi=150)
    plt.close(fig)

    # 2 ── Perplexity ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(steps, perps, color=COLORS[1], lw=1.4)
    ax.fill_between(steps, perps, alpha=0.12, color=COLORS[1])
    if len(losses) >= w:
        sp = np.minimum(np.exp(_smooth(losses, w)), 500)
        ax.plot(steps[w - 1:], sp, color=COLORS[1], lw=2.2, alpha=0.8)
    ax.set_title("FinanceGPT — Perplexity  (lower = better)", fontsize=14, pad=12)
    ax.set_xlabel("Global step"); ax.set_ylabel("Perplexity")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "02_perplexity.png"), dpi=150)
    plt.close(fig)

    # 3 ── Train vs Val loss per epoch ───────────────────────────────
    epochs     = history.get("val_epochs", [])
    ep_nums    = [e["epoch"]    for e in epochs]
    ep_tr      = [e["train_loss"] for e in epochs]
    ep_vl      = [e["val_loss"]   for e in epochs]
    if ep_nums:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(ep_nums, ep_tr, "o-",  color=COLORS[0], lw=2, ms=6, label="train")
        ax.plot(ep_nums, ep_vl, "s--", color=COLORS[2], lw=2, ms=6, label="val")
        ax.set_title("FinanceGPT — Train vs Validation Loss per Epoch",
                     fontsize=14, pad=12)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "03_train_vs_val.png"), dpi=150)
        plt.close(fig)

    # 4 ── Learning rate schedule ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(steps, lrs, color=COLORS[3], lw=2)
    ax.set_title("FinanceGPT — Learning Rate Schedule", fontsize=14, pad=12)
    ax.set_xlabel("Global step"); ax.set_ylabel("Learning rate")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "04_learning_rate.png"), dpi=150)
    plt.close(fig)

    print(f"\n  Plots saved → {PLOTS_DIR}/")


# ── Evaluation ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, label_smoothing, use_amp):
    model.eval()
    total_loss, n = 0.0, 0
    ctx = torch.autocast(device.type, dtype=torch.float16) if use_amp else contextlib.nullcontext()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with ctx:
            _, loss = model(x, y, label_smoothing=label_smoothing)
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


# ── Main train ─────────────────────────────────────────────────────────

def train(csv_file: str = None):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp   = TRAIN_CONFIG["mixed_precision"] and device.type == "cuda"
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Use all CPU cores for faster CPU-side ops
    torch.set_num_threads(os.cpu_count() or 4)

    sep_line = "═" * 62
    print(f"\n{sep_line}")
    print("  FinanceGPT — Training Session")
    print(f"  Device : {device}  |  AMP : {use_amp}")
    print(f"{sep_line}\n")

    # ── Data ──────────────────────────────────────────────────────────
    print("[1/5] Loading data…")
    texts = load_all_csv(specific=csv_file)
    if not texts:
        print("  No data found. Aborting.")
        return
    print(f"  Total samples: {len(texts):,}\n")

    # ── Tokenizer ────────────────────────────────────────────────────
    print("[2/5] Tokenizer…")
    tok = BPETokenizer()
    if os.path.exists(TOKENIZER):
        tok.load(TOKENIZER)
        print(f"  Existing vocab: {tok.vocab_size}")
        # Re-train to absorb new text (cumulative BPE)
        old_vocab = dict(tok.vocab)
        tok.train(texts, vocab_size=10_000)
        added = tok.vocab_size - len(old_vocab)
        if added > 0:
            print(f"  +{added} new tokens → {tok.vocab_size} total")
    else:
        tok.train(texts, vocab_size=10_000)
    tok.save(TOKENIZER)

    # ── Dataset ───────────────────────────────────────────────────────
    print("\n[3/5] Building dataset…")
    train_ds, val_ds = make_datasets(
        texts, tok,
        block_size=TRAIN_CONFIG["block_size"],
        val_split=TRAIN_CONFIG["val_split"],
    )
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=TRAIN_CONFIG["batch_size"],
                              shuffle=True,  drop_last=True,  num_workers=0,
                              pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=TRAIN_CONFIG["batch_size"],
                              shuffle=False, drop_last=False, num_workers=0,
                              pin_memory=pin)

    # ── Model ─────────────────────────────────────────────────────────
    print("\n[4/5] Model…")
    history     = {"steps": [], "losses": [], "lrs": [], "val_epochs": []}
    step_offset = 0
    best_val    = float("inf")
    no_improve  = 0

    if os.path.exists(CHECKPOINT):
        print("  Resuming from checkpoint…")
        model, ckpt = FinanceGPT.load(CHECKPOINT)
        if "history" in ckpt:
            history     = ckpt["history"]
            step_offset = history["steps"][-1] if history["steps"] else 0
            best_val    = min((e["val_loss"] for e in history.get("val_epochs", [])),
                              default=float("inf"))
        if ckpt["config"]["vocab_size"] != tok.vocab_size:
            print(f"  Vocab mismatch → rebuilding model head "
                  f"({ckpt['config']['vocab_size']} → {tok.vocab_size})")
            cfg   = {**MODEL_CONFIG, "vocab_size": tok.vocab_size}
            model = FinanceGPT(cfg)
    else:
        cfg   = {**MODEL_CONFIG, "vocab_size": tok.vocab_size}
        model = FinanceGPT(cfg)

    model = model.to(device)
    print(f"  Parameters : {model.num_params():,}")

    optimizer    = torch.optim.AdamW(
        model.parameters(), lr=TRAIN_CONFIG["lr"],
        weight_decay=0.1, betas=(0.9, 0.95),
    )
    total_steps  = len(train_loader) * TRAIN_CONFIG["epochs"]
    global_step  = 0
    grad_accum   = TRAIN_CONFIG["grad_accum"]

    # ── Training loop ─────────────────────────────────────────────────
    print(f"\n[5/5] Training "
          f"({TRAIN_CONFIG['epochs']} epochs × {len(train_loader)} steps, "
          f"eff-batch={TRAIN_CONFIG['batch_size'] * grad_accum})…\n")
    t0 = time.time()

    for epoch in range(1, TRAIN_CONFIG["epochs"] + 1):
        model.train()
        ep_losses = []
        optimizer.zero_grad()

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch:02d}/{TRAIN_CONFIG['epochs']}",
                    ncols=82)
        for step_in_epoch, (x, y) in enumerate(pbar, 1):
            x, y = x.to(device), y.to(device)
            lr   = _lr(global_step, total_steps, TRAIN_CONFIG)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            if use_amp:
                with torch.autocast(device.type, dtype=torch.float16):
                    _, loss = model(x, y, label_smoothing=TRAIN_CONFIG["label_smoothing"])
                scaler.scale(loss / grad_accum).backward()
            else:
                _, loss = model(x, y, label_smoothing=TRAIN_CONFIG["label_smoothing"])
                (loss / grad_accum).backward()

            lv = loss.item()
            ep_losses.append(lv)
            history["steps"].append(step_offset + global_step + 1)
            history["losses"].append(lv)
            history["lrs"].append(lr)
            global_step += 1

            if global_step % grad_accum == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               TRAIN_CONFIG["grad_clip"])
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            pbar.set_postfix(
                loss=f"{lv:.4f}",
                ppl=f"{math.exp(min(lv, 10)):.1f}",
                lr=f"{lr:.1e}",
            )

        # Flush remaining gradients
        if global_step % grad_accum != 0:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["grad_clip"])
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        avg_train = sum(ep_losses) / len(ep_losses)
        val_loss  = evaluate(model, val_loader, device,
                             TRAIN_CONFIG["label_smoothing"], use_amp)
        history["val_epochs"].append({
            "epoch":      epoch,
            "step":       step_offset + global_step,
            "train_loss": avg_train,
            "val_loss":   val_loss,
            "train_ppl":  math.exp(min(avg_train, 10)),
            "val_ppl":    math.exp(min(val_loss,  10)),
        })

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:02d} | "
            f"train={avg_train:.4f} ppl={math.exp(min(avg_train,10)):.1f} | "
            f"val={val_loss:.4f} ppl={math.exp(min(val_loss,10)):.1f} | "
            f"{elapsed:.0f}s"
        )

        # Early stopping
        if val_loss < best_val - 1e-4:
            best_val   = val_loss
            no_improve = 0
            model.save(CHECKPOINT, extra={"history": history})
            print(f"  New best val loss {best_val:.4f} — checkpoint saved.")
        else:
            no_improve += 1
            if no_improve >= TRAIN_CONFIG["patience"]:
                print(f"\n  Early stopping after {TRAIN_CONFIG['patience']} "
                      "epochs without improvement.")
                break

    # Final save (in case early stopping never triggered)
    model.save(CHECKPOINT, extra={"history": history})

    save_plots(history)

    final_tr  = history["val_epochs"][-1]["train_loss"]
    final_val = history["val_epochs"][-1]["val_loss"]
    print(f"\n{sep_line}")
    print("  Training complete!")
    print(f"  Train loss : {final_tr:.4f}   ppl : {math.exp(min(final_tr,10)):.2f}")
    print(f"  Val   loss : {final_val:.4f}   ppl : {math.exp(min(final_val,10)):.2f}")
    print(f"  Best val   : {best_val:.4f}")
    print(f"  Plots      : {PLOTS_DIR}/")
    print(f"{sep_line}\n")
