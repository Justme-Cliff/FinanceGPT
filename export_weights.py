"""
export_weights.py -- Convert PyTorch FinanceGPT checkpoint to C binary format.

Usage:
    python export_weights.py
    python export_weights.py --input checkpoints/finance_gpt.pt --output checkpoints/finance_gpt.bin

Binary format (FGPT v1):
    magic:       char[4] = "FGPT"
    version:     uint32 = 1
    vocab_size:  uint32
    d_model:     uint32
    n_heads:     uint32
    n_layers:    uint32
    d_ff:        uint32
    max_seq_len: uint32
    n_tensors:   uint32
    for each tensor:
        name_len: uint32
        name:     char[name_len]  (no null)
        n_dims:   uint32
        dims:     uint32[n_dims]
        n_elems:  uint32
        data:     float32[n_elems]  (little-endian)
"""
import argparse
import os
import struct
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Export FinanceGPT PyTorch checkpoint to C binary"
    )
    parser.add_argument(
        "--input",
        default="checkpoints/finance_gpt.pt",
        help="PyTorch checkpoint (default: checkpoints/finance_gpt.pt)",
    )
    parser.add_argument(
        "--output",
        default="checkpoints/finance_gpt.bin",
        help="C binary output path (default: checkpoints/finance_gpt.bin)",
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch not installed.  Run: pip install torch")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(
            f"ERROR: {args.input} not found.\n"
            f"Train first with: python main.py /train"
        )
        sys.exit(1)

    print(f"Loading {args.input} ...")
    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    sd   = ckpt["model"]

    print(f"Model config   : {cfg}")
    print(f"State dict keys: {len(sd)} tensors")

    vocab_size  = int(cfg["vocab_size"])
    d_model     = int(cfg["d_model"])
    n_heads     = int(cfg["n_heads"])
    n_layers    = int(cfg["n_layers"])
    d_ff        = int(cfg["d_ff"])
    max_seq_len = int(cfg["max_seq_len"])

    # ------------------------------------------------------------------
    # Map PyTorch state_dict keys -> C tensor names.
    #
    # PyTorch model layout (from model.py):
    #   embed.weight                      -> embed
    #   ln_f.weight                       -> ln_f
    #   blocks.{L}.ln1.weight             -> blocks.{L}.ln1_w
    #   blocks.{L}.attn.c_attn.weight     -> blocks.{L}.attn_qkv
    #   blocks.{L}.attn.c_proj.weight     -> blocks.{L}.attn_proj
    #   blocks.{L}.ln2.weight             -> blocks.{L}.ln2_w
    #   blocks.{L}.mlp.gate.weight        -> blocks.{L}.ffn_gate
    #   blocks.{L}.mlp.up.weight          -> blocks.{L}.ffn_up
    #   blocks.{L}.mlp.down.weight        -> blocks.{L}.ffn_down
    #
    # Weight-tied lm_head is the same tensor as embed -- we only write it
    # once (under the name "embed").
    # ------------------------------------------------------------------

    def get_tensor(key):
        """Look up key in state dict; handles torch.compile _orig_mod. prefix."""
        for prefix in ("", "_orig_mod."):
            full_key = prefix + key
            if full_key in sd:
                return sd[full_key].float().detach().cpu()
        print(f"  WARNING: key not found: {key}")
        return None

    tensors = []  # list of (c_name, numpy_array)

    def add(c_name, pt_key):
        t = get_tensor(pt_key)
        if t is not None:
            arr = t.numpy()
            tensors.append((c_name, arr))
            shape_str = "x".join(str(d) for d in arr.shape)
            print(f"  {c_name:45s}  [{shape_str}]")

    print("\nExporting tensors:")
    add("embed",  "embed.weight")
    add("ln_f",   "ln_f.weight")

    for layer in range(n_layers):
        add(f"blocks.{layer}.ln1_w",    f"blocks.{layer}.ln1.weight")
        add(f"blocks.{layer}.attn_qkv", f"blocks.{layer}.attn.c_attn.weight")
        add(f"blocks.{layer}.attn_proj",f"blocks.{layer}.attn.c_proj.weight")
        add(f"blocks.{layer}.ln2_w",    f"blocks.{layer}.ln2.weight")
        add(f"blocks.{layer}.ffn_gate", f"blocks.{layer}.mlp.gate.weight")
        add(f"blocks.{layer}.ffn_up",   f"blocks.{layer}.mlp.up.weight")
        add(f"blocks.{layer}.ffn_down", f"blocks.{layer}.mlp.down.weight")

    print(f"\nTotal tensors : {len(tensors)}")

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"\nWriting {args.output} ...")
    import numpy as np

    with open(args.output, "wb") as f:
        # ── Fixed header ────────────────────────────────────────────
        f.write(b"FGPT")                            # magic (4 bytes)
        f.write(struct.pack("<I", 1))               # version
        f.write(struct.pack("<I", vocab_size))
        f.write(struct.pack("<I", d_model))
        f.write(struct.pack("<I", n_heads))
        f.write(struct.pack("<I", n_layers))
        f.write(struct.pack("<I", d_ff))
        f.write(struct.pack("<I", max_seq_len))
        f.write(struct.pack("<I", len(tensors)))    # n_tensors

        # ── Per-tensor records ───────────────────────────────────────
        for c_name, arr in tensors:
            name_bytes = c_name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))  # name_len
            f.write(name_bytes)                           # name (no NUL)
            f.write(struct.pack("<I", arr.ndim))          # n_dims
            for dim in arr.shape:
                f.write(struct.pack("<I", int(dim)))      # each dimension
            n_elems = int(arr.size)
            f.write(struct.pack("<I", n_elems))           # n_elems
            f.write(arr.astype(np.float32).tobytes())     # raw float32 LE

    file_size = os.path.getsize(args.output)
    print(f"\nDone!")
    print(f"Output    : {args.output}")
    print(f"File size : {file_size / 1024 / 1024:.1f} MB")
    print(f"\nNext steps:")
    print(f"  make")
    print(f"  ./financegpt /chat")


if __name__ == "__main__":
    main()
