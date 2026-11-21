# export.py — ckpt.pt -> model.bin
import argparse
import os
import struct

import numpy as np
import torch


def write_vec(f, t):
    v = t.detach().cpu().numpy().astype(np.float32)
    f.write(v.tobytes())
    return v.size * 4


def write_mat(f, t, quant):
    w = t.detach().cpu().numpy().astype(np.float32)   # [out, in]
    rows, cols = w.shape
    if quant:
        scale = np.maximum(np.abs(w).max(axis=1) / 127.0, 1e-8).astype(np.float32)
        q = np.clip(np.rint(w / scale[:, None]), -127, 127).astype(np.int8)
        f.write(scale.tobytes())
        f.write(q.tobytes())
        return rows * cols + rows * 4
    f.write(w.tobytes())
    return rows * cols * 4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpt.pt")
    p.add_argument("--out", default="model.bin")
    p.add_argument("--quant", choices=["none", "int8"], default="int8")
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg, chars, sd = ck["config"], ck["chars"], ck["model"]
    V, D = cfg["vocab_size"], cfg["n_embd"]
    L, H, S, HID = cfg["n_layer"], cfg["n_head"], cfg["block_size"], cfg["hidden"]
    assert len(chars) == V
    quant = args.quant == "int8"

    total = 0
    with open(args.out, "wb") as f:
        f.write(b"TLLM")
        f.write(struct.pack("<8i", 1, V, D, L, H, S, HID, int(quant)))
        total += 4 + 32

        for ch in chars:                      # словарь: u16 длина + UTF-8 байты
            b = ch.encode("utf-8")
            f.write(struct.pack("<H", len(b)))
            f.write(b)
            total += 2 + len(b)

        total += write_vec(f, sd["tok_emb.weight"])          # [V, D]
        total += write_vec(f, sd["pos_emb.weight"])          # [S, D]
        for i in range(L):
            b = f"blocks.{i}."
            total += write_vec(f, sd[b + "rms_att.weight"])
            for name in ("wq", "wk", "wv", "wo"):
                total += write_mat(f, sd[b + f"attn.{name}.weight"], quant)
            total += write_vec(f, sd[b + "rms_ffn.weight"])
            total += write_mat(f, sd[b + "mlp.w1.weight"], quant)   # [HID, D]
            total += write_mat(f, sd[b + "mlp.w2.weight"], quant)   # [D, HID]
        total += write_vec(f, sd["rms_final.weight"])
        total += write_mat(f, sd["lm_head.weight"], quant)          # [V, D]

    size = os.path.getsize(args.out)
    assert size == total, (size, total)
    print(f"{args.out}: {size:,} байт ({size / 1e6:.2f} МБ), quant={args.quant}, "
          f"val_loss={ck.get('val_loss', float('nan')):.4f}")


if __name__ == "__main__":
    main()