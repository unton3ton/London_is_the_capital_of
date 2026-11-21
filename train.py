# train.py — обучение на CPU с прогресс-баром без внешних зависимостей

# python train.py --data dataset.txt --steps 5000 --dim 128 --layers 4 --heads 4

import argparse
import os
import time
from dataclasses import asdict
import math
import torch

from model import Config, TinyTransformer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset.txt")
    p.add_argument("--out", default="ckpt.pt")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--block", type=int, default=128)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cpu")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    return p.parse_args()


def fmt_eta(sec):
    sec = int(max(0, sec))
    return f"{sec // 60:d}m{sec % 60:02d}s"


def progress(step, total, loss, tps, eta, width=28):
    filled = int(width * step / total)
    bar = "█" * filled + "░" * (width - filled)
    print(f"\r[{bar}] {100 * step / total:5.1f}% | loss {loss:.4f} "
          f"| {tps:,.0f} tok/s | ETA {fmt_eta(eta)}", end="", flush=True)

def lr_at(step):
    args = parse_args()
    if step <= args.warmup:
        return args.lr * step / max(1, args.warmup)
    t = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    text = open(args.data, encoding="utf-8").read().replace("\r\n", "\n")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    if len(data) < args.block * 4:
        raise SystemExit(f"Датасет слишком мал: {len(data)} символов. Нужно хотя бы {args.block * 4}.")

    cfg = Config(vocab_size=len(chars), block_size=args.block,
                     n_layer=args.layers, n_head=args.heads, n_embd=args.dim)

    n = int((1 - args.val_frac) * len(data))
    train_data = data[:n]
    tail = data[n:]
    # если хвост короче окна — валидируемся на хвосте train (для заучивания это нормально)
    val_data = tail if len(tail) > cfg.block_size + 2 else train_data[-(cfg.block_size + 2):]
    train_data, val_data = data[:n], data[n:]
    print(f"символов: {len(data):,} | словарь: {len(chars)} | потоков: {args.threads}")


    model = TinyTransformer(cfg).to(args.device)
    print(f"параметров: {model.num_params():,} "
          f"(~{model.num_params() * 4 / 1e6:.2f} МБ f32, ~{model.num_params() / 1e6:.2f} МБ int8)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.01)

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - cfg.block_size - 1, (args.batch,))
        x = torch.stack([d[i:i + cfg.block_size] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + cfg.block_size] for i in ix])
        return x.to(args.device), y.to(args.device)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            acc = 0.0
            for _ in range(args.eval_batches):
                x, y = get_batch(split)
                acc += model(x, y)[1].item()
            out[split] = acc / args.eval_batches
        model.train()
        return out

    @torch.no_grad()
    def peek(prompt="В: ", max_new=160, temp=0.7):
        model.eval()
        # idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long)
        idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long).to(args.device)
        for _ in range(max_new):
            logits, _ = model(idx[:, -cfg.block_size:])
            probs = torch.softmax(logits[:, -1, :] / temp, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        model.train()
        return "".join(chars[i] for i in idx[0].tolist())

    best_val = float("inf")
    t0 = time.time()
    smooth = None
    model.train()

    for step in range(1, args.steps + 1):
        x, y = get_batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.step()

        lv = loss.item()
        smooth = lv if smooth is None else 0.95 * smooth + 0.05 * lv
        done = step * args.batch * cfg.block_size
        elapsed = time.time() - t0
        progress(step, args.steps, smooth, done / elapsed,
                 elapsed / step * (args.steps - step))

        if step % args.eval_every == 0 or step == args.steps:
            m = estimate_loss()
            print(f"\nшаг {step:6d} | train {m['train']:.4f} | val {m['val']:.4f}")
            if m["val"] < best_val:
                best_val = m["val"]
                torch.save({"model": model.state_dict(),
                            "config": asdict(cfg),
                            "chars": chars,
                            "val_loss": best_val,
                            "step": step}, args.out)
                print(f"  ↳ сохранён лучший чекпоинт: {args.out}")
            if step % (args.eval_every * 4) == 0 or step == args.steps:
                print("  пример:", repr(peek()[:160]))

    print(f"\nготово за {fmt_eta(time.time() - t0)} | лучший val {best_val:.4f}")


if __name__ == "__main__":
    main()