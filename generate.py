# generate.py — эталонная генерация на PyTorch
import argparse
import torch
from model import Config, TinyTransformer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpt.pt")
    p.add_argument("--prompt", default="В: Какая столица у Франции?\nО: ")
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--temp", type=float, default=0.0)   # 0 = greedy
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = Config(**ck["config"])
    chars = ck["chars"]
    stoi = {c: i for i, c in enumerate(chars)}
    model = TinyTransformer(cfg)
    model.load_state_dict(ck["model"])
    model.eval()

    ids = [stoi[c] for c in args.prompt if c in stoi] or [0]
    idx = torch.tensor([ids], dtype=torch.long)

    print(args.prompt, end="", flush=True)
    with torch.no_grad():
        for k in range(args.tokens):
            logits, _ = model(idx[:, -cfg.block_size:])
            logits = logits[:, -1, :]
            if args.debug and k == 0:
                v, i = torch.topk(logits[0], 5)
                for li, ii in zip(v.tolist(), i.tolist()):
                    print(f"\n[debug] {chars[ii]!r} id={ii} logit={li:.5f}", end="")
                print()
            if args.temp <= 0:
                nxt = int(torch.argmax(logits, dim=-1))
            else:
                lg = logits / args.temp
                if args.top_k > 0:
                    thr = torch.topk(lg, min(args.top_k, lg.size(-1))).values[..., -1, None]
                    lg = lg.masked_fill(lg < thr, float("-inf"))
                nxt = int(torch.multinomial(torch.softmax(lg, dim=-1), 1))
            print(chars[nxt], end="", flush=True)
            idx = torch.cat([idx, torch.tensor([[nxt]])], dim=1)
            if "".join(chars[t] for t in idx[0, -2:].tolist()) == "\n\n":
                break
    print()


if __name__ == "__main__":
    main()