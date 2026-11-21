# eval_accuracy.py — доля правильных ответов по всем вопросам (greedy) + CSV
import argparse
import csv
import re

import torch

from model import Config, TinyTransformer


def norm(s):
    return re.sub(r"\s+", " ", s.lower().replace("ё", "е")).strip(" .")


def parse_pairs(path):
    lines = [l.strip() for l in open(path, encoding="utf-8").read().splitlines() if l.strip()]
    return [(lines[i][2:].strip(), lines[i + 1][2:].strip())
            for i in range(len(lines) - 1)
            if lines[i].startswith("В:") and lines[i + 1].startswith("О:")]


@torch.no_grad()
def answer(model, cfg, stoi, chars, q, max_new=90):
    prompt = f"В: {q}\nО: "
    idx = torch.tensor([[stoi[c] for c in prompt if c in stoi]], dtype=torch.long)
    out = []
    for _ in range(max_new):
        logits, _ = model(idx[:, -cfg.block_size:])
        nxt = int(torch.argmax(logits[0, -1]))
        ch = chars[nxt]
        if ch == "\n":
            break
        out.append(ch)
        idx = torch.cat([idx, torch.tensor([[nxt]])], dim=1)
    return "".join(out).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpt.pt")
    p.add_argument("--data", default="dataset.txt")
    p.add_argument("--csv", default="results.csv")
    p.add_argument("--show", type=int, default=15)
    args = p.parse_args()

    torch.set_num_threads(4)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg, chars = Config(**ck["config"]), ck["chars"]
    stoi = {c: i for i, c in enumerate(chars)}
    model = TinyTransformer(cfg)
    model.load_state_dict(ck["model"])
    model.eval()

    pairs = parse_pairs(args.data)
    rows, exact, contains = [], 0, 0
    for i, (q, gold) in enumerate(pairs, 1):
        pred = answer(model, cfg, stoi, chars, q)
        ok_e = norm(pred) == norm(gold)
        ok_c = norm(gold) in norm(pred)
        exact += ok_e
        contains += ok_c
        rows.append({"question": q, "gold": gold, "pred": pred,
                     "exact": int(ok_e), "contains": int(ok_c)})
        print(f"\r[{i}/{len(pairs)}] exact {100 * exact / i:5.1f}%", end="", flush=True)

    print(f"\n\nточных ответов: {exact}/{len(pairs)} = {100 * exact / len(pairs):.1f}%")
    print(f"содержат верный ответ: {contains}/{len(pairs)} = {100 * contains / len(pairs):.1f}%")

    bad = [r for r in rows if not r["exact"]]
    for r in bad[:args.show]:
        print(f"  ✗ {r['question']}\n     ждут: {r['gold']}\n     выдал: {r['pred']}")

    with open(args.csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question", "gold", "pred", "exact", "contains"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV: {args.csv}")


if __name__ == "__main__":
    main()