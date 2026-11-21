# augment.py — размножение шаблонов Q&A, нормализация типографики, перемешивание
import argparse
import random
import re

Q_TEMPLATES = [
    "Какая столица у {c}?",
    "Назови столицу {c}.",
    "У {c} какая столица?",
    "Какой город является столицей {c}?",
    "Столица {c} - какой город?",
    "Подскажи столицу {c}.",
    "Какая столица государства {c}?",
]
A_TEMPLATES = ["{a}", "Столица - {a}", "Это {a}"]


def normalize(s: str) -> str:
    return (s.replace("\u00a0", " ")
             .replace("’", "'").replace("‘", "'")
             .replace("—", "-").replace("–", "-"))


def parse_pairs(text: str):
    lines = [l.strip() for l in normalize(text).splitlines() if l.strip()]
    pairs = []
    for i in range(len(lines) - 1):
        if lines[i].startswith("В:") and lines[i + 1].startswith("О:"):
            pairs.append((lines[i][2:].strip(), lines[i + 1][2:].strip()))
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="src", default="dataset.txt")
    p.add_argument("--out", dest="dst", default="dataset_aug.txt")
    p.add_argument("--variants", type=int, default=6, help="вариантов вопроса на пару")
    p.add_argument("--lower", type=float, default=0.25, help="доля строчных вариантов")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    rnd = random.Random(args.seed)
    text = open(args.src, encoding="utf-8").read()
    pairs = parse_pairs(text)
    print(f"исходных пар: {len(pairs)}")

    out = []
    for q, a in pairs:
        m = re.match(r"^Какая столица у (.+)\?$", q)
        country = m.group(1) if m else None
        if country is None:                       # нестандартный вопрос — оставляем как есть
            out.append(f"В: {q}\nО: {a}")
            continue
        for tpl in rnd.sample(Q_TEMPLATES, min(args.variants, len(Q_TEMPLATES))):
            qq = tpl.format(c=country)
            aa = rnd.choice(A_TEMPLATES).format(a=a)
            if rnd.random() < args.lower:
                qq, aa = qq.lower(), aa.lower()
            out.append(f"В: {qq}\nО: {aa}")

    rnd.shuffle(out)                              # ломаем алфавитную утечку
    result = "\n\n".join(out) + "\n"              # пустая строка = чистый стоп-сигнал
    open(args.dst, "w", encoding="utf-8").write(result)

    print(f"пар после аугментации: {len(out)}")
    print(f"символов: {len(result):,} | словарь: {len(set(result))} "
          f"| размер: {len(result.encode('utf-8')) / 1024:.1f} КБ")


if __name__ == "__main__":
    main()

# исходных пар: 195
# пар после аугментации: 1170
# символов: 63,799 | словарь: 69 | размер: 108.2 КБ