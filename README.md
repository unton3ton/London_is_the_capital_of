```bash
mkdir -p ~/my-tiny-llm/src && cd ~/my-tiny-llm
python3 -m venv venv
source venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

my-tiny-llm/
├── dataset.txt
├── model.py        # архитектура (общая для train/export/generate)
├── train.py        # обучение + прогресс-бар
├── export.py       # ckpt.pt -> model.bin (f32 / int8)
├── generate.py     # эталонная генерация на PyTorch (для сверки)
├── Cargo.toml
└── src/main.rs     # движок на Rust

# быстрый прогон — проверить, что всё работает
python3.10 augment.py --variants 6

python3.10 train.py --data dataset_aug.txt --dim 96 --layers 3 --heads 4 \
                --block 128 --batch 32 --steps 1200 --lr 3e-3 --warmup 80 \
                --val-frac 0.05 --threads 4 --device cuda

python3.10 eval_accuracy.py --ckpt ckpt.pt --data dataset.txt

# рабочий вариант
python3.10 train.py --data dataset_aug.txt --dim 128 --layers 4 --heads 4 \
                --block 128 --batch 32 --steps 3000 --lr 3e-3 --warmup 100 \
                --val-frac 0.05 --threads 4 --out ckpt.pt --device cuda

python3.10 eval_accuracy.py --ckpt ckpt.pt --data dataset.txt --csv results.csv


# 1. экспорт весов
# python3.10 export.py --ckpt ckpt.pt --out model.bin --quant int8
python3.10 export.py --ckpt ckpt.pt --out model_f32.bin --quant none

# 2. сборка движка
RUSTFLAGS="-C target-cpu=native" cargo build --release

# 3. запуск
./target/release/tiny_llm model_f32.bin --prompt "В: Какая столица у России? О: " --temp 0.7

# модель: vocab=69 dim=128 layers=4 heads=4 ctx=128 quant=false
# В: Какая столица у России? О: Это Москва.
# [13 токенов за 0.00 с = 2911.1 tok/s]


./target/release/tiny_llm model_f32.bin --chat

# модель: vocab=69 dim=128 layers=4 heads=4 ctx=128 quant=false
# Чат с микро-моделью (пустая строка или quit — выход).

# В: Какая столица у Германии
# О: Бейрлин.
# [9 токенов за 0.00 с = 2533.5 tok/s]


./target/release/tiny_llm model_f32.bin --bench --tokens 512 --temp 0
# модель: vocab=69 dim=128 layers=4 heads=4 ctx=128 quant=false
# [512 токенов за 0.25 с = 2029.0 tok/s]

python3.10 generate.py --ckpt ckpt.pt --temp 0 --tokens 120 --debug --prompt "В: Какая столица у Великобритании? О: "
В: Какая столица у Великобритании? О: 
[debug] 'Э' id=35 logit=13.19154
[debug] 'С' id=27 logit=12.16045
[debug] 'Б' id=12 logit=11.54877
[debug] 'Т' id=28 logit=8.79673
[debug] 'Л' id=21 logit=8.63011
Это Браззавилиль.


testuser@pop-os:~/Рабочий стол/Rust Inference Engine$ ./target/release/tiny_llm model_f32.bin --temp 0 --tokens 120 --debug --prompt "В: Какая столица у Великобритании? О: "
модель: vocab=69 dim=128 layers=4 heads=4 ctx=128 quant=false
В: Какая столица у Великобритании? О: 
[debug] "Э" id=35 logit=13.19154
[debug] "С" id=27 logit=12.16045
[debug] "Б" id=12 logit=11.54877
[debug] "Т" id=28 logit=8.79673
[debug] "Л" id=21 logit=8.63011
Это Браззавилиль.
[19 токенов за 0.01 с = 3181.7 tok/s]


./tiny_llm model_f32.bin --temp 0 --tokens 120 --prompt "В: Какая столица у Японии? О: "
модель: vocab=69 dim=128 layers=4 heads=4 ctx=128 quant=false
В: Какая столица у Японии? О: Это Токио.
[12 токенов за 0.00 с = 3509.9 tok/s]
```

# Source

* [TINY 2.1 — a transformer for DOS](https://github.com/WollyMonjac/Tiny-2.1)
