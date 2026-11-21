// src/main.rs — инференс микро-трансформера. Только std, без крейтов.
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// ------------------------- формат модели -------------------------

#[derive(Clone, Copy, Debug)]
struct Config {
    vocab: usize,
    dim: usize,
    layers: usize,
    heads: usize,
    seq: usize,
    hidden: usize,
    quant: bool,
}

enum Matrix {
    F32 { w: Vec<f32>, rows: usize, cols: usize },
    I8 { q: Vec<i8>, s: Vec<f32>, rows: usize, cols: usize },
}

impl Matrix {
    #[inline]
    fn matmul(&self, y: &mut [f32], x: &[f32]) {
        match self {
            Matrix::F32 { w, rows, cols } => {
                for o in 0..*rows {
                    let r = &w[o * cols..(o + 1) * cols];
                    y[o] = r.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                }
            }
            Matrix::I8 { q, s, rows, cols } => {
                for o in 0..*rows {
                    let r = &q[o * cols..(o + 1) * cols];
                    let acc: f32 = r.iter().zip(x.iter()).map(|(a, b)| (*a as f32) * b).sum();
                    y[o] = acc * s[o]; // дековантование одним умножением на строку
                }
            }
        }
    }
}

struct Layer {
    rms_att: Vec<f32>,
    wq: Matrix,
    wk: Matrix,
    wv: Matrix,
    wo: Matrix,
    rms_ffn: Vec<f32>,
    w1: Matrix,
    w2: Matrix,
}

struct Weights {
    tok_emb: Vec<f32>,
    pos_emb: Vec<f32>,
    layers: Vec<Layer>,
    rms_final: Vec<f32>,
    wcls: Matrix,
}

struct Reader {
    r: BufReader<File>,
}

impl Reader {
    fn open(path: &str) -> Self {
        let f = File::open(path).unwrap_or_else(|e| panic!("не открыть {path}: {e}"));
        Reader { r: BufReader::new(f) }
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        let mut b = vec![0u8; n];
        self.r.read_exact(&mut b).expect("файл модели обрезан");
        b
    }
    fn i32(&mut self) -> i32 {
        let b = self.bytes(4);
        i32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }
    fn u16(&mut self) -> u16 {
        let b = self.bytes(2);
        u16::from_le_bytes([b[0], b[1]])
    }
    fn f32s(&mut self, n: usize) -> Vec<f32> {
        self.bytes(n * 4)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
    fn string(&mut self) -> String {
        let n = self.u16() as usize;
        String::from_utf8(self.bytes(n)).expect("символ не UTF-8")
    }
    fn matrix(&mut self, quant: bool, rows: usize, cols: usize) -> Matrix {
        if quant {
            let s = self.f32s(rows);
            let q: Vec<i8> = self.bytes(rows * cols).into_iter().map(|b| b as i8).collect();
            Matrix::I8 { q, s, rows, cols }
        } else {
            Matrix::F32 { w: self.f32s(rows * cols), rows, cols }
        }
    }
}

// ------------------------- состояние / forward -------------------------

struct State {
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    hb: Vec<f32>,
    logits: Vec<f32>,
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
}

struct Transformer {
    cfg: Config,
    w: Weights,
    s: State,
    tokens: Vec<String>,
    map: HashMap<char, usize>,
}

#[inline]
fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32]) {
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let f = 1.0 / (ss + 1e-5).sqrt();
    for i in 0..x.len() {
        out[i] = weight[i] * x[i] * f;
    }
}

#[inline]
fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

impl Transformer {
    fn load(path: &str) -> Self {
        let mut r = Reader::open(path);
        assert_eq!(&r.bytes(4), b"TLLM", "это не model.bin");
        let version = r.i32();
        assert_eq!(version, 1, "неподдерживаемая версия формата: {version}");
        let cfg = Config {
            vocab: r.i32() as usize,
            dim: r.i32() as usize,
            layers: r.i32() as usize,
            heads: r.i32() as usize,
            seq: r.i32() as usize,
            hidden: r.i32() as usize,
            quant: r.i32() != 0,
        };
        assert!(cfg.dim % cfg.heads == 0, "dim должен делиться на heads");

        let mut tokens = Vec::with_capacity(cfg.vocab);
        let mut map = HashMap::new();
        for i in 0..cfg.vocab {
            let s = r.string();
            let mut it = s.chars();
            if let (Some(c), None) = (it.next(), it.next()) {
                map.insert(c, i);
            }
            tokens.push(s);
        }

        let d = cfg.dim;
        let tok_emb = r.f32s(cfg.vocab * d);
        let pos_emb = r.f32s(cfg.seq * d);
        let mut layers = Vec::with_capacity(cfg.layers);
        for _ in 0..cfg.layers {
            let rms_att = r.f32s(d);
            let wq = r.matrix(cfg.quant, d, d);
            let wk = r.matrix(cfg.quant, d, d);
            let wv = r.matrix(cfg.quant, d, d);
            let wo = r.matrix(cfg.quant, d, d);
            let rms_ffn = r.f32s(d);
            let w1 = r.matrix(cfg.quant, cfg.hidden, d);
            let w2 = r.matrix(cfg.quant, d, cfg.hidden);
            layers.push(Layer { rms_att, wq, wk, wv, wo, rms_ffn, w1, w2 });
        }
        let rms_final = r.f32s(d);
        let wcls = r.matrix(cfg.quant, cfg.vocab, d);

        let cache = cfg.layers * cfg.seq * d;
        let s = State {
            x: vec![0.0; d],
            xb: vec![0.0; d],
            xb2: vec![0.0; d],
            q: vec![0.0; d],
            k: vec![0.0; d],
            v: vec![0.0; d],
            att: vec![0.0; cfg.heads * cfg.seq],
            hb: vec![0.0; cfg.hidden],
            logits: vec![0.0; cfg.vocab],
            key_cache: vec![0.0; cache],
            value_cache: vec![0.0; cache],
        };
        Transformer { cfg, w: Weights { tok_emb, pos_emb, layers, rms_final, wcls }, s, tokens, map }
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::new();
        for c in text.chars() {
            match self.map.get(&c) {
                Some(&i) => ids.push(i),
                None => eprintln!("предупреждение: символа {c:?} нет в словаре, пропуск"),
            }
        }
        ids
    }

    fn clear_cache(&mut self) {
        self.s.key_cache.fill(0.0);
        self.s.value_cache.fill(0.0);
    }

    /// Один токен на позиции pos. Результат — в self.s.logits.
    fn forward(&mut self, token: usize, pos: usize) {
        let c = self.cfg;
        let d = c.dim;
        let hs = d / c.heads;
        let scale = 1.0 / (hs as f32).sqrt();
        let w = &self.w;

        self.s.x.copy_from_slice(&w.tok_emb[token * d..(token + 1) * d]);
        for i in 0..d {
            self.s.x[i] += w.pos_emb[pos * d + i];
        }

        for l in 0..c.layers {
            let lw = &w.layers[l];
            let base = l * c.seq * d;

            // --- attention ---
            rmsnorm(&mut self.s.xb, &self.s.x, &lw.rms_att);
            lw.wq.matmul(&mut self.s.q, &self.s.xb);
            lw.wk.matmul(&mut self.s.k, &self.s.xb);
            lw.wv.matmul(&mut self.s.v, &self.s.xb);
            let off = base + pos * d;
            self.s.key_cache[off..off + d].copy_from_slice(&self.s.k);
            self.s.value_cache[off..off + d].copy_from_slice(&self.s.v);

            for h in 0..c.heads {
                let qh = &self.s.q[h * hs..(h + 1) * hs];
                for t in 0..=pos {
                    let ko = base + t * d + h * hs;
                    let kh = &self.s.key_cache[ko..ko + hs];
                    let score: f32 = qh.iter().zip(kh.iter()).map(|(a, b)| a * b).sum();
                    self.s.att[h * c.seq + t] = score * scale;
                }
                let a0 = h * c.seq;
                softmax(&mut self.s.att[a0..a0 + pos + 1]);
                let out = &mut self.s.xb[h * hs..(h + 1) * hs];
                out.fill(0.0);
                for t in 0..=pos {
                    let a = self.s.att[a0 + t];
                    let vo = base + t * d + h * hs;
                    for i in 0..hs {
                        out[i] += a * self.s.value_cache[vo + i];
                    }
                }
            }
            lw.wo.matmul(&mut self.s.xb2, &self.s.xb);
            for i in 0..d {
                self.s.x[i] += self.s.xb2[i];
            }

            // --- MLP (SiLU) ---
            rmsnorm(&mut self.s.xb, &self.s.x, &lw.rms_ffn);
            lw.w1.matmul(&mut self.s.hb, &self.s.xb);
            for v in self.s.hb.iter_mut() {
                *v = *v / (1.0 + (-*v).exp());
            }
            lw.w2.matmul(&mut self.s.xb2, &self.s.hb);
            for i in 0..d {
                self.s.x[i] += self.s.xb2[i];
            }
        }

        rmsnorm(&mut self.s.xb, &self.s.x, &w.rms_final);
        w.wcls.matmul(&mut self.s.logits, &self.s.xb);
    }
}

// ------------------------- сэмплинг -------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        let s = if seed != 0 {
            seed
        } else {
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_nanos() as u64
        };
        Rng(s | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn argmax(x: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..x.len() {
        if x[i] > x[best] {
            best = i;
        }
    }
    best
}

fn sample(logits: &[f32], temp: f32, top_k: usize, top_p: f32, rng: &mut Rng) -> usize {
    if temp <= 0.0 {
        return argmax(logits);
    }
    let mut p: Vec<f32> = logits.iter().map(|v| v / temp).collect();
    softmax(&mut p);
    let mut idx: Vec<usize> = (0..p.len()).collect();
    idx.sort_unstable_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap_or(std::cmp::Ordering::Equal));

    if top_k > 0 && top_k < idx.len() {
        idx.truncate(top_k);
    }
    if top_p > 0.0 && top_p < 1.0 {
        let mut cum = 0.0;
        let mut keep = 0;
        for &i in &idx {
            cum += p[i];
            keep += 1;
            if cum >= top_p {
                break;
            }
        }
        idx.truncate(keep.max(1));
    }
    let total: f32 = idx.iter().map(|&i| p[i]).sum();
    let mut r = rng.next_f32() * total;
    for &i in &idx {
        r -= p[i];
        if r <= 0.0 {
            return i;
        }
    }
    *idx.last().unwrap()
}

// ------------------------- потоковый вывод со стоп-строками -------------------------

struct Emitter {
    stops: Vec<String>,
    pending: String,
    hold: usize,
}

impl Emitter {
    fn new(stops: &[String]) -> Self {
        let hold = stops.iter().map(|s| s.len()).max().unwrap_or(0);
        Emitter { stops: stops.to_vec(), pending: String::new(), hold }
    }
    /// false -> встретилась стоп-строка, генерацию пора остановить
    fn push(&mut self, s: &str) -> bool {
        self.pending.push_str(s);
        for st in &self.stops {
            if let Some(p) = self.pending.find(st.as_str()) {
                print!("{}", &self.pending[..p]);
                self.pending.clear();
                let _ = io::stdout().flush();
                return false;
            }
        }
        let mut cut = self.pending.len().saturating_sub(self.hold);
        while cut > 0 && !self.pending.is_char_boundary(cut) {
            cut -= 1;
        }
        if cut > 0 {
            print!("{}", &self.pending[..cut]);
            self.pending.drain(..cut);
            let _ = io::stdout().flush();
        }
        true
    }
    fn flush(&mut self) {
        print!("{}", self.pending);
        self.pending.clear();
        let _ = io::stdout().flush();
    }
}

// ------------------------- генерация -------------------------

fn generate(t: &mut Transformer, a: &Args, prompt: &str, echo: &str, stops: &[String]) {
    let mut hist = t.encode(prompt);
    if hist.is_empty() {
        hist.push(0);
    }
    if hist.len() > t.cfg.seq {
        hist = hist[hist.len() - t.cfg.seq..].to_vec();
    }
    if !a.bench {
        print!("{echo}");
        let _ = io::stdout().flush();
    }

    t.clear_cache();
    let mut pos = hist.len() - 1;
    for i in 0..pos {
        t.forward(hist[i], i);
    }

    let mut rng = Rng::new(a.seed);
    let mut emit = Emitter::new(stops);
    let mut debug_done = false;
    let start = Instant::now();
    let mut n = 0usize;

    while n < a.tokens {
        let last = *hist.last().unwrap();
        t.forward(last, pos);

        if a.debug && !debug_done {
            let mut idx: Vec<usize> = (0..t.s.logits.len()).collect();
            idx.sort_unstable_by(|&x, &y| t.s.logits[y].partial_cmp(&t.s.logits[x]).unwrap());
            eprintln!();
            for &i in idx.iter().take(5) {
                eprintln!("[debug] {:?} id={} logit={:.5}", t.tokens[i], i, t.s.logits[i]);
            }
            debug_done = true;
        }

        let next = sample(&t.s.logits, a.temp, a.top_k, a.top_p, &mut rng);
        hist.push(next);
        n += 1;

        if !a.bench {
            let piece = t.tokens[next].clone();
            if !emit.push(&piece) {
                break;
            }
        }

        pos += 1;
        // скользящий контекст: пересобираем кэш из последней половины окна
        if pos >= t.cfg.seq {
            let keep = t.cfg.seq / 2;
            let tail = hist[hist.len() - keep..].to_vec();
            t.clear_cache();
            for i in 0..keep - 1 {
                t.forward(tail[i], i);
            }
            hist = tail;
            pos = keep - 1;
        }
    }
    if !a.bench {
        emit.flush();
        println!();
    }
    let dt = start.elapsed().as_secs_f64();
    eprintln!("[{n} токенов за {dt:.2} с = {:.1} tok/s]", n as f64 / dt.max(1e-9));
}

fn chat(t: &mut Transformer, a: &Args) {
    eprintln!("Чат с микро-моделью (пустая строка или quit — выход).");
    let stops = vec!["\n".to_string(), "В:".to_string()];
    let stdin = io::stdin();
    loop {
        print!("\nВ: ");
        let _ = io::stdout().flush();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let q = line.trim();
        if q.is_empty() || q == "quit" || q == "exit" {
            break;
        }
        let prompt = format!("В: {q}\nО: ");
        generate(t, a, &prompt, "О: ", &stops);
    }
    eprintln!("пока!");
}

// ------------------------- CLI -------------------------

struct Args {
    model: String,
    prompt: String,
    tokens: usize,
    temp: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
    chat: bool,
    debug: bool,
    bench: bool,
    stops: Vec<String>,
    no_stop: bool,
}

impl Args {
    fn parse() -> Self {
        let mut a = Args {
            model: String::new(),
            prompt: "В: ".to_string(),
            tokens: 300,
            temp: 0.8,
            top_k: 0,
            top_p: 0.9,
            seed: 0,
            chat: false,
            debug: false,
            bench: false,
            stops: vec![],
            no_stop: false,
        };
        let argv: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        let next = |i: &mut usize, name: &str| -> String {
            *i += 1;
            argv.get(*i).cloned().unwrap_or_else(|| panic!("нет значения у {name}"))
        };
        while i < argv.len() {
            let arg = argv[i].as_str();
            match arg {
                "--prompt" => a.prompt = next(&mut i, arg),
                "--tokens" => a.tokens = next(&mut i, arg).parse().expect("--tokens: число"),
                "--temp" => a.temp = next(&mut i, arg).parse().expect("--temp: число"),
                "--top-k" => a.top_k = next(&mut i, arg).parse().expect("--top-k: число"),
                "--top-p" => a.top_p = next(&mut i, arg).parse().expect("--top-p: число"),
                "--seed" => a.seed = next(&mut i, arg).parse().expect("--seed: число"),
                "--stop" => a.stops.push(next(&mut i, arg)),
                "--chat" => a.chat = true,
                "--debug" => a.debug = true,
                "--bench" => a.bench = true,
                "--no-stop" => a.no_stop = true,
                "-h" | "--help" => {
                    println!(
                        "tiny_llm model.bin [опции]\n\
                         --prompt S    стартовый текст (по умолчанию \"В: \")\n\
                         --tokens N    максимум токенов (300)\n\
                         --temp F      температура, 0 = greedy (0.8)\n\
                         --top-k N     top-k, 0 = выкл\n\
                         --top-p F     nucleus, 1.0 = выкл (0.9)\n\
                         --seed N      0 = случайный\n\
                         --stop S      стоп-строка (можно несколько)\n\
                         --no-stop     не останавливаться\n\
                         --chat        интерактивный режим\n\
                         --debug       top-5 логитов первого шага\n\
                         --bench       без вывода текста, только скорость"
                    );
                    std::process::exit(0);
                }
                other if other.starts_with("--") => panic!("неизвестный флаг {other}"),
                other => a.model = other.to_string(),
            }
            i += 1;
        }
        if a.model.is_empty() {
            eprintln!("использование: tiny_llm model.bin [--prompt ...] [--chat]");
            std::process::exit(1);
        }
        if a.stops.is_empty() && !a.no_stop {
            a.stops = vec!["\n\n".to_string(), "\nВ:".to_string()];
        }
        if a.no_stop {
            a.stops.clear();
        }
        a
    }
}

fn main() {
    let a = Args::parse();
    let mut t = Transformer::load(&a.model);
    eprintln!(
        "модель: vocab={} dim={} layers={} heads={} ctx={} quant={}",
        t.cfg.vocab, t.cfg.dim, t.cfg.layers, t.cfg.heads, t.cfg.seq, t.cfg.quant
    );
    if a.chat {
        chat(&mut t, &a);
    } else {
        let stops = a.stops.clone();
        let prompt = a.prompt.clone();
        generate(&mut t, &a, &prompt, &prompt, &stops);
    }
}
