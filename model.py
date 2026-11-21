# model.py — минимальный decoder-only трансформер (без bias, RMSNorm, SiLU)
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int = 0
    block_size: int = 128      # длина контекста в символах
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    hidden: int = 0            # 0 -> 4 * n_embd
    norm_eps: float = 1e-5

    def __post_init__(self):
        if self.hidden == 0:
            self.hidden = 4 * self.n_embd
        assert self.n_embd % self.n_head == 0


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_size = cfg.n_embd // cfg.n_head
        self.wq = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.wk = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.wv = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.wo = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.register_buffer("mask",
                             torch.tril(torch.ones(cfg.block_size, cfg.block_size))
                             .view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (self.head_size ** 0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.w1 = nn.Linear(cfg.n_embd, cfg.hidden, bias=False)
        self.w2 = nn.Linear(cfg.hidden, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.rms_att = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.rms_ffn = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.rms_att(x))
        x = x + self.mlp(self.rms_ffn(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.rms_final = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.rms_final(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())