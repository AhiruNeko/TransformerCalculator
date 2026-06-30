import torch
import torch.nn as nn

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        out = self.token_emb(x)
        return self.dropout(out)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=196, base=10000):
        super().__init__()
        self.dim = dim

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)

    def _rotate_half(self, x):
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
        seq_len = x.size(2)
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        return (x * cos) + (self._rotate_half(x) * sin)