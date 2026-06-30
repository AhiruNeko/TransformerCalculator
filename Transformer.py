from Embedding import *


class TransformerLayer(nn.Module):
    def __init__(self, d_model, ffn_dim, num_heads, dropout=0.1, max_len=192):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(dim=self.head_dim, max_len=max_len)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        norm_x = self.norm1(x)
        batch_size, seq_len, d_model = norm_x.size()

        q = self.q_proj(norm_x)
        k = self.k_proj(norm_x)
        v = self.v_proj(norm_x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        q = q.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        k = k.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        attn_out, _ = self.attn(query=q, key=k, value=v, attn_mask=mask)
        x = x + self.dropout1(attn_out)

        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        out = x + self.dropout2(ffn_out)
        out = out.contiguous()
        return out


class Transformer(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, ffn_dim, num_heads, dropout=0.1, num_layers=3):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, ffn_dim, num_heads, dropout, max_len=max_len)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        device = x.device
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

        out = self.embedding(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.norm(out)
        out = self.output(out)
        out = out.contiguous()
        return out