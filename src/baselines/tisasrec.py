from types import SimpleNamespace
import torch.nn as nn
import torch
import math
import torch.nn.functional as F


args = SimpleNamespace(
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    hidden_units=64,
    dropout_rate=0.20,
    maxlen=100,
    time_span=256,
    num_heads=2,
    num_blocks=2
)


class PWFFN(nn.Module):
    def __init__(self, hidden_units: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden_units, hidden_units, 1),  # 1×1
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Conv1d(hidden_units, hidden_units, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x):  # [B, L, C]
        out = self.net(x.transpose(1, 2)).transpose(1, 2)
        return out + x  # residual


# ---------- time-aware multi-head attention ------------------
class TimeAwareMHA(nn.Module):
    def __init__(self, d_model, n_heads, dropout, max_len, time_bins, device):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.scale = math.sqrt(self.dk)

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

        # embeddings
        self.abs_pos_K = nn.Embedding(max_len, d_model)
        self.abs_pos_V = nn.Embedding(max_len, d_model)
        self.int_K = nn.Embedding(time_bins + 1, d_model)
        self.int_V = nn.Embedding(time_bins + 1, d_model)

        self.dropout = nn.Dropout(dropout)
        self.device = device

    # helper: split & concat heads
    def _split(self, x):  # [B, L, C] -> [B*h, L, dk]
        B, L, _ = x.size()
        return x.view(B, L, self.h, self.dk).transpose(1, 2).reshape(B * self.h, L, self.dk)

    def _merge(self, x, B):  # [B*h, L, dk] -> [B, L, C]
        _, L, _ = x.size()
        return x.view(B, self.h, L, self.dk).transpose(1, 2).reshape(B, L, self.h * self.dk)

    def forward(self, seq, timemat, pad_mask):
        """
        seq      : [B, L, d]     (item embeddings + residuals)
        timemat  : [B, L, L]     (bucketed interval indices, 0..T)
        pad_mask : [B, L]  True where PAD
        returns  : [B, L, d]
        """
        B, L, _ = seq.size()
        Q, K, V = self.Wq(seq), self.Wk(seq), self.Wv(seq)

        # add absolute position emb to K/V
        pos_ids = torch.arange(L, device=self.device)
        abs_K = self.abs_pos_K(pos_ids)  # [L,d]
        abs_V = self.abs_pos_V(pos_ids)

        # add relative-time emb to K/V
        int_K = self.int_K(timemat)  # [B,L,L,d]
        int_V = self.int_V(timemat)

        # split heads
        q = self._split(Q)
        k = self._split(K + abs_K)
        v = self._split(V + abs_V)

        int_K_ = int_K.view(B, L, L, self.h, self.dk)
        int_K_ = int_K_.permute(0, 3, 1, 2, 4)
        int_K_ = int_K_.reshape(B * self.h, L, L, self.dk)
        k = k + int_K_[:, :, 0, :]

        int_V_ = int_V.view(B, L, L, self.h, self.dk)
        int_V_ = int_V_.permute(0, 3, 1, 2, 4)
        int_V_ = int_V_.reshape(B * self.h, L, L, self.dk)
        v = v + int_V_[:, :, 0, :]

        # scaled dot-product attention
        attn = torch.bmm(q, k.transpose(1, 2)) / self.scale

        # causal + padding masks
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=self.device), 1)
        attn.masked_fill_(causal, -1e9)
        pad = pad_mask.repeat_interleave(self.h, dim=0).unsqueeze(1)
        attn.masked_fill_(pad, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.bmm(attn, v)
        out = self._merge(out, B)
        return self.Wo(out)


# ---------- TiSASRec block -----------------------------------
class TiSASRecBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout, max_len, time_bins, device):
        super().__init__()
        self.mha = TimeAwareMHA(d_model, n_heads, dropout, max_len, time_bins, device)
        self.ln1 = nn.LayerNorm(d_model, eps=1e-8)
        self.ffn = PWFFN(d_model, dropout)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-8)
        self.dro = nn.Dropout(dropout)

    def forward(self, x, timemat, pad_mask):
        h = self.mha(self.ln1(x), timemat, pad_mask)
        x = x + self.dro(h)
        h = self.ffn(self.ln2(x))
        return x + self.dro(h)


# ---------- TiSASRec top-level model -------------------------
class TiSASRec(nn.Module):
    def __init__(self, item_num, args):  # args carries hyper-params
        super().__init__()
        self.pad_idx = 0
        self.max_len = args.maxlen
        self.time_bins = args.time_span
        d = args.hidden_units

        self.item_emb = nn.Embedding(item_num + 1, d, padding_idx=self.pad_idx)
        self.emb_drop = nn.Dropout(args.dropout_rate)

        self.blocks = nn.ModuleList([
            TiSASRecBlock(d, args.num_heads, args.dropout_rate,
                          args.maxlen, args.time_span, args.device)
            for _ in range(args.num_blocks)
        ])
        self.last_ln = nn.LayerNorm(d, eps=1e-8)
        self.out_bias = nn.Embedding(item_num + 1, 1, padding_idx=self.pad_idx)

    # ---- helper: build pair-wise bucketed intervals -----------
    def build_timemat(self, ts):
        # ts : [B,L] absolute timestamps (0 for PAD)
        diff = (ts.unsqueeze(2) - ts.unsqueeze(1)).abs()  # [B,L,L]
        buckets = torch.clamp((diff.float().log1p()).floor().long(),
                              0, self.time_bins)
        return buckets

    # ---- forward for next-item logits -------------------------
    def forward(self, seq_items, seq_times):
        """
        seq_items : [B, L] left-padded item ids  (PAD=0)
        seq_times : [B, L] aligned timestamps    (0 for PAD)
        returns   : logits [B, item_num+1]
        """
        pad_mask = seq_items.eq(self.pad_idx)  # [B,L] True where PAD
        timemat = self.build_timemat(seq_times)  # [B,L,L]

        x = self.item_emb(seq_items) * (self.item_emb.embedding_dim ** 0.5)
        x = self.emb_drop(x)

        for blk in self.blocks:
            x = blk(x, timemat, pad_mask)

        x = self.last_ln(x)

        user_vec = x[:, -1, :]  # last position
        logits = user_vec @ self.item_emb.weight.T + self.out_bias.weight.squeeze(-1)
        logits[:, self.pad_idx] = -1e9
        return logits
