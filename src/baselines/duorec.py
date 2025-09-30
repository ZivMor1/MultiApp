import torch
import torch.nn as nn
import torch.nn.functional as F


class DuoRecConfig:
    def __init__(self,
                 vocab_size: int,
                 max_len: int = 50,
                 d_model: int = 64,
                 n_heads: int = 2,
                 n_layers: int = 2,
                 dropout: float = 0.2,
                 temperature: float = 0.6
                 ):
        assert d_model % n_heads == 0
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.temperature = temperature


class DuoRec(nn.Module):
    """
  DuoRec (RecSys'22):
  - Transformer encoder to get sequence-level rep (last non-pad token)
  - Next-item softmax over whole item set using tied item embeddings
  - Contrastive regularizer:
  anchor = Dropout-augmented encoding of sequence A
  positive = Dropout-augmented encoding of sequence B with same next-item target as A
  negatives = other sequences in batch, excluding any that share the same target
  symmetric InfoNCE with temperature τ
  - Total loss: L = CE + λ * CL
  """

    def __init__(self, cfg: DuoRecConfig):
        super().__init__()
        self.cfg = cfg

        # Item embedding and learned positional embedding
        self.item_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        self.emb_dropout = nn.Dropout(cfg.dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.d_model,
            dropout=cfg.dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)

        # Small projection head for contrastive learning
        self.proj_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.d_model, cfg.d_model)
        )

        # Tie output projection to item embeddings for ŷ = softmax(V h)
        self.tie_weights = True

    def _seq_repr(self, seq_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
      Compute sequence representation as the last non-pad token’s hidden state.
      seq_ids: [B, L] LongTensor
      pad_mask: [B, L] BoolTensor (True where PAD)
      """
        B, L = seq_ids.size()
        pos_ids = torch.arange(L, device=seq_ids.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seq_ids) + self.pos_emb(pos_ids)
        x = self.emb_dropout(x)  # Dropout-based augmentation happens here (train mode)

        # src_key_padding_mask: True for pads
        h = self.encoder(x, src_key_padding_mask=pad_mask)  # [B, L, d]

        # last non-pad index per sample
        last_idx = (~pad_mask).sum(dim=1) - 1  # [B]
        rep = h[torch.arange(B, device=h.device), last_idx]  # [B, d]
        return rep

    def forward(self, seq_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
      Next-item logits over whole item set.
      """
        rep = self._seq_repr(seq_ids, pad_mask)  # [B, d]
        if self.tie_weights:
            logits = rep @ self.item_emb.weight.T  # [B, V]
        else:
            # Optional separate output layer if you prefer not to tie
            if not hasattr(self, "out_proj"):
                self.out_proj = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)
            logits = self.out_proj(rep)
        return logits

    def contrastive_loss(self,
                         anc_seq: torch.Tensor, anc_mask: torch.Tensor,
                         pos_seq: torch.Tensor, pos_mask: torch.Tensor,
                         targets: torch.Tensor) -> torch.Tensor:
        """
      DuoRec contrastive regularizer (symmetric InfoNCE) with:
        anchor = Dropout-augmented encoding of anc_seq
        positive = Dropout-augmented encoding of pos_seq (same next-item targets)
        negatives = in-batch others, excluding any same-target entries (except the positive)
      targets: [B] LongTensor of next-item ids for anchors (same as for positives by construction)
      """
        # Encode two views (Dropout creates stochasticity)
        za = F.normalize(self.proj_head(self._seq_repr(anc_seq, anc_mask)), dim=-1)  # [B, d]
        zp = F.normalize(self.proj_head(self._seq_repr(pos_seq, pos_mask)), dim=-1)  # [B, d]

        tau = float(self.cfg.temperature)
        logits = za @ zp.t() / tau  # [B, B]
        B = logits.size(0)
        labels = torch.arange(B, device=logits.device)  # positives are on the diagonal

        # Build mask to exclude false negatives: any pair with the same target item
        same_tgt = targets.unsqueeze(1).eq(targets.unsqueeze(0))  # [B, B]
        same_tgt.fill_diagonal_(False)  # keep the diagonal (true positive)
        # Valid entries in denominator = not same-target or the diagonal
        keep = ~same_tgt
        keep[torch.arange(B), torch.arange(B)] = True
        logits1 = logits.masked_fill(~keep, float("-inf"))
        loss1 = F.cross_entropy(logits1, labels)

        # Symmetric term (swap roles)
        logits2 = (zp @ za.t()) / tau
        logits2 = logits2.masked_fill(~keep, float("-inf"))
        loss2 = F.cross_entropy(logits2, labels)

        return 0.5 * (loss1 + loss2)
