from __future__ import annotations
import torch
from typing import Tuple


# ───────────────────────── metrics ──────────────────────────
def hit_at_k(logits: torch.Tensor,
             targets: torch.Tensor,
             k: int = 10) -> float:
    """
    Hit@k (a.k.a. Recall@k, HR@k) for a *single* batch.
    `logits`: [B, C]  (`C` = full catalogue or sampled pool)
    `targets`: [B]    (index of the positive within that second dim)
    """
    topk = torch.topk(logits, k=min(k, logits.size(1)), dim=1).indices
    hits = (topk == targets.unsqueeze(1)).any(dim=1).float()
    return hits.mean().item()


def ndcg_at_k(logits: torch.Tensor,
              targets: torch.Tensor,
              k: int = 10) -> float:
    """
    NDCG@k for a *single* batch.  Same expectations as `hit_at_k`.
    """
    ranks = logits.argsort(dim=1, descending=True)  # [B,C]
    pos = (ranks == targets.unsqueeze(1)).nonzero(as_tuple=True)[1]
    k_eff = min(k, logits.size(1))
    dcg = 1.0 / torch.log2(pos.float() + 2.0)  # rank starts at 0
    dcg[pos >= k_eff] = 0.0
    return dcg.mean().item()


# ──────────────────────── evaluation ────────────────────────
@torch.no_grad()
def evaluate_recommender(model: torch.nn.Module,
                         data_loader: torch.utils.data.DataLoader,
                         device: torch.device,
                         k: int = 10,
                         loss_fn: torch.nn.Module | None = None,
                         n_sampling_eval: int | None = None
                         ) -> Tuple[float, float, float]:
    """
    Evaluate all batches and return (avg_loss, HR@k, NDCG@k).
    """
    model.eval()

    total_loss, total_examples = 0.0, 0
    hit_sum, ndcg_sum = 0.0, 0.0

    # for inputs, mask, targets in data_loader: # todo erase
    #     inputs, mask = inputs.to(device), mask.to(device)

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits_full = model(inputs)  # [B, V]
        B, V = logits_full.shape

        # --------------- optional negative sampling ---------------
        if n_sampling_eval is not None:
            n_neg = min(n_sampling_eval, V - 1)
            # sample negatives uniformly *excluding* the target id
            # trick: sample in [0,V-2] then shift indices ≥ target by +1
            neg = torch.randint(0, V - 1, (B, n_neg), device=device)
            neg[neg >= targets.unsqueeze(1)] += 1  # shift
            cand_idx = torch.cat([targets.unsqueeze(1), neg], dim=1)  # [B,1+n]
            logits = logits_full.gather(1, cand_idx)  # [B,1+n]
            new_tgt = torch.zeros(B, dtype=torch.long, device=device)
        else:
            logits, new_tgt = logits_full, targets  # [B,V], [B]

        # --------------------- loss -------------------------------
        if loss_fn is not None:
            loss = loss_fn(logits, new_tgt)
            total_loss += loss.item() * inputs.size(0)

        # ------------------- metrics ------------------------------
        hit_k = hit_at_k(logits, new_tgt, k)
        ndcg_k = ndcg_at_k(logits, new_tgt, k)

        batch_sz = inputs.size(0)
        hit_sum += hit_k * batch_sz
        ndcg_sum += ndcg_k * batch_sz
        total_examples += batch_sz

    avg_loss = (total_loss / total_examples) if loss_fn is not None else -1.0
    hit_k = hit_sum / total_examples
    ndcg_k = ndcg_sum / total_examples
    return avg_loss, hit_k, ndcg_k
