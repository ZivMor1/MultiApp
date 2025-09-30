from src.utils.temporal_embeddings import TimeSeqPreprocessor
import gc
import torch
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
import pandas as pd


def replace_last_token_with_sep_token_if_needed(input_ids: torch.Tensor, hf_tokenizer: PreTrainedTokenizerFast):
    last_tokens = input_ids[:, -1]  # Get the last token for all sequences
    masked_longer_sentences = last_tokens != hf_tokenizer.pad_token_id
    input_ids[masked_longer_sentences, -1] = hf_tokenizer.sep_token_id
    return input_ids


def modify_sentences_with_cls_sep_tokens(sentences_list, cls_token='[CLS]', sep_token='[SEP]'):
    sentences_modified = [
        f"{cls_token} {' '.join(sentence_list)} {sep_token}"
        for sentence_list in tqdm(sentences_list, desc="Modifying sentences with CLS and SEP tokens")
    ]
    return sentences_modified


def apply_tokenizer_with_cls_and_sep_tokens(sentences_list: list[str], hf_tokenizer: PreTrainedTokenizerFast,
                                            tokenizing_args_dict: dict):
    print(f'Got {len(sentences_list)} sentences to tokenize.')

    sentences_modified = modify_sentences_with_cls_sep_tokens(sentences_list=sentences_list)

    print(f'Tokenizing flows by tokenizer')
    encoding = hf_tokenizer(sentences_modified, **tokenizing_args_dict)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    input_ids = replace_last_token_with_sep_token_if_needed(input_ids=input_ids, hf_tokenizer=hf_tokenizer)
    res = {'input_ids': input_ids, 'attention_mask': attention_mask}

    del sentences_modified
    gc.collect()

    return res


def convert_time_and_actions_sequences_to_tokens(actions_seq, times_seq, tokenizer, padding_side="right",
                                                 tokenizer_add_special_tokens=False):
    tokenizer.padding_side = padding_side
    if not tokenizer_add_special_tokens:
        tokenizer.add_special_tokens = False

    tokenizer_args = {"return_tensors": "pt", "truncation": True, "padding": "max_length", "max_length": 100}
    time_processor = TimeSeqPreprocessor(max_time=10080, max_length_of_sequence=100, padding_side=padding_side)
    flows_tokens = apply_tokenizer_with_cls_and_sep_tokens(sentences_list=actions_seq, hf_tokenizer=tokenizer,
                                                           tokenizing_args_dict=tokenizer_args)
    src_padding_masking = ~flows_tokens['attention_mask'].bool()
    time_encodings = time_processor(times_seq)
    return flows_tokens['input_ids'], time_encodings, src_padding_masking


def explode_to_windows(df: pd.DataFrame, min_hist: int = 3, keep_last_only: bool = False) -> pd.DataFrame:
    """
    Input df columns (per user row):
      - user_id
      - seq_actions : list[int]      (full sequence of movies)
      - seq_times   : list[float]|None  (aligned with seq_actions)  # optional
      - target      : int            (ignored; we recompute per-window)
      - seq_length  : int            (len(seq_actions))

    Output: new DataFrame with one row per (history -> next) window:
      - user_id
      - hist_actions : list[int]     (prefix of seq_actions)
      - hist_times   : list[float]|None (prefix of seq_times, if provided)
      - target       : int           (the next movie after hist)
      - hist_len     : int           (#items in hist_actions)
    """
    rows = []
    for _, r in df.iterrows():
        acts = r["seq_actions"]
        times = r.get("seq_times", None)
        T = int(r["sequence_length"])

        # need at least 'min_hist' history items + 1 target
        if T < min_hist + 1:
            continue

        # t is the history length; we predict acts[t]
        t_iter = range(min_hist, T) if not keep_last_only else range(T - 1, T)
        for t in t_iter:
            rows.append({
                "user_id": r["user_id"],
                "seq_actions": acts[:t],
                "seq_times": (times[:t] if times is not None else None),
                "target": acts[t],
                "hist_len": t,
            })
    return pd.DataFrame(rows)
