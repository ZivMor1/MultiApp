from tqdm import tqdm
import torch
import torch.nn as nn
import math
from typing import List
from src.utils.general import ModelConfig


class TimeSeqPreprocessor:
    def __init__(self, max_time, max_length_of_sequence, time_pad_token="[TIME_PAD]",
                 padding_side="left", return_attention_mask: bool = False):

        self.max_time = max_time
        self.time_pad_id = max_time + 1
        self.time_pad_token = time_pad_token
        self.max_length_of_sequence = max_length_of_sequence
        self.padding_side = padding_side
        self.min_time = 0
        self.return_attention_mask = return_attention_mask

    def _clamp(self, seq: List[int]) -> List[int]:
        # clamp times to [min_time, max_time]
        return [min(self.max_time, max(self.min_time, int(t))) for t in seq]

    def _process_sequence(self, seq: list[int]):
        seq = self._clamp(seq)
        L = len(seq)
        M = self.max_length_of_sequence

        if L >= M:
            # truncation
            if self.padding_side == "left":
                seq = seq[-M:]
            else:  # right
                seq = seq[:M]
            mask = [1] * M
            return seq, mask

        # padding
        pad_len = M - L
        if self.padding_side == "left":
            padded = [self.time_pad_id] * pad_len + seq
            mask = [0] * pad_len + [1] * L
        else:  # right
            padded = seq + [self.time_pad_id] * pad_len
            mask = [1] * L + [0] * pad_len

        return padded, mask

    def __call__(self, times_seq_list: list[list[int]], return_dtype=torch.int32):
        seq_times = []
        masks = [] if self.return_attention_mask else None

        for seq in tqdm(times_seq_list, total=len(times_seq_list),
                        desc=f"Processing list of times with padding token id ({self.time_pad_id})"):
            padded, mask = self._process_sequence(seq)
            seq_times.append(padded)
            if self.return_attention_mask:
                masks.append(mask)

        times_tensor = torch.tensor(seq_times, dtype=return_dtype)
        if self.return_attention_mask:
            mask_tensor = torch.tensor(masks, dtype=torch.bool)
            return times_tensor, mask_tensor

        return times_tensor


class TemporalEmbedding(nn.Module):
    def __init__(self, model_config: ModelConfig, include_time_pad_token=True):
        super(TemporalEmbedding, self).__init__()
        self.max_time = model_config.max_time_within_model
        self.d_model = model_config.model_dim
        # Create the positional encoding matrix
        self.time_encoding = self._create_time_encoding(self.max_time + 1,
                                                        self.d_model)  #including 0 which is also a token

        self.time_pad_token_id = None
        if include_time_pad_token:
            # adding vector of zeros to the last position
            self.time_encoding = torch.cat((self.time_encoding, torch.zeros(1, 1, self.d_model)), dim=1)
            self.time_pad_token_id = self.time_encoding.shape[1] - 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.time_encoding = self.time_encoding.to(device)

    def _create_time_encoding(self, max_time, d_model):
        position = torch.arange(max_time, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe = torch.zeros(max_time, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices

        return pe.unsqueeze(0)  # Add batch dimension

    def forward(self, accumulated_times):
        return self.time_encoding[0, accumulated_times]
