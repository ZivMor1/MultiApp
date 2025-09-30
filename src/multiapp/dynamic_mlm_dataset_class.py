import random
import torch
from torch.utils.data import Dataset
import os
from src.multiapp.tokenizer import MultiAppBehaviorTokenizer


class MultiAppDataset(Dataset):
    def __init__(self, pretrained_tokenizer: MultiAppBehaviorTokenizer,
                 event_ids_seq_files_folder_path: str, times_seq_files_folder_path: str,
                 padding_mask_files_folder_path: str, check_accurate_length_of_data=False,
                 is_validation=False, validation_seed: int = 42,
                 masking_prob=0.15, device=None):
        """
    Initializes the MultiAppDataset.

    Args:
        pretrained_tokenizer: Tokenizer for encoding sequences.
        event_ids_seq_files_folder_path: Path to folder with event ID files.
        times_seq_files_folder_path: Path to folder with time sequence files.
        padding_mask_files_folder_path: Path to folder with attention mask files.
        check_accurate_length_of_data: If True, checks the accurate length of data.
        apply_dynamic_masking: If True, applies dynamic masking to event IDs.
    """
        self.validation_seed = validation_seed
        self.is_validation = is_validation
        self.tokenizer = pretrained_tokenizer
        self.masking_prob = masking_prob

        # specific files paths
        self.event_ids_paths = self.get_all_files_paths_from_folder_path(event_ids_seq_files_folder_path)
        self.times_seq_paths = self.get_all_files_paths_from_folder_path(times_seq_files_folder_path)
        self.padding_mask_paths = self.get_all_files_paths_from_folder_path(padding_mask_files_folder_path)

        self.idx_in_file = 0
        self.current_file_idx = 0

        # loading first files
        self._load_current_file_data()

        if check_accurate_length_of_data:
            self.total_len = self._get_total_len()
        else:
            # heuristically approach
            self.total_len = self.current_file_event_ids.shape[0] * len(self.event_ids_paths)

        self.device = device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_all_files_paths_from_folder_path(data_folder_path: str, file_endswith='.pt',
                                             sort_files_by_index=True):
        files_paths = [os.path.join(data_folder_path, file_name)
                       for file_name in os.listdir(data_folder_path) if file_name.endswith(file_endswith)]

        if sort_files_by_index:
            files_paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        return files_paths

    def _get_total_len(self):
        print("""Compute the total length across all event id files...""")
        total_length = 0
        for file_path in self.event_ids_paths:
            # Load the file and get its shape, then release memory
            temp_tensor = torch.load(file_path, weights_only=False)
            total_length += temp_tensor.shape[0]
            del temp_tensor  # Explicitly delete to free memory
        return total_length

    def __len__(self):
        return self.total_len

    def _load_current_file_data(self):
        """Load the current file data into memory."""
        self.current_file_event_ids = torch.load(self.event_ids_paths[self.current_file_idx], weights_only=False)
        self.current_file_times_seq = torch.load(self.times_seq_paths[self.current_file_idx], weights_only=False)
        self.current_file_padding_mask = torch.load(self.padding_mask_paths[self.current_file_idx], weights_only=False)

    def apply_dynamic_masking_to_event_ids(self, event_ids):
        """
    Applies dynamic masking to tokenized sentences for MLM training.

    input_ids: Tensor of shape [batch_size, sequence_length] (tokenized sentences)
    tokenizer: Hugging Face tokenizer (must be a fast tokenizer with pre-tokenization)

    Returns:
    - masked_input_ids: Tensor of shape [batch_size, sequence_length] with masked tokens
    - labels: Tensor of shape [batch_size, sequence_length] where masked tokens are their original value, and unmasked are -100
    """
        mask_token_id = self.tokenizer.mask_token_id
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        pad_token_id = self.tokenizer.pad_token_id

        labels = event_ids.clone()  # Clone input_ids to create labels
        masked_input_ids = event_ids.clone()  # Clone input_ids to create the masked version

        # Set a deterministic seed for validation, no seed for training
        if self.is_validation:
            torch.manual_seed(self.validation_seed)
            random.seed(self.validation_seed)
        else:
            #refreshing seed for the next dynamic masking
            torch.manual_seed(random.randint(0, 1000))
            random.seed(random.randint(0, 1000))

        # Generate a mask for which tokens to apply masking
        probability_matrix = torch.full(event_ids.shape, self.masking_prob)
        special_tokens_mask = (event_ids == cls_token_id) | (event_ids == sep_token_id) | (event_ids == pad_token_id)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)  # Don't mask special tokens
        masked_indices = torch.bernoulli(probability_matrix).bool()  # Create mask for each token

        # Apply [MASK] token to selected indices
        masked_input_ids[masked_indices] = mask_token_id

        # The labels are -100 for unmasked tokens (so they don't contribute to the loss)
        labels[~masked_indices] = -100

        return masked_input_ids, labels

    def __getitem__(self, idx):
        if self.idx_in_file >= len(self.current_file_event_ids):
            # Move to next file
            self.current_file_idx += 1
            if self.current_file_idx >= len(
                    self.event_ids_paths):  # If all files have been iterated (AKA one epoch), start from the beginning
                self.current_file_idx = 0
                self.idx_in_file = 0

            self._load_current_file_data()
            self.idx_in_file = 0

        batch_event_ids = self.current_file_event_ids[self.idx_in_file]
        batch_times_seq = self.current_file_times_seq[self.idx_in_file]
        batch_attention_mask = self.current_file_padding_mask[self.idx_in_file]

        # apply dynamic masking
        batch_event_ids, labels = self.apply_dynamic_masking_to_event_ids(batch_event_ids)

        self.idx_in_file += 1
        return batch_event_ids, labels.long(), batch_times_seq, batch_attention_mask
