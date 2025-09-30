import json
import os
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import WordLevel


class MultiAppBehaviorTokenizer(PreTrainedTokenizerFast):
    def __init__(self, vocab: dict, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                 sep_token="[SEP]", mask_token="[MASK]", model_max_length=100, **kwargs):
        """
      vocab: dictionary file, {word_value_i : token_number_i}
      """
        # Ensure special tokens are in the vocabulary
        special_tokens = {
            unk_token: len(vocab),
            pad_token: len(vocab) + 1,
            cls_token: len(vocab) + 2,
            sep_token: len(vocab) + 3,
            mask_token: len(vocab) + 4,
        }

        for token, idx in special_tokens.items():
            if token not in vocab:
                vocab[token] = idx

        # Initialize the WordLevel tokenizer with the updated vocabulary
        tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=unk_token))

        # Use a pre-tokenizer to split the text into words based on whitespace
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        # Wrap the tokenizer in a PreTrainedTokenizerFast object
        super().__init__(tokenizer_object=tokenizer, clean_up_tokenization_spaces=False,
                         model_max_length=model_max_length, **kwargs)

        # Add special tokens
        self.add_special_tokens({
            'unk_token': unk_token,
            'pad_token': pad_token,
            'cls_token': cls_token,
            'sep_token': sep_token,
            'mask_token': mask_token
        })

    def save_pretrained(self, save_directory):
        """Save the tokenizer including the vocabulary."""
        # Save the tokenizer files (this will save the vocab inside)
        super().save_pretrained(save_directory)

        # Save the vocabulary as a separate file
        vocab_file = os.path.join(save_directory, 'vocab.json')
        with open(vocab_file, 'w') as f:
            json.dump(self.get_vocab(), f)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Load the tokenizer with the correct vocab."""
        # Load the vocabulary from the vocab.json file
        vocab_file = os.path.join(pretrained_model_name_or_path, 'vocab.json')
        with open(vocab_file, 'r') as f:
            vocab = json.load(f)

        # Pass the loaded vocab to the __init__ method
        return cls(vocab=vocab, *args, **kwargs)

    def get_id_to_token_vocab(self):
        dic = {token_id: token for token, token_id in self.vocab.items()}
        return dic
