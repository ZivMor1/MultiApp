import random
import numpy as np
import torch
import os



class ModelConfig:
    def __init__(self, vocab_size, max_time_within_model=10080, model_dim=512):
        self.vocab_size = vocab_size
        self.max_time_within_model = max_time_within_model
        self.model_dim = model_dim



def set_seed(seed: int = 42):
    """Full reproducibility for PyTorch + CUDA """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # cuDNN / cuBLAS deterministic paths
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # <-- add this line

    # Optional: disable TF-32 for bit-exact runs
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
