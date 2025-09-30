# MultiApp: Foundation Model for Learning Generalized User Behavior Patterns Across Multiple Mobile Apps

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)](https://pytorch.org/)

## Quick Start

### 1. Get the Data
Download pre-trained MovieLens-1M embeddings from [v0.1-data release](https://github.com/ZivMor1/MultiApp/releases/tag/train_val_test_dfs):
- `x_train_embeddings.pt`, `x_val_embeddings.pt`, `x_test_embeddings.pt` (512-dim embeddings)
- `y_train.pt`, `y_val.pt`, `y_test.pt` (target labels)

### 2. Run the Notebooks
- **`src/notebooks/train_mlp_movielens.ipynb`**: Train recommendation models with pre-trained embeddings
- **`src/notebooks/movielens_preprocessing_notebook.ipynb`**: Process your own data

### 3. What You Get
- **Pre-trained embeddings** extracted from MultiApp foundation model (1,300+ apps, 1.5B tokens)
- **Complete baseline implementations**: TiSASRec, DuoRec, TBiLSTM
- **Fast convergence**: ~10× faster than training from scratch

## Architecture

### Core Components

- **MultiAppBehaviorTokenizer**: Custom tokenizer for behavior sequences with special tokens
- **TemporalEmbedding**: Time-aware positional encodings for user journey context
- **MultiAppDataset**: Dynamic masked language modeling dataset for pre-training
- **MLP Architectures**: Task-specific heads for downstream applications

### Baseline Implementations

- **TBiLSTM**: Time-aware bidirectional LSTM baseline
- **TiSASRec**: Time-aware self-attention for sequential recommendation
- **DuoRec**: Transformer-based sequential recommendation with contrastive learning loss

## Repository Contents

### Core Implementation (`src/multiapp/`)
- **`tokenizer.py`**: Custom MultiAppBehaviorTokenizer for behavior sequences
- **`temporal_embeddings.py`**: Time-aware processing and positional encodings
- **`dynamic_mlm_dataset_class.py`**: MultiAppDataset for masked language modeling
- **`mlp_architecture.py`**: MLP models for next-item prediction

### Baseline Implementations (`src/baselines/`)
- **`duorec.py`**: Complete DuoRec implementation (RecSys'22)
- **`tisasrec.py`**: Complete TiSASRec implementation with time-aware attention
- **`tbiltsm_baseline.py`**: TBiLSTM baseline implementation

### Utilities (`src/utils/`)
- **`evaluation_functions.py`**: Hit@K, NDCG@K evaluation metrics
- **`training_functions.py`**: Training loops and model evaluation
- **`general.py`**: Model configuration and reproducibility utilities
- **`preprocessing_seq.py`**: Sequence preprocessing helpers

### Ready-to-Use Notebooks (`src/notebooks/`)
- **`movielens_preprocessing_notebook.ipynb`**: Complete data preprocessing pipeline
- **`train_mlp_movielens.ipynb`**: End-to-end training example with pre-trained embeddings

### Shared Data (Download Required)
- **Pre-trained embeddings**: MovieLens-1M embeddings extracted from MultiApp model
- **Train/Val/Test splits**: Ready-to-use data splits for immediate experimentation
- **Vocabulary files**: Token mappings and metadata

## Installation

```bash
git clone https://github.com/ZivMor1/MultiApp.git
cd MultiApp
pip install -r requirements.txt
```

**Requirements**: Python 3.8+, PyTorch 2.8.0, Transformers 4.53.0, NumPy, Pandas, tqdm

### Basic Usage (Advanced)

```python
from src.multiapp.tokenizer import MultiAppBehaviorTokenizer
from src.multiapp.temporal_embeddings import TemporalEmbedding
from src.multiapp.mlp_architecture import MLPNextItemPrediction

# Initialize tokenizer
vocab = {"action_1": 0, "action_2": 1, ...}  # Your app vocabulary
tokenizer = MultiAppBehaviorTokenizer(vocab=vocab)

# Create temporal embeddings
temporal_emb = TemporalEmbedding(model_config)

# Initialize recommendation model
model = MLPNextItemPrediction(recs_size=len(vocab))
```

## Data & Notebooks

### Pre-trained Embeddings
**Download**: [v0.1-data release](https://github.com/ZivMor1/MultiApp/releases/tag/train_val_test_dfs)

**Files:**
- `x_train_embeddings.pt` - Train Representations extracted from MultiApp 
- `x_val_embeddings.pt` 
- `x_test_embeddings.pt` 
- `y_train.pt`, `y_val.pt`, `y_test.pt` (targets)


### Ready-to-Use Notebooks
- **`train_mlp_movielens.ipynb`**: Train models with pre-trained embeddings (start here!)
- **`movielens_preprocessing_notebook.ipynb`**: Process your own data

## Usage Examples

### Quick Start with Pre-trained Data
1. Download embeddings from [v0.1-data release](https://github.com/ZivMor1/MultiApp/releases/tag/train_val_test_dfs)
2. Run `src/notebooks/train_mlp_movielens.ipynb`
3. Get Hit@10, NDCG@10 results in minutes

### Train Baselines
- **TiSASRec**: `src/baselines/tisasrec.py`
- **DuoRec**: `src/baselines/duorec.py` 
- **TBiLSTM**: `src/baselines/tbiltsm_baseline.py`

### Advanced Usage (Code-based)

#### Pre-training
Train the foundation model on your app data:

```python
from src.multiapp.dynamic_mlm_dataset_class import MultiAppDataset

# Create dataset
dataset = MultiAppDataset(
    pretrained_tokenizer=tokenizer,
    event_ids_seq_files_folder_path="path/to/event_ids",
    times_seq_files_folder_path="path/to/times",
    padding_mask_files_folder_path="path/to/masks"
)
```

#### Fine-tuning
Fine-tune on downstream tasks:

```python
from src.utils.training_functions import train_model

# Train recommendation model
best_val = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=30,
    learning_rate=3e-4
)
```

## Evaluation

The repository includes comprehensive evaluation metrics:

- **Hit@K**: Hit rate at top-K recommendations
- **NDCG@K**: Normalized Discounted Cumulative Gain

## Supported Tasks

1. **Next-Item Prediction**: Sequential recommendation (main focus)

## Paper

**"Foundation Model for Learning Generalized User Behavior Patterns Across Multiple Mobile Apps"**

## Limitations

1. **Privacy Constraints**: MultiApp model cannot be shared publicly due to privacy constraints
2. **Pre-trained Embeddings**: Only extracted embeddings are available for fast convergence
3. **Baseline Comparisons**: Focused on available implementations with reproducible pipelines

## Contact

For questions or issues, please contact: **zivmord@post.bgu.ac.il**