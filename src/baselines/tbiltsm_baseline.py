from src.utils.temporal_embeddings import TemporalEmbedding
from src.utils.general import ModelConfig
import torch
import torch.nn as nn


class TBiLSTMModel(nn.Module):
    def __init__(self, total_words, embedding_dim, emb_mat: torch.Tensor = None, lstm_units=50):
        super(TBiLSTMModel, self).__init__()

        # Embedding layer
        if emb_mat is None:
            self.embedding = nn.Embedding(total_words, embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(emb_mat, freeze=False)

        # Time embedding layer
        self.tbilstm_model_config = ModelConfig(vocab_size=total_words)
        self.time_embedding = TemporalEmbedding(model_config=self.tbilstm_model_config)

        # Bidirectional LSTM layer
        self.bilstm = nn.LSTM(embedding_dim, lstm_units, bidirectional=True, batch_first=True)

        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(2 * lstm_units)  # For LSTM output
        self.layer_norm2 = nn.LayerNorm(100)  # For dense layer 1

        # Dense layers
        self.dense1 = nn.Linear(2 * lstm_units, 100)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(100, 50)
        self.dense3 = nn.Linear(50, 1)

    def forward(self, x, times_seq):
        # Embedding layer: times and events
        tokens_embedded = self.embedding(x)
        time_embeddings = self.time_embedding(times_seq)

        # sum times before bilstm layer
        embedded = tokens_embedded + time_embeddings

        # biLSTM layer
        bilstm_output, _ = self.bilstm(embedded)

        # Layer Normalization
        normalized = self.layer_norm1(bilstm_output)

        # Sum over sequence length
        sum_over_time = torch.sum(normalized, dim=1)

        # Dense layers
        out = self.dense1(sum_over_time)
        out = self.relu(out)
        out = self.layer_norm2(out)
        out = self.dense2(out)
        out = self.relu(out)
        out = self.dense3(out)

        return out

    def set_pretrained_embedding(self, embedding_matrix, freeze_weights=False):
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embedding_matrix).float(), freeze=freeze_weights)

    def extend_event_embedding_matrix(self, embedding_extension, device, freeze_embeddings_layer=True):
        x = self.embedding.weight.clone().detach()
        x_ext = torch.from_numpy(embedding_extension).to(device)
        extended_event_embeddings = torch.cat((x, x_ext), dim=0)
        self.embedding = nn.Embedding.from_pretrained(extended_event_embeddings, freeze=freeze_embeddings_layer)

    def freeze_all_but_dense_layers(self):
        """
      Freezes all layers in the model except the dense relu layers (dense1, dense2, dense3).
      """
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze dense layers
        for dense_layer in [self.dense1, self.dense2, self.dense3]:
            for param in dense_layer.parameters():
                param.requires_grad = True

        # Unfreeze activations (if any gradients propagate through ReLU or similar)
        for layer_norm in [self.layer_norm2]:
            for param in layer_norm.parameters():
                param.requires_grad = True
