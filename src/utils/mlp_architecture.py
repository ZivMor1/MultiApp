import torch.nn as nn


class MLPNextItemPrediction(nn.Module):
    def __init__(self, recs_size: int, classifier_input_size=512):
        super(MLPNextItemPrediction, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(classifier_input_size, 150),
            nn.ReLU(),
            nn.LayerNorm(150),
            nn.Dropout(0.15),
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.LayerNorm(100),
            nn.Dropout(0.15),
            nn.Linear(100, recs_size)
        )

    def forward(self, x):
        return self.mlp(x)


class MLPBinaryClassifier(nn.Module):
    def __init__(self, classifier_input_size=512):
        super(MLPBinaryClassifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(classifier_input_size, 150),
            nn.ReLU(),
            nn.LayerNorm(150),
            nn.Dropout(0.3),
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.LayerNorm(100),
            nn.Dropout(0.3),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.LayerNorm(50),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(50, 1),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.output_layer(x)
        return x
