# models.py
"""
Defines the use of CompressAI VAE architectures and custom classification models.
"""
import torch
import torch.nn as nn
from compressai.zoo import bmshj2018_hyperprior, cheng2020_anchor  # Example models

# Load pre-trained CompressAI models
def get_pretrained_vae(model_name='bmshj2018_hyperprior', quality=3):
    """Load a pretrained VAE model from CompressAI."""
    if model_name == 'bmshj2018_hyperprior':
        return bmshj2018_hyperprior(quality=quality, pretrained=True)
    elif model_name == 'cheng2020_anchor':
        return cheng2020_anchor(quality=quality, pretrained=True)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, 256)
        self.positional_encoding = PositionalEncoding(256)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
