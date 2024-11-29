# src/autoencoder.py
import torch

class AutoEncoder(torch.nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, encoding_dim * 3),
            torch.nn.ReLU(),
            torch.nn.Linear(encoding_dim * 3, encoding_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(encoding_dim * 2, encoding_dim),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(encoding_dim, input_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim * 2, input_dim * 3),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim * 3, input_dim),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
