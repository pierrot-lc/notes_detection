"""Simplistic baseline as an MLP.
"""
import torch
import torch.nn as nn


class AMTMLP(nn.Module):
    def __init__(self, window_size: int, hidden_size: int, n_layers: int, n_labels: int):
        super().__init__()

        self.project_raw = nn.Sequential(
            nn.Linear(window_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
            )
            for layer_id in range(n_layers)
        ])

        self.head = nn.Sequential(
            nn.Linear(hidden_size, n_labels),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Do a simple forward pass.

        Input
        -----
            x: Batch of samples.
                Shape of [batch_size, window_size].

        Output
        ------
            y: Batch of predictions.
                Shape of [batch_size, n_labels].
        """
        x = self.project_raw(x)
        for layer in self.hidden_layers:
            x = x + layer(x)
        return self.head(x)


if __name__ == '__main__':
    from torchinfo import summary
    model = AMTMLP(2048, 200, 3, 88)

    summary(model, input_size=(128, 2048))
