"""Simplistic baseline as an MLP.
"""
import torch
import torch.nn as nn


class AMTMLP(nn.Module):
    def __init__(self, window_size: int, hidden_size: int, n_layers: int, n_labels: int):
        super().__init__()
        self.window_size = window_size

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
        The input is unfolded through a sliding window over the `window_size`.
        This allows to predict multiple values over a single larger interval than
        the specified `window_size`.

        Input
        -----
            x: Batch of samples.
                Shape of [batch_size, input_size].

        Output
        ------
            y: Batch of predictions.
                Shape of [batch_size, input_size, n_labels].
        """
        x = self.unfold_x(x)
        x = self.project_raw(x)
        for layer in self.hidden_layers:
            x = x + layer(x)
        return self.head(x)

    def unfold_x(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Add padding to `x` and unfold it so it has as much windows
        as it has inputs.

        Input
        -----
            x: Batch of samples.
                Shape of [batch_size, input_size].

        Output
        ------
            x: Batch of unfolded samples.
                Shape of [batch_size, input_size, window_size].
        """
        is_even = (self.window_size % 2) == 0
        pad_b = torch.zeros((x.shape[0], self.window_size // 2 - int(is_even))).to(x.device)
        pad_e = torch.zeros((x.shape[0], self.window_size // 2)).to(x.device)
        x = torch.cat((pad_b, x, pad_e), dim=1)
        return x.unfold(1, self.window_size, 1)


def from_config(config: dict) -> AMTMLP:
    return AMTMLP(
        config['dataset']['window_size'] // 2,
        config['model_params']['hidden_size'],
        config['model_params']['n_layers'],
        config['dataset']['stats']['note']['max']
    )


if __name__ == '__main__':
    from torchinfo import summary
    model = AMTMLP(1024, 200, 3, 88)

    summary(model, input_size=(128, 2048))
