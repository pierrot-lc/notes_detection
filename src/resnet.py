"""1D ResNet with big kernel sizes.
"""
import torch
import torch.nn as nn

from einops.layers.torch import Rearrange

class AMTResNet(nn.Module):
    """ResNet with strides of only 1.
    """
    def __init__(
            self,
            kernel_size: int,
            n_filters: int,
            n_layers: int,
            n_pitches: int
        ):
        super().__init__()

        # Small sliding window that project the 1D features to n_filters
        self.project = nn.Sequential(
            nn.Conv1d(1, n_filters // 4, 3, 1, padding='same', bias=False),
            nn.BatchNorm1d(n_filters // 4),
        )

        # Global attending convs
        # High kernel size and low number of filters
        self.global_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    n_filters // 4,
                    n_filters // 4,
                    kernel_size=kernel_size,
                    stride=1,
                    padding='same',
                    bias=False,
                ),
                nn.BatchNorm1d(n_filters // 4),
                nn.LeakyReLU(),
            )
            for _ in range(n_layers)
        ])

        self.glob_to_loc = nn.Sequential(
                nn.Conv1d(n_filters // 4, n_filters, 3, 1, padding='same', bias=False),
                nn.BatchNorm1d(n_filters),
            )

        # Local attending convs
        # Low kernel size and high number of filters
        self.local_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    n_filters,
                    n_filters,
                    kernel_size=kernel_size // 4,
                    stride=1,
                    padding='same',
                    bias=False,
                ),
                nn.BatchNorm1d(n_filters),
                nn.LeakyReLU(),
            )
            for _ in range(n_layers)
        ])

        self.head = nn.Sequential(
            nn.Conv1d(n_filters, n_pitches, 3, 1, padding='same'),
            Rearrange('b p i -> b i p'),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Pass `x` through the CNN and predict the activated pitches.

        Input
        -----
            x: Batch of windows.
                Shape of [batch_size, input_size].

        Output
        ------
            y: Batch of one-hot pitches logits.
                Shape of [batch_size, input_size, n_piches].
        """
        x = x.unsqueeze(1)  # [batch_size, 1, input_size]
        x = self.project(x)

        for conv_layer in self.global_convs:
            x = conv_layer(x) + x

        x = self.glob_to_loc(x)

        for conv_layer in self.local_convs:
            x = conv_layer(x) + x

        return self.head(x)


def from_config(config: dict) -> AMTResNet:
    return AMTResNet(
        config['model_params']['kernel_size'],
        config['model_params']['n_filters'],
        config['model_params']['n_layers'],
        config['dataset']['stats']['note']['max']
    )


if __name__ == '__main__':
    from torchinfo import summary
    n_filters = 10
    kernel_size = 1023
    stride = 8
    n_layers = 3
    n_pitches = 94

    batch_size = 196
    window_size = 16384

    model = AMTResNet(
        kernel_size,
        n_filters,
        n_layers,
        n_pitches
    )
    summary(model, input_size=(batch_size, window_size))
