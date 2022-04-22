"""Basic CNN as a baseline architecture.
"""
import torch
import torch.nn as nn


class AMTCNN(nn.Module):
    def __init__(
            self,
            kernel_size: int,
            n_filters: int,
            stride: int,
            n_layers: int,
            window_size: int,
            n_pitches: int
        ):
        super().__init__()

        # Small sliding window that project the 1D features to n_filters
        self.project = nn.Sequential(
            nn.Conv1d(1, n_filters, 3, 1, 2),
            nn.BatchNorm1d(n_filters),
        )

        # Main convolutional layers
        self.convs = [
            nn.Sequential(
                nn.Conv1d(n_filters, n_filters, kernel_size >> layer_id, stride, bias=False),
                nn.BatchNorm1d(n_filters),
                nn.LeakyReLU(),
                nn.MaxPool1d(3, 2),
            )
            for layer_id in range(n_layers)
        ]

        self.convs += [nn.Flatten()]
        self.convs = nn.Sequential(*self.convs)

        # Compute the output size and create the final head
        with torch.no_grad():
            x = torch.zeros((64, 1, window_size))
            x = self.project(x)
            x = self.convs(x)
            hidden_dim = x.shape[-1]

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, n_pitches)
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Pass `x` through the CNN and predict the activated pitches.

        Input
        -----
            x: Batch of windows.
                Shape of [batch_size, window_size].

        Output
        ------
            y: Batch of one-hot pitches logits.
                Shape of [batch_size, n_piches].
        """
        x = x.unsqueeze(1)  # [batch_size, 1, window_size]
        x = self.project(x)
        x = self.convs(x)
        return self.head(x)


if __name__ == '__main__':
    from torchinfo import summary
    n_filters = 10
    kernel_size = 1024
    stride = 3
    n_layers = 3
    n_pitches = 94

    batch_size = 196
    input_size = 16384

    model = AMTCNN(
        kernel_size,
        n_filters,
        stride,
        n_layers,
        input_size,
        n_pitches
    )
    summary(model, input_size=(batch_size, input_size))
