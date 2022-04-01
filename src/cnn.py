"""Basic CNN as a baseline architecture.
"""
import torch
import torch.nn as nn


class AMTCNN(nn.Module):
    def __init__(
            self,
            window_size: int,
            n_filters: int,
            kernel_size: int,
            n_res_layers: int,
            n_head_layers: int,
            n_pitches: int
        ):
        super().__init__()
        intermediate_shape = (n_filters, window_size - kernel_size + 1)

        self.project_input = nn.Sequential(
            nn.Conv1d(1, n_filters, kernel_size=kernel_size, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(n_filters),
            nn.LeakyReLU(),
        )

        self.res_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, stride=1, padding='same', bias=False),
                nn.BatchNorm1d(n_filters),
                nn.LeakyReLU(),
            )
            for _ in range(n_res_layers)
        ])

        reduce_layers = []
        output_size = compute_output_size(window_size, kernel_size, stride=1, padding=0)
        for _ in range(n_head_layers):
            layer = nn.Sequential(
                nn.Conv1d(n_filters, n_filters, 2 * kernel_size, stride=1, bias=False),
                nn.BatchNorm1d(n_filters),
                nn.LeakyReLU(),
            )

            reduce_layers.append(layer)
            output_size = compute_output_size(output_size, 2 * kernel_size, stride=1, padding=0)

        self.head = nn.Sequential(
            *reduce_layers,
            nn.Flatten(),
            nn.Linear(n_filters * output_size, n_pitches),
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
        x = self.project_input(x)
        for conv_layer in self.res_layers:
            x = x + conv_layer(x)
        return self.head(x)


def compute_output_size(input_size: int, kernel_size: int, stride: int, padding: int):
    """Compute the output size of a convolution.
    """
    return (input_size + 2 * padding - kernel_size) // stride + 1


if __name__ == '__main__':
    from torchinfo import summary
    window_size = 2048
    n_filters = 10
    kernel_size = 128
    n_res_layers = 5
    n_head_layers = 5
    n_pitches = 94

    batch_size = 196

    model = AMTCNN(
        window_size,
        n_filters,
        kernel_size,
        n_res_layers,
        n_head_layers,
        n_pitches
    )
    summary(model, input_size=(batch_size, window_size))
