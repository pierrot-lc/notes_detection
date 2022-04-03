"""Basic CNN as a baseline architecture.
"""
import torch
import torch.nn as nn


class ResLayer(nn.Module):
    """Apply residual layers to the input.
    """
    def __init__(
            self,
            input_size: int,
            kernel_size: int,
            stride: int,
            n_filters: int,
            n_layers: int,
        ):
        super().__init__()
        padding = compute_same_padding(input_size, kernel_size, stride)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(n_filters, n_filters, kernel_size, stride, padding, bias=False),
                nn.BatchNorm1d(n_filters),
                nn.LeakyReLU(),
            )
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for layer in self.layers:
            x = x + layer(x)
        return x


class AMTCNN(nn.Module):
    def __init__(
            self,
            window_size: int,
            n_filters: int,
            kernel_size: int,
            n_res_layers: int,
            n_main_layers: int,
            n_pitches: int
        ):
        super().__init__()
        stride = 2

        # First we project the input to a shape where we can apply some residual layers
        padding = compute_same_padding(window_size, kernel_size, stride)
        self.project_input = nn.Sequential(
            nn.Conv1d(1, n_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(n_filters),
            nn.LeakyReLU(),
        )

        main_layers = []
        input_size = window_size
        for _ in range(n_main_layers):
            # Add Residual layers
            main_layers.append(ResLayer(input_size, kernel_size, stride, n_filters, n_res_layers))

            # Divide input size by 4
            padding = compute_divide_by_4_padding(input_size, kernel_size, stride=2 * stride)
            main_layers.append(nn.Conv1d(n_filters, n_filters << 1, kernel_size, stride=2 * stride, padding=padding))
            input_size = compute_output_size(input_size, kernel_size, 2 * stride, padding)
            n_filters  = n_filters << 1

        self.main = nn.Sequential(
            *main_layers,
            nn.Flatten(),
            nn.Linear(n_filters * input_size, n_pitches),
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
        return self.main(x)


def compute_output_size(input_size: int, kernel_size: int, stride: int, padding: int):
    """Compute the output size of a convolution.
    """
    assert (input_size + 2 * padding - kernel_size) % stride == 0, f'L={input_size}, S={stride}, K={kernel_size}, P={padding}'
    return (input_size + 2 * padding - kernel_size) // stride + 1


def compute_same_padding(input_size: int, kernel_size: int, stride: int):
    """Compute the padding value to get the same output size as the input.
    """
    assert (input_size * (stride - 1) + kernel_size - stride) % 2 == 0, f'L={input_size}, S={stride}, K={kernel_size}'
    return (input_size * (stride - 1) + kernel_size - stride) // 2

def compute_divide_by_4_padding(input_size: int, kernel_size: int, stride: int):
    """Compute the padding value to get the output size divided by 2 w.r.t. the input size.
    """
    assert input_size % 4 == 0, f'L={input_size}'
    assert (stride * (input_size // 4 - 1) - input_size + kernel_size) % 2 == 0, f'L={input_size}, S={stride}, K={kernel_size}'
    return (stride * (input_size // 4 - 1) - input_size + kernel_size) // 2


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
