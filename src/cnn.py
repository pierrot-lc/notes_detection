"""Basic CNN as a baseline architecture.
"""
import torch
import torch.nn as nn

from einops.layers.torch import Rearrange

class AMTCNN(nn.Module):
    """CNN with strides of only 1.
    This allows the model to produce more outputs linearly proportional to the input shape.
    Based on the basic `window_size`, it supposes that the input shape will be
    [window_size + n_middle - 1], where `n_middle` is the number of labels to predict.
    """
    def __init__(
            self,
            kernel_size: int,
            stride: int,
            n_filters: int,
            n_layers: int,
            n_pitches: int
        ):
        super().__init__()

        # Small sliding window that project the 1D features to n_filters
        self.project = nn.Sequential(
            nn.Conv1d(1, n_filters, 3, 1, padding='same'),
            nn.BatchNorm1d(n_filters),
        )

        hparams = {
            'kernel_size': [kernel_size >> layer_id for layer_id in range(n_layers)],
            'stride': [stride >> layer_id for layer_id in range(n_layers)],
        }
        self.hparams = hparams

        # Main convolutional layers
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    n_filters << layer_id,
                    n_filters << (layer_id + 1),
                    kernel_size = hparams['kernel_size'][layer_id],
                    stride = hparams['stride'][layer_id],
                    bias = False
                ),
                nn.BatchNorm1d(n_filters << (layer_id + 1)),
                nn.LeakyReLU(),
            )
            for layer_id in range(n_layers)
        ])

        self.deconvs = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(
                    n_filters << (layer_id + 1),
                    n_filters << layer_id,
                    kernel_size = hparams['kernel_size'][layer_id],
                    stride = hparams['stride'][layer_id],
                    bias = False,
                ),
                nn.BatchNorm1d(n_filters << layer_id),
                nn.LeakyReLU(),
            )
            for layer_id in reversed(range(n_layers))
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

        # Encoder forward
        activations = []
        paddings = []
        for layer_id, conv_layer in enumerate(self.convs):
            x, pad = self.pad_input(x, layer_id)
            activations.insert(0, x)
            paddings.insert(0, pad)

            x = conv_layer(x)

        # Decoder forward
        for deconv_layer, act, pad in zip(self.deconvs, activations, paddings):
            x = deconv_layer(x) + act
            x = self.unpad_input(x, pad)

        del activations
        return self.head(x)


    def pad_input(self, x: torch.FloatTensor, layer_id: int) -> tuple[torch.FloatTensor, int]:
        """Add zero-padding to `x` so that the output of the convolution is a perfect integer.

        Input
        -----
            x: Batch of samples.
                Shape of [batch_size, n_filters, input_size].
            layer_id: Id of the convolution layer, used to fetch the hyperparameters.

        Output
        ------
            x: Batch of padded samples.
                Shape of [batch_size, n_filters, input_size + padding].
            padding: Number of padding added to `x`.
        """
        padding_total = (x.shape[-1] - self.hparams['kernel_size'][layer_id]) % self.hparams['stride'][layer_id]
        padding_total = self.hparams['stride'][layer_id] - padding_total if padding_total != 0 else 0

        padd_b = torch.zeros((x.shape[0], x.shape[1], padding_total // 2 + padding_total % 2)).to(x.device)
        padd_e = torch.zeros((x.shape[0], x.shape[1], padding_total // 2)).to(x.device)
        x = torch.cat((padd_b, x, padd_e), dim=-1)
        return x, padding_total

    def unpad_input(self, x: torch.FloatTensor, padding_total: int) -> torch.FloatTensor:
        """Remove the extra padding to `x`.

        Input
        -----
            x: Batch of padded samples.
                Shape of [batch_size, n_filters, input_size].
            padding_total: Number of padding to remove.

        Output
        ------
            x: Batch of unpadded samples.
                Shape of [batch_size, n_filters, input_size - padding_total].
        """
        padd_b = padding_total // 2 + padding_total % 2
        padd_e = padding_total // 2
        return x[:, :, padd_b : x.shape[-1]-padd_e]


if __name__ == '__main__':
    from torchinfo import summary
    n_filters = 10
    kernel_size = 1024
    stride = 8
    n_layers = 3
    n_pitches = 94

    batch_size = 196
    window_size = 16384

    model = AMTCNN(
        kernel_size,
        stride,
        n_filters,
        n_layers,
        n_pitches
    )
    summary(model, input_size=(batch_size, window_size))
