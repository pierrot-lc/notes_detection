"""Basic CNN as a baseline architecture.
"""
import torch
import torch.nn as nn


class AMTCNN(nn.Module):
    def __init__(self, window_size: int, receptive_field: int, n_layers: int, n_labels: int):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 3, kernel_size=100, stride=10),
            nn.ReLU(),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = x.unsqueeze(1)
        return self.cnn(x)


if __name__ == '__main__':
    from torchinfo import summary

    model = AMTCNN(2048, 210, 3, 88)
    summary(model, input_size=(196, 2048))
