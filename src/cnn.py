"""Basic CNN as a baseline architecture.
"""
import torch
import torch.nn as nn


class AMTCNN(nn.Module):
    def __init__(self, window_size: int, receptive_field: int):
        super().__init__()
