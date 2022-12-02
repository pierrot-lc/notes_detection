import numpy as np
import pandas as pd
import pytest
import torch

from src.dataset.dataset import AMTDataset
from src.dataset.label import SongLabels


@pytest.mark.parametrize(
    "filepath, timesteps",
    [
        ("./data/musicnet/train_labels/1727.csv", [0, 1000, 10000]),  # Base case.
        (
            "./data/musicnet/train_labels/1727.csv",
            [0, 272727, 1000000000, 1000000001, 1000000002],
        ),  # Extreme case with no labels.
        (
            "./data/musicnet/train_labels/1727.csv",
            [0, 9182, 62430],
        ),  # Test on exact start_time and end_time.
        (
            "./data/musicnet/train_labels/1727.csv",
            [i for i in range(1000)],
        ),  # Test on exact start_time and end_time.
    ],
)
def test_from_timesteps(filepath: str, timesteps: list[int]):
    df = pd.read_csv(filepath)
    labels = SongLabels(df, df["note"].max(), df["instrument"].max())
    labels = labels.from_timesteps(np.array(timesteps))

    for timestep, label in zip(timesteps, labels):
        true_label = np.zeros_like(label)
        for start_time, end_time, instrument_id, note_id in df[
            ["start_time", "end_time", "instrument", "note"]
        ].values:
            if start_time <= timestep < end_time:
                true_label[instrument_id - 1, note_id - 1] = 1

        assert (true_label == label).all()


@pytest.mark.parametrize(
    "window_size, n_windows, sample_rate",
    [(128, 5, 22050), (128, 1, 8820), (1, 5, 17640)],
)
def test_dataset(window_size: int, n_windows: int, sample_rate: int):
    wav_paths = [
        "./data/musicnet/test_data/1759.wav",
        "./data/musicnet/test_data/1819.wav",
        "./data/musicnet/test_data/2106.wav",
        "./data/musicnet/test_data/2303.wav",
    ]
    labels = [
        "./data/musicnet/test_labels/1759.csv",
        "./data/musicnet/test_labels/1819.csv",
        "./data/musicnet/test_labels/2106.csv",
        "./data/musicnet/test_labels/2303.csv",
    ]
    labels = [pd.read_csv(lp) for lp in labels]
    max_notes = max(df["note"].max() for df in labels)
    max_instru = max(df["instrument"].max() for df in labels)
    dataset = AMTDataset(wav_paths, labels, window_size, n_windows, sample_rate)
    for waves, labels in dataset:
        assert len(waves.shape) == 3
        assert waves.shape == torch.Size([n_windows, 1, window_size])
        assert len(labels.shape) == 4
        assert labels.shape == torch.Size(
            [n_windows, window_size, max_instru, max_notes]
        )
