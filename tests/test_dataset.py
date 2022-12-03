import numpy as np
import pandas as pd
import pytest
import torch
import torchaudio

from src.dataset.dataset import AMTDataset
from src.dataset.label import SongLabels
from src.midi.parser import frames_to_timesteps


@pytest.mark.parametrize(
    "filepath, timesteps",
    [
        ("./data/musicnet/train_labels/1727.csv", [0, 1000, 10000]),  # Basic case.
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
            [i for i in range(10000, 10300)],
        ),  # Test on consecutive frames.
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
def test_dataset_shapes(window_size: int, n_windows: int, sample_rate: int):
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


@pytest.mark.parametrize(
    "labels_path, wav_path, sampling_rate",
    [
        (
            "./data/musicnet/test_labels/1759.csv",
            "./data/musicnet/test_data/1759.wav",
            22050,
        )
    ],
)
def test_labels_downsampling(labels_path: str, wav_path: str, sampling_rate: int):
    df = pd.read_csv(labels_path)
    dataset = AMTDataset([wav_path], [df], 100, 5, sampling_rate)
    original_sr = torchaudio.info(wav_path).sample_rate

    head_df = df.head(5)
    for start_time, end_time, instrument, note in head_df[
        ["start_time", "end_time", "instrument", "note"]
    ].values:
        # Read labels according to the event start and end time.
        waves, labels = dataset.get_windows(
            0, np.array([start_time]), end_time - start_time
        )
        _, labels = dataset.downsample(waves, labels, original_sr)

        # Extract event.
        labels = labels[
            0, :, instrument - 1
        ].numpy()  # Shape is [n_downsampled_points, n_notes].
        labels[:, : note - 1] = 0
        labels[:, note:] = 0
        timesteps = frames_to_timesteps(labels, sampling_rate)

        assert len(timesteps) == 1
        event = timesteps[0]

        # Round the number to account for numerical approximation.
        start_time = start_time / original_sr
        end_time = end_time / original_sr
        event = (
            event[0],
            round(start_time + event[1], 2),
            round(start_time + event[2], 2),
        )
        start_time = round(start_time, 2)
        end_time = round(end_time, 2)

        assert event == (note - 1, start_time, end_time)
