import os

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.functional as TF
from torch.utils.data import Dataset

from .label import SongLabels

DEFAULT_SAMPLE_RATE = 44100


class AMTDataset(Dataset):
    """Divide all musics into samples of fixed length (window size).
    Each sample is coupled with it's activated notes in the middle.

    Lists of `ids`, `wav_paths` and `labels` must have matching entries.

    Parameters
    ----------
        ids:            List of music ids.
        wav_paths:      List of music paths.
        labels:         List of labels.
        window_size:    Number of samples necessary to predict the middle labels.
        sampling_rate:  Targeted sampling rate. Each wav will be converted to
                        this sampling rate.
        n_pitches:      Number of possible pitches for the labels' dimension.
        n_instruments:  Number of possible instruments for the labels' dimension.
        n_windows:      Number of random starting points. When sampling from
                        this dataset, it will return one window for each starting point.
    """

    def __init__(
        self,
        wav_paths: list[str],
        labels: list[pd.DataFrame],
        window_size: int,
        n_windows: int,
    ):
        self.wav_paths = wav_paths
        self.window_size = window_size
        self.n_windows = n_windows

        max_notes = max(df["note"].max() for df in labels)
        max_instru = max(df["instrument"].max() for df in labels)
        self.labels = [SongLabels(df, max_notes, max_instru) for df in labels]

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        wav_path = self.wav_paths[index]
        infos = torchaudio.info(wav_path)
        num_frames, sample_rate = infos.num_frames, infos.sample_rate

        begin_frames = torch.randint(
            low=0,
            high=num_frames - self.window_size,
            size=(self.n_windows,),
        ).numpy()

        waves = [
            torchaudio.load(wav_path, frame_offset=begin, num_frames=self.window_size)
            for begin in begin_frames
        ]  # Load only the frames needed.
        waves = [w[0] for w in waves]  # Get the waves.
        waves = torch.stack(waves)  # To full tensor.

        window_timestep = np.arange(self.window_size)
        labels = [
            self.labels[index].from_timesteps(window_timestep + begin)
            for begin in begin_frames
        ]
        labels = torch.stack([torch.CharTensor(label) for label in labels])

        # waves = TF.resample(waves, sample_rate, DEFAULT_SAMPLE_RATE)

        return waves, labels


def load(
    path: str,
    train: bool,
    max_songs: int = -1,
    piano_only: bool = False,
) -> dict:
    """Read the labels' dataframes and store the path to
    the wav files.

    You can choose either loading the training or testing dataset.

    -------
    Returns
        data: Dictionary containing:
            - 'id': List of music ids.
            - 'wav_path': List of paths to the .wav files.
            - 'labels': List of loaded DataFrame labels.
    """
    data = {
        "id": [],
        "wav_path": [],
        "labels": [],
    }

    dir_path = "train_" if train else "test_"
    path_wavs = os.path.join(path, dir_path + "data")
    path_labels = os.path.join(path, dir_path + "labels")

    for filename in os.listdir(path_wavs):
        music_id = int(filename[: -len(".wav")])
        path_df = os.path.join(path_labels, f"{music_id}.csv")
        path_wav = os.path.join(path_wavs, filename)

        df = pd.read_csv(path_df)

        # Filter non-piano songs if needed
        if piano_only and (df["instrument"] == 1).mean() < 0.5:
            # This song has less than 50% of piano labels
            # so we do not consider this song as a piano song
            continue

        data["labels"].append(df)
        data["id"].append(music_id)
        data["wav_path"].append(path_wav)

    if max_songs > 0:
        for key in data:
            data[key] = data[key][:max_songs]

    return data


def merge_instruments(labels: torch.Tensor) -> torch.Tensor:
    """Merge all instruments at each timestep, by doing a logical OR
    between each of them.

    Input
    -----
        labels: Tensor containing the one-hot activation of each instrument.
            Shape of [n_windows, n_middle, n_instruments, n_pitches].

    Output
    ------
        notes: Tensor with the merged instruments.
            Shape of [n_windows, n_middle, n_pitches].
    """
    notes = labels[:, :, 0]
    for instrument_id in range(labels.shape[2]):
        notes |= labels[:, :, instrument_id]
    return notes.char()


def load_dataloader():
    pass
