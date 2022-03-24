import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import decimate
from tqdm import tqdm

import torch
from torch.utils.data import Dataset


class SongLabels:
    def __init__(self, labels: pd.DataFrame, n_labels: int, downsampling_factor: int=1):
        self.n_labels = n_labels
        self.events = []  # Tuples (time, pitch, is_pressed)

        for start, end, note in labels[['start_time', 'end_time', 'note']].values:
            self.events.append((start, note, True))
            self.events.append((end, note, False))

        self.events = list(sorted(self.events, key=lambda el: el[0]))  # Sort by time

        self.curr_index = 0
        self.curr_labels = np.zeros(n_labels, dtype=int)
        self.downsampling_factor = downsampling_factor
        self.n_labels = n_labels

    def from_middle_points(self, middle_points: np.ndarray) -> np.ndarray:
        labels = np.zeros(
            (len(middle_points), self.n_labels),
            dtype=int
        )

        point_id = 0
        previous_labels = np.zeros(self.n_labels, dtype=int)
        for curr_labels, time in self:
            if time >= middle_points[point_id]:  # We passed the middle point
                labels[point_id] = previous_labels

                point_id += 1
                if point_id >= len(middle_points):
                    break

            previous_labels = curr_labels

        return labels

    def __len__(self):
        return len(self.events)

    def __iter__(self):
        self.curr_index = 0
        return self

    def __next__(self):
        if self.curr_index >= len(self):
            raise StopIteration

        time, note, is_pressed = self.events[self.curr_index]
        self.curr_labels[note - 1] = int(is_pressed)
        self.curr_index += 1
        return self.curr_labels.copy(), time // self.downsampling_factor


class AMTDataset(Dataset):
    """Divide all musics into samples of fixed length (window size).
    Each sample is coupled with it's activated notes in the middle.
    """
    def __init__(
            self,
            ids: list,
            wav_paths: list,
            labels: list,
            window_size: int,
            sampling_rate: int,
            n_samples_by_item: int,
            n_labels: int,
        ):
        self.ids = ids
        self.wav_paths = wav_paths
        self.labels = labels

        self.window_size = window_size
        self.target_sr = sampling_rate
        self.n_samples = n_samples_by_item
        self.n_labels = n_labels

        self.prepare()

    def prepare(self):
        """Downsample all .wav files, and store them into the class.
        """
        print('Downsampling .wav files...')
        self.factors = []

        futures = []
        with ThreadPoolExecutor() as executor:
            for path in tqdm(self.wav_paths):
                original_sr, data = wavfile.read(path)
                data = data / np.max(np.abs(data))
                downsampling_factor = original_sr // self.target_sr

                futures.append(executor.submit(decimate, data, downsampling_factor))
                self.factors.append(downsampling_factor)

        self.wavs = [
            f.result()
            for f in futures
        ]

        self.labels = [
            SongLabels(l, self.n_labels, f)
            for l, f in zip(self.labels, self.factors)
        ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int):
        # Load data
        data = self.wavs[index]
        downsampling_factor = self.factors[index]

        # Create subsamples
        start_indices = np.random.randint(low=0, high=len(data) - self.window_size, size=self.n_samples)
        start_indices.sort()
        samples = np.array([
            data[index:index + self.window_size]
            for index in start_indices
        ])
        samples = torch.FloatTensor(samples)

        # Compute corresponding labels
        labels = self.labels[index].from_middle_points(start_indices + self.window_size // 2)
        labels = torch.FloatTensor(labels)

        return samples, labels

    def get_all(self, index: int, max_idx: int):
        """Return all windows of the given index.
        """
        # Load data
        data = self.wavs[index]
        labels = self.labels[index]
        downsampling_factor = self.factors[index]

        # Create subsamples
        start_indices = np.arange(len(data) - self.window_size)
        start_indices = start_indices[:max_idx]

        samples = np.array([
            data[start_index:start_index + self.window_size]
            for start_index in start_indices
        ])
        samples = torch.FloatTensor(samples)

        # Compute corresponding labels
        labels = self.labels[index].from_middle_points(start_indices + self.window_size // 2)
        labels = torch.FloatTensor(labels)

        return samples, labels

    def collate_fn(batch: list) -> tuple:
        """Collate function a dataloader must call for this dataset.
        """
        samples, labels = [], []
        for s, l in batch:
            samples.append(s)
            labels.append(l)
        samples = torch.cat(samples)
        labels = torch.cat(labels)
        return samples, labels


def number_of_labels(labels: list) -> int:
    """Compute the number of labels in the given list of labels.
    This is essentially the maximum note encountered among all dataframes.
    """
    max_note, min_note = labels[0]['note'].max(), labels[0]['note'].min()
    for df in labels:
        max_note = max(max_note, df['note'].max())
        min_note = min(min_note, df['note'].min())
    return max_note


def load(
        path: str,
        train: bool,
        max_songs: int=-1,
    ) -> dict:
    """Read the labels' dataframes and store the path to
    the wav files.

    You can choose either loading the training or testing dataset.

    Output
    ------
        data: Dictionnary containing:
            - 'id': List of music ids.
            - 'wav_path': List of paths to the .wav files.
            - 'labels': List of loaded DataFrame labels.
    """
    data = {
        'id': [],
        'wav_path': [],
        'labels': [],
    }

    if train:
        path_wavs = os.path.join(path, 'train_data')
        path_labels = os.path.join(path, 'train_labels')
    else:
        path_wavs = os.path.join(path, 'test_data')
        path_labels = os.path.join(path, 'test_labels')

    for filename in os.listdir(path_wavs):
        music_id = int(filename[:-len('.wav')])
        data['id'].append(music_id)

        data['wav_path'].append(os.path.join(path_wavs, filename))

        path_df = os.path.join(path_labels, f'{music_id}.csv')
        df = pd.read_csv(path_df)
        data['labels'].append(df)

    if max_songs > 0:
        for key in data:
            data[key] = data[key][:max_songs]

    return data


if __name__ == '__main__':
    data = load('../MusicNet/musicnet/musicnet/', train=True)
    n_labels = number_of_labels(data['labels'])
    dataset = AMTDataset(data['id'], data['wav_path'], data['labels'], 2048, 11000, 10, n_labels)

    samples, labels = dataset[0]
    print(samples.shape, labels.shape)
    samples, labels = dataset.get_all(0, 10000)
    print(samples.shape, labels.shape)
