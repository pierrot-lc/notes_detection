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
    def __init__(
            self,
            labels: pd.DataFrame,
            n_pitches: int,
            n_instruments: int,
            downsampling_factor: int=1
        ):
        self.n_pitches = n_pitches
        self.n_instruments = n_instruments
        self.downsampling_factor = downsampling_factor
        self.events = []  # Tuples (time, pitch, instrument, is_pressed)

        for start, end, note, instrument in labels[['start_time', 'end_time', 'note', 'instrument']].values:
            self.events.append((start, note, instrument, True))
            self.events.append((end, note, instrument, False))

        self.events = list(sorted(self.events, key=lambda el: el[0]))  # Sort by time

    def from_middle_points(self, middle_points: np.ndarray) -> np.ndarray:
        labels = np.zeros(
            (len(middle_points), self.n_instruments, self.n_pitches),
            dtype=int
        )

        point_id = 0
        previous_labels = np.zeros((self.n_instruments, self.n_pitches), dtype=int)
        iter_label = iter(self)
        curr_labels, curr_time = next(iter_label)

        while point_id < len(middle_points):
            if curr_time >= middle_points[point_id]:
                labels[point_id] = previous_labels
                point_id += 1
            else:
                previous_labels = curr_labels
                curr_labels, curr_time = next(iter_label)

        return labels

    def __len__(self):
        return len(self.events)

    def __iter__(self):
        self.curr_index = 0
        self.curr_labels = np.zeros((self.n_instruments, self.n_pitches), dtype=int)
        self.first_iter = True
        return self

    def __next__(self):
        if self.curr_index >= len(self):
            raise StopIteration

        if self.first_iter:  # Return the labels for the first iteration
            self.first_iter = False
            return self.curr_labels.copy(), 0

        # Do update the labels while the time event does not change
        time, note, instrument, is_pressed = self.events[self.curr_index]
        self.curr_labels[instrument - 1, note - 1] = int(is_pressed)
        self.curr_index += 1

        while self.curr_index < len(self) and self.events[self.curr_index - 1][0] == self.events[self.curr_index][0]:
            time, note, instrument, is_pressed = self.events[self.curr_index]
            self.curr_labels[instrument - 1, note - 1] = int(is_pressed)
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
            n_pitches: int,
            n_instruments: int,
        ):
        self.ids = ids
        self.wav_paths = wav_paths

        self.window_size = window_size
        self.target_sr = sampling_rate
        self.n_samples = n_samples_by_item

        self.prepare(labels, n_pitches, n_instruments)

    def prepare(self, labels: list, n_pitches: int, n_instruments: int):
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
            SongLabels(l, n_pitches, n_instruments, f)
            for l, f in zip(labels, self.factors)
        ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int):
        data = self.wavs[index]

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
        labels = torch.LongTensor(labels)

        return samples, labels

    def get_all(self, index: int, max_idx: int):
        """Return all windows of the given index.
        """
        data = self.wavs[index]

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
        labels = torch.LongTensor(labels)

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


def get_stats(labels: list) -> int:
    """Compute the number of pitches and instruments in the given list of labels.
    """
    stats = {
        col: dict()
        for col in ['note', 'instrument']
    }

    for col in stats:
        max_c, min_c = labels[0][col].max(), labels[0][col].min()

        for df in labels:
            max_c = max(max_c, df[col].max())
            min_c = min(min_c, df[col].min())

        stats[col]['max'] = max_c
        stats[col]['min'] = min_c

    return stats


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
    sampling_rate = 11000
    window_size = 2048
    n_samples_by_item = 10

    data = load('../MusicNet/musicnet/musicnet/', train=False)
    stats = get_stats(data['labels'])
    dataset = AMTDataset(
        data['id'],
        data['wav_path'],
        data['labels'],
        window_size,
        sampling_rate,
        n_samples_by_item,
        stats['note']['max'],
        stats['instrument']['max'],
    )

    music_id = 0
    samples, labels = dataset[music_id]
    print(f'\nn_samples_by_item = {n_samples_by_item} window_size = {window_size}')
    print('Shape of one example:', samples.shape, labels.shape)

    n_samples = 1530
    samples, labels = dataset.get_all(music_id, n_samples)
    print(f'Shape of the first {n_samples} samples:', samples.shape, labels.shape)

    print(f'\nFor music id: {dataset.ids[music_id]}')
    pitches_id = torch.arange(labels.shape[2])
    for time, label in enumerate(labels):
        if label.sum() == 0:
            continue

        print(f'T{time}')
        for instrument_id, pitches in enumerate(label):
            if (pitches == 1).sum() > 0:
                print(f'{instrument_id} - {pitches_id[pitches == 1]}')

        print('')
