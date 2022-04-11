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
        """Return the labels for each points given in that song.
        This is done in `O(n)` time, where `n` is `len(self)`.

        Input
        -----
            middle_points: Array of timesteps.
                Shape of [n_points,].

        Output
        ------
            labels: One-hot vectors for each timesteps.
                Shape of [n_points, n_instruments, n_pitches].
        """
        labels = np.zeros(
            (len(middle_points), self.n_instruments, self.n_pitches),
            dtype=int
        )

        point_id = 0
        previous_labels = np.zeros((self.n_instruments, self.n_pitches), dtype=int)
        iter_label = iter(self)
        curr_labels, curr_time = next(iter_label)

        try:
            # We go through all events and fill the labels until
            # all are processed or events are terminated.
            while point_id < len(middle_points):
                if curr_time >= middle_points[point_id]:
                    labels[point_id] = previous_labels
                    point_id += 1
                else:
                    previous_labels = curr_labels
                    curr_labels, curr_time = next(iter_label)
        except StopIteration:
            # The last middle points are after the end of the song,
            # so we assume the end of the song is containing no labels.
            while point_id < len(middle_points):
                labels[point_id] = np.zeros((self.n_instruments, self.n_pitches), dtype=int)
                point_id += 1

        return labels

    def __len__(self):
        """Return the number of events in this song.
        One event is a key being pressed on or off for a specific instrument.
        This is one row of the MusicNet dataframes.
        """
        return len(self.events)

    def __iter__(self):
        """Returns itself.
        It initializes some variables for a properly fresh iteration through all events.
        """
        self.curr_index = 0
        self.curr_labels = np.zeros((self.n_instruments, self.n_pitches), dtype=int)
        self.first_iter = True
        return self

    def __next__(self):
        """Next iteration through the events.
        It allows the user to iterate in chronological order.
        If some events occurs at the same time, they will be gathered into one
        iteration.

        Output
        -----
            label: One-hot vector for the current time step.
                Shape of [n_instruments, n_pitches].
            time: Time step corresponding to this label. This time step is
                rescaled to take into account the `downsampling_factor`.
        """
        if self.curr_index >= len(self):
            raise StopIteration

        if self.first_iter:  # Return the labels for the first iteration
            self.first_iter = False
            return self.curr_labels.copy(), 0

        # Do update the labels while the time event does not change
        first_iter = True  # Do-while loop
        while first_iter or (
                self.curr_index < len(self) and\
                self.events[self.curr_index - 1][0] == self.events[self.curr_index][0]  # Is the current event at the same time as the preceeding one?
            ):
            time, note, instrument, is_pressed = self.events[self.curr_index]
            self.curr_labels[instrument - 1, note - 1] = int(is_pressed)

            self.curr_index += 1
            first_iter = False

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
        """Read the songs data and labels, instanciate all the `SongLabels` objects.
        """
        self.factors = []
        for path in self.wav_paths:
            original_sr, _ = wavfile.read(path)
            self.factors.append(original_sr // self.target_sr)

        self.labels = [
            SongLabels(l, n_pitches, n_instruments, f)
            for l, f in zip(labels, self.factors)
        ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int):
        """Compute a random set of windows with their corresponding labels.

        Input
        -----
            index: Index of the song to exctract the windows from.

        Output
        ------
            samples: Windows from the song.
                Shape of [n_samples, window_size].
            labels: Corresponding labels from the extracted windows.
                The labels are the ones at the middle of each window.
                Shape of [n_samples, n_instruments, n_pitches].
        """
        # data = self.wavs[index]
        original_sr, data = wavfile.read(self.wav_paths[index])
        downsampling_factor = original_sr // self.target_sr
        data = decimate(data, downsampling_factor)
        data = data / np.max(np.abs(data))

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

        The starting window is picked at random.
        """
        # data = self.wavs[index]
        original_sr, data = wavfile.read(self.wav_paths[index])
        downsampling_factor = original_sr // self.target_sr
        data = decimate(data, downsampling_factor)
        data = data / np.max(np.abs(data))

        # Create subsamples
        start_indices = np.arange(len(data) - self.window_size)
        offset = np.random.randint(0, start_indices[-1] - max_idx)
        start_indices = start_indices[offset:max_idx + offset]

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
        piano_only: bool=False,
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

    dir_path = 'train_' if train else 'test_'
    path_wavs = os.path.join(path, dir_path + 'data')
    path_labels = os.path.join(path, dir_path + 'labels')

    for filename in os.listdir(path_wavs):
        music_id = int(filename[:-len('.wav')])
        path_df = os.path.join(path_labels, f'{music_id}.csv')
        path_wav = os.path.join(path_wavs, filename)

        df = pd.read_csv(path_df)

        # Filter non-piano songs if needed
        if piano_only and (df['instrument'] == 1).mean() < 0.5:
            # This song has less than 50% of piano labels
            # so we do not consider this song as a piano song
            continue

        data['labels'].append(df)
        data['id'].append(music_id)
        data['wav_path'].append(path_wav)

    if max_songs > 0:
        for key in data:
            data[key] = data[key][:max_songs]

    return data


def merge_instruments(labels: torch.LongTensor) -> torch.LongTensor:
    """Merge all instruments at each timestep, by doing a logical or
    between each of them.

    Input
    -----
        labels: Tensor containing the one-hot activation of each instrument.
            Shape of [n_samples, n_instruments, n_pitches].

    Output
    ------
        notes: Tensor with the merged instruments.
            Shape of [n_samples, n_pitches].
    """
    notes = labels[:, 0]
    for instrument_id in range(labels.shape[1]):
        notes |= labels[:, instrument_id]
    return notes


if __name__ == '__main__':
    sampling_rate = 11000
    window_size = 2048
    n_samples_by_item = 10

    data = load(
        '../data/musicnet/',
        train=True,
        piano_only=True
    )
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
    print(f'{len(dataset):,} songs.')

    music_id = 0
    samples, labels = dataset[music_id]
    print(f'\nn_samples_by_item = {n_samples_by_item} window_size = {window_size}')
    print('Shape of one example:', samples.shape, labels.shape)

    n_samples = 30
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
