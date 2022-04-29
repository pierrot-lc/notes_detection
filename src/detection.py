"""Use a trained model to do the AMT task.
"""
import numpy as np
from music21.stream import Stream, Score
from music21.note import Note, Rest

import torch
import torch.nn as nn


def predict_labels(
        model: nn.Module,
        x: torch.FloatTensor,
        positive_threshold: float,

    ) -> np.ndarray:
    """Use the trained model to predict the notes played in each window.

    Input
    -----
        model: The trained model.
        x: Input windows.
            Shape of [1, window_size + n_middle - 1].

    Output
    ------
        labels: Predicted one-hot pitches for all samples.
            Shape of [n_middle, n_pitches].
    """
    with torch.no_grad():
        output = model(x)  # Shape is [1, n_middle, n_pitches]
        output = torch.sigmoid(output) >= positive_threshold
        labels = output.char().cpu().numpy()

    return labels[0]


def fill_blanks(labels: np.ndarray, delta: int) -> np.ndarray:
    """Add missing pitchs when a serie of the same pitch is predicted,
    with some noisy small blanks between them.
    The parameter delta is setting how much long a blank can be.

    This algorithm has time complexity of O(n).

    Input
    -----
        labels: One-hot pitches for all samples.
            Shape of [n_samples, n_pitches].
        delta: Number of consecutive blanks accepted to get filled.

    Output
    ------
        filled: One-hot pitches for all samples, where 0's are replaced
            by 1's if they are no more than `delta` consecutives.
            Shape of [n_samples, n_pitches].
    """
    labels = 1 - labels

    curr_sum = np.zeros(labels.shape[1], dtype=int)
    for sample_id, sample in enumerate(labels):
        curr_sum += sample
        curr_sum[sample == 0] = 0

        labels[sample_id] = curr_sum

    labels = np.flip(labels, axis=0)  # Take the samples in reverse order
    between_notes = np.zeros(labels.shape[1], dtype=int)  # Wether we are between pressed pitches or not
    for sample_id, sample in enumerate(labels):
        between_notes[sample == 0] = 1
        between_notes[sample > delta] = 0
        labels[sample_id] = between_notes

    labels = np.flip(labels, axis=0)  # Go back to the original order

    return labels


def postprocess(
        labels: np.ndarray,
        max_zeros: int,
        max_ones: int
    ) -> np.ndarray:
    """Fill low consecutives 0's by 1's, and remove low consecutives 1's by 0's.
    Low consecutives 0's are considered as false negative and low consecutives 1's
    are considered as false positive.

    Input
    -----
        labels: One-hot pitches for all samples.
            Shape of [n_samples, n_pitches].
        max_zeros: Number of consecutive 0's accepted to get filled.
        max_ones: Number of consecutive 1's accepted to get replaced.

    Output
    ------
        labels: One-hot pitches postprocessed.
            Shape of [n_samples, n_pitches].
    """
    labels = fill_blanks(labels, max_zeros)
    labels = 1 - labels
    labels = fill_blanks(labels, max_ones)
    labels = 1 - labels
    return labels


def streamify(labels: np.ndarray) -> list:
    """Parse the labels into a stream of played notes.

    Input
    -----
        labels: One-hot pitches for all samples.
            Shape of [n_samples, n_pitches].

    Output
    ------
        stream: List of pressed and unpressed pitches.
            For each pitch, a list is associated with its onset history and duration.
            Shape of [n_pitches].
    """
    durations = np.zeros(labels.shape[1], dtype=int)
    onsets = np.zeros(labels.shape[1], dtype=bool)
    stream = [[] for _ in range(labels.shape[1])]

    for sample in labels:
        for pitch_id in range(labels.shape[1]):
            # Add notes if the onset is changing and if durations of the current pitch is not 0
            # Durations can be 0 at the beginning of the samples
            if onsets[pitch_id] != sample[pitch_id] and durations[pitch_id] != 0:
                stream[pitch_id].append((onsets[pitch_id], durations[pitch_id]))

        durations += 1
        durations[sample != onsets] = 1
        onsets = sample.astype('bool')

    # Add the pending notes
    for pitch_id in range(labels.shape[1]):
        stream[pitch_id].append((onsets[pitch_id], durations[pitch_id]))

    return stream


def to_midi(stream: list, sampling_rate: int) -> Stream:
    """Produce a midi object from a list of pressed and unpressed
    pitches.

    Input
    -----
        stream: List of pressed and unpressed pitches.
            For each pitch, a list is associated with its onset history and duration.
            Shape of [n_pitches].
        sampling_rate: Number of points in 1 second of sound.

    Output
    ------
        midi: Stream of multiple substreams, where each substream
            is following one different pitch history.
    """
    midi = Score(id='mainScore')

    for pitch_id, pitch_history in enumerate(stream):
        pitch_midi = Stream()
        for onset, duration in pitch_history:
            duration = 2 * duration / sampling_rate
            if onset:
                note = Note(pitch_id+1, quarterLength=duration)
            else:
                note = Rest(quarterLength=duration)

            pitch_midi.append(note)

        midi.insert(0, pitch_midi)

    return midi


def convert_labels_to_midi(
        labels: np.ndarray,
        sampling_rate: int,
        max_zeros: int=100,
        max_ones: int=100,
    ):
    """Pipeline predicting the labels detected by the model,
    and converting those labels into one midi object.

    Input
    -----
        labels: Input windows.
            Shape of [n_samples, window_size].
        sampling_rate: Number of points in 1 second of sound.
        max_zeros: Number of consecutive 0's accepted to get filled.
        max_ones: Number of consecutive 1's accepted to get replaced.

    Output
    ------
        midi: Stream of multiple substreams, where each substream
            is following one different pitch history.
    """
    labels = postprocess(labels, max_zeros, max_ones)
    stream = streamify(labels)
    midi = to_midi(stream, sampling_rate)
    return midi


def convert_samples_to_midi(
        model: nn.Module,
        samples: torch.FloatTensor,
        sampling_rate: int,
        max_zeros: int=100,
        max_ones: int=100,
        positive_threshold: float=0.5,
    ) -> Stream:
    """Pipeline predicting the labels detected by the model,
    and converting those labels into one midi object.

    Input
    -----
        model: The trained model.
        samples: Input windows.
            Shape of [n_samples, window_size].
        sampling_rate: Number of points in 1 second of sound.
        max_zeros: Number of consecutive 0's accepted to get filled.
        max_ones: Number of consecutive 1's accepted to get replaced.

    Output
    ------
        midi: Stream of multiple substreams, where each substream
            is following one different pitch history.
    """
    labels = predict_labels(model, samples, positive_threshold)
    midi = convert_labels_to_midi(
        labels,
        sampling_rate,
        max_zeros,
        max_ones,
    )
    return midi


if __name__ == '__main__':
    from mlp import AMTMLP
    from data import load, get_stats, AMTDataset, merge_instruments

    data = load('../data/musicnet/', train=False)
    sampling_rate = 11000
    stats = get_stats(data['labels'])
    dataset = AMTDataset(
        data['id'],
        data['wav_path'],
        data['labels'],
        2048,
        sampling_rate,
        stats['note']['max'],
        stats['instrument']['max'],
        10,
    )
    samples, labels = dataset.__getitem__(0, n_windows=1, window_size=100)
    # samples, labels = samples[0], labels[0]

    model = AMTMLP(2048, 200, 3, stats['note']['max'])

    labels = predict_labels(model, samples, 0.7)
    labels = np.array([
        [1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
    ]).transpose(1, 0)
    filled = postprocess(labels, max_zeros=1, max_ones=2)

    print('For max_zeros = 1 and max_ones = 2')
    for original, final in zip(
        labels.transpose(1, 0),
        filled.transpose(1, 0),
    ):
        print('Original:\t', original)
        print('Final:\t\t', final)
        print('')

    stream = streamify(filled)
    midi = to_midi(stream, 1)
    midi.show('text')

    # Get the first 10 secs
    idx = 0
    midi_id = dataset.ids[idx]
    samples, labels = dataset.__getitem__(idx, n_windows=1, window_size=15 * sampling_rate)
    samples, labels = samples[0], labels[0]

    notes = merge_instruments(labels)
    notes = notes.char().cpu().numpy()

    midi = convert_labels_to_midi(notes, sampling_rate, 1, 1)
    midi.write('midi', f'{midi_id}.mid')
