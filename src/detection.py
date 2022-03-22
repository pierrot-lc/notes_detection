"""Use a trained model to do the AMT task.
"""
import numpy as np

import torch
import torch.nn as nn


def predict_labels(
        model: nn.Module,
        x: torch.FloatTensor,

    ) -> np.ndarray:
    """Use the trained model to predict the notes played in each window.

    Input
    -----
        model: The trained model.
        x: Input windows.
            Shape of [1, n_samples, window_size].

    Output
    ------
        labels: Predicted one-hot pitches for all samples.
            Shape of [n_samples, pitches].
    """
    output = model(x)  # [1, n_windows, n_labels]
    output = output > 0.0
    labels =  output.long().cpu().numpy()
    return labels
    return labels.transpose(1, 0)
    notes_id = torch.arange(output.shape[-1]).to(output.device)
    labels = [
        notes_id[o].cpu().numpy()
        for o in output
    ]
    return labels


def fill_blanks(labels: np.ndarray, delta: int):
    """Add missing pitchs when a serie of the same pitch is predicted,
    with some noisy small blanks between them.
    The parameter delta is setting how much long a blank can be.

    Input
    -----
        - labels: One-hot pitches for all samples.
            Shape of [n_samples, pitches].
        - delta: Number of consecutive blanks accepted to get filled.
    """
    labels = 1 - labels
    # print('Reversed:\t', *labels)

    curr_sum = np.zeros(labels.shape[1], dtype=int)
    for sample_id, sample in enumerate(labels):
        curr_sum += sample
        curr_sum[sample == 0] = 0

        labels[sample_id] = curr_sum

    # print('Cumulated:\t', *labels)
    labels = np.flip(labels, axis=0)  # Take the samples in reverse order
    # print('Flipped:\t', *labels)
    between_notes = np.zeros(labels.shape[1], dtype=int)  # Wether we are between pressed pitches or not
    for sample_id, sample in enumerate(labels):
        between_notes[sample == 0] = 1
        between_notes[sample > delta] = 0
        labels[sample_id] = between_notes

    labels = np.flip(labels, axis=0)  # Go back to the original order

    return labels


if __name__ == '__main__':
    from mlp import AMTMLP
    from data import load, number_of_labels, AMTDataset

    data = load('../MusicNet/musicnet/musicnet/', train=False)
    n_labels = number_of_labels(data['labels'])
    dataset = AMTDataset(data['id'], data['wav_path'], data['labels'], 2048, 11000, 10, n_labels)
    samples, labels = dataset.get_all(0, 100)

    model = AMTMLP(2048, 200, 3, n_labels)

    labels = predict_labels(model, samples)
    labels = np.array([
        [1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
    ]).transpose(1, 0)
    filled = fill_blanks(labels, 2)

    for original, final in zip(
        labels.transpose(1, 0),
        filled.transpose(1, 0),
    ):
        print('Original:\t', original)
        print('Final:\t\t', final)
        print('')

