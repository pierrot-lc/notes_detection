"""Use a trained model to do the AMT task.
"""
import torch
import torch.nn as nn


def predict_labels(
        model: nn.Module,
        x: torch.FloatTensor,

    ) -> list:
    """Use the trained model to predict the notes played in each window.

    Input
    -----
        model: The trained model.
        x: Input windows.
            Shape of [1, n_windows, window_size].

    Output
    ------
        labels: List of all predicted labels for each window.
    """
    output = model(x)  # [1, n_windows, n_labels]
    output = output > 0.0
    notes_id = torch.arange(output.shape[-1]).to(output.device)
    labels = [
        notes_id[o].cpu().numpy()
        for o in output
    ]
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
