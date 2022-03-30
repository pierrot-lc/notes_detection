"""Training loop.
"""
from collections import defaultdict

import wandb
import numpy as np
from tqdm import tqdm
from midi2audio import FluidSynth

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.detection import convert_samples_to_midi, convert_labels_to_midi
from src.data import merge_instruments


def loss_batch(
        model: nn.Module,
        samples: torch.FloatTensor,
        labels: torch.LongTensor,
        config: dict
    ) -> dict:
    """Compute the loss for one batch of samples.

    It returns a bunch of metrics, along with the loss, into
    a dictionnary.
    """
    device = config['device']
    metrics = dict()

    samples = samples.to(device)
    labels = merge_instruments(labels)
    labels = labels.float().to(device).flatten()

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.ones(len(labels)) * config['pos_weight'],
    ).to(device)


    pred = model(samples).flatten()  # [batch_size * n_labels, ]
    metrics['loss'] = loss_fn(pred, labels)

    pred = pred > 0.0
    labels = labels.bool()
    metrics['acc'] = (pred == labels).float().mean()
    metrics['precision'] = ( pred & labels ).sum() / pred.sum()
    metrics['recall'] = ( pred & labels ).sum() / labels.sum()

    return metrics


def eval_loader(model: nn.Module, loader: DataLoader, config: dict) -> dict:
    device = config['device']
    metrics = defaultdict(list)

    model.to(device)
    model.eval()
    with torch.no_grad():
        for samples, labels in loader:
            metrics_batch = loss_batch(model, samples, labels, config)

            for name, value in metrics_batch.items():
                metrics[name].append(value.item())

    for name, values in metrics.items():
        metrics[name] = np.mean(values)

    return metrics


def train(model: nn.Module, config: dict):
    train_loader, test_loader = config['train_loader'], config['test_loader']
    optimizer = config['optimizer']
    epochs = config['epochs']
    device = config['device']

    converter = FluidSynth(sample_rate=config['sampling_rate'])

    model.to(device)
    for epoch_id in tqdm(range(1, epochs+1)):
        model.train()
        for samples, labels in train_loader:
            optimizer.zero_grad()

            metrics = loss_batch(model, samples, labels, config)
            loss = metrics['loss']

            loss.backward()
            optimizer.step()

        logs = dict()
        metrics_train = eval_loader(model, train_loader, config)
        metrics_test = eval_loader(model, test_loader, config)
        for group, metrics in zip(['Train', 'Test'], [metrics_train, metrics_test]):
            for name, value in metrics.items():
                logs[f'{group} - {name}'] = value

        if epoch_id % config['convert_frequence'] == 0:
            music_idx = torch.randint(len(train_loader.dataset), (1,) )[0]
            samples, labels_real = train_loader.dataset.get_all(music_idx, 5 * config['sampling_rate'])  # Gather the first 10 seconds
            labels_real = merge_instruments(labels_real)

            samples = samples.to(device)
            midi = convert_samples_to_midi(model, samples, config['sampling_rate'])
            midi.write('midi', 'artifacts/out_pred.mid')
            converter.midi_to_audio('artifacts/out_pred.mid', 'artifacts/out_pred.wav')
            logs['Predicted'] = wandb.Audio('artifacts/out_pred.wav')

            labels_real = labels_real.long().cpu().numpy()
            midi = convert_labels_to_midi(labels_real, config['sampling_rate'], 1, 1)
            midi.write('midi', 'artifacts/out_real.mid')
            converter.midi_to_audio('artifacts/out_real.mid', 'artifacts/out_real.wav')
            logs['Real'] = wandb.Audio('artifacts/out_real.wav')

        wandb.log(logs)
