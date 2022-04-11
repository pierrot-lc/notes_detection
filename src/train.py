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


def save_checkpoint(model: nn.Module, epoch_id: int, config: dict):
    """Save every training objects.
    """
    torch.save({
        'model': model.state_dict(),
        'optimizer': config['optimizer'].state_dict(),
        'epoch': epoch_id,
    }, 'artifacts/checkpoint.json')


def load_checkpoint(model: nn.Module, config: dict) -> int:
    """Load the model and the optimizer.
    Return the starting epoch.
    """
    device = config['device']
    checkpoint = torch.load('artifacts/checkpoint.json', map_location=device)

    model.load_state_dict(checkpoint['model'])
    model.to(device)
    config['optimizer'].load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']


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

    pred = torch.sigmoid(pred) >= config['positive_threshold']
    labels = labels.bool()
    metrics['acc'] = (pred == labels).float().mean()
    metrics['precision'] = ( pred & labels ).sum() / pred.sum()
    metrics['recall'] = ( pred & labels ).sum() / labels.sum()
    metrics['F1'] = metrics['precision'] * metrics['recall']

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

    converter = FluidSynth(
        sound_font='data/midi.sf2',
        sample_rate=config['sampling_rate']
    )

    model.to(device)
    for epoch_id in tqdm(range(config['starting_epoch'], epochs+1)):
        model.train()
        logs = dict()

        for samples, labels in train_loader:
            optimizer.zero_grad()

            metrics = loss_batch(model, samples, labels, config)
            loss = metrics['loss']

            loss.backward()
            optimizer.step()

            for name, value in metrics.items():
                logs['Train - ' + name] = value.item()

        metrics = eval_loader(model, test_loader, config)
        for name, value in metrics.items():
            logs[f'Test - {name}'] = value

        if epoch_id % config['convert_rate'] == 0:
            # Predict the beggining of a random sample.
            # Logs the results into WandB.
            with torch.autocast('cpu'):
                first_seconds = 2
                music_idx = torch.randint(len(train_loader.dataset), (1,) )[0]
                samples, labels_real = train_loader.dataset.get_all(music_idx, first_seconds * config['sampling_rate'])  # Gather the first seconds
                labels_real = merge_instruments(labels_real)

                samples = samples.to('cpu')
                model.to('cpu')
                midi = convert_samples_to_midi(model, samples, config['sampling_rate'], config['positive_threshold'])
                midi.write('midi', 'artifacts/out_pred.mid')
                model.to(device)

                labels_real = labels_real.long().cpu().numpy()
                midi = convert_labels_to_midi(labels_real, config['sampling_rate'], 1, 1)
                midi.write('midi', 'artifacts/out_real.mid')

            converter.midi_to_audio('artifacts/out_pred.mid', 'artifacts/out_pred.wav')
            converter.midi_to_audio('artifacts/out_real.mid', 'artifacts/out_real.wav')

            logs['Predicted'] = wandb.Audio('artifacts/out_pred.wav')
            logs['Real'] = wandb.Audio('artifacts/out_real.wav')

        wandb.log(logs)
        save_checkpoint(model, epoch_id, config)
