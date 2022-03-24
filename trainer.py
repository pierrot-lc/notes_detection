"""Train a model based on config files.
"""
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.train import train
from src.mlp import AMTMLP
from src.data import load, number_of_labels, AMTDataset


def init_config() -> dict:
    config = {
        'group': 'Test',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'sampling_rate': 11000,
        'convert_frequence': 10,

        'epochs': 30,
        'batch_size': 10,
        'n_samples_by_item': 50,
        'lr': 1e-3,
        'pos_weight': 15,

        'model_type': 'MLP',
        'window_size': 4098,
        'hidden_size': 500,
        'n_layers': 10,
    }

    return config


def load_config(config: dict):
    train_dataset = load('MusicNet/musicnet/musicnet/', train=True, max_songs=200)
    config['n_labels'] = number_of_labels(train_dataset['labels'])
    train_dataset = AMTDataset(
        train_dataset['id'],
        train_dataset['wav_path'],
        train_dataset['labels'],
        config['window_size'],
        config['sampling_rate'],
        config['n_samples_by_item'],
        config['n_labels'],
    )

    config['train_loader'] = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=AMTDataset.collate_fn,
    )

    test_dataset = load('MusicNet/musicnet/musicnet/', train=False)
    test_dataset = AMTDataset(
        test_dataset['id'],
        test_dataset['wav_path'],
        test_dataset['labels'],
        config['window_size'],
        config['sampling_rate'],
        config['n_samples_by_item'],
        config['n_labels'],
    )

    config['test_loader'] = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=AMTDataset.collate_fn,
    )

    if config['model_type'] == 'MLP':
        model = AMTMLP(config['window_size'], config['hidden_size'], config['n_layers'], config['n_labels'])

    config['model'] = model
    config['optimizer'] = optim.Adam(
        model.parameters(),
        lr=config['lr']
    )


if __name__ == '__main__':
    config = init_config()
    load_config(config)

    with wandb.init(
        project='Automatic Music Transcription',
        entity='pierrotlc',
        group=config['group'],
        config=config,
    ):
        train(config['model'], config)
