"""Train a model based on config files.
"""
import sys
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary

from src.train import train
from src.data import load, get_stats, AMTDataset
from src.mlp import AMTMLP
from src.cnn import AMTCNN


def init_config(model_type: str) -> dict:
    config = {
        'group': 'Pitch prediction',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'window_size': 4098,
        'sampling_rate': 11000,
        'convert_frequence': 5,
        'positive_threshold': 0.7,

        'epochs': 10,
        'batch_size': 4,
        'n_samples_by_item': 100,
        'lr': 1e-4,
        'pos_weight': 10,
        'model_type': model_type,
    }

    return config


def config_mlp(config: dict):
    """Config for MLP model.
    """
    config['hidden_size'] = 500
    config['n_layers'] = 10


def config_cnn(config: dict):
    """Config for CNN model.
    """
    config['n_filters'] = 20
    config['kernel_size'] = 256
    config['n_res_layers'] = 5
    config['n_head_layers'] = 3


def load_config(config: dict):
    train_dataset = load('MusicNet/musicnet/musicnet/', train=True)
    config['stats'] = get_stats(train_dataset['labels'])
    train_dataset = AMTDataset(
        train_dataset['id'],
        train_dataset['wav_path'],
        train_dataset['labels'],
        config['window_size'],
        config['sampling_rate'],
        config['n_samples_by_item'],
        config['stats']['note']['max'],
        config['stats']['instrument']['max'],
    )

    config['train_loader'] = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
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
        config['stats']['note']['max'],
        config['stats']['instrument']['max'],
    )

    config['test_loader'] = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=AMTDataset.collate_fn,
    )

    if config['model_type'] == 'MLP':
        config_mlp(config)
        model = AMTMLP(
            config['window_size'],
            config['hidden_size'],
            config['n_layers'],
            config['stats']['note']['max'],
        )
    elif config['model_type'] == 'CNN':
        config_cnn(config)
        model = AMTCNN(
            config['window_size'],
            config['n_filters'],
            config['kernel_size'],
            config['n_res_layers'],
            config['n_head_layers'],
            config['stats']['note']['max'],
        )


    config['model'] = model
    config['optimizer'] = optim.Adam(
        model.parameters(),
        lr=config['lr']
    )


def print_infos(config: dict):
    """Show the config file informations.
    Also print the model summary.
    """
    ignore = {'model', 'optimizer', 'train_loader', 'test_loader'}
    tabsize = 30

    print(f'{2 * tabsize * "-"} Notes Predictions {2 * tabsize * "-"}\n\n')

    for param, value in config.items():
        if param in ignore:
            continue

        param_exp = f'[{param}]\t'.expandtabs(tabsize)
        print(f'     {param_exp}-\t\t{value}')

    print('')

    summary(
        config['model'],
        input_size=(
            config['batch_size'] * config['n_samples_by_item'],
            config['window_size'],
        ),
        depth=2,
    )


if __name__ == '__main__':
    available_types = ['MLP', 'CNN']
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} [model_type]')
        print('Available model types:', *available_types)
        sys.exit(0)

    model_type = sys.argv[1]
    config = init_config(model_type)
    load_config(config)
    print_infos(config)

    continue_training = input('\n\nContinue? [y/n] ')
    if continue_training != 'y':
        sys.exit(0)

    with wandb.init(
        project='Automatic Music Transcription',
        entity='pierrotlc',
        group=config['group'],
        config=config,
    ):
        train(config['model'], config)
