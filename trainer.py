"""Train a model based on config files.
"""
import sys
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary

from src.train import train, load_checkpoint
from src.data import load, get_stats, AMTDataset
from src.mlp import AMTMLP
from src.cnn import AMTCNN


def init_config(model_type: str) -> dict:
    config = {
        'group': 'Pitch prediction',
        'piano_only': True,

        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'reload_checkpoint': False,

        'window_size': 16384, #2048, #16384, # 4096
        'sampling_rate': 11000,
        'convert_rate': 10,
        'convert_seconds': 5,
        'positive_threshold': 0.7,

        'epochs': 50,
        'batch_size': 5,
        'n_windows': 2,
        'lr': 1e-4,
        'pos_weight': 5,
        'model_type': model_type,
    }

    return config


def config_mlp(config: dict) -> nn.Module:
    """Config for MLP model.
    """
    config['hidden_size'] = 500
    config['n_layers'] = 10

    return AMTMLP(
        config['window_size'] // 2,
        config['hidden_size'],
        config['n_layers'],
        config['stats']['note']['max'],
    )


def config_cnn(config: dict) -> nn.Module:
    """Config for CNN model.
    """
    config['kernel_size'] = 1024
    config['stride'] = 6
    config['n_filters'] = 10
    config['n_layers'] = 3

    return AMTCNN(
        config['kernel_size'],
        config['stride'],
        config['n_filters'],
        config['n_layers'],
        config['stats']['note']['max'],
    )


def load_config(config: dict):
    train_dataset = load(
        'data/musicnet/',
        train=True,
        piano_only=config['piano_only'],
    )
    config['stats'] = get_stats(train_dataset['labels'])
    train_dataset = AMTDataset(
        train_dataset['id'],
        train_dataset['wav_path'],
        train_dataset['labels'],
        config['window_size'],
        config['sampling_rate'],
        config['stats']['note']['max'],
        config['stats']['instrument']['max'],
        config['n_windows'],
    )

    config['train_loader'] = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=1,
        collate_fn=AMTDataset.collate_fn,
    )

    test_dataset = load(
        'data/musicnet/',
        train=False,
        piano_only=config['piano_only']
    )
    test_dataset = AMTDataset(
        test_dataset['id'],
        test_dataset['wav_path'],
        test_dataset['labels'],
        config['window_size'],
        config['sampling_rate'],
        config['stats']['note']['max'],
        config['stats']['instrument']['max'],
        config['n_windows'],
    )

    config['test_loader'] = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=1,
        collate_fn=AMTDataset.collate_fn,
    )

    model_map = {
        'MLP': config_mlp,
        'CNN': config_cnn,
    }
    model = model_map[config['model_type']](config)
    config['model'] = model

    config['optimizer'] = optim.Adam(
        model.parameters(),
        lr=config['lr']
    )

    config['starting_epoch'] = 1
    if config['reload_checkpoint']:
        config['starting_epoch'] = load_checkpoint(config['model'], config)


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


    print(f'\n\n{2 * tabsize * "-"} {config["model_type"]} {2 * tabsize * "-"}\n')

    summary(
        config['model'],
        input_size=(
            config['batch_size'] * config['n_windows'],
            config['train_loader'].dataset.window_size,
        ),
        depth=2,
    )

    print(f'\nTrain songs: {len(config["train_loader"].dataset):,}')
    print(f'Test songs: {len(config["test_loader"].dataset):,}')


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
