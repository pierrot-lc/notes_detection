"""Train a model based on config files.
"""
import sys
import wandb
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary

import src.mlp as mlp
import src.cnn as cnn
import src.resnet as resnet
from src.train import train, load_checkpoint
from src.data import load, get_stats, AMTDataset


def load_datasets(config: dict):
    """Load the dataset objects.
    Store everything in the `config['dataset']` dictionnary.
    """
    train_dataset = load(
        'data/musicnet/',
        train=True,
        piano_only=config['dataset']['piano_only'],
    )
    config['dataset']['stats'] = get_stats(train_dataset['labels'])
    train_dataset = AMTDataset(
        train_dataset['id'],
        train_dataset['wav_path'],
        train_dataset['labels'],
        config['dataset']['window_size'],
        config['dataset']['sampling_rate'],
        config['dataset']['stats']['note']['max'],
        config['dataset']['stats']['instrument']['max'],
        config['dataset']['n_windows'],
    )

    config['dataset']['train_loader'] = DataLoader(
        train_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=AMTDataset.collate_fn,
    )

    test_dataset = load(
        'data/musicnet/',
        train=False,
        piano_only=config['dataset']['piano_only']
    )
    test_dataset = AMTDataset(
        test_dataset['id'],
        test_dataset['wav_path'],
        test_dataset['labels'],
        config['dataset']['window_size'],
        config['dataset']['sampling_rate'],
        config['dataset']['stats']['note']['max'],
        config['dataset']['stats']['instrument']['max'],
        config['dataset']['n_windows'],
    )

    config['dataset']['test_loader'] = DataLoader(
        test_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=1,
        collate_fn=AMTDataset.collate_fn,
    )


def load_config(config_path: str, reload_checkpoint: bool):
    """Load the yaml config file and prepare all the
    necessaries objects for the training.

    Everything is stored inside the config dictionnary.
    """
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['training']['reload_checkpoint'] = reload_checkpoint

    load_datasets(config)
    model_map = {
        'MLP': mlp.from_config,
        'Unet': cnn.from_config,
        'ResNet': resnet.from_config,
    }
    model = model_map[config['model_params']['model_type']](config)
    config['model'] = model

    config['training']['optimizer'] = optim.Adam(
        model.parameters(),
        lr=config['training']['lr']
    )

    config['training']['starting_epoch'] = 1
    if config['training']['reload_checkpoint']:
        config['training']['starting_epoch'] = load_checkpoint(config['model'], config)

    return config


def print_infos(config: dict):
    """Show the config file informations.
    Also print the model summary.
    """

    def print_dict(my_dict: dict, front_space: int, ignore: set):
        for param, value in my_dict.items():
            if param in ignore:
                continue

            param_exp = f'{" " * front_space}{param}'
            if type(value) is dict:
                print(f'{param_exp}:')
                print_dict(value, front_space + 4, ignore)
            else:
                print(f'{param_exp:<30}-\t\t{value}')


    ignore = {'model', 'optimizer', 'train_loader', 'test_loader'}

    print(f'{60 * "-"} Notes Predictions {60 * "-"}\n\n')
    print_dict(config, 0, ignore)
    print(f'\n\n{60 * "-"} {config["model_params"]["model_type"]} {60 * "-"}\n')

    summary(
        config['model'],
        input_size=(
            config['dataset']['batch_size'] * config['dataset']['n_windows'],
            config['dataset']['train_loader'].dataset.window_size,
        ),
        depth=2,
    )

    print(f'\nTrain songs: {len(config["dataset"]["train_loader"].dataset):,}')
    print(f'Test songs: {len(config["dataset"]["test_loader"].dataset):,}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} [config_file]')
        sys.exit(0)

    config_path = sys.argv[1]
    reload_checkpoint = False
    config = load_config(config_path, reload_checkpoint)
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
