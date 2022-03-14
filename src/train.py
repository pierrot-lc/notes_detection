"""Training loop.
"""
from collections import defaultdict

import wandb
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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
    samples = samples.to(device)
    labels = labels.to(device).flatten()
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.ones(len(labels)) * config['pos_weight'],
    ).to(device)
    metrics = dict()

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

    model.to(device)
    for _ in tqdm(range(epochs)):
        model.train()
        for samples, labels in train_loader:
            optimizer.zero_grad()

            metrics = loss_batch(model, samples, labels, config)
            loss = metrics['loss']

            loss.backward()
            optimizer.step()

        metrics_train = eval_loader(model, train_loader, config)
        metrics_test = eval_loader(model, test_loader, config)

        logs = dict()
        for group, metrics in zip(['Train', 'Test'], [metrics_train, metrics_test]):
            for name, value in metrics.items():
                logs[f'{group} - {name}'] = value

        wandb.log(logs)
