"""A small 1D CNN classifier — the "model under interpretation" for the
time-series notebooks. Deliberately simple so training is fast and the
learned filters stay easy to reason about.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TSClassifier(nn.Module):
    def __init__(self, n_channels: int = 1, n_classes: int = 2, hidden: int = 16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, hidden, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=9, padding=4),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.pool(h).squeeze(-1)
        return self.classifier(h)


def train(
    model: TSClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 0,
) -> list[float]:
    torch.manual_seed(seed)
    x_t = torch.from_numpy(x_train)
    y_t = torch.from_numpy(y_train)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    n = len(y_t)
    losses = []
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            logits = model(x_t[idx])
            loss = loss_fn(logits, y_t[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        losses.append(epoch_loss / n)
    return losses


@torch.no_grad()
def accuracy(model: TSClassifier, x: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    logits = model(torch.from_numpy(x))
    preds = logits.argmax(dim=-1).numpy()
    return float((preds == y).mean())
