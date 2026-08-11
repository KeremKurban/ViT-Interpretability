"""Two feature-attribution methods for the 1D classifier, plus a metric to
score them against the known injected event window.

- occlusion_1d: slide a window over the signal, replace it with a baseline
  value, and measure how much the target-class probability drops. This is
  the naive perturbation baseline referenced in the tutorial notebook.
- integrated_gradients_1d: Captum's IntegratedGradients applied directly to
  the (batch, channel, time) tensor — the same API used for images, just
  with a 1D input.

Both return a per-timestep importance vector so they can be plotted and
compared on equal footing.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients


@torch.no_grad()
def occlusion_1d(
    model: torch.nn.Module,
    x: np.ndarray,
    target: int,
    window: int = 8,
    stride: int = 4,
    baseline_value: float = 0.0,
) -> np.ndarray:
    """x: (1, length) single-channel signal. Returns (length,) attribution."""
    model.eval()
    x_t = torch.from_numpy(x).unsqueeze(0)  # (1, 1, length)
    length = x_t.shape[-1]

    base_prob = F.softmax(model(x_t), dim=-1)[0, target].item()

    attr = np.zeros(length, dtype=np.float32)
    counts = np.zeros(length, dtype=np.float32)

    for start in range(0, length - window + 1, stride):
        end = start + window
        occluded = x_t.clone()
        occluded[..., start:end] = baseline_value
        prob = F.softmax(model(occluded), dim=-1)[0, target].item()
        drop = base_prob - prob  # positive => that window mattered
        attr[start:end] += drop
        counts[start:end] += 1

    counts[counts == 0] = 1
    return attr / counts


def integrated_gradients_1d(
    model: torch.nn.Module,
    x: np.ndarray,
    target: int,
    baseline_value: float = 0.0,
    steps: int = 64,
) -> np.ndarray:
    """x: (1, length) single-channel signal. Returns (length,) attribution."""
    model.eval()
    x_t = torch.from_numpy(x).unsqueeze(0).requires_grad_()
    baseline = torch.full_like(x_t, baseline_value)

    ig = IntegratedGradients(model)
    attributions = ig.attribute(x_t, baseline, target=target, n_steps=steps)
    return attributions.squeeze(0).squeeze(0).detach().numpy()


def top_k_overlap(attr: np.ndarray, event_window: tuple[int, int], top_frac: float = 0.2) -> float:
    """IoU between the top-`top_frac` attributed timesteps and the true event
    window — a proxy for 'did the explanation point at the right thing'.
    """
    start, end = event_window
    if start < 0:
        return float("nan")  # sample has no injected event

    length = len(attr)
    k = max(1, int(length * top_frac))
    top_idx = set(np.argsort(attr)[-k:].tolist())
    true_idx = set(range(start, end))

    intersection = len(top_idx & true_idx)
    union = len(top_idx | true_idx)
    return intersection / union if union else 0.0
