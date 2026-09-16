"""Feature-attribution methods for the 1D classifier, plus metrics that score
them against the known injected event window and channel.

Core methods, all operating on a single sample of shape ``(n_channels, length)``
and returning an attribution of the same shape:

- ``occlusion`` — slide a window over time, replace it with a baseline value,
  measure the drop in target-class probability. The naive perturbation baseline.
- ``integrated_gradients`` — Captum's ``IntegratedGradients`` applied directly
  to the ``(batch, channel, time)`` tensor. Same API as for images.
- ``temporal_saliency_rescaling`` — the TSR correction of Ismail et al.
  (NeurIPS 2020), which separates *when* something mattered from *which
  channel* mattered before recombining them.

``occlusion_1d`` / ``integrated_gradients_1d`` are single-channel convenience
wrappers returning a flat ``(length,)`` vector.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients


# --------------------------------------------------------------------------
# Base attribution methods
# --------------------------------------------------------------------------


@torch.no_grad()
def occlusion(
    model: torch.nn.Module,
    x: np.ndarray,
    target: int,
    window: int = 8,
    stride: int = 4,
    baseline_value: float = 0.0,
    per_channel: bool = True,
) -> np.ndarray:
    """Sliding-window occlusion.

    x: ``(n_channels, length)``. Returns attribution of the same shape.

    With ``per_channel=True`` each channel is occluded independently, so the
    result distinguishes which channel a time window mattered in. With
    ``per_channel=False`` all channels are occluded together and every channel
    receives the same score (cheaper, but channel-blind).
    """
    model.eval()
    x_t = torch.from_numpy(x).unsqueeze(0)  # (1, C, T)
    n_channels, length = x.shape

    base_prob = F.softmax(model(x_t), dim=-1)[0, target].item()

    attr = np.zeros((n_channels, length), dtype=np.float32)
    counts = np.zeros((n_channels, length), dtype=np.float32)

    channel_sets = (
        [[c] for c in range(n_channels)] if per_channel else [list(range(n_channels))]
    )

    for channels in channel_sets:
        for start in range(0, length - window + 1, stride):
            end = start + window
            occluded = x_t.clone()
            occluded[0, channels, start:end] = baseline_value
            prob = F.softmax(model(occluded), dim=-1)[0, target].item()
            drop = base_prob - prob  # positive => that window mattered
            attr[channels, start:end] += drop
            counts[channels, start:end] += 1

    counts[counts == 0] = 1
    return attr / counts


def integrated_gradients(
    model: torch.nn.Module,
    x: np.ndarray,
    target: int,
    baseline_value: float = 0.0,
    steps: int = 64,
) -> np.ndarray:
    """Integrated Gradients. x: ``(n_channels, length)`` -> same shape."""
    model.eval()
    x_t = torch.from_numpy(x).unsqueeze(0).requires_grad_()
    baseline = torch.full_like(x_t, baseline_value)

    ig = IntegratedGradients(model)
    attributions = ig.attribute(x_t, baseline, target=target, n_steps=steps)
    return attributions.squeeze(0).detach().numpy()


def occlusion_1d(model, x: np.ndarray, target: int, **kwargs) -> np.ndarray:
    """Single-channel wrapper: ``(1, length)`` -> flat ``(length,)``."""
    return occlusion(model, x, target, **kwargs)[0]


def integrated_gradients_1d(model, x: np.ndarray, target: int, **kwargs) -> np.ndarray:
    """Single-channel wrapper: ``(1, length)`` -> flat ``(length,)``."""
    return integrated_gradients(model, x, target, **kwargs)[0]


# --------------------------------------------------------------------------
# Temporal Saliency Rescaling (Ismail et al., NeurIPS 2020)
# --------------------------------------------------------------------------


def temporal_saliency_rescaling(
    model: torch.nn.Module,
    x: np.ndarray,
    target: int,
    base_method: str = "integrated_gradients",
    time_stride: int = 8,
    threshold_percentile: float = 50.0,
    baseline_value: float = 0.0,
    ig_steps: int = 16,
) -> np.ndarray:
    """TSR: rescale a base saliency map by separating time from feature relevance.

    The algorithm (paper Algorithm 1), per sample:

    1. **Time relevance** ``dt`` for each time block: mask *all channels* in
       that block, recompute the base saliency, and take the total absolute
       change. Large change => that time block matters.
    2. **Feature relevance** ``d_ct``: only for time blocks that clear a
       threshold, mask *one channel* in the block and again measure the
       saliency change. This asks "given this moment matters, which channel
       carries it?"
    3. Final map is the product ``dt * d_ct``.

    Why it exists: raw saliency conflates the two axes, so a genuinely
    important value can score low merely because of *when* it occurs. The
    product form forces a point to be important on both axes to survive.

    Cost note: this is O(T/stride x C) full saliency recomputations, which is
    why ``time_stride`` and a low ``ig_steps`` matter. Exact per-timestep TSR
    on a long signal is expensive — that is a real property of the method, not
    a shortcut in this implementation.
    """
    n_channels, length = x.shape

    def base_saliency(sample: np.ndarray) -> np.ndarray:
        if base_method == "integrated_gradients":
            return integrated_gradients(
                model, sample, target, baseline_value=baseline_value, steps=ig_steps
            )
        if base_method == "occlusion":
            return occlusion(model, sample, target, baseline_value=baseline_value)
        raise ValueError(f"unknown base_method: {base_method}")

    reference = base_saliency(x)

    blocks = [(s, min(s + time_stride, length)) for s in range(0, length, time_stride)]

    # --- step 1: time relevance -------------------------------------------
    time_relevance = np.zeros(len(blocks), dtype=np.float32)
    for bi, (s, e) in enumerate(blocks):
        masked = x.copy()
        masked[:, s:e] = baseline_value
        time_relevance[bi] = np.abs(reference - base_saliency(masked)).sum()

    # --- step 2: feature relevance, only where time relevance clears bar ---
    threshold = np.percentile(time_relevance, threshold_percentile)
    feature_relevance = np.zeros((n_channels, len(blocks)), dtype=np.float32)

    for bi, (s, e) in enumerate(blocks):
        if time_relevance[bi] <= threshold:
            continue  # below threshold => contributes zero, skip the compute
        for c in range(n_channels):
            masked = x.copy()
            masked[c, s:e] = baseline_value
            feature_relevance[c, bi] = np.abs(reference - base_saliency(masked)).sum()

    # --- step 3: combine and expand blocks back to per-timestep -----------
    combined = time_relevance[None, :] * feature_relevance  # (C, n_blocks)

    attr = np.zeros((n_channels, length), dtype=np.float32)
    for bi, (s, e) in enumerate(blocks):
        attr[:, s:e] = combined[:, bi : bi + 1]
    return attr


# --------------------------------------------------------------------------
# Scoring against ground truth
# --------------------------------------------------------------------------


def top_k_overlap(
    attr: np.ndarray,
    event_window: tuple[int, int],
    top_frac: float | None = None,
) -> float:
    """IoU between the top-attributed timesteps and the true event window —
    a proxy for "did the explanation point at the right time".

    By default ``k`` is set to the **true event length**, which keeps the score
    comparable across events of different durations. A fixed ``top_frac``
    silently caps the achievable IoU whenever the event is longer than
    ``top_frac * length`` (select 38 timesteps against a 128-sample event and
    the best possible score is 38/128), so pass ``top_frac`` only when
    comparing events of equal length.

    Accepts either a flat ``(length,)`` map or a ``(n_channels, length)`` map;
    multi-channel maps are collapsed over channels first, since this metric
    scores the time axis only.
    """
    start, end = event_window
    if start < 0:
        return float("nan")  # sample has no injected event

    if attr.ndim == 2:
        attr = np.abs(attr).max(axis=0)

    length = len(attr)
    k = (end - start) if top_frac is None else max(1, int(length * top_frac))
    k = int(np.clip(k, 1, length))
    top_idx = set(np.argsort(attr)[-k:].tolist())
    true_idx = set(range(start, end))

    intersection = len(top_idx & true_idx)
    union = len(top_idx | true_idx)
    return intersection / union if union else 0.0


def channel_hit(attr: np.ndarray, event_channel: int) -> float:
    """Did the attribution put the most mass on the channel that carried the event?

    Returns 1.0 / 0.0, or NaN when the sample has no event. This is the second
    axis TSR is meant to fix — a method can localize the right *moment* while
    crediting the wrong *sensor*.
    """
    if event_channel < 0:
        return float("nan")
    mass = np.abs(attr).sum(axis=1)
    return float(int(np.argmax(mass)) == int(event_channel))
