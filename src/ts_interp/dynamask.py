"""Dynamask: a learned, smooth perturbation mask over time.

Reference: Crabbé & van der Schaar, *Explaining Time Series Predictions with
Dynamic Masks* (ICML 2021), https://arxiv.org/abs/2106.05303

Occlusion asks "what happens if I blank out this fixed rectangular window?",
which bakes in two assumptions: that the important region is contiguous, and
that it is exactly ``window`` samples wide. Dynamask drops both. It optimises a
mask ``M in [0,1]^(C x T)`` directly by gradient descent, where the perturbed
input is

    x_perturbed = M * x + (1 - M) * baseline

and the objective balances three terms:

1. **Fidelity** — keep the model's prediction for the target class high when
   the *kept* part of the signal is shown. The mask must retain what matters.
2. **Mask size** — an L1 penalty pushing the mask toward zero, so it keeps only
   what it must. This is what makes the result sparse and readable.
3. **Temporal smoothness** — a total-variation penalty on the time axis, which
   encodes the prior that physiological events are contiguous in time rather
   than a scatter of isolated samples.

The result is a soft, per-timestep, per-channel importance map with no window
size to choose.

This is a compact reimplementation for teaching purposes, not the authors'
reference code — it keeps the core objective and drops the extras (their
adaptive area-constraint and several perturbation operators). For real work,
use the published implementation or ``tsinterpret``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_dynamask(
    model: torch.nn.Module,
    x: np.ndarray,
    target: int,
    baseline: str = "zero",
    n_epochs: int = 300,
    lr: float = 0.05,
    size_reg: float = 1.0,
    tv_reg: float = 1.0,
    seed: int = 0,
    return_history: bool = False,
):
    """Learn a saliency mask for one sample.

    x: ``(n_channels, length)``. Returns a mask of the same shape in [0, 1],
    where higher = more important to keep. If ``return_history`` is set, also
    returns the per-epoch loss components for plotting.

    ``baseline`` selects what the masked-out regions are replaced with:
    ``"zero"`` blanks them, ``"mean"`` uses each channel's mean (a less
    out-of-distribution perturbation, usually the better default for signals
    that are not zero-centred).
    """
    torch.manual_seed(seed)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    x_t = torch.from_numpy(x).unsqueeze(0)  # (1, C, T)

    if baseline == "zero":
        base = torch.zeros_like(x_t)
    elif baseline == "mean":
        base = x_t.mean(dim=-1, keepdim=True).expand_as(x_t).clone()
    else:
        raise ValueError(f"unknown baseline: {baseline}")

    # optimise in logit space so the mask stays in [0, 1] without clamping
    mask_logits = torch.zeros_like(x_t, requires_grad=True)
    opt = torch.optim.Adam([mask_logits], lr=lr)

    history = {"fidelity": [], "size": [], "tv": []}

    for _ in range(n_epochs):
        opt.zero_grad()
        mask = torch.sigmoid(mask_logits)

        perturbed = mask * x_t + (1 - mask) * base
        logits = model(perturbed)
        log_probs = F.log_softmax(logits, dim=-1)

        # 1. fidelity: keep the target class probable under the kept signal
        fidelity = -log_probs[0, target]
        # 2. mask size: prefer small masks
        size = mask.mean()
        # 3. temporal smoothness: penalise change along the time axis
        tv = (mask[..., 1:] - mask[..., :-1]).abs().mean()

        loss = fidelity + size_reg * size + tv_reg * tv
        loss.backward()
        opt.step()

        history["fidelity"].append(fidelity.item())
        history["size"].append(size.item())
        history["tv"].append(tv.item())

    for p in model.parameters():
        p.requires_grad_(True)

    mask = torch.sigmoid(mask_logits).detach().squeeze(0).numpy()
    return (mask, history) if return_history else mask
