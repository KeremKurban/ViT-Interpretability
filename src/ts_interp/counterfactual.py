"""Counterfactual explanations for time-series classification.

Where attribution answers "which timesteps did the model weigh?", a
counterfactual answers "what is the smallest change to this signal that would
have changed the decision?" — much closer to the question a clinician actually
asks, and a step toward causal reasoning rather than correlational saliency.

Implemented here is a Native-Guide / NUN-CF style method (Delaney et al.,
*Instance-based Counterfactual Explanations for Time Series Classification*,
2021): find the nearest training instance of a *different* class (the "nearest
unlike neighbour"), then splice the shortest contiguous segment from it into
the query that flips the prediction.

The appeal is that the counterfactual stays on the data manifold — every value
in it came from a real signal of the target class, so it is a plausible
trace rather than an adversarial perturbation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Counterfactual:
    x_cf: np.ndarray  # (n_channels, length) the counterfactual signal
    segment: tuple[int, int]  # (start, end) of the spliced region
    nun_index: int  # index into the reference set of the donor instance
    original_class: int
    cf_class: int
    n_changed: int  # length of the spliced segment
    success: bool


def nearest_unlike_neighbour(
    x: np.ndarray, x_ref: np.ndarray, y_ref: np.ndarray, exclude_class: int
) -> int:
    """Index of the closest instance in ``x_ref`` whose label != ``exclude_class``.

    Distance is plain Euclidean over the flattened signal. For signals that are
    phase-shifted relative to one another, DTW would be the better metric — this
    stays Euclidean for simplicity and speed.
    """
    candidates = np.where(y_ref != exclude_class)[0]
    if len(candidates) == 0:
        raise ValueError("no instances of a different class in the reference set")
    dists = np.linalg.norm(
        x_ref[candidates].reshape(len(candidates), -1) - x.reshape(1, -1), axis=1
    )
    return int(candidates[int(np.argmin(dists))])


@torch.no_grad()
def _predict(model: torch.nn.Module, x: np.ndarray) -> tuple[int, float]:
    logits = model(torch.from_numpy(x).unsqueeze(0))
    probs = F.softmax(logits, dim=-1)[0]
    cls = int(probs.argmax())
    return cls, float(probs[cls])


def nun_counterfactual(
    model: torch.nn.Module,
    x: np.ndarray,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    target_class: int | None = None,
    min_len: int = 4,
    step: int = 4,
) -> Counterfactual:
    """Find the shortest contiguous splice from the NUN that flips the prediction.

    Searches segment lengths from ``min_len`` upward; for each length it tries
    every start position and returns the first splice that changes the predicted
    class. Because lengths are tried in increasing order, the first hit is the
    shortest one found on this grid (subject to ``step``).
    """
    model.eval()
    orig_class, _ = _predict(model, x)
    nun_idx = nearest_unlike_neighbour(x, x_ref, y_ref, exclude_class=orig_class)
    donor = x_ref[nun_idx]

    length = x.shape[-1]

    for seg_len in range(min_len, length + 1, step):
        for start in range(0, length - seg_len + 1, step):
            end = start + seg_len
            candidate = x.copy()
            candidate[:, start:end] = donor[:, start:end]
            cls, _ = _predict(model, candidate)

            if cls != orig_class and (target_class is None or cls == target_class):
                return Counterfactual(
                    x_cf=candidate,
                    segment=(start, end),
                    nun_index=nun_idx,
                    original_class=orig_class,
                    cf_class=cls,
                    n_changed=seg_len,
                    success=True,
                )

    # nothing flipped it — return the full-donor substitution as a failure record
    return Counterfactual(
        x_cf=donor.copy(),
        segment=(0, length),
        nun_index=nun_idx,
        original_class=orig_class,
        cf_class=_predict(model, donor)[0],
        n_changed=length,
        success=False,
    )
